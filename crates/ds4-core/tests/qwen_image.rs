//! P0 layout tests for the image engine: the two artifact contracts, the
//! refusal set and the plan.
//!
//! Model-free by construction. The contract tables and the refusal set are
//! checked without any artifact; the two artifacts themselves are checked when
//! their paths are provided, in the style the other families use:
//!
//!     DS4_QWEN_IMAGE_DIT=/path/qwen-image-2.1-Q6_K.gguf \
//!     DS4_QWEN_IMAGE_VAE=/path/vae-decode-bf16.gguf \
//!         cargo test -p ds4-core --test qwen_image

use std::path::{Path, PathBuf};

use ds4_core::qwen_image::{
    ar_control, dit_contract, identify_dit, identify_vae, resolve_image_plan,
    vae_decode_contract, ArControl, ImageArtifactKind, ImageRequest, AR_CONTROLS, ENGINE,
    DIT_BF16_COUNT, DIT_LAYERS, DIT_Q6K_COUNT, DIT_TENSOR_COUNT, TYPE_BF16, TYPE_Q6_K,
    VAE_DECODE_TENSOR_COUNT,
};

/// Writes a minimal GGUF v3 with one tensor, so a wrong artifact can be
/// presented to identification without a real model.
fn write_tiny_gguf(path: &Path, name: &str, dims: &[u64], typ: u32) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("create");
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
    f.write_all(&0u64.to_le_bytes()).unwrap(); // n_kv
    f.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
    f.write_all(name.as_bytes()).unwrap();
    f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
    for d in dims {
        f.write_all(&d.to_le_bytes()).unwrap();
    }
    f.write_all(&typ.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap(); // offset
    // GGUF aligns the data section: data_pos = align_up(dir_end, 32).
    let dir_end = f.metadata().unwrap().len();
    for _ in 0..(32 - dir_end % 32) % 32 {
        f.write_all(&[0u8]).unwrap();
    }
    let elements: u64 = dims.iter().product();
    let bytes = match typ {
        TYPE_BF16 => elements * 2,
        TYPE_Q6_K => elements / 256 * 210,
        _ => elements * 2,
    };
    f.write_all(&vec![0u8; bytes as usize]).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ds4-qwen-image-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn dit_contract_pins_the_measured_quant_split() {
    let c = dit_contract();
    assert_eq!(c.len(), DIT_TENSOR_COUNT);
    assert_eq!(c.iter().filter(|t| t.typ == TYPE_Q6_K).count(), DIT_Q6K_COUNT);
    assert_eq!(c.iter().filter(|t| t.typ == TYPE_BF16).count(), DIT_BF16_COUNT);
    assert_eq!(c.iter().filter(|t| t.name.starts_with("transformer_blocks.")).count(), 9 * DIT_LAYERS as usize);
    // The refusal table is a fixed set; the plan must not grow it silently.
    assert_eq!(plan_refusals().len(), AR_CONTROLS.len());
}

fn plan_refusals() -> &'static [ArControl] {
    resolve_image_plan(&ImageRequest::default(), &[]).qualified.refusals
}

#[test]
fn vae_contract_covers_the_decode_path_only() {
    let c = vae_decode_contract();
    assert_eq!(c.len(), VAE_DECODE_TENSOR_COUNT);
    assert!(c.iter().all(|t| t.typ == TYPE_BF16));
    for t in &c {
        assert!(
            t.name.starts_with("decoder.") || t.name.starts_with("conv2."),
            "{} is not on the decode path",
            t.name
        );
    }
}

#[test]
fn every_ar_control_is_refused_by_name() {
    for c in AR_CONTROLS {
        let req = ImageRequest {
            ar_controls: vec![c.flag.to_string()],
            ..Default::default()
        };
        let plan = resolve_image_plan(&req, &[]);
        assert!(!plan.may_load(), "{} must refuse", c.flag);
        let issue = plan
            .issues
            .iter()
            .find(|i| i.code == c.code)
            .unwrap_or_else(|| panic!("{} refused without its code", c.flag));
        assert!(issue.message.contains(c.flag), "{} not named in the message", c.flag);
    }
    assert!(ar_control("--ctx=").is_some());
    assert!(ar_control("--temperature").is_none());
}

/// A GGUF whose single KV is `general.architecture`, i.e. a text model's
/// marker. The image path must refuse it before comparing tensor names.
fn write_text_gguf(path: &Path, arch: &str) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap(); // n_tensors
    f.write_all(&1u64.to_le_bytes()).unwrap(); // n_kv
    let key = "general.architecture";
    f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key.as_bytes()).unwrap();
    f.write_all(&8u32.to_le_bytes()).unwrap(); // GGUF_VALUE_STRING
    f.write_all(&(arch.len() as u64).to_le_bytes()).unwrap();
    f.write_all(arch.as_bytes()).unwrap();
    let end = f.metadata().unwrap().len();
    for _ in 0..(32 - end % 32) % 32 {
        f.write_all(&[0u8]).unwrap();
    }
}

#[test]
fn a_text_artifact_is_refused_by_name() {
    let p = tmp("a-text-model.gguf");
    write_text_gguf(&p, "deepseek4");
    let err = identify_dit(&p).expect_err("a text GGUF is not a DiT");
    assert!(err.token().contains("image-unsupported"), "{}", err.token());
    assert!(err.token().contains("text artifact"), "{}", err.token());

    let p2 = tmp("an-image-arch.gguf");
    write_text_gguf(&p2, "qwen_image_vae");
    let err = identify_vae(&p2).expect_err("an architecture key is not the pinned layout");
    assert!(err.token().contains("not one of the pinned layouts"), "{}", err.token());
}

#[test]
fn image_artifact_is_refused_a_tensor_directory_that_is_not_the_contract() {
    let p = tmp("not-a-dit.gguf");
    write_tiny_gguf(&p, "img_in.weight", &[64, 4096], TYPE_BF16);
    let err = identify_dit(&p).expect_err("a one-tensor file is not a DiT");
    let token = err.token();
    assert!(token.contains("image-contract"), "{token}");
    assert!(token.contains("missing"), "{token}");
}

#[test]
fn dit_artifact_when_configured() {
    let Ok(path) = std::env::var("DS4_QWEN_IMAGE_DIT") else {
        return;
    };
    let id = identify_dit(Path::new(&path)).expect("identify the DiT artifact");
    assert_eq!(id.kind, ImageArtifactKind::DitQ6K);
    assert_eq!(id.tensors, DIT_TENSOR_COUNT);
    assert_eq!(id.type_count(TYPE_Q6_K), DIT_Q6K_COUNT);
    assert_eq!(id.type_count(TYPE_BF16), DIT_BF16_COUNT);
    assert!(id.bytes > 5_000_000_000, "DiT weights are ~5.6 GB, got {}", id.bytes);
}

#[test]
fn vae_artifact_when_configured() {
    let Ok(path) = std::env::var("DS4_QWEN_IMAGE_VAE") else {
        return;
    };
    let id = identify_vae(Path::new(&path)).expect("identify the converted VAE");
    assert_eq!(id.kind, ImageArtifactKind::VaeDecodeBf16);
    assert_eq!(id.tensors, VAE_DECODE_TENSOR_COUNT);
    assert_eq!(id.type_count(TYPE_BF16), VAE_DECODE_TENSOR_COUNT);
}

#[test]
fn plan_reports_the_engine_and_the_placement() {
    let plan = resolve_image_plan(&ImageRequest::default(), &[]);
    assert_eq!(plan.engine, ENGINE);
    assert!(plan.may_load());
    let report = plan.report();
    assert!(report.contains("engine: qwen-image-2.1"), "{report}");
    assert!(report.contains("diffusion=cuda0,vram"), "{report}");
    assert!(report.contains("note: contract only"), "{report}");
    let json = plan.to_json();
    assert!(json.contains("\"engine\":\"qwen-image-2.1\""), "{json}");
    assert!(json.contains("\"tier\":\"host\""), "{json}");
}
