//! P0 gate for the image engine: the `--check-config` surface, at argv level.
//!
//! Model-free where it can be: a one-tensor GGUF is enough to exercise the
//! refusal path, because every autoregressive control must be refused by name
//! whatever the artifacts are. The full run over both real artifacts happens
//! when their paths are provided:
//!
//!     DS4_QWEN_IMAGE_DIT=/path/dit.gguf DS4_QWEN_IMAGE_VAE=/path/vae.gguf \
//!         cargo test -p ds4-server --test image_cli

use std::path::{Path, PathBuf};

use ds4_server::image_cli::{check_image, parse_placement};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ds4-qwen-image-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A minimal GGUF v3 with a single BF16 tensor, aligned the way the reader
/// expects, so identification runs and refuses on content.
fn write_tiny_gguf(path: &Path, name: &str, dims: &[u64]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
    f.write_all(name.as_bytes()).unwrap();
    f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
    for d in dims {
        f.write_all(&d.to_le_bytes()).unwrap();
    }
    f.write_all(&30u32.to_le_bytes()).unwrap(); // BF16
    f.write_all(&0u64.to_le_bytes()).unwrap();
    let dir_end = f.metadata().unwrap().len();
    for _ in 0..(32 - dir_end % 32) % 32 {
        f.write_all(&[0u8]).unwrap();
    }
    let bytes: u64 = dims.iter().product::<u64>() * 2;
    f.write_all(&vec![0u8; bytes as usize]).unwrap();
}

const AR_ARGV: [&str; 17] = [
    "--check-config",
    "--ctx",
    "8192",
    "--max-seqs",
    "4",
    "--prefix-reuse",
    "exact",
    "--mtp-mode",
    "on",
    "--mtp",
    "/tmp/mtp.gguf",
    "--mtp-draft",
    "2",
    "--kv-disk-dir",
    "/tmp/kv",
    "--cont-width",
    "8",
];

#[test]
fn every_ar_control_is_refused_by_name() {
    let dit = tmp("not-a-dit.gguf");
    write_tiny_gguf(&dit, "img_in.weight", &[64, 4096]);
    let argv: Vec<String> = AR_ARGV.iter().map(|s| s.to_string()).collect();
    let out = check_image(Some(dit.to_str().unwrap()), None, false, &[], true, &argv);
    assert_eq!(out.exit_code, 2, "a run carrying AR controls must not pass");
    for flag in [
        "--ctx",
        "--max-seqs",
        "--prefix-reuse",
        "--mtp-mode",
        "--mtp-draft",
        "--kv-disk-dir",
        "--cont-width",
    ] {
        assert!(
            out.report.contains(flag),
            "{flag} was not refused by name:\n{}",
            out.report
        );
    }
    assert!(out.report.contains("image_artifact_invalid"), "{}", out.report);
    assert!(out.json.contains("\"level\":\"error\""), "{}", out.json);
}

#[test]
fn a_check_without_artifacts_refuses_the_run() {
    let out = check_image(Some("absent.gguf"), None, false, &[], true, &[]);
    assert_eq!(out.exit_code, 2);
    assert!(out.stderr.iter().any(|l| l.contains("absent.gguf")));
}

#[test]
fn placement_refusals_come_from_the_measured_rules() {
    let dit = tmp("not-a-dit.gguf");
    write_tiny_gguf(&dit, "img_in.weight", &[64, 4096]);
    let placement = parse_placement("vae=cuda0:vram").unwrap();
    let out = check_image(Some(dit.to_str().unwrap()), None, false, &placement, true, &[]);
    assert!(out.report.contains("image_double_pin_unsupported"), "{}", out.report);
}

#[test]
fn image_artifacts_when_configured() {
    let (Ok(dit), Ok(vae)) = (
        std::env::var("DS4_QWEN_IMAGE_DIT"),
        std::env::var("DS4_QWEN_IMAGE_VAE"),
    ) else {
        return;
    };
    let out = check_image(Some(&dit), Some(&vae), false, &[], true, &["--check-config".into()]);
    assert_eq!(out.exit_code, 0, "clean check failed:\n{}\n{}", out.report, out.stderr.join("\n"));
    assert!(out.report.contains("dit_tensors=297"), "{}", out.report);
    assert!(out.report.contains("dit_q6_k=229"), "{}", out.report);
    assert!(out.report.contains("vae_tensors=134"), "{}", out.report);
    assert!(out.report.contains("diffusion=cuda0,vram"), "{}", out.report);

    // The same artifacts with one AR control must now fail by name.
    let out = check_image(
        Some(&dit),
        Some(&vae),
        false,
        &[],
        true,
        &["--check-config".into(), "--ctx".into(), "8192".into()],
    );
    assert_eq!(out.exit_code, 2);
    assert!(out.report.contains("ar_ctx_unsupported"), "{}", out.report);
}
