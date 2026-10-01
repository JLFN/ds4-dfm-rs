//! The image engine's configuration surface (P0): argv in, a resolved plan out.
//!
//! The engine is a sibling of the text families and has no serving surface yet,
//! so `--check-config` is its only entry point: it identifies the two artifacts,
//! resolves the placement and refuses every autoregressive control by name.
//! Keeping it here (not in the binary) is what lets the refusal set be tested
//! at argv level without a linked native binary.

use std::path::Path;

use ds4_core::qwen_image::{
    ar_control, identify_dit, identify_vae, resolve_image_plan, ImageDevice, ImageIssue,
    ImageLevel, ImageModule, ImageRequest, ModulePlacement, ParamTier,
};

/// The outcome of a check: what to print, and the process's exit code.
#[derive(Clone, Debug)]
pub struct ImageCheck {
    pub report: String,
    pub json: String,
    pub exit_code: i32,
    /// Diagnostics the caller prints before the report, one per line.
    pub stderr: Vec<String>,
}

pub fn parse_module(name: &str) -> Option<ImageModule> {
    ImageModule::ALL.iter().copied().find(|m| m.name() == name)
}

pub fn parse_device(name: &str) -> Option<ImageDevice> {
    if name == "cpu" {
        return Some(ImageDevice::Cpu);
    }
    name.strip_prefix("cuda").and_then(|i| i.parse().ok()).map(ImageDevice::Cuda)
}

pub fn parse_tier(name: &str) -> Option<ParamTier> {
    match name {
        "vram" => Some(ParamTier::Vram),
        "host" => Some(ParamTier::HostRam),
        "disk" => Some(ParamTier::Disk),
        _ => None,
    }
}

/// `--max-vram <GiB>`: the per-device budget for managed weights and runner
/// buffers, in the reference's own unit, returned in MiB.
pub fn parse_max_vram_gib(value: &str) -> Result<u64, String> {
    let gib: u64 = value
        .parse()
        .map_err(|_| format!("--max-vram {value}: expected a whole number of GiB"))?;
    if gib == 0 {
        return Err("--max-vram 0: a zero budget pins nothing".into());
    }
    Ok(gib * 1024)
}

/// `--image-placement te=cpu:host,diffusion=cuda0:vram,vae=cpu:host`
pub fn parse_placement(spec: &str) -> Result<Vec<ModulePlacement>, String> {
    let mut out = Vec::new();
    for item in spec.split(',') {
        let (module, rest) = item
            .split_once('=')
            .ok_or_else(|| format!("--image-placement {item}: expected module=device:tier"))?;
        let (device, tier) = rest
            .split_once(':')
            .ok_or_else(|| format!("--image-placement {item}: expected device:tier"))?;
        let module =
            parse_module(module).ok_or_else(|| format!("--image-placement: unknown module {module}"))?;
        let device =
            parse_device(device).ok_or_else(|| format!("--image-placement: unknown device {device}"))?;
        let tier = parse_tier(tier).ok_or_else(|| format!("--image-placement: unknown tier {tier}"))?;
        out.push(ModulePlacement { module, device, tier });
    }
    Ok(out)
}

/// Refuses the run itself: the engine cannot listen before P5.
fn refuse(message: String) -> ImageCheck {
    ImageCheck {
        report: String::new(),
        json: String::new(),
        exit_code: 2,
        stderr: vec![message],
    }
}

/// Runs the image configuration check. `argv` is the full argument list after
/// the program name, so every autoregressive control the caller passed is seen
/// and refused by name.
pub fn check_image(
    dit: Option<&str>,
    vae: Option<&str>,
    offload: bool,
    placement: &[ModulePlacement],
    vram_budget_mib: Option<u64>,
    check_config: bool,
    argv: &[String],
) -> ImageCheck {
    if !check_config {
        return refuse("the image engine has no serving surface yet (P5); use --check-config".into());
    }
    let Some(dit) = dit else {
        return refuse("--image-vae requires --image-dit".into());
    };

    let mut ids = Vec::new();
    let mut stderr = Vec::new();
    let mut artifact_error = None;
    match identify_dit(Path::new(dit)) {
        Ok(id) => ids.push(id),
        Err(error) => {
            stderr.push(format!("--image-dit {dit}: {error}"));
            artifact_error = Some(format!("--image-dit {dit}: {error}"));
        }
    }
    if let Some(path) = vae {
        match identify_vae(Path::new(path)) {
            Ok(id) => ids.push(id),
            Err(error) => {
                stderr.push(format!("--image-vae {path}: {error}"));
                artifact_error = Some(format!("--image-vae {path}: {error}"));
            }
        }
    }

    let controls: Vec<String> = argv
        .iter()
        .filter(|a| ar_control(a).is_some())
        .cloned()
        .collect();
    let req = ImageRequest {
        ar_controls: controls,
        placement: placement.to_vec(),
        offload,
        vram_budget_mib,
    };
    let mut plan = resolve_image_plan(&req, &ids);
    if let Some(message) = artifact_error {
        plan.issues.push(ImageIssue {
            level: ImageLevel::Error,
            code: "image_artifact_invalid",
            message,
        });
    }
    if vae.is_none() {
        plan.issues.push(ImageIssue {
            level: ImageLevel::Warn,
            code: "image_vae_absent",
            message: "no --image-vae: the decode path was not validated".into(),
        });
    }

    ImageCheck {
        report: plan.report(),
        json: plan.to_json(),
        exit_code: if plan.may_load() { 0 } else { 2 },
        stderr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_spec_parses_and_refuses_unknown_parts() {
        let ok = parse_placement("te=cpu:host,diffusion=cuda0:vram,vae=cpu:disk").unwrap();
        assert_eq!(ok.len(), 3);
        assert_eq!(ok[1].device, ImageDevice::Cuda(0));
        assert_eq!(ok[1].tier, ParamTier::Vram);
        assert_eq!(ok[2].tier, ParamTier::Disk);
        assert!(parse_placement("te=gpu:host").is_err());
        assert!(parse_placement("te=cpu:ram").is_err());
        assert!(parse_placement("te=cpu").is_err());
    }

    #[test]
    fn image_flags_require_check_config() {
        let out = check_image(Some("x.gguf"), None, false, &[], None, false, &[]);
        assert_eq!(out.exit_code, 2);
        assert!(out.stderr[0].contains("--check-config"));
    }

    /// The budget arrives in GiB on the command line, as the reference's own
    /// flag does, and is carried in MiB.
    #[test]
    fn max_vram_parses_gib_and_refuses_zero_or_junk() {
        assert_eq!(parse_max_vram_gib("140").unwrap(), 140 * 1024);
        assert_eq!(parse_max_vram_gib("12").unwrap(), 12288);
        assert!(parse_max_vram_gib("0").is_err());
        assert!(parse_max_vram_gib("12.5").is_err());
        assert!(parse_max_vram_gib("twelve").is_err());
    }
}
