//! Named render recipes, as data the Rust binaries read.
//!
//! `tools/h3-profiles.json` holds one entry per recipe and this module is the
//! only reader: the route script asks `h3-rs --print-profile NAME` for the shell
//! values, the CLI asks `h3-rs --profile NAME` to apply one to a request, and
//! `h3-server-rs` serves the same names through the API's `profile` field. No
//! second implementation exists, so a recipe cannot mean one thing to the script
//! and another to the API.
//!
//! The file is deliberately small and explicit: a profile is a set of the
//! flags the binaries already have, and an unknown key is an error rather than
//! an ignored line, because a typo in a recipe is exactly the failure a recipe
//! file exists to prevent.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use super::dit::{VaeAttention, VaeGemm};

/// One recipe. Every field is optional except its name, so a profile states only
/// what it changes; what it omits is the binary's own default.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub name: String,
    pub doc: String,
    pub steps: Option<usize>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seconds: Option<u32>,
    pub one_pass: bool,
    pub draft_scale: Option<f32>,
    /// The draft leg's data type: `bf16` or `e4m3`, the two words the CLI takes.
    pub draft_attention: Option<String>,
    /// The refine grid: one sigma, or the reference's two-step `ts_h`.
    pub refine_sigmas: Vec<f32>,
    pub dit_gemm: Option<String>,
    pub vae_gemm: Option<VaeGemm>,
    pub vae_attention: Option<VaeAttention>,
    pub crf: Option<u32>,
    pub preset: Option<String>,
    pub prompt: Option<String>,
    pub prompt_file: Option<String>,
}

impl Profile {
    /// The draft leg this profile asks for, or `None` when it renders in one
    /// stage. A profile that states no scale is the released one.
    pub fn draft(&self) -> Option<super::engine::Draft> {
        if self.one_pass {
            return None;
        }
        // A recipe that states no route key is settings-only and renders in one
        // stage: defaulting it to the released route would silently turn a
        // canvas recipe into a two-stage render.
        if self.draft_scale.is_none() && self.refine_sigmas.is_empty() {
            return None;
        }
        let mut draft = super::engine::Draft::RELEASED;
        if let Some(scale) = self.draft_scale {
            draft.scale = scale;
        }
        if let Some(first) = self.refine_sigmas.first() {
            draft.sigma = *first;
            draft.next = self.refine_sigmas.get(1).copied();
        }
        Some(draft)
    }

    /// The `KEY=value` lines `tools/h3-rapid.sh` evaluates, in the shell's own
    /// names. This is the whole interface to the script.
    pub fn shell(&self) -> Vec<(String, String)> {
        let mut lines = Vec::new();
        let mut push = |key: &str, value: String| lines.push((key.to_string(), value));
        if let Some(steps) = self.steps {
            push("STEPS", steps.to_string());
        }
        if let Some(width) = self.width {
            push("WIDTH", width.to_string());
        }
        if let Some(height) = self.height {
            push("HEIGHT", height.to_string());
        }
        if let Some(seconds) = self.seconds {
            push("SECONDS_ARG", seconds.to_string());
        }
        if self.one_pass {
            push("ONE_PASS", "1".to_string());
        }
        if let Some(scale) = self.draft_scale {
            push("DRAFT_SCALE", scale.to_string());
        }
        if let Some(attention) = &self.draft_attention {
            push("DRAFT_ATTENTION", attention.clone());
        }
        if !self.refine_sigmas.is_empty() {
            push(
                "REFINE_SIGMA",
                self.refine_sigmas
                    .iter()
                    .map(|sigma| sigma.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        if let Some(gemm) = &self.dit_gemm {
            push("H3_DIT_GEMM", gemm.clone());
        }
        if let Some(gemm) = self.vae_gemm {
            push("H3_VAE_GEMM", vae_gemm_name(gemm).to_string());
        }
        if let Some(attention) = self.vae_attention {
            push(
                "H3_VAE_ATTENTION",
                match attention {
                    VaeAttention::Scalar => "scalar",
                    VaeAttention::Tensor => "tensor",
                }
                .to_string(),
            );
        }
        if let Some(crf) = self.crf {
            push("CRF", crf.to_string());
        }
        if let Some(preset) = &self.preset {
            push("PRESET", preset.clone());
        }
        if let Some(prompt) = &self.prompt {
            push("PROMPT", prompt.clone());
        }
        if let Some(file) = &self.prompt_file {
            push("PROMPT_FILE", file.clone());
        }
        lines
    }
}

/// The arm's word, the same one the CLI and the environment take.
pub fn vae_gemm_name(gemm: VaeGemm) -> &'static str {
    match gemm {
        VaeGemm::Fp32 => "fp32",
        VaeGemm::Reduced => "bf16",
        VaeGemm::Full => "bf16i",
        VaeGemm::Chain => "bf16a",
    }
}

fn vae_gemm(word: &str) -> Option<VaeGemm> {
    match word {
        "fp32" => Some(VaeGemm::Fp32),
        "bf16" => Some(VaeGemm::Reduced),
        "bf16i" => Some(VaeGemm::Full),
        "bf16a" => Some(VaeGemm::Chain),
        _ => None,
    }
}

fn vae_attention(word: &str) -> Option<VaeAttention> {
    match word {
        "scalar" => Some(VaeAttention::Scalar),
        "tensor" => Some(VaeAttention::Tensor),
        _ => None,
    }
}

/// Quote a value for the shell: single quotes, with embedded quotes escaped, so
/// a prompt with an apostrophe cannot break the script's `eval`.
pub fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// The recipes in one file, by name.
#[derive(Debug, Clone)]
pub struct Profiles {
    entries: BTreeMap<String, Profile>,
}

impl Profiles {
    pub fn load(path: &Path) -> Result<Profiles, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Self::parse(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Profiles, String> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| format!("the file is not JSON: {error}"))?;
        let object = root
            .get("profiles")
            .and_then(Value::as_object)
            .ok_or_else(|| "no `profiles` object".to_string())?;
        let mut entries = BTreeMap::new();
        for (name, value) in object {
            let mut profile = Profile {
                name: name.clone(),
                ..Profile::default()
            };
            let fields = value
                .as_object()
                .ok_or_else(|| format!("profile {name:?} is not an object"))?;
            for (key, field) in fields {
                match key.as_str() {
                    "doc" => profile.doc = string(field, name, key)?,
                    "steps" => profile.steps = Some(number(field, name, key)? as usize),
                    "width" => profile.width = Some(number(field, name, key)? as u32),
                    "height" => profile.height = Some(number(field, name, key)? as u32),
                    "seconds" => profile.seconds = Some(number(field, name, key)? as u32),
                    "crf" => profile.crf = Some(number(field, name, key)? as u32),
                    "one_pass" => {
                        profile.one_pass = field
                            .as_bool()
                            .ok_or_else(|| format!("profile {name:?}: {key} is not a boolean"))?
                    }
                    "draft_scale" => profile.draft_scale = Some(float(field, name, key)?),
                    "draft_attention" => {
                        let word = string(field, name, key)?;
                        if word != "bf16" && word != "e4m3" && word != "fp8" {
                            return Err(format!(
                                "profile {name:?}: draft_attention {word:?} is not bf16 or e4m3"
                            ));
                        }
                        profile.draft_attention = Some(word);
                    }
                    "refine_sigma" => profile.refine_sigmas = vec![float(field, name, key)?],
                    "refine_sigmas" => {
                        let list = field
                            .as_array()
                            .ok_or_else(|| format!("profile {name:?}: {key} is not an array"))?;
                        let mut sigmas = Vec::with_capacity(list.len());
                        for entry in list {
                            sigmas.push(entry.as_f64().map(|value| value as f32).ok_or_else(
                                || format!("profile {name:?}: {key} holds a non-number"),
                            )?);
                        }
                        // The reference writes the grid with a terminal zero
                        // (`"0.78, 0.45, 0"`); it ends the schedule and is not
                        // a sigma, so it is dropped here once.
                        if sigmas.last() == Some(&0.0) {
                            sigmas.pop();
                        }
                        profile.refine_sigmas = sigmas;
                    }
                    "dit_gemm" => profile.dit_gemm = Some(string(field, name, key)?),
                    "vae_gemm" => {
                        let word = string(field, name, key)?;
                        profile.vae_gemm = Some(vae_gemm(&word).ok_or_else(|| {
                            format!("profile {name:?}: vae_gemm {word:?} is not fp32, bf16, bf16i or bf16a")
                        })?);
                    }
                    "vae_attention" => {
                        let word = string(field, name, key)?;
                        profile.vae_attention = Some(vae_attention(&word).ok_or_else(|| {
                            format!(
                                "profile {name:?}: vae_attention {word:?} is not scalar or tensor"
                            )
                        })?);
                    }
                    "preset" => profile.preset = Some(string(field, name, key)?),
                    "prompt" => profile.prompt = Some(string(field, name, key)?),
                    "prompt_file" => profile.prompt_file = Some(string(field, name, key)?),
                    other => return Err(format!(
                        "profile {name:?}: unknown key {other:?}; have doc, steps, width, height, \
                             seconds, crf, one_pass, draft_scale, draft_attention, refine_sigma, \
                             refine_sigmas, \
                             dit_gemm, vae_gemm, vae_attention, preset, prompt, prompt_file"
                    )),
                }
            }
            entries.insert(name.clone(), profile);
        }
        if entries.is_empty() {
            return Err("no profiles in the file".to_string());
        }
        Ok(Profiles { entries })
    }

    pub fn get(&self, name: &str) -> Option<&Profile> {
        self.entries.get(name)
    }

    /// Every recipe with its one-line description, in name order.
    pub fn names(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries
            .iter()
            .map(|(name, profile)| (name.as_str(), profile.doc.as_str()))
    }
}

fn string(value: &Value, name: &str, key: &str) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("profile {name:?}: {key} is not a string"))
}

fn number(value: &Value, name: &str, key: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("profile {name:?}: {key} is not a non-negative integer"))
}

fn float(value: &Value, name: &str, key: &str) -> Result<f32, String> {
    value
        .as_f64()
        .map(|value| value as f32)
        .ok_or_else(|| format!("profile {name:?}: {key} is not a number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"{
      "profiles": {
        "two": {"doc": "two-step", "steps": 6, "width": 768, "height": 1344,
                "draft_scale": 0.7, "refine_sigmas": [0.78, 0.45, 0.0],
                "vae_gemm": "bf16i"},
        "one": {"doc": "one stage", "one_pass": true, "steps": 6, "vae_gemm": "bf16a"}
      }
    }"#;

    #[test]
    fn a_recipe_is_its_flags_and_the_terminal_zero_is_not_a_sigma() {
        let profiles = Profiles::parse(FILE).expect("the file parses");
        let two = profiles.get("two").expect("two is there");
        assert_eq!(two.steps, Some(6));
        assert_eq!(two.width, Some(768));
        assert_eq!(two.refine_sigmas, vec![0.78, 0.45]);
        let draft = two.draft().expect("two states no one_pass");
        assert_eq!(draft.sigma, 0.78);
        assert_eq!(draft.next, Some(0.45));
        assert_eq!(draft.scale, 0.7);

        let one = profiles.get("one").expect("one is there");
        assert!(one.draft().is_none(), "one_pass renders in one stage");
    }

    #[test]
    fn the_shell_lines_are_the_scripts_own_names() {
        let profiles = Profiles::parse(FILE).expect("the file parses");
        let lines = profiles.get("two").expect("two is there").shell();
        let line = |key: &str| {
            lines
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.clone())
        };
        assert_eq!(line("STEPS").as_deref(), Some("6"));
        assert_eq!(line("WIDTH").as_deref(), Some("768"));
        assert_eq!(line("REFINE_SIGMA").as_deref(), Some("0.78,0.45"));
        assert_eq!(line("H3_VAE_GEMM").as_deref(), Some("bf16i"));
        assert_eq!(line("ONE_PASS"), None);
    }

    #[test]
    fn an_unknown_key_is_refused_by_name() {
        let error = Profiles::parse(r#"{"profiles": {"x": {"stepps": 4}}}"#)
            .expect_err("a typo must not pass");
        assert!(error.contains("stepps"), "{error}");
    }

    #[test]
    fn a_quoted_prompt_survives_the_shell() {
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }
}
