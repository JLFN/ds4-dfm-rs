//! The stage-2 prompt cache: the fixed offline conditioning file the refiner
//! loads, one unit after the connector that produces the two contexts.
//!
//! Source of truth: the release's own `runtime/prompt_cache.py` and
//! `runtime/cache_builder.py`, read first-hand this session. Every rule below
//! cites it:
//!
//! - `prompt_cache.py:16-28` `RECIPE`; `:17` `FIXED_PROMPT`; `:29`
//!   `MAX_CACHE_BYTES` 64 MiB; `:30` `CONTEXT_SHAPES` video `(1, 1024, 4096)`
//!   and audio `(1, 1024, 2048)`.
//! - `prompt_cache.py:32-33` `fingerprint`: the sha256 of
//!   `json.dumps({"prompt": prompt, "recipe": RECIPE}, sort_keys=True,
//!   separators=(",", ":"))`.
//! - `prompt_cache.py:38-61` `validate_payload`: `schema_version` 1; the prompt
//!   and the recipe equal to `RECIPE`; the fingerprint equal to
//!   `fingerprint(prompt)`; both contexts present; each on CPU, `torch.bfloat16`,
//!   the shape in `CONTEXT_SHAPES`, every size >= 1, its bytes at most
//!   `MAX_CACHE_BYTES`, `torch.isfinite(...).all()`; a matching
//!   `tensor_manifest` entry; and `prompt_stats` with
//!   `characters == len(prompt)` and `truncated is False`.
//! - `prompt_cache.py:93-100` `load_cache`: the file must be a file of at most
//!   `MAX_CACHE_BYTES`, read through the container's `load`.
//! - `cache_builder.py:129-150` the payload the release writes, including
//!   `generation`; `:153` the `torch.save` write; `:94-96` it refuses an
//!   existing destination.
//!
//! The container itself is `super::torch_file`.
//!
//! Deviations, named:
//!
//! - `characters` is compared as code points (`str::chars().count()`), which
//!   is Python's `len(str)`.
//! - The canonical JSON escapes the way `json.dumps(ensure_ascii=True)` does:
//!   `"` and `\`, the short control escapes, and everything at or above 0x7F as
//!   `\uXXXX`. The one prompt this unit uses is ASCII, so no escape is
//!   exercised; a non-ASCII prompt is escaped rather than refused.
//! - `generation` is not checked by `validate_payload`; a cache this port
//!   writes carries its own honest block.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::minimax_h3::torch_file::{self, TorchDtype, TorchError, Value};

/// `prompt_cache.py:17`.
pub const FIXED_PROMPT: &str =
    "4K, refined, high quality, cinematic detail, clean textures, natural motion.";
/// `prompt_cache.py:29`.
pub const MAX_CACHE_BYTES: u64 = 64 * 1024 * 1024;
/// `prompt_cache.py:16-28`, in the sorted order the fingerprint hashes.
pub const RECIPE: &[(&str, RecipeValue)] = &[
    (
        "boundary",
        RecipeValue::Text("post_INT8_connector_video_and_audio_encoding"),
    ),
    (
        "connector",
        RecipeValue::Text("ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"),
    ),
    ("connector_bytes", RecipeValue::Int(21504034224)),
    ("connector_int8_linears", RecipeValue::Int(96)),
    ("context_dtype", RecipeValue::Text("torch.bfloat16")),
    ("gemma_int8_linears", RecipeValue::Int(328)),
    ("joint_audio_video", RecipeValue::Bool(true)),
    (
        "mask_boundary",
        RecipeValue::Text("consumed_inside_process_hidden_states_not_denoiser_input"),
    ),
    (
        "text_encoder",
        RecipeValue::Text("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    ),
    ("text_encoder_bytes", RecipeValue::Int(15372969374)),
    ("weight_scale_dtype", RecipeValue::Text("torch.float32")),
];

/// The scalar types `RECIPE` holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RecipeValue {
    Text(&'static str),
    Int(i64),
    Bool(bool),
}

/// `CONTEXT_SHAPES` (`prompt_cache.py:30`).
pub fn context_shape(name: &str) -> Option<[u64; 3]> {
    match name {
        "video" => Some([1, 1024, 4096]),
        "audio" => Some([1, 1024, 2048]),
        _ => None,
    }
}

#[derive(Debug)]
pub enum CacheError {
    Container(TorchError),
    Format(String),
    Mismatch(String),
}

impl CacheError {
    pub fn token(&self) -> String {
        match self {
            Self::Container(error) => error.token(),
            Self::Format(detail) => format!("format: {detail}"),
            Self::Mismatch(detail) => format!("mismatch: {detail}"),
        }
    }
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.token())
    }
}

impl std::error::Error for CacheError {}

impl From<TorchError> for CacheError {
    fn from(error: TorchError) -> Self {
        Self::Container(error)
    }
}

/// `prompt_cache.py:32-33`, the exact byte string hashed then sha256'd.
pub fn canonical_json(prompt: &str) -> String {
    let mut out = String::with_capacity(prompt.len() + 512);
    out.push_str("{\"prompt\":");
    json_string(prompt, &mut out);
    out.push_str(",\"recipe\":{");
    for (index, (key, value)) in RECIPE.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        json_string(key, &mut out);
        out.push(':');
        match value {
            RecipeValue::Text(text) => json_string(text, &mut out),
            RecipeValue::Int(number) => out.push_str(&number.to_string()),
            RecipeValue::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        }
    }
    out.push_str("}}");
    out
}

/// `prompt_cache.py:32-33`: the sha256 hex digest of `canonical_json`.
pub fn fingerprint(prompt: &str) -> String {
    let digest = Sha256::digest(canonical_json(prompt).as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Python's `json.dumps(..., ensure_ascii=True)` string escaping.
fn json_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) >= 0x7F => {
                let code = c as u32;
                if code > 0xFFFF {
                    let value = code - 0x1_0000;
                    out.push_str(&format!(
                        "\\u{:04x}\\u{:04x}",
                        0xD800 + (value >> 10),
                        0xDC00 + (value & 0x3FF)
                    ));
                } else {
                    out.push_str(&format!("\\u{code:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `prompt_stats` (`cache_builder.py:20-35`). Only `characters` and
/// `truncated` are required by `validate_payload`; the rest are carried when
/// this port writes a cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptStats {
    pub characters: u64,
    pub whitespace_words: Option<u64>,
    pub raw_tokens_including_bos: Option<u64>,
    pub kept_tokens: Option<u64>,
    pub max_tokens: Option<u64>,
    pub truncated: bool,
}

/// A validated cache: the two contexts as the refiner consumes them.
#[derive(Clone, Debug)]
pub struct Cache {
    pub prompt: String,
    pub fingerprint: String,
    pub video: torch_file::Tensor,
    pub audio: torch_file::Tensor,
    pub stats: PromptStats,
    pub generation: Value,
}

/// `prompt_cache.py:38-61`, every rule, on a container `Value`.
pub fn validate(value: &Value, prompt: &str) -> Result<Cache, CacheError> {
    let map = value
        .as_map()
        .ok_or_else(|| CacheError::Format("payload is not a dict".to_string()))?;
    if map.is_empty() {
        return Err(CacheError::Format("payload is an empty dict".to_string()));
    }

    // `schema_version` 1, the prompt, the recipe and the fingerprint.
    if value.get("schema_version").and_then(Value::as_i64) != Some(1) {
        return Err(CacheError::Mismatch("schema_version is not 1".to_string()));
    }
    if prompt.trim().is_empty() {
        return Err(CacheError::Mismatch("the prompt is empty".to_string()));
    }
    if value.get("prompt").and_then(Value::as_str) != Some(prompt) {
        return Err(CacheError::Mismatch(
            "the cached prompt is not the expected one".to_string(),
        ));
    }
    let recipe = value
        .get("recipe")
        .ok_or_else(|| CacheError::Format("no recipe".to_string()))?;
    if !recipe_matches(recipe) {
        return Err(CacheError::Mismatch("the recipe is not RECIPE".to_string()));
    }
    let expected = fingerprint(prompt);
    if value.get("fingerprint").and_then(Value::as_str) != Some(expected.as_str()) {
        return Err(CacheError::Mismatch(
            "the fingerprint is not fingerprint(prompt)".to_string(),
        ));
    }

    // Both contexts, each CPU bfloat16, the shape, the size bound, finite, and
    // a matching manifest.
    let contexts = value
        .get("contexts")
        .and_then(Value::as_map)
        .ok_or_else(|| CacheError::Format("no contexts".to_string()))?;
    if contexts.len() != 2
        || !contexts.iter().any(|(k, _)| k.as_str() == Some("video"))
        || !contexts.iter().any(|(k, _)| k.as_str() == Some("audio"))
    {
        return Err(CacheError::Mismatch(
            "contexts must be exactly video and audio".to_string(),
        ));
    }
    let manifest = value
        .get("tensor_manifest")
        .and_then(Value::as_map)
        .ok_or_else(|| CacheError::Format("no tensor_manifest".to_string()))?;

    let mut contexts_out: Vec<(String, torch_file::Tensor)> = Vec::new();
    for (name, tensor) in contexts.iter() {
        let name = name
            .as_str()
            .ok_or_else(|| CacheError::Format("a context key is not a string".to_string()))?;
        let expected = context_shape(name)
            .ok_or_else(|| CacheError::Mismatch(format!("unexpected context {name}")))?;
        let tensor = match tensor {
            Value::Tensor(tensor) => tensor,
            _ => return Err(CacheError::Format(format!("{name} is not a tensor"))),
        };
        if tensor.dtype != TorchDtype::BFloat16 {
            return Err(CacheError::Mismatch(format!(
                "{name} is {}, not torch.bfloat16",
                tensor.dtype.name()
            )));
        }
        if tensor.shape != expected {
            return Err(CacheError::Mismatch(format!(
                "{name} shape {:?} is not {expected:?}",
                tensor.shape
            )));
        }
        if tensor.shape.iter().any(|size| *size < 1) {
            return Err(CacheError::Mismatch(format!("{name} has a zero size")));
        }
        let bytes = tensor.numel() * tensor.dtype.size() as u64;
        if bytes > MAX_CACHE_BYTES {
            return Err(CacheError::Mismatch(format!(
                "{name} holds {bytes} bytes, over the {MAX_CACHE_BYTES} bound"
            )));
        }
        if !tensor.all_finite() {
            return Err(CacheError::Mismatch(format!(
                "{name} holds a non-finite value"
            )));
        }
        let entry = manifest
            .iter()
            .find(|(k, _)| k.as_str() == Some(name))
            .map(|(_, v)| v)
            .ok_or_else(|| CacheError::Mismatch(format!("no manifest entry for {name}")))?;
        let shape_ok = entry
            .get("shape")
            .map(|shape| manifest_shape(shape) == Some(tensor.shape.clone()))
            .unwrap_or(false);
        let dtype_ok = entry.get("dtype").and_then(Value::as_str) == Some(tensor.dtype.name());
        if !shape_ok || !dtype_ok {
            return Err(CacheError::Mismatch(format!(
                "the manifest entry for {name} does not match the tensor"
            )));
        }
        contexts_out.push((name.to_string(), tensor.clone()));
    }

    // `prompt_stats`: the character count is `len(prompt)`, truncation is off.
    let stats = value
        .get("prompt_stats")
        .and_then(Value::as_map)
        .ok_or_else(|| CacheError::Format("no prompt_stats".to_string()))?;
    let characters = stats
        .iter()
        .find(|(k, _)| k.as_str() == Some("characters"))
        .and_then(|(_, v)| v.as_i64())
        .ok_or_else(|| CacheError::Mismatch("prompt_stats has no characters".to_string()))?;
    if characters as usize != prompt.chars().count() {
        return Err(CacheError::Mismatch(
            "prompt_stats characters is not len(prompt)".to_string(),
        ));
    }
    let truncated = stats
        .iter()
        .find(|(k, _)| k.as_str() == Some("truncated"))
        .and_then(|(_, v)| v.as_bool())
        .ok_or_else(|| CacheError::Mismatch("prompt_stats has no truncated".to_string()))?;
    if truncated {
        return Err(CacheError::Mismatch(
            "the cache was built from a truncated prompt".to_string(),
        ));
    }

    let take = |name: &str| {
        contexts_out
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, tensor)| tensor.clone())
            .expect("video and audio were proved present")
    };
    Ok(Cache {
        prompt: prompt.to_string(),
        fingerprint: expected,
        video: take("video"),
        audio: take("audio"),
        stats: PromptStats {
            characters: characters as u64,
            whitespace_words: stat_int(stats, "whitespace_words"),
            raw_tokens_including_bos: stat_int(stats, "raw_tokens_including_bos"),
            kept_tokens: stat_int(stats, "kept_tokens"),
            max_tokens: stat_int(stats, "max_tokens"),
            truncated,
        },
        generation: value.get("generation").cloned().unwrap_or(Value::None),
    })
}

fn stat_int(stats: &[(Value, Value)], key: &str) -> Option<u64> {
    stats
        .iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .and_then(|(_, v)| v.as_i64())
        .filter(|value| *value >= 0)
        .map(|value| value as u64)
}

/// A `torch.Size` or a plain list of ints, as a shape.
fn manifest_shape(value: &Value) -> Option<Vec<u64>> {
    let items = match value {
        Value::Size(items) => return Some(items.clone()),
        Value::List(items) | Value::Tuple(items) => items,
        _ => return None,
    };
    items
        .iter()
        .map(|item| {
            item.as_i64()
                .filter(|value| *value >= 0)
                .map(|value| value as u64)
        })
        .collect()
}

/// Whether a payload's recipe is `RECIPE`, order-insensitively (`dict`
/// equality in the reference).
fn recipe_matches(value: &Value) -> bool {
    let entries = match value.as_map() {
        Some(entries) => entries,
        None => return false,
    };
    if entries.len() != RECIPE.len() {
        return false;
    }
    let mut seen: BTreeMap<&str, &Value> = BTreeMap::new();
    for (key, item) in entries {
        match key.as_str() {
            Some(key) => {
                if seen.insert(key, item).is_some() {
                    return false;
                }
            }
            None => return false,
        }
    }
    RECIPE.iter().all(|(key, expected)| {
        seen.get(key).is_some_and(|item| match expected {
            RecipeValue::Text(text) => item.as_str() == Some(*text),
            RecipeValue::Int(number) => item.as_i64() == Some(*number),
            RecipeValue::Bool(flag) => item.as_bool() == Some(*flag),
        })
    })
}

/// `prompt_cache.py:93-100` and `cache_builder.py:99`, then `validate`.
pub fn load(path: &Path, prompt: &str) -> Result<Cache, CacheError> {
    let meta = std::fs::metadata(path).map_err(|error| {
        CacheError::Container(TorchError::Io {
            path: path.to_path_buf(),
            detail: error.to_string(),
        })
    })?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_CACHE_BYTES {
        return Err(CacheError::Mismatch(format!(
            "{} is not a cache-sized file ({} bytes)",
            path.display(),
            meta.len()
        )));
    }
    let archive = torch_file::Archive::open(path)?;
    archive.version()?;
    let value = archive.load()?;
    validate(&value, prompt)
}

/// Build the payload `cache_builder.py:129-150` builds, and write it with the
/// container. The destination must not exist (`cache_builder.py:94-96`).
pub fn save(path: &Path, cache: &Cache) -> Result<(), CacheError> {
    if path.exists() {
        return Err(CacheError::Mismatch(format!(
            "{} already exists",
            path.display()
        )));
    }
    let value = payload(cache);
    torch_file::write(path, &value)?;
    Ok(())
}

/// The payload tree, in the reference's key order.
pub fn payload(cache: &Cache) -> Value {
    let manifest = |tensor: &torch_file::Tensor| {
        torch_file::map(vec![
            ("shape", Value::Size(tensor.shape.clone())),
            ("dtype", Value::Str(tensor.dtype.name().to_string())),
        ])
    };
    torch_file::map(vec![
        ("schema_version", Value::Int(1)),
        ("prompt", Value::Str(cache.prompt.clone())),
        ("recipe", recipe_value()),
        ("fingerprint", Value::Str(cache.fingerprint.clone())),
        (
            "contexts",
            torch_file::map(vec![
                ("video", Value::Tensor(cache.video.clone())),
                ("audio", Value::Tensor(cache.audio.clone())),
            ]),
        ),
        ("prompt_stats", stats_value(&cache.stats)),
        (
            "tensor_manifest",
            torch_file::map(vec![
                ("video", manifest(&cache.video)),
                ("audio", manifest(&cache.audio)),
            ]),
        ),
        ("generation", cache.generation.clone()),
    ])
}

fn recipe_value() -> Value {
    Value::Map(
        RECIPE
            .iter()
            .map(|(key, value)| {
                (
                    Value::Str((*key).to_string()),
                    match value {
                        RecipeValue::Text(text) => Value::Str((*text).to_string()),
                        RecipeValue::Int(number) => Value::Int(*number),
                        RecipeValue::Bool(flag) => Value::Bool(*flag),
                    },
                )
            })
            .collect(),
    )
}

fn stats_value(stats: &PromptStats) -> Value {
    let mut entries = vec![
        ("characters", Value::Int(stats.characters as i64)),
        ("truncated", Value::Bool(stats.truncated)),
    ];
    let extras = [
        ("whitespace_words", stats.whitespace_words),
        ("raw_tokens_including_bos", stats.raw_tokens_including_bos),
        ("kept_tokens", stats.kept_tokens),
        ("max_tokens", stats.max_tokens),
    ];
    for (key, value) in extras {
        if let Some(value) = value {
            entries.push((key, Value::Int(value as i64)));
        }
    }
    torch_file::map(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canonical_json_is_the_python_dumps_form() {
        // Sorted keys, `(",", ":")` separators, `true` for the bool and plain
        // integers, as `json.dumps(..., sort_keys=True, separators=(",", ":"))`
        // writes them.
        assert_eq!(
            canonical_json(FIXED_PROMPT),
            concat!(
                r#"{"prompt":"4K, refined, high quality, cinematic detail, "#,
                r#"clean textures, natural motion.","recipe":{"boundary":"#,
                r#""post_INT8_connector_video_and_audio_encoding","connector":"#,
                r#""ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors","#,
                r#""connector_bytes":21504034224,"connector_int8_linears":96,"#,
                r#""context_dtype":"torch.bfloat16","gemma_int8_linears":328,"#,
                r#""joint_audio_video":true,"mask_boundary":"#,
                r#""consumed_inside_process_hidden_states_not_denoiser_input","#,
                r#""text_encoder":"#,
                r#""gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors","#,
                r#""text_encoder_bytes":15372969374,"#,
                r#""weight_scale_dtype":"torch.float32"}}"#
            )
        );
    }

    #[test]
    fn the_fingerprint_is_the_reference_digest() {
        // The sha256 of the canonical bytes above, cross-checked with the
        // system's own `sha256sum` over the same 594 bytes.
        assert_eq!(
            fingerprint(FIXED_PROMPT),
            "d37b38a96aa8f851cdb2782a3ca403e1f7d1328762a97da53c43e46a2b1b1e0e"
        );
    }

    #[test]
    fn the_string_escapes_match_python() {
        let mut out = String::new();
        json_string("a\"b\\c\nd\t\u{7f}\u{1F600}", &mut out);
        assert_eq!(out, "\"a\\\"b\\\\c\\nd\\t\\u007f\\ud83d\\ude00\"");
    }
}
