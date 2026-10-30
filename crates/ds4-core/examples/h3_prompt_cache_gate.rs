//! Gate the stage-2 prompt cache: the container round trip, every
//! `validate_payload` rule (`runtime/prompt_cache.py:38-61`), and the
//! `fingerprint` (`:32-33`).
//!
//! Usage:
//!
//!     h3_prompt_cache_gate [--write FILE] [--inspect FILE] [--prompt TEXT]
//!
//! Default (no `--inspect`): the self test. It builds a valid payload with two
//! deterministic bfloat16 contexts of the recipe's own shape
//! (`CONTEXT_SHAPES`, `prompt_cache.py:30`), writes it through the container to
//! a scratch path, reads it back, and asserts the round trip is byte-exact and
//! that every validation rule passes. It then applies one mutation per rule and
//! asserts each is refused, so the gate is a gate. `--write FILE` keeps the
//! cache it built at FILE instead of a scratch path.
//!
//! `--inspect FILE` reads a cache the release's Python wrote (the operator's
//! golden) with this port's own container reader and reports every rule's
//! verdict plus a per-context digest, without needing the model or a device.
//!
//! Output, in order:
//!
//! - `FINGERPRINT` the prompt's reference digest.
//! - `CONTAINER` the archive name, byte count, record count and `version`.
//! - `CONTEXT` one line per context: shape, dtype, bytes and sha256.
//! - `ROUNDTRIP` whether the read-back tensors equal the written ones.
//! - `RULE` one line per validation rule, with the refusal's token.
//! - `PROMPT-CACHE-OK` or `PROMPT-CACHE-REFUSED`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ds4_core::minimax_h3::prompt_cache::{self, Cache, PromptStats, FIXED_PROMPT};
use ds4_core::minimax_h3::torch_file::{self, Archive, Tensor, TorchDtype, Value};

/// sha256 hex of a byte slice.
fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(bytes);
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A deterministic bfloat16 context of `shape`, finite by construction.
fn synthetic_context(seed: u64, shape: &[u64]) -> Tensor {
    let numel: u64 = shape.iter().product();
    let mut bytes = Vec::with_capacity(numel as usize * 2);
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for _ in 0..numel {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut raw = (state >> 48) as u16;
        // Keep the exponent away from all-ones so no value is inf or nan.
        if raw & 0x7F80 == 0x7F80 {
            raw &= 0xFBFF;
        }
        bytes.extend_from_slice(&raw.to_le_bytes());
    }
    Tensor::contiguous(TorchDtype::BFloat16, shape.to_vec(), bytes)
}

fn synthetic_cache(prompt: &str) -> Cache {
    let video = synthetic_context(1, &[1, 1024, 4096]);
    let audio = synthetic_context(2, &[1, 1024, 2048]);
    Cache {
        prompt: prompt.to_string(),
        fingerprint: prompt_cache::fingerprint(prompt),
        video,
        audio,
        stats: PromptStats {
            characters: prompt.chars().count() as u64,
            whitespace_words: Some(11),
            raw_tokens_including_bos: Some(17),
            kept_tokens: Some(17),
            max_tokens: Some(1024),
            truncated: false,
        },
        generation: torch_file::map(vec![
            ("builder", Value::Str("ds4-dfm-rs minimax_h3".to_string())),
            (
                "source",
                Value::Str("port connector graph, synthetic contexts".to_string()),
            ),
        ]),
    }
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut write: Option<PathBuf> = None;
    let mut inspect: Option<PathBuf> = None;
    let mut prompt = FIXED_PROMPT.to_string();

    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--write" => {
                index += 1;
                match arguments.get(index) {
                    Some(value) => write = Some(PathBuf::from(value)),
                    None => return usage("--write needs a path"),
                }
            }
            "--inspect" => {
                index += 1;
                match arguments.get(index) {
                    Some(value) => inspect = Some(PathBuf::from(value)),
                    None => return usage("--inspect needs a path"),
                }
            }
            "--prompt" => {
                index += 1;
                match arguments.get(index) {
                    Some(value) => prompt = value.clone(),
                    None => return usage("--prompt needs text"),
                }
            }
            "--help" | "-h" => return usage(""),
            value => return usage(&format!("unexpected argument {value}")),
        }
        index += 1;
    }

    println!("FINGERPRINT\t{}", prompt_cache::fingerprint(&prompt));

    match inspect {
        Some(path) => inspect_cache(&path, &prompt),
        None => self_test(&prompt, write.as_deref()),
    }
}

fn inspect_cache(path: &Path, prompt: &str) -> ExitCode {
    let archive = match Archive::open(path) {
        Ok(archive) => archive,
        Err(error) => {
            println!("PROMPT-CACHE-REFUSED\t{}", error.token());
            return ExitCode::FAILURE;
        }
    };
    let names = archive.records();
    let version = archive
        .version()
        .map(|v| v.to_string())
        .unwrap_or_else(|e| e.token());
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!(
        "CONTAINER\tarchive={}\tbytes={bytes}\trecords={}\tversion={version}",
        archive.archive_name,
        names.len()
    );

    let value = match archive.load() {
        Ok(value) => value,
        Err(error) => {
            println!("PROMPT-CACHE-REFUSED\t{}", error.token());
            return ExitCode::FAILURE;
        }
    };
    match prompt_cache::validate(&value, prompt) {
        Ok(cache) => {
            report_contexts(&cache);
            println!(
                "STATS\tcharacters={}\tkept={:?}\tmax={:?}\ttruncated={}",
                cache.stats.characters,
                cache.stats.kept_tokens,
                cache.stats.max_tokens,
                cache.stats.truncated
            );
            println!(
                "PROMPT-CACHE-OK\tpath={}\tfingerprint={}",
                path.display(),
                cache.fingerprint
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            println!("PROMPT-CACHE-REFUSED\t{}", error.token());
            ExitCode::FAILURE
        }
    }
}

fn report_contexts(cache: &Cache) {
    for (name, tensor) in [("video", &cache.video), ("audio", &cache.audio)] {
        let bytes = tensor.contiguous_bytes().unwrap_or(&[]);
        let shape = tensor
            .shape
            .iter()
            .map(|axis| axis.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "CONTEXT\t{name}\tshape={shape}\tdtype={}\tbytes={}\tsha256={}",
            tensor.dtype.name(),
            bytes.len(),
            digest(bytes)
        );
    }
}

fn self_test(prompt: &str, write: Option<&Path>) -> ExitCode {
    let cache = synthetic_cache(prompt);
    let (path, cleanup) = match write {
        Some(path) => (path.to_path_buf(), false),
        None => (
            std::env::temp_dir().join(format!("h3-prompt-cache-{}.pt", std::process::id())),
            true,
        ),
    };
    std::fs::remove_file(&path).ok();

    if let Err(error) = prompt_cache::save(&path, &cache) {
        println!("PROMPT-CACHE-REFUSED\twrite: {}", error.token());
        return ExitCode::FAILURE;
    }

    // The container's own structure and version.
    let archive = match Archive::open(&path) {
        Ok(archive) => archive,
        Err(error) => {
            println!("PROMPT-CACHE-REFUSED\topen: {}", error.token());
            return ExitCode::FAILURE;
        }
    };
    let names = archive.records();
    let version = archive
        .version()
        .map(|value| value.to_string())
        .unwrap_or_else(|error| error.token());
    let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
    println!(
        "CONTAINER\tarchive={}\tbytes={bytes}\trecords={}\tversion={version}\tlist={}",
        archive.archive_name,
        names.len(),
        names.join(",")
    );

    // The read-back payload validates and is byte-exact.
    let read = match prompt_cache::load(&path, prompt) {
        Ok(read) => read,
        Err(error) => {
            println!("PROMPT-CACHE-REFUSED\tload: {}", error.token());
            return ExitCode::FAILURE;
        }
    };
    report_contexts(&read);
    let video_same = read.video == cache.video;
    let audio_same = read.audio == cache.audio;
    let fingerprint_same = read.fingerprint == cache.fingerprint;
    println!(
        "ROUNDTRIP\tvideo={}\taudio={}\tfingerprint={}",
        if video_same { "same" } else { "differs" },
        if audio_same { "same" } else { "differs" },
        if fingerprint_same { "same" } else { "differs" }
    );

    // One mutation per validation rule: each must be refused.
    let mut rules = 0usize;
    let mut failures = Vec::new();
    for (name, payload) in broken_payloads(&cache) {
        rules += 1;
        match prompt_cache::validate(&payload, prompt) {
            Ok(_) => failures.push(format!("{name} was accepted")),
            Err(error) => println!("RULE\t{name}\trefused\t{}", error.token()),
        }
    }
    // Plus the positive control.
    let accepted = prompt_cache::validate(&prompt_cache::payload(&cache), prompt).is_ok();
    println!(
        "RULE\tvalid_payload\t{}\trules={rules}",
        if accepted { "accepted" } else { "refused" }
    );

    if cleanup {
        std::fs::remove_file(&path).ok();
    }
    if !video_same || !audio_same || !fingerprint_same || !accepted || !failures.is_empty() {
        for failure in &failures {
            println!("PROMPT-CACHE-REFUSED\t{failure}");
        }
        return ExitCode::FAILURE;
    }
    println!(
        "PROMPT-CACHE-OK\tfile={}\tfingerprint={}\trules={rules}",
        path.display(),
        cache.fingerprint
    );
    ExitCode::SUCCESS
}

/// The valid payload with one rule broken at a time, named after the rule.
fn broken_payloads(cache: &Cache) -> Vec<(&'static str, Value)> {
    let mut out = Vec::new();
    push_case(&mut out, cache, "schema_version=2", |entries| {
        entry_set(entries, "schema_version", Value::Int(2));
    });
    push_case(&mut out, cache, "prompt=other", |entries| {
        entry_set(entries, "prompt", Value::Str("another prompt".to_string()));
    });
    push_case(&mut out, cache, "recipe_boundary_changed", |entries| {
        if let Some(Value::Map(recipe)) = entry_get(entries, "recipe") {
            entry_set(recipe, "boundary", Value::Str("something_else".to_string()));
        }
    });
    push_case(&mut out, cache, "recipe_key_missing", |entries| {
        if let Some(Value::Map(recipe)) = entry_get(entries, "recipe") {
            recipe.retain(|(key, _)| key.as_str() != Some("connector"));
        }
    });
    push_case(&mut out, cache, "fingerprint_changed", |entries| {
        entry_set(entries, "fingerprint", Value::Str("0".repeat(64)));
    });
    push_case(&mut out, cache, "audio_context_missing", |entries| {
        if let Some(Value::Map(contexts)) = entry_get(entries, "contexts") {
            contexts.retain(|(key, _)| key.as_str() != Some("audio"));
        }
    });
    push_case(&mut out, cache, "audio_dtype_float32", |entries| {
        tensor_mut(entries, "audio").dtype = TorchDtype::Float32;
    });
    push_case(&mut out, cache, "video_shape_changed", |entries| {
        tensor_mut(entries, "video").shape = vec![1, 512, 4096];
    });
    push_case(&mut out, cache, "video_value_nan", |entries| {
        // bfloat16 0x7FC0: exponent all ones, mantissa nonzero.
        let tensor = tensor_mut(entries, "video");
        tensor.bytes[0] = 0xC0;
        tensor.bytes[1] = 0x7F;
    });
    push_case(&mut out, cache, "manifest_dtype_mismatch", |entries| {
        if let Some(Value::Map(manifest)) = entry_get(entries, "tensor_manifest") {
            if let Some(Value::Map(entry)) = entry_get(manifest, "video") {
                entry_set(entry, "dtype", Value::Str("torch.float32".to_string()));
            }
        }
    });
    push_case(&mut out, cache, "characters_off_by_one", |entries| {
        if let Some(Value::Map(stats)) = entry_get(entries, "prompt_stats") {
            entry_set(stats, "characters", Value::Int(1));
        }
    });
    push_case(&mut out, cache, "truncated_true", |entries| {
        if let Some(Value::Map(stats)) = entry_get(entries, "prompt_stats") {
            entry_set(stats, "truncated", Value::Bool(true));
        }
    });
    out
}

fn push_case(
    out: &mut Vec<(&'static str, Value)>,
    cache: &Cache,
    name: &'static str,
    edit: fn(&mut Vec<(Value, Value)>),
) {
    let mut payload = prompt_cache::payload(cache);
    if let Value::Map(entries) = &mut payload {
        edit(entries);
    }
    out.push((name, payload));
}

fn entry_get<'a>(entries: &'a mut Vec<(Value, Value)>, key: &str) -> Option<&'a mut Value> {
    entries
        .iter_mut()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, value)| value)
}

fn entry_set(entries: &mut Vec<(Value, Value)>, key: &str, value: Value) {
    match entry_get(entries, key) {
        Some(slot) => *slot = value,
        None => entries.push((Value::Str(key.to_string()), value)),
    }
}

fn tensor_mut<'a>(entries: &'a mut Vec<(Value, Value)>, name: &str) -> &'a mut Tensor {
    let contexts = match entry_get(entries, "contexts") {
        Some(Value::Map(contexts)) => contexts,
        _ => unreachable!("payload has contexts"),
    };
    match entry_get(contexts, name) {
        Some(Value::Tensor(tensor)) => tensor,
        _ => unreachable!("payload has {name}"),
    }
}

fn usage(detail: &str) -> ExitCode {
    if !detail.is_empty() {
        eprintln!("{detail}");
    }
    eprintln!(
        "usage: h3_prompt_cache_gate [--write FILE] [--inspect FILE] [--prompt TEXT]\n\
         \x20 default  build the recipe's payload, round trip it, and refuse one\n\
         \x20          mutation per validate_payload rule\n\
         \x20 --inspect  read a release-written cache and report every rule\n\
         \x20 --write    keep the self test's cache at FILE"
    );
    ExitCode::from(2)
}
