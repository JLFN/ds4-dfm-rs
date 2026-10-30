//! The Gemma4 tokenizer and the processor that pads for it.
//!
//! Unit 3. The asset is inside the checkpoint - `tokenizer_json` and
//! `hf_asset__tokenizer_config.json`, both U8 tensors - so nothing here needs a
//! file from outside it. Source of truth:
//!
//! - the wrapper the release calls,
//!   `/data/ltx2-pin/packages/ltx-core/src/ltx_core/text_encoders/gemma/tokenizer.py:39-63`,
//!   for the strip, the truncate, the BOS and the left padding;
//! - HuggingFace's `tokenizers` 0.22.1 - what `transformers` 5.14.1 requires
//!   (`>=0.22.0,<=0.23.0`) and what the pinned Python wraps - unpacked
//!   read-only at `/data/refs/tokenizers/tokenizers-0.22.1`, for every step
//!   under it. The citations below are that crate.
//!
//! This is a re-implementation, not a reuse: the operator's decision is the
//! repo's own BPE, so the deviation is named in
//! `docs/minimax-h3/gemma-tokenizer-port-map-2026-10-30.md` and measured
//! against the golden's ids.
//!
//! The asset's pipeline, as its own JSON states it:
//!
//!     normalizer     Replace   " " -> "▁"
//!     pre_tokenizer  Split     " ", MergedWithPrevious
//!     post_processor TemplateProcessing with no special tokens
//!     model          BPE, unk "<unk>", fuse_unk, byte_fallback, no ignore_merges
//!
//! which is SentencePiece-shaped, not the GPT-2 byte-level pipeline
//! `minimax_h3::tokenizer` implements for the MiniMax encoder: the alphabet is
//! characters, the space is U+2581, and the byte codec has no part in it.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::minimax_h3::gemma::GemmaError;
use crate::minimax_h3::safetensors::{self, Dtype};

/// `TOKENIZER_MAX_LENGTH` (`gemma_assets.py:162`), and the width every prompt
/// is padded to before the tower sees it.
pub const MAX_LENGTH: usize = 1024;

/// The `▁` the normalizer writes in place of a space (`U+2581`).
const SPACE_MARKER: char = '\u{2581}';

/// The two assets' names inside the checkpoint.
const TOKENIZER_JSON: &str = "tokenizer_json";
const TOKENIZER_CONFIG: &str = "hf_asset__tokenizer_config.json";

/// The Gemma4 tokenizer, built from the checkpoint's own `tokenizer.json`.
pub struct Tokenizer {
    /// `model.vocab`, which already carries the special tokens.
    vocab: HashMap<String, u32>,
    /// `(left id, right id) -> (rank, merged id)`, the reference's own keying
    /// (`models/bpe/model.rs:174-192`, where the rank is the merges index and
    /// the merged id is the vocabulary's entry for the concatenation).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// The byte-fallback symbols the vocab actually carries, keyed by the
    /// spelling the reference builds: `format!("<{byte:#04X}>")`, uppercase
    /// (`model.rs:417-431`).
    byte_tokens: HashMap<String, u32>,
    /// The added tokens, longest first, so a literal scan finds the longest
    /// match at a position.
    added: Vec<(String, u32)>,
    /// `model.unk_token`'s id.
    unk: u32,
    /// `tokenizer_config.json`'s `bos_token` and `pad_token`, resolved through
    /// the vocabulary rather than written down here.
    bos: u32,
    pad: u32,
    /// The vocabulary's own size, so a decode-side check can refuse an id the
    /// checkpoint does not define.
    width: usize,
}

impl Tokenizer {
    /// Read the two assets out of the checkpoint and check the pipeline shape.
    pub fn open(path: &Path) -> Result<Tokenizer, GemmaError> {
        let shard = safetensors::read_shard(path)
            .map_err(|error| GemmaError::Token(format!("reader: {}", error.token())))?;
        let json = read_asset(&shard, TOKENIZER_JSON)?;
        let config = read_asset(&shard, TOKENIZER_CONFIG)?;
        let root: Value = serde_json::from_slice(&json)
            .map_err(|error| GemmaError::Token(format!("tokenizer.json: {error}")))?;
        let config: Value = serde_json::from_slice(&config)
            .map_err(|error| GemmaError::Token(format!("tokenizer_config: {error}")))?;

        Tokenizer::from_json(&root, &config)
    }

    /// The same, from already-parsed values, so a test needs no checkpoint.
    pub fn from_json(root: &Value, config: &Value) -> Result<Tokenizer, GemmaError> {
        check_pipeline(root)?;

        let model = root.get("model").expect("checked");
        let vocab = read_vocab(model)?;
        let merges = read_merges(model, &vocab)?;
        let byte_tokens = read_byte_tokens(&vocab);
        let mut added = read_added(root, &vocab)?;

        // Leftmost-longest, so the longest content is tried first at each
        // position.
        added.sort_by(|left, right| right.0.len().cmp(&left.0.len()));

        let unk = model
            .get("unk_token")
            .and_then(Value::as_str)
            .and_then(|name| vocab.get(name).copied())
            .ok_or_else(|| GemmaError::Token("model.unk_token is not in the vocabulary".into()))?;
        let resolve = |key: &str| -> Result<u32, GemmaError> {
            let name = config
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| GemmaError::Token(format!("tokenizer_config.json has no {key}")))?;
            vocab.get(name).copied().ok_or_else(|| {
                GemmaError::Token(format!("{key} = {name} is not in the vocabulary"))
            })
        };
        let bos = resolve("bos_token")?;
        let pad = resolve("pad_token")?;
        let width = vocab.len();

        Ok(Tokenizer {
            vocab,
            merges,
            byte_tokens,
            added,
            unk,
            bos,
            pad,
            width,
        })
    }

    /// The vocabulary's size.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The BOS id the wrapper prepends.
    pub fn bos(&self) -> u32 {
        self.bos
    }

    /// The model's own ids for a text, with no BOS and no padding: the
    /// reference's `tokenizer(text, padding=False, truncation=False)`.
    pub fn ids(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut start = 0;

        // Added tokens are split out of the raw text first
        // (`tokenizer/added_vocabulary.rs:462-465`), and the gaps between them
        // go through the pipeline.
        while start < text.len() {
            let Some((offset, length, id)) = self.added_match(text, start) else {
                break;
            };

            if offset > start {
                self.encode_plain(&text[start..offset], &mut ids);
            }

            ids.push(id);
            start = offset + length;
        }

        if start < text.len() {
            self.encode_plain(&text[start..], &mut ids);
        }

        ids
    }

    /// `LTXGemmaTokenizer.tokenize_with_weights` (`tokenizer.py:39-63`): strip,
    /// encode, truncate to [`MAX_LENGTH`], prepend BOS unless it is already
    /// first, truncate again, left-pad with the pad id. Returns the ids and the
    /// attention mask, which is what `encoders/base_encoder.py:56-66` forwards.
    pub fn tokenize(&self, prompt: &str) -> (Vec<u32>, Vec<u32>) {
        let mut ids = self.ids(prompt.trim());

        ids.truncate(MAX_LENGTH);

        if ids.first() != Some(&self.bos) {
            ids.insert(0, self.bos);
            ids.truncate(MAX_LENGTH);
        }

        let padding = MAX_LENGTH - ids.len();
        let mut padded = vec![self.pad; padding];
        padded.extend_from_slice(&ids);

        let mut mask = vec![0u32; padding];
        mask.resize(MAX_LENGTH, 1);

        (padded, mask)
    }

    /// Normalize, pre-tokenize, then BPE - the three stages between the added
    /// tokens and the ids.
    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        // `normalizers/replace.rs:82-86` over a literal pattern: every space
        // becomes `▁`.
        let normalized: String = text.replace(' ', &SPACE_MARKER.to_string());

        // `pre_tokenizers/split.rs:141-149` splits on a literal `" "`, and the
        // normalizer above has just removed every one of them, so the split is
        // the identity. Asserted rather than assumed: a string that still held
        // a space would need the split's `MergedWithPrevious` arm, which this
        // port does not implement because this asset cannot reach it.
        debug_assert!(!normalized.contains(' '));

        self.bpe(&normalized, ids);
    }

    /// `models/bpe/model.rs:382-466` `merge_word`: characters to ids with the
    /// byte fallback, then the merges.
    fn bpe(&self, piece: &str, ids: &mut Vec<u32>) {
        let mut symbols: Vec<u32> = Vec::with_capacity(piece.len());
        let mut pending: Option<u32> = None;

        for character in piece.chars() {
            let text = character.to_string();

            if let Some(id) = self.vocab.get(&text) {
                if let Some(unk) = pending.take() {
                    symbols.push(unk);
                }
                symbols.push(*id);
                continue;
            }

            // `:417-431`: the character's bytes as `<0xXX>` tokens, but only
            // when every one of them is in the vocabulary. The asset carries
            // 100 of the 256 byte symbols, so this arm reaches a byte whose
            // uppercase hex has both digits in 0-9 and nothing else.
            let bytes: Option<Vec<u32>> = text
                .bytes()
                .map(|byte| self.byte_tokens.get(&byte_symbol(byte)).copied())
                .collect();

            if let Some(bytes) = bytes {
                // The reference adds the byte tokens here and leaves any
                // pending `unk` pending (`:423-430` has no flush before the
                // `continue`), so with `fuse_unk` the unk is emitted after
                // them. Mirrored deliberately.
                symbols.extend(bytes);
                continue;
            }

            // `:437-452`, `fuse_unk` true: a run of unknown characters becomes
            // one unk id.
            pending = Some(pending.unwrap_or(self.unk));
        }

        if let Some(unk) = pending {
            symbols.push(unk);
        }

        self.merge(&mut symbols);

        ids.extend_from_slice(&symbols);
    }

    /// `Word::merge_all` (`models/bpe/word.rs:162-...`) with `Merge`'s ordering
    /// (`:26-35`): the lowest rank first and the leftmost position on a tie,
    /// one occurrence at a time, with the pairs either side of a merge
    /// reconsidered. The reference keeps that order in a min-heap; a scan finds
    /// the same element each round, so the sequence of merges is the same.
    fn merge(&self, symbols: &mut Vec<u32>) {
        loop {
            let mut best: Option<(usize, u32, u32)> = None;

            for index in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[index], symbols[index + 1]);

                let Some((rank, merged)) = self.merges.get(&pair) else {
                    continue;
                };

                if best.is_none_or(|(_, best_rank, _)| *rank < best_rank) {
                    best = Some((index, *rank, *merged));
                }
            }

            let Some((position, _, merged)) = best else {
                return;
            };

            symbols[position] = merged;
            symbols.remove(position + 1);
        }
    }

    /// The leftmost added token at or after `start`, longest on a tie
    /// (`added_vocabulary.rs:367-...`, whose `single_word`, `lstrip` and
    /// `rstrip` arms are all off in this asset).
    fn added_match(&self, text: &str, start: usize) -> Option<(usize, usize, u32)> {
        for offset in char_offsets(text, start) {
            let rest = &text[offset..];

            for (content, id) in &self.added {
                if rest.starts_with(content.as_str()) {
                    return Some((offset, content.len(), *id));
                }
            }
        }

        None
    }
}

/// Every byte offset at or after `start` that is a character boundary.
fn char_offsets(text: &str, start: usize) -> impl Iterator<Item = usize> + '_ {
    text.char_indices()
        .map(|(offset, _)| offset)
        .filter(move |offset| *offset >= start)
}

/// `format!("<{byte:#04X}>")` (`models/bpe/model.rs:421`): the `0x` prefix, two
/// digits, uppercase.
fn byte_symbol(byte: u8) -> String {
    format!("<{byte:#04X}>")
}

/// The whole asset, checked before a byte of it is trusted.
fn read_asset(shard: &safetensors::Shard, name: &str) -> Result<Vec<u8>, GemmaError> {
    let tensor = shard
        .find(name)
        .ok_or_else(|| GemmaError::Token(format!("{name} is not in the checkpoint")))?;

    if tensor.dtype != Dtype::U8 {
        return Err(GemmaError::Token(format!(
            "{name} is {}, not a U8 blob",
            tensor.dtype.name()
        )));
    }

    let mut bytes = vec![0u8; tensor.bytes() as usize];
    shard
        .read_data(tensor, &mut bytes)
        .map_err(|error| GemmaError::Token(format!("reader: {}", error.token())))?;

    Ok(bytes)
}

/// The pipeline this port implements, refused by name if the asset differs.
fn check_pipeline(root: &Value) -> Result<(), GemmaError> {
    let normalizer = root
        .get("normalizer")
        .ok_or_else(|| GemmaError::Token("no normalizer".into()))?;

    if normalizer.get("type").and_then(Value::as_str) != Some("Replace")
        || normalizer
            .pointer("/pattern/String")
            .and_then(Value::as_str)
            != Some(" ")
        || normalizer.get("content").and_then(Value::as_str) != Some(&SPACE_MARKER.to_string())
    {
        return Err(GemmaError::Token(format!(
            "normalizer is not Replace(\" \" -> \"{SPACE_MARKER}\"): {normalizer}"
        )));
    }

    let pretokenizer = root
        .get("pre_tokenizer")
        .ok_or_else(|| GemmaError::Token("no pre_tokenizer".into()))?;

    if pretokenizer.get("type").and_then(Value::as_str) != Some("Split")
        || pretokenizer
            .pointer("/pattern/String")
            .and_then(Value::as_str)
            != Some(" ")
        || pretokenizer.get("behavior").and_then(Value::as_str) != Some("MergedWithPrevious")
        || pretokenizer.get("invert").and_then(Value::as_bool) != Some(false)
    {
        return Err(GemmaError::Token(format!(
            "pre_tokenizer is not Split(\" \", MergedWithPrevious): {pretokenizer}"
        )));
    }

    // The wrapper prepends the BOS itself, so the post-processor must add
    // nothing; a template that inserted a token would double it.
    let processor = root
        .get("post_processor")
        .ok_or_else(|| GemmaError::Token("no post_processor".into()))?;

    if processor.get("type").and_then(Value::as_str) != Some("TemplateProcessing")
        || processor
            .get("special_tokens")
            .and_then(Value::as_object)
            .is_none_or(|tokens| !tokens.is_empty())
    {
        return Err(GemmaError::Token(format!(
            "post_processor is not a token-free TemplateProcessing: {processor}"
        )));
    }

    let model = root
        .get("model")
        .ok_or_else(|| GemmaError::Token("no model".into()))?;
    let flag = |key: &str| model.get(key).and_then(Value::as_bool);
    let null = |key: &str| model.get(key).is_none_or(Value::is_null);

    if model.get("type").and_then(Value::as_str) != Some("BPE")
        || flag("byte_fallback") != Some(true)
        || flag("fuse_unk") != Some(true)
        || flag("ignore_merges") != Some(false)
        || !null("dropout")
        || !null("continuing_subword_prefix")
        || !null("end_of_word_suffix")
    {
        return Err(GemmaError::Token(format!(
            "model is not the BPE this port implements: {model}"
        )));
    }

    Ok(())
}

fn read_vocab(model: &Value) -> Result<HashMap<String, u32>, GemmaError> {
    let raw = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| GemmaError::Token("model.vocab is not an object".into()))?;
    let mut vocab = HashMap::with_capacity(raw.len());

    for (symbol, id) in raw {
        let id = id
            .as_u64()
            .ok_or_else(|| GemmaError::Token(format!("vocab {symbol:?} has no integer id")))?;
        vocab.insert(
            symbol.clone(),
            u32::try_from(id)
                .map_err(|_| GemmaError::Token(format!("vocab id {id} does not fit u32")))?,
        );
    }

    Ok(vocab)
}

/// The ranks, as `models/bpe/model.rs:174-192` builds them: each merge's index
/// is its rank and its result is the vocabulary's entry for the concatenation,
/// which must exist.
fn read_merges(
    model: &Value,
    vocab: &HashMap<String, u32>,
) -> Result<HashMap<(u32, u32), (u32, u32)>, GemmaError> {
    let raw = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| GemmaError::Token("model.merges is not an array".into()))?;
    let mut merges = HashMap::with_capacity(raw.len());

    for (rank, entry) in raw.iter().enumerate() {
        // `h3_parse_merges`'s two spellings, kept because the reference crate
        // accepts both in its own loader.
        let (left, right) = match entry {
            Value::Array(pair) if pair.len() == 2 => (
                pair[0].as_str().unwrap_or_default().to_string(),
                pair[1].as_str().unwrap_or_default().to_string(),
            ),
            Value::String(line) => match line.split_once(' ') {
                Some((left, right)) => (left.to_string(), right.to_string()),
                None => {
                    return Err(GemmaError::Token(format!(
                        "merge {rank} is not a pair: {line:?}"
                    )));
                }
            },
            other => {
                return Err(GemmaError::Token(format!(
                    "merge {rank} is not a pair: {other}"
                )));
            }
        };

        let at = |symbol: &str| -> Result<u32, GemmaError> {
            vocab.get(symbol).copied().ok_or_else(|| {
                GemmaError::Token(format!("merge {rank} token {symbol:?} is absent"))
            })
        };
        let merged = format!("{left}{right}");
        let rank = u32::try_from(rank)
            .map_err(|_| GemmaError::Token("more merges than u32 ranks".into()))?;

        merges.insert((at(&left)?, at(&right)?), (rank, at(&merged)?));
    }

    Ok(merges)
}

/// The byte symbols the vocabulary carries, keyed by the uppercase spelling the
/// reference looks up.
fn read_byte_tokens(vocab: &HashMap<String, u32>) -> HashMap<String, u32> {
    let mut tokens = HashMap::new();

    for (symbol, id) in vocab {
        let Some(hex) = symbol
            .strip_prefix("<0x")
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };

        if let Ok(byte) = u8::from_str_radix(hex, 16) {
            tokens.insert(byte_symbol(byte), *id);
        }
    }

    tokens
}

fn read_added(
    root: &Value,
    vocab: &HashMap<String, u32>,
) -> Result<Vec<(String, u32)>, GemmaError> {
    let raw = root
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| GemmaError::Token("added_tokens is not an array".into()))?;
    let mut added = Vec::with_capacity(raw.len());

    for entry in raw {
        let content = entry
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| GemmaError::Token("an added token has no content".into()))?;
        let id = entry
            .get("id")
            .and_then(Value::as_u64)
            .ok_or_else(|| GemmaError::Token(format!("added token {content:?} has no id")))?;
        let id = u32::try_from(id).map_err(|_| {
            GemmaError::Token(format!("added token {content:?} id does not fit u32"))
        })?;

        if vocab.get(content) != Some(&id) {
            return Err(GemmaError::Token(format!(
                "added token {content:?} is {id} in added_tokens but \
                 {:?} in the vocabulary",
                vocab.get(content)
            )));
        }

        // Every added token in this asset is `normalized: false` with the
        // three trimming flags off, which is what makes the raw-text scan
        // right; anything else is refused rather than half-implemented.
        for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
            if entry.get(flag).and_then(Value::as_bool) != Some(false) {
                return Err(GemmaError::Token(format!(
                    "added token {content:?} sets {flag}, which this port does not implement"
                )));
            }
        }

        added.push((content.to_string(), id));
    }

    Ok(added)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A miniature asset with the same pipeline shape, so the tests exercise
    /// this port and not the checkpoint: the alphabet is a few characters, the
    /// space is `▁`, and one byte symbol exists so both fallback arms can run.
    fn asset() -> (Value, Value) {
        let root = json!({
            "normalizer": {
                "type": "Replace",
                "pattern": {"String": " "},
                "content": "▁"
            },
            "pre_tokenizer": {
                "type": "Split",
                "pattern": {"String": " "},
                "behavior": "MergedWithPrevious",
                "invert": false
            },
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [],
                "special_tokens": {}
            },
            "added_tokens": [
                {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<unk>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 3, "content": "<eos>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<unk>",
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": true,
                "byte_fallback": true,
                "ignore_merges": false,
                "vocab": {
                    "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                    "▁": 4, "a": 5, "b": 6, "c": 7, "d": 8,
                    "▁a": 9, "ab": 10, "▁ab": 11, "abc": 12, "cd": 13,
                    "<0x41>": 14
                },
                "merges": [
                    ["▁", "a"],
                    ["a", "b"],
                    ["▁a", "b"],
                    ["c", "d"],
                    ["ab", "c"]
                ]
            }
        });
        let config = json!({
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "unk_token": "<unk>"
        });

        (root, config)
    }

    fn tokenizer() -> Tokenizer {
        let (root, config) = asset();
        Tokenizer::from_json(&root, &config).expect("the miniature asset loads")
    }

    #[test]
    fn the_space_becomes_the_marker_before_the_merges() {
        // " a b" -> "▁a▁b": the marker merges with 'a' (rank 0), so two ids.
        assert_eq!(tokenizer().ids(" a b"), vec![9, 4, 6]);
    }

    #[test]
    fn the_lowest_rank_wins_and_the_leftmost_position_breaks_a_tie() {
        // "abab" -> [(a,b) rank 1 at 0, (a,b) rank 1 at 2]; the leftmost is
        // merged first, then the pair either side is reconsidered. Both
        // occurrences merge, and "ab" has no further merge.
        assert_eq!(tokenizer().ids("abab"), vec![10, 10]);
        // "▁ab" has ("▁","a") rank 0 and ("a","b") rank 1: the lower rank
        // first gives "▁a" then ("▁a","b") rank 2, so one id, not two.
        assert_eq!(tokenizer().ids(" ab"), vec![11]);
    }

    #[test]
    fn an_unknown_character_without_byte_symbols_is_one_fused_unk() {
        // "z" is absent from the vocabulary and "<0x7A>" is absent too, so the
        // character is an unk; three in a row fuse into one id.
        assert_eq!(tokenizer().ids("zzz"), vec![1]);
        // A known character after an unknown one flushes the unk first.
        assert_eq!(tokenizer().ids("zab"), vec![1, 10]);
    }

    #[test]
    fn the_byte_fallback_reaches_only_the_symbols_the_vocab_carries() {
        // 'A' is absent, but "<0x41>" is present, so the character becomes that
        // byte symbol rather than an unk.
        assert_eq!(tokenizer().ids("A"), vec![14]);
    }

    #[test]
    fn added_tokens_are_split_out_of_the_raw_text_before_normalizing() {
        let tokenizer = tokenizer();
        // "<bos>" is matched as an added token, so it does not go through the
        // merges that would otherwise build it from "<", "b", "o", "s".
        assert_eq!(tokenizer.ids("<bos>abc"), vec![2, 12]);
    }

    #[test]
    fn the_wrapper_pads_left_and_keeps_the_bos_first() {
        let tokenizer = tokenizer();
        let (ids, mask) = tokenizer.tokenize("ab");
        let body = MAX_LENGTH - 2;

        assert_eq!(ids.len(), MAX_LENGTH);
        assert_eq!(mask.len(), MAX_LENGTH);
        assert_eq!(ids[body], tokenizer.bos());
        assert_eq!(&ids[body + 1..], &[10]);
        assert_eq!(mask[body..], [1, 1]);
        assert!(mask[..body].iter().all(|value| *value == 0));
        assert!(ids[..body].iter().all(|id| *id == tokenizer.pad));
    }

    #[test]
    fn a_prompt_that_already_starts_with_bos_is_not_doubled() {
        let tokenizer = tokenizer();
        let (ids, _) = tokenizer.tokenize("<bos>ab");
        let body = MAX_LENGTH - 2;

        assert_eq!(ids[body], tokenizer.bos());
        assert_eq!(&ids[body + 1..], &[10]);
    }

    #[test]
    fn a_pipeline_this_port_does_not_implement_is_refused_by_name() {
        let (mut root, config) = asset();
        root["normalizer"]["content"] = json!("_");

        let error = refuse(&root, &config);
        assert!(
            error.token().starts_with("token: normalizer"),
            "{}",
            error.token()
        );

        let (mut root, config) = asset();
        root["model"]["byte_fallback"] = json!(false);

        let error = refuse(&root, &config);
        assert!(
            error.token().starts_with("token: model"),
            "{}",
            error.token()
        );
    }

    /// The refusal of a pipeline the port does not implement.
    fn refuse(root: &Value, config: &Value) -> GemmaError {
        match Tokenizer::from_json(root, config) {
            Ok(_) => panic!("that asset should have been refused"),
            Err(error) => error,
        }
    }
}
