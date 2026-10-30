# The Gemma4 tokenizer and prompt assembly (2026-10-30)

Unit 3. `minimax_h3::gemma_token` reads the checkpoint's own tokenizer assets and
runs the release's prompt path; the source read is
[`gemma-tokenizer-port-map-2026-10-30.md`](gemma-tokenizer-port-map-2026-10-30.md).

## 1. What landed

- `crates/ds4-core/src/minimax_h3/gemma_token.rs` (new): the pipeline check, the
  vocabulary and merge-rank maps, the raw-text added-token split, the `Replace`
  normalizer, the character alphabet with the reference's own byte-fallback
  reach, the merge order, and the wrapper's strip / truncate / BOS / left-pad.
- `crates/ds4-core/src/minimax_h3/gemma.rs`: `GemmaError::Token`.
- `crates/ds4-core/examples/h3_gemma_gate.rs`: `--prompt TEXT` runs the tower on
  the checkpoint's own tokenizer, and `--tokens-only` prints the ids and stops
  before any device work, so the tokenizer is checkable without loading 12.9 GB.

The operator's decision (2026-10-30) is the repo's own BPE rather than the
`tokenizers` crate, so this is a re-implementation and the deviation is named
and measured against the golden rather than assumed equivalent.

## 2. The pipeline, and the three points taken from the reference

The asset is SentencePiece-shaped:

    normalizer     Replace  " " -> "▁"
    pre_tokenizer  Split    " ", MergedWithPrevious
    post_processor TemplateProcessing with no special tokens
    model          BPE, unk "<unk>", fuse_unk, byte_fallback, no ignore_merges

Three consequences the port depends on, each read from the reference rather
than inferred:

1. **The merge order is rank first and position second**
   (`tokenizers-0.22.1/src/models/bpe/word.rs:26-35`), so the leftmost
   lowest-rank pair merges first, one occurrence at a time. The reference keeps
   that order in a min-heap; a scan finds the same element each round, so the
   sequence of merges is identical and a heap is not needed.
2. **The byte fallback is spelled uppercase** (`models/bpe/model.rs:421`,
   `format!("<{b:#04X}>")`) and reaches only the symbols the vocabulary carries
   - 100 of the 256, `<0x00>` to `<0x99>`. A byte whose spelling is absent makes
   the `Option<Vec<_>>` at `:419-427` `None`, so the whole **character** becomes
   an unk; a port that minted a byte token on demand would silently differ.
3. **The pre-tokenizer's split is the identity here**: it splits on a literal
   space (`pre_tokenizers/split.rs:141-149`), and the normalizer ahead of it has
   already replaced every space with `▁` (`normalizers/replace.rs:82-86`). The
   port asserts that instead of implementing a `MergedWithPrevious` arm this
   asset cannot reach.

## 3. Verification

**Unit tests**, eight of them, on a miniature asset with the same pipeline shape:
the marker's interaction with the merges, the rank-then-position order, the
fused unk, the byte fallback's reach, the added-token split, the wrapper's
left padding and BOS guard, and two refusals of a pipeline the port does not
implement. `cargo test -p ds4-core --lib` is 734 passing.

**The real asset**, over the mount: vocabulary 262144, `bos` 2, `pad` 0, and
`A golden retriever running on a beach at sunset.` becomes

    2 236776 13935 102202 4710 580 496 7642 657 14711 236761

which the vocabulary spells `A`, `▁golden`, `▁retriever`, `▁running`, `▁on`,
`▁a`, `▁beach`, `▁at`, `▁sunset`, `.` - the marker exactly where the text had a
space, and the BOS the wrapper prepends. Three more, each checked the same way:

| Prompt | Result |
|---|---|
| `En golden retriever springer på en strand i solnedgång.` | `▁En ▁golden ▁retriever ▁springer ▁på ▁en ▁strand ▁i ▁sol` `ned` `gång` `.` - the compound splits, as SentencePiece does |
| `a 🐕 on the beach` | `a` `▁` `🐕` `▁on ▁the ▁beach` - a standalone space marker before the emoji, which has no merge partner |
| `  leading spaces  ` | `leading` `▁spaces` - stripped, so no leading marker |
| 1,500 words | 1024 ids, `padding=0`: the truncation is before the BOS insert, as `tokenizer.py:51-55` has it |

**Both hosts agree on the ids**, byte for byte, for every prompt above
(`--tokens-only` on the workstation and on the Spark, `IDS` lines diffed).

**The end to end**, Spark, the same prompt through the tokenizer and then the
whole tower: `--prompt 'A golden retriever running on a beach at sunset.'
--layers 48` ran 994 tensors / 12,923,715,168 B, load 2.280 s, run 2.161 s and
printed 49 states, `GEMMA-TOWER-OK`. Its first, second and last states are

    STATE 0  8eb1abc7748aa2cd
    STATE 1  2260ca0969c8fb52
    STATE 48 f57b4042aea422d6

These are the run on this prompt, not the synthetic one
([`gemma-tower-graph-2026-10-30.md`](gemma-tower-graph-2026-10-30.md), section
7); the golden's prompt replaces both.

## 4. Limits

- **The gate is the operator's golden.** There is no reference tokenizer on
  either host - neither has Python with `tokenizers`, and the operator's
  decision keeps the crate out of the workspace - so what is checked here is the
  pipeline shape, the algorithm against its own unit tests, and the ids against
  the vocabulary they came from. Only the golden's ids and mask state what the
  reference produces.
- The read is of `tokenizers` 0.22.1: `transformers` 5.14.1 requires
  `>=0.22.0,<=0.23.0`, 0.23.0 is not on crates.io, and the algorithm is stable
  across that range - but the reference environment's installed patch version is
  not pinned anywhere, and the golden is what tests that assumption.
- Decode is not implemented: the conditioning path only encodes, and the
  reference's decoder (a `Sequence` of `Replace`, `ByteFallback` and `Fuse`) is
  not on it.
- The added-token matcher is a leftmost, longest-on-ties literal scan rather
  than the reference's regex set. They agree unless two added tokens overlap,
  which none in this asset do.

## 5. What unit 4 needs

Nothing from this unit: the connector consumes the tower's 49 hidden states
(`gemma-tower-graph-2026-10-30.md`, section 7). Its gate is the golden's
connector contexts, and its weights are BF16 until the missing int8 file lands.
