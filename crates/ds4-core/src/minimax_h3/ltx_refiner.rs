//! The LTX-2.5 refiner's transformer body: unit 6, the block the whole
//! stage-2 graph is 48 copies of.
//!
//! Source of truth: LTX's own `ltx-core/src/ltx_core/model/transformer/`, read
//! first-hand at `/data/ltx2-pin/packages` (the pin the release runs,
//! `d151147788a9284cca791edc6ce898007e727fe6` = tag v1.2.0). Every step below
//! cites it:
//!
//! - `transformer.py:87-190` `BasicAVTransformerBlock`: six attention modules
//!   (video self/cross, audio self/cross, A2V, V2A), two feed-forwards and six
//!   AdaLN tables, replicated 48 times with no per-layer variation
//!   (`model.py:407-417`).
//! - `transformer.py:253-417` the forward order: video self-attention, video
//!   text cross-attention, audio self- and text cross-attention, the A2V/V2A
//!   pair from a pre-mutation snapshot (`:331-334`), then the two
//!   feed-forwards.
//! - `ops.py:22-36` `PytorchPreAttention` (the Q/K RMS norm over the whole
//!   inner dimension, then the rotary pair), `ops.py:50-58`
//!   `PytorchAdaZeroFunction`, `ops.py:74-82` `PytorchPostSAFunction`,
//!   `ops.py:94-106` `PytorchGatedAttention`.
//! - `attention.py:478-580` `Attention`: biased projections, `q_norm`/`k_norm`
//!   as `RMSNorm(inner_dim)`, the per-head gate taken from the module's own
//!   input *before* `to_out`, and an ungated output projection.
//! - `transformer.py:190-220` the AdaLN value (`table row + modulation row`),
//!   `:239-252` and `:421-446` the text cross-attention's query/KV modulation,
//!   `:328-397` the AV pair.
//!
//! Four facts this port must not lose:
//!
//! - **Torch rounds BF16 elementwise ops one at a time.** `opmath_type<
//!   BFloat16>` is `float`, so each op computes in FP32 and rounds once, and
//!   every expression in the source is a chain of separate ops:
//!   `rms_norm(x) * (1 + scale) + shift` is four ops and four roundings. The
//!   kernels round where the expression does; a fused single-rounding form is
//!   a different function.
//! - **A2V and V2A both read the state before either mutates it**
//!   (`transformer.py:331-334`, so the direction order does not bias the
//!   result). `ada_zero` is pure, so this port forms all four modulated inputs
//!   before running either attention - the reference's function without its
//!   two 132 MB copies.
//! - **The AdaLN tables are F32 in the checkpoint** and the reference casts
//!   them to the timestep's BF16 before adding (`transformer.py:195-197`), so
//!   they are rounded once at load and the block's add runs on BF16 operands.
//! - **The per-head gate reads the module's own input**, not the context, and
//!   is applied before `to_out` (`attention.py:514`, `:576-582`).
//!
//! The block reads and writes its modality state in place, exactly as the
//! reference's `vx = vx + ...` chain does; every intermediate lives in
//! [`RefinerScratch`], which 48 blocks share.
//!
//! Not this unit: the preprocessing (patchify, positions and RoPE tables, the
//! AdaLN MLPs, the keyframe term) and the output head. The block takes its
//! modulation and positional tables as [`Signals`] - exactly what a block reads
//! besides its own weights - so it is verifiable alone and the preprocessing
//! lands without touching it.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype as Device, Gpu, Tensor};

use crate::minimax_h3::gemma::bf16_round;
use crate::minimax_h3::safetensors::{self, Dtype, Shard};

/// `num_layers` (`model_configurator.py:222-225`).
pub const LAYERS: usize = 48;
/// `num_attention_heads` (`model.py:53`), every attention module.
pub const HEADS: usize = 32;
/// `attention_head_dim` (`model.py:54`).
pub const VIDEO_HEAD_DIM: usize = 128;
/// `audio_attention_head_dim` (`model.py:65`): the audio and both AV modules.
pub const AUDIO_HEAD_DIM: usize = 64;
/// `attention_head_dim * num_attention_heads` (`model.py:108`).
pub const VIDEO_DIM: usize = 4096;
/// `audio_attention_head_dim * audio_num_attention_heads`.
pub const AUDIO_DIM: usize = 2048;
/// `FeedForward`'s `mult` (`feed_forward.py:8`).
pub const MULTIPLIER: usize = 4;
/// The latent width both streams are patchified to (`in_channels` 128).
pub const CHANNELS: usize = 128;
/// `cross_attention_dim` (`model.py:57`), the video text context's width.
pub const VIDEO_CONTEXT_DIM: usize = 4096;
/// `audio_cross_attention_dim` (`model.py:68`), the audio text context's.
pub const AUDIO_CONTEXT_DIM: usize = 2048;
/// `adaln_embedding_coefficient(True)` (`adaln.py:11-22`): 6 base + 3 cross.
pub const COEFFICIENT: usize = 9;
/// `_init_audio_video(num_scale_shift_values=4)` (`model.py:132`): the A2V/V2A
/// scale/shift row pairs, with the gate on the table's fifth row.
pub const CROSS_COEFFICIENT: usize = 4;
/// `embedding_coefficient=1` for the two AV gate AdaLN singles.
pub const GATE_COEFFICIENT: usize = 1;
/// `embedding_coefficient=2` for the two prompt AdaLN singles.
pub const PROMPT_COEFFICIENT: usize = 2;
/// `norm_eps` (`model.py:58`), every RMS and Layer norm in the block.
pub const RMS_EPSILON: f32 = 1e-6;
/// `positional_embedding_theta` (`model.py:60`).
pub const ROPE_BASE: f64 = 10_000.0;
/// `cross_pe_max_pos` = `max(max_pos[0], audio_max_pos[0])` (`model.py:130`):
/// the cross-attention's RoPE is the time axis only.
pub const CROSS_MAX_POS: f32 = 20.0;
/// The release's video geometry: the refiner latent is `(1, 128, 16, 24, 42)`
/// and patchifies at patch size 1 to `16 * 24 * 42` tokens
/// (`runtime/stage2_ops/h3_upscale.py`, `components/patchifiers.py:31-38`).
pub const VIDEO_TOKENS: u32 = 16 * 24 * 42;
/// `AudioLatentShape.from_video_pixel_shape` at 121 frames
/// (`core/types.py:185-201`): `round(121 / 24 * 16000 / 160 / 4)`.
pub const AUDIO_TOKENS: u32 = 126;
/// The tokenizer's fixed prompt length (`gemma_assets.py:161`), the prompt
/// context's row count.
pub const CONTEXT_ROWS: u32 = 1024;

/// The AV pair's table rows: `slice(0, 2)` for A2V and `slice(2, 4)` for V2A
/// (`transformer.py:341`, `:369`).
const A2V_ROWS: u32 = 0;
const V2A_ROWS: u32 = 2;
/// The gate's row of the five-row AV tables (`transformer.py:211-212`).
const AV_GATE_ROW: u32 = 4;

/// Why the refiner's body could not be loaded or run.
#[derive(Debug)]
pub enum RefinerError {
    Reader(String),
    Missing(String),
    Shape { name: String, detail: String },
    Run(String),
}

impl RefinerError {
    /// The one-line token a record quotes.
    pub fn token(&self) -> String {
        match self {
            RefinerError::Reader(detail) => format!("reader: {detail}"),
            RefinerError::Missing(name) => format!("missing: {name}"),
            RefinerError::Shape { name, detail } => format!("shape: {name}: {detail}"),
            RefinerError::Run(detail) => format!("run: {detail}"),
        }
    }
}

impl fmt::Display for RefinerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for RefinerError {}

/// Whether a `torch.nn.Linear` in this checkpoint carries a bias. The video
/// feed-forward is bias-free while the audio one is not (`model.py:76-81`:
/// `ff_bias false` with `audio_ff_bias` absent, so it keeps its `True`
/// default).
#[derive(Clone, Copy)]
enum Bias {
    Present,
    Absent,
}

/// Which of a modality's two AV operands a modulation writes: A2V queries with
/// the video and keys with the audio, V2A the other way round
/// (`transformer.py:151-177`).
#[derive(Clone, Copy)]
enum Side {
    Query,
    Context,
}

/// One BF16 `torch.nn.Linear`. The reference fuses its bias into the GEMM's
/// accumulate; [`Gpu::linear`] adds it after the matmul's rounding (a named
/// deviation, the same one unit 4 carries).
struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    input_dim: u32,
    output_dim: u32,
}

impl Linear {
    fn apply(
        &self,
        gpu: &Gpu,
        output: &Tensor,
        input: &Tensor,
        rows: u32,
    ) -> Result<(), RefinerError> {
        gpu.linear(
            output,
            input,
            &self.weight,
            self.bias.as_ref(),
            rows,
            self.input_dim,
            self.output_dim,
            Device::Bf16,
        )
        .map_err(RefinerError::Run)
    }
}

/// One `FeedForward` (`feed_forward.py:4-15`).
struct FeedForward {
    up: Linear,
    down: Linear,
}

/// One `Attention` (`attention.py:478-580`).
struct Attention {
    /// The checkpoint's module name, so a shape fault names itself.
    name: String,
    q_norm: Tensor,
    k_norm: Tensor,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    /// `to_gate_logits = Linear(query_dim, heads)` (`attention.py:514`), read
    /// off the module's own input rather than the context.
    gate: Linear,
    inner: u32,
    heads: u32,
    head_dim: u32,
}

/// Everything one modality contributes to a block besides its weights: the
/// AdaLN modulation rows the preprocessing computed, the text context and the
/// positional tables. `modulation` is `[rows, 9, dim]` and `cross_scale_shift`
/// `[rows, 4, dim]` - the shapes `timestep.reshape(batch, T, num_ada_params,
/// -1)` produces (`transformer.py:197-198`) - while `cross_gate` and
/// `prompt_modulation` are per-request, one row each (`transformer.py:207-220`,
/// `:441-443`).
pub struct Signals {
    pub rows: u32,
    pub dim: u32,
    pub modulation: Tensor,
    pub cross_scale_shift: Tensor,
    pub cross_gate: Tensor,
    pub prompt_modulation: Tensor,
    pub context: Tensor,
    pub context_rows: u32,
    pub self_cos: Tensor,
    pub self_sin: Tensor,
    /// One `[rows, 32, head_dim / 2]` pair per modality for the cross RoPE:
    /// the time axis at `audio_cross_attention_dim` 2048 with one position
    /// dimension (`transformer_args.py:365-370`).
    pub cross_cos: Tensor,
    pub cross_sin: Tensor,
}

/// The model's body: the blocks this run loaded, in shard order.
pub struct Refiner {
    blocks: Vec<Block>,
    tensors: usize,
    bytes: u64,
}

impl Refiner {
    /// Read the transformer shard's `layers`, in the order given. The pipeline
    /// loads `0..LAYERS`; a gate loads one block and pays for one.
    pub fn open(gpu: &Gpu, path: &Path, layers: &[usize]) -> Result<Refiner, RefinerError> {
        let shard =
            safetensors::read_shard(path).map_err(|error| RefinerError::Reader(error.token()))?;

        if layers.is_empty() {
            return Err(RefinerError::Run("no refiner layer selected".into()));
        }

        let mut reader = Reader {
            gpu,
            buffer: Vec::new(),
            tensors: 0,
            bytes: 0,
        };
        let mut blocks = Vec::with_capacity(layers.len());

        for layer in layers {
            if *layer >= LAYERS {
                return Err(RefinerError::Run(format!(
                    "layer {layer} is past the model's {LAYERS}"
                )));
            }

            blocks.push(Block::load(&mut reader, &shard, *layer)?);
        }

        Ok(Refiner {
            blocks,
            tensors: reader.tensors,
            bytes: reader.bytes,
        })
    }

    /// The tensors read, and their bytes, for the gate's record.
    pub fn loaded(&self) -> (usize, u64) {
        (self.tensors, self.bytes)
    }

    /// The blocks held.
    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    /// `LTXModel._process_transformer_blocks` (`model.py:434-470`) over the
    /// blocks this run loaded. `video` and `audio` are the modality states,
    /// read and written in place.
    pub fn apply(
        &self,
        gpu: &Gpu,
        video: &Tensor,
        audio: &Tensor,
        video_signals: &Signals,
        audio_signals: &Signals,
        scratch: &RefinerScratch,
    ) -> Result<(), RefinerError> {
        for block in &self.blocks {
            block.apply(gpu, video, audio, video_signals, audio_signals, scratch)?;
        }

        Ok(())
    }

    /// One block, by index into the loaded set.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_one(
        &self,
        index: usize,
        gpu: &Gpu,
        video: &Tensor,
        audio: &Tensor,
        video_signals: &Signals,
        audio_signals: &Signals,
        scratch: &RefinerScratch,
    ) -> Result<(), RefinerError> {
        let block = self
            .blocks
            .get(index)
            .ok_or_else(|| RefinerError::Run(format!("no block {index} is loaded")))?;

        block.apply(gpu, video, audio, video_signals, audio_signals, scratch)
    }
}

/// One `BasicAVTransformerBlock` (`transformer.py:87`).
struct Block {
    video_self: Attention,
    video_cross: Attention,
    audio_self: Attention,
    audio_cross: Attention,
    audio_to_video: Attention,
    video_to_audio: Attention,
    video_ff: FeedForward,
    audio_ff: FeedForward,
    video_table: Tensor,
    audio_table: Tensor,
    av_video_table: Tensor,
    av_audio_table: Tensor,
    video_prompt_table: Tensor,
    audio_prompt_table: Tensor,
}

impl Block {
    fn load(reader: &mut Reader<'_>, shard: &Shard, layer: usize) -> Result<Block, RefinerError> {
        let at = format!("model.diffusion_model.transformer_blocks.{layer}");
        let video = (HEADS, VIDEO_HEAD_DIM);
        let audio = (HEADS, AUDIO_HEAD_DIM);

        Ok(Block {
            video_self: reader.attention(
                shard,
                &format!("{at}.attn1"),
                VIDEO_DIM,
                VIDEO_DIM,
                video,
            )?,
            video_cross: reader.attention(
                shard,
                &format!("{at}.attn2"),
                VIDEO_DIM,
                VIDEO_CONTEXT_DIM,
                video,
            )?,
            audio_self: reader.attention(
                shard,
                &format!("{at}.audio_attn1"),
                AUDIO_DIM,
                AUDIO_DIM,
                audio,
            )?,
            audio_cross: reader.attention(
                shard,
                &format!("{at}.audio_attn2"),
                AUDIO_DIM,
                AUDIO_CONTEXT_DIM,
                audio,
            )?,
            // Both AV modules take their heads and head width from the *audio*
            // config (`transformer.py:151-177`).
            audio_to_video: reader.attention(
                shard,
                &format!("{at}.audio_to_video_attn"),
                VIDEO_DIM,
                AUDIO_DIM,
                audio,
            )?,
            video_to_audio: reader.attention(
                shard,
                &format!("{at}.video_to_audio_attn"),
                AUDIO_DIM,
                VIDEO_DIM,
                audio,
            )?,
            video_ff: FeedForward {
                up: reader.linear(
                    shard,
                    &format!("{at}.ff.net.0.proj"),
                    VIDEO_DIM,
                    VIDEO_DIM * MULTIPLIER,
                    Bias::Absent,
                )?,
                down: reader.linear(
                    shard,
                    &format!("{at}.ff.net.2"),
                    VIDEO_DIM * MULTIPLIER,
                    VIDEO_DIM,
                    Bias::Absent,
                )?,
            },
            audio_ff: FeedForward {
                up: reader.linear(
                    shard,
                    &format!("{at}.audio_ff.net.0.proj"),
                    AUDIO_DIM,
                    AUDIO_DIM * MULTIPLIER,
                    Bias::Present,
                )?,
                down: reader.linear(
                    shard,
                    &format!("{at}.audio_ff.net.2"),
                    AUDIO_DIM * MULTIPLIER,
                    AUDIO_DIM,
                    Bias::Present,
                )?,
            },
            video_table: reader.table(
                shard,
                &format!("{at}.scale_shift_table"),
                COEFFICIENT,
                VIDEO_DIM,
            )?,
            audio_table: reader.table(
                shard,
                &format!("{at}.audio_scale_shift_table"),
                COEFFICIENT,
                AUDIO_DIM,
            )?,
            av_video_table: reader.table(
                shard,
                &format!("{at}.scale_shift_table_a2v_ca_video"),
                CROSS_COEFFICIENT + 1,
                VIDEO_DIM,
            )?,
            av_audio_table: reader.table(
                shard,
                &format!("{at}.scale_shift_table_a2v_ca_audio"),
                CROSS_COEFFICIENT + 1,
                AUDIO_DIM,
            )?,
            video_prompt_table: reader.table(
                shard,
                &format!("{at}.prompt_scale_shift_table"),
                PROMPT_COEFFICIENT,
                VIDEO_DIM,
            )?,
            audio_prompt_table: reader.table(
                shard,
                &format!("{at}.audio_prompt_scale_shift_table"),
                PROMPT_COEFFICIENT,
                AUDIO_DIM,
            )?,
        })
    }

    /// `BasicAVTransformerBlock.forward` (`transformer.py:253-417`). `video`
    /// and `audio` are `vx` and `ax`, read and written in place.
    ///
    /// Every branch runs: `run_vx` and `run_ax` are always true on this route
    /// (the release builds an `AudioVideo` model with both latents non-empty,
    /// `runtime/stage2_ops/models.py:47-134`), and `cross_attn_skip_all` is
    /// false because the perturbation config is empty.
    fn apply(
        &self,
        gpu: &Gpu,
        video: &Tensor,
        audio: &Tensor,
        v: &Signals,
        a: &Signals,
        s: &RefinerScratch,
    ) -> Result<(), RefinerError> {
        // 1-2. Video self-attention, then the text cross-attention
        // (`transformer.py:270-298`).
        self_and_text(
            gpu,
            video,
            &self.video_self,
            &self.video_cross,
            &self.video_table,
            &self.video_prompt_table,
            v,
            &s.video,
        )?;

        // 3. Audio self-attention and text cross-attention (`:300-326`).
        self_and_text(
            gpu,
            audio,
            &self.audio_self,
            &self.audio_cross,
            &self.audio_table,
            &self.audio_prompt_table,
            a,
            &s.audio,
        )?;

        // 4. A2V then V2A (`:328-397`): all four modulated operands are formed
        // from the pre-A2V state, before either attention runs.
        av_modulate(
            gpu,
            video,
            v,
            &self.av_video_table,
            A2V_ROWS,
            Side::Query,
            &s.video,
        )?;
        av_modulate(
            gpu,
            audio,
            a,
            &self.av_audio_table,
            A2V_ROWS,
            Side::Context,
            &s.audio,
        )?;
        av_modulate(
            gpu,
            audio,
            a,
            &self.av_audio_table,
            V2A_ROWS,
            Side::Query,
            &s.audio,
        )?;
        av_modulate(
            gpu,
            video,
            v,
            &self.av_video_table,
            V2A_ROWS,
            Side::Context,
            &s.video,
        )?;

        av_direction(
            gpu,
            &self.audio_to_video,
            &self.av_video_table,
            v,
            video,
            &s.video,
            a,
            &s.audio,
        )?;
        av_direction(
            gpu,
            &self.video_to_audio,
            &self.av_audio_table,
            a,
            audio,
            &s.audio,
            v,
            &s.video,
        )?;

        // 5. The two feed-forwards (`:399-414`), rows 3..5, re-formed here
        // because the AV gate borrowed the same buffers.
        feed_forward(gpu, video, &self.video_ff, &self.video_table, v, &s.video)?;
        feed_forward(gpu, audio, &self.audio_ff, &self.audio_table, a, &s.audio)
    }
}

/// `transformer.py:270-298` (video) and `:300-326` (audio): the self-attention
/// step, then the text cross-attention step, both gated into the residual.
///
/// `post_sa_function` (`ops.py:74-82`) returns `x + y * gate` *and* the RMS
/// norm of that sum, which is the cross-attention's input - normalized once,
/// not twice (`transformer.py:283-289`).
#[allow(clippy::too_many_arguments)]
fn self_and_text(
    gpu: &Gpu,
    state: &Tensor,
    self_attention: &Attention,
    cross_attention: &Attention,
    table: &Tensor,
    prompt_table: &Tensor,
    signals: &Signals,
    m: &ModalityScratch,
) -> Result<(), RefinerError> {
    let (rows, dim) = (signals.rows, signals.dim);

    for row in 0..3u32 {
        ada(
            gpu,
            m.ada(row),
            table,
            &signals.modulation,
            rows,
            dim,
            COEFFICIENT as u32,
            row,
        )?;
    }

    gpu.ada_zero(
        &m.normed,
        state,
        &m.scale,
        &m.shift,
        rows,
        dim,
        rows,
        RMS_EPSILON,
    )
    .map_err(RefinerError::Run)?;

    self_attention.run(
        gpu,
        &m.scratch,
        &m.attention,
        &m.normed,
        &m.normed,
        rows,
        rows,
        Some((
            &signals.self_cos,
            &signals.self_sin,
            &signals.self_cos,
            &signals.self_sin,
        )),
    )?;

    gpu.mul(&m.gated, &m.attention, &m.gate)
        .map_err(RefinerError::Run)?;
    gpu.add(&m.residual, state, &m.gated)
        .map_err(RefinerError::Run)?;
    gpu.rms_norm(&m.post_normed, &m.residual, None, rows, dim, RMS_EPSILON)
        .map_err(RefinerError::Run)?;

    // Rows 6..8 on the query and the 2-row prompt table on the context, with
    // no RoPE: `apply_cross_attention_adaln` is called without `pe`
    // (`transformer.py:239-252`), so `attn(...)` gets none.
    for row in 6..9u32 {
        ada(
            gpu,
            m.ada(row - 6),
            table,
            &signals.modulation,
            rows,
            dim,
            COEFFICIENT as u32,
            row,
        )?;
    }

    gpu.modulate(
        &m.text_input,
        &m.post_normed,
        &m.scale,
        &m.shift,
        rows,
        dim,
        rows,
    )
    .map_err(RefinerError::Run)?;

    for row in 0..2u32 {
        ada(
            gpu,
            m.ada(row),
            prompt_table,
            &signals.prompt_modulation,
            1,
            dim,
            PROMPT_COEFFICIENT as u32,
            row,
        )?;
    }

    gpu.modulate(
        &m.context,
        &signals.context,
        &m.scale,
        &m.shift,
        signals.context_rows,
        dim,
        1,
    )
    .map_err(RefinerError::Run)?;

    cross_attention.run(
        gpu,
        &m.scratch,
        &m.attention,
        &m.text_input,
        &m.context,
        rows,
        signals.context_rows,
        None,
    )?;

    // `attn(...) * q_gate` (`transformer.py:252`), then the residual.
    gpu.mul_rows(&m.gated, &m.attention, &m.gate, rows, dim, rows)
        .map_err(RefinerError::Run)?;
    gpu.add(&m.crossed, &m.residual, &m.gated)
        .map_err(RefinerError::Run)
}

/// `transformer.py:339-357` and `:371-386`: one direction's modulated operands.
/// `pair` is the AV table's row pair - 0 for A2V, 2 for V2A - and `side`
/// selects which of the two buffers this modality fills. Both halves are
/// computed before either attention runs, so the V2A keys are the pre-A2V
/// video state.
fn av_modulate(
    gpu: &Gpu,
    state: &Tensor,
    signals: &Signals,
    table: &Tensor,
    pair: u32,
    side: Side,
    m: &ModalityScratch,
) -> Result<(), RefinerError> {
    let (rows, dim) = (signals.rows, signals.dim);
    let target = match side {
        Side::Query => &m.cross_query,
        Side::Context => &m.cross_context,
    };

    for offset in 0..2u32 {
        ada(
            gpu,
            m.ada(offset),
            table,
            &signals.cross_scale_shift,
            rows,
            dim,
            CROSS_COEFFICIENT as u32,
            pair + offset,
        )?;
    }

    gpu.ada_zero(
        target,
        state,
        &m.scale,
        &m.shift,
        rows,
        dim,
        rows,
        RMS_EPSILON,
    )
    .map_err(RefinerError::Run)
}

/// `transformer.py:341-364` and `:368-394`: one direction of the AV pair. The
/// gate is the five-row table's fourth row plus this modality's gate
/// modulation, one row per request (`:207-220`), and the cross RoPE is the time
/// axis - this modality's table on the queries, the other's on the keys.
#[allow(clippy::too_many_arguments)]
fn av_direction(
    gpu: &Gpu,
    attention: &Attention,
    table: &Tensor,
    own_signals: &Signals,
    own_state: &Tensor,
    own: &ModalityScratch,
    other_signals: &Signals,
    other: &ModalityScratch,
) -> Result<(), RefinerError> {
    // The gate is the table's fifth row plus the gate modulation's own first
    // row: `get_ada_values(scale_shift_table[4:, :], ..., slice(None, None))`
    // (`transformer.py:211-218`), the one caller whose two row indices differ.
    gpu.ada_add(
        &own.gate,
        table,
        &own_signals.cross_gate,
        1,
        own_signals.dim,
        GATE_COEFFICIENT as u32,
        AV_GATE_ROW,
        0,
    )
    .map_err(RefinerError::Run)?;

    attention.run(
        gpu,
        &own.scratch,
        &own.attention,
        &own.cross_query,
        &other.cross_context,
        own_signals.rows,
        other_signals.rows,
        Some((
            &own_signals.cross_cos,
            &own_signals.cross_sin,
            &other_signals.cross_cos,
            &other_signals.cross_sin,
        )),
    )?;

    gpu.mul_rows(
        &own.gated,
        &own.attention,
        &own.gate,
        own_signals.rows,
        own_signals.dim,
        1,
    )
    .map_err(RefinerError::Run)?;
    gpu.add(own_state, own_state, &own.gated)
        .map_err(RefinerError::Run)
}

/// `transformer.py:399-414`: `x = x + ff(rms_norm(x) * (1 + scale) + shift) *
/// gate`, the FF a `Linear -> GELU(tanh) -> Linear` at `mult` 4
/// (`feed_forward.py:4-15`, `gelu_approx.py:10-11`).
fn feed_forward(
    gpu: &Gpu,
    state: &Tensor,
    ff: &FeedForward,
    table: &Tensor,
    signals: &Signals,
    m: &ModalityScratch,
) -> Result<(), RefinerError> {
    let (rows, dim) = (signals.rows, signals.dim);

    for row in 3..6u32 {
        ada(
            gpu,
            m.ada(row - 3),
            table,
            &signals.modulation,
            rows,
            dim,
            COEFFICIENT as u32,
            row,
        )?;
    }

    gpu.ada_zero(
        &m.normed,
        state,
        &m.scale,
        &m.shift,
        rows,
        dim,
        rows,
        RMS_EPSILON,
    )
    .map_err(RefinerError::Run)?;

    ff.up.apply(gpu, &m.hidden, &m.normed, rows)?;
    gpu.gelu_tanh(&m.hidden, &m.hidden)
        .map_err(RefinerError::Run)?;
    ff.down.apply(gpu, &m.down, &m.hidden, rows)?;
    gpu.mul_rows(&m.down, &m.down, &m.gate, rows, dim, rows)
        .map_err(RefinerError::Run)?;
    gpu.add(state, state, &m.down).map_err(RefinerError::Run)
}

/// `get_ada_values` (`transformer.py:190-200`): one table row plus the AdaLN
/// MLP's matching coefficient row, rounded once to BF16. The AV gate is the
/// only caller whose two row indices differ, and it calls `ada_add` directly.
#[allow(clippy::too_many_arguments)]
fn ada(
    gpu: &Gpu,
    output: &Tensor,
    table: &Tensor,
    modulation: &Tensor,
    rows: u32,
    width: u32,
    coefficient: u32,
    row: u32,
) -> Result<(), RefinerError> {
    gpu.ada_add(
        output,
        table,
        modulation,
        rows,
        width,
        coefficient,
        row,
        row,
    )
    .map_err(RefinerError::Run)
}

/// The attention module's own step: `Attention.forward` (`attention.py:
/// 521-579`) and `PytorchPreAttention` (`ops.py:22-36`). `input` is the
/// module's operand - the source of both the query projection and the gate
/// logits - and `context` the key/value source, which is `input` for
/// self-attention.
impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        gpu: &Gpu,
        scratch: &AttentionScratch,
        output: &Tensor,
        input: &Tensor,
        context: &Tensor,
        q_rows: u32,
        kv_rows: u32,
        rope: Option<(&Tensor, &Tensor, &Tensor, &Tensor)>,
    ) -> Result<(), RefinerError> {
        let named = |detail: String| RefinerError::Run(format!("{}: {detail}", self.name));

        self.query
            .apply(gpu, &scratch.query, input, q_rows)
            .map_err(|error| named(error.token()))?;
        self.key
            .apply(gpu, &scratch.key, context, kv_rows)
            .map_err(|error| named(error.token()))?;
        self.value
            .apply(gpu, &scratch.value, context, kv_rows)
            .map_err(|error| named(error.token()))?;

        // `q_norm`/`k_norm` are `RMSNorm(inner_dim)` over the whole inner
        // dimension and *before* the head split (`attention.py:505-506`).
        gpu.rms_norm(
            &scratch.query,
            &scratch.query,
            Some(&self.q_norm),
            q_rows,
            self.inner,
            RMS_EPSILON,
        )
        .map_err(named)?;
        gpu.rms_norm(
            &scratch.key,
            &scratch.key,
            Some(&self.k_norm),
            kv_rows,
            self.inner,
            RMS_EPSILON,
        )
        .map_err(named)?;

        if let Some((q_cos, q_sin, k_cos, k_sin)) = rope {
            gpu.ltx_rope_pair(
                &scratch.query,
                &scratch.key,
                q_cos,
                q_sin,
                k_cos,
                k_sin,
                q_rows,
                kv_rows,
                self.heads,
                self.head_dim,
            )
            .map_err(named)?;
        }

        // `scaled_dot_product_attention` at its default scale with no mask
        // (`attention.py:99-100`), the operands token-major, which is what the
        // projections above produce.
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        gpu.ltx_attention(
            &scratch.attention,
            &scratch.query,
            &scratch.key,
            &scratch.value,
            q_rows,
            kv_rows,
            self.heads,
            self.head_dim,
            scale,
        )
        .map_err(named)?;

        // `PytorchGatedAttention` (`ops.py:94-106`): the logits come from the
        // module's input, not the context, and the gate is applied before
        // `to_out`, which is therefore ungated (`attention.py:576-582`).
        self.gate
            .apply(gpu, &scratch.logits, input, q_rows)
            .map_err(|error| named(error.token()))?;
        gpu.per_head_gate(
            &scratch.gated,
            &scratch.attention,
            &scratch.logits,
            q_rows,
            self.heads,
            self.head_dim,
        )
        .map_err(named)?;

        self.output
            .apply(gpu, output, &scratch.gated, q_rows)
            .map_err(|error| named(error.token()))
    }
}

/// The block-sized buffers one refiner run reuses, so 48 blocks allocate one
/// set rather than 48.
pub struct RefinerScratch {
    video: ModalityScratch,
    audio: ModalityScratch,
}

impl RefinerScratch {
    /// `rows` is the video token count and `audio_rows` the audio one, which
    /// must not exceed it (126 against 16,128 on the release's geometry: the
    /// AV pair gives the audio's attention scratch the video's query rows).
    /// `context_rows` is the prompt length, 1024.
    pub fn new(
        gpu: &Gpu,
        rows: u32,
        audio_rows: u32,
        context_rows: u32,
    ) -> Result<RefinerScratch, RefinerError> {
        Ok(RefinerScratch {
            video: ModalityScratch::new(gpu, rows, VIDEO_DIM, context_rows, rows, context_rows)?,
            // The audio's attention scratch takes the video's row count: the AV
            // pair queries 16,128 video rows and keys the audio's 126, and V2A
            // is the other way round.
            audio: ModalityScratch::new(
                gpu,
                audio_rows,
                AUDIO_DIM,
                context_rows,
                rows,
                context_rows,
            )?,
        })
    }
}

/// One modality's working set. `dim` is the modality's inner width, `rows` its
/// token count, `scratch_rows` the largest query row count any attention in
/// the block sees, and `key_rows` the largest key/value count - which the text
/// cross-attention sets to the prompt length.
struct ModalityScratch {
    normed: Tensor,
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
    gated: Tensor,
    residual: Tensor,
    post_normed: Tensor,
    text_input: Tensor,
    context: Tensor,
    crossed: Tensor,
    cross_query: Tensor,
    cross_context: Tensor,
    hidden: Tensor,
    down: Tensor,
    attention: Tensor,
    scratch: AttentionScratch,
}

impl ModalityScratch {
    fn new(
        gpu: &Gpu,
        rows: u32,
        dim: usize,
        context_rows: u32,
        scratch_rows: u32,
        key_rows: u32,
    ) -> Result<ModalityScratch, RefinerError> {
        let row = rows as usize;
        let context = context_rows as usize;

        Ok(ModalityScratch {
            normed: alloc(gpu, row * dim)?,
            shift: alloc(gpu, row * dim)?,
            scale: alloc(gpu, row * dim)?,
            gate: alloc(gpu, row * dim)?,
            gated: alloc(gpu, row * dim)?,
            residual: alloc(gpu, row * dim)?,
            post_normed: alloc(gpu, row * dim)?,
            text_input: alloc(gpu, row * dim)?,
            context: alloc(gpu, context * dim)?,
            crossed: alloc(gpu, row * dim)?,
            cross_query: alloc(gpu, row * dim)?,
            cross_context: alloc(gpu, row * dim)?,
            hidden: alloc(gpu, row * dim * MULTIPLIER)?,
            down: alloc(gpu, row * dim)?,
            attention: alloc(gpu, row * dim)?,
            scratch: AttentionScratch::new(gpu, scratch_rows.max(key_rows), dim)?,
        })
    }

    /// The three AdaLN values one step needs, so a caller can walk
    /// `(shift, scale, gate)` instead of naming the fields.
    fn ada(&self, index: u32) -> &Tensor {
        match index {
            0 => &self.shift,
            1 => &self.scale,
            _ => &self.gate,
        }
    }
}

/// The working tensors one attention module moves its operands through. Every
/// kernel that consumes them takes the row count explicitly, so surplus rows
/// are inert.
struct AttentionScratch {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention: Tensor,
    logits: Tensor,
    gated: Tensor,
}

impl AttentionScratch {
    fn new(gpu: &Gpu, rows: u32, inner: usize) -> Result<AttentionScratch, RefinerError> {
        let rows = rows as usize;

        Ok(AttentionScratch {
            query: alloc(gpu, rows * inner)?,
            key: alloc(gpu, rows * inner)?,
            value: alloc(gpu, rows * inner)?,
            attention: alloc(gpu, rows * inner)?,
            logits: alloc(gpu, rows * HEADS)?,
            gated: alloc(gpu, rows * inner)?,
        })
    }
}

/// The shard's reader: one reusable host buffer, and the counters a gate
/// reports. Every read checks the checkpoint's own descriptor against what the
/// graph will do with the tensor before anything reaches the device.
struct Reader<'a> {
    gpu: &'a Gpu,
    buffer: Vec<u8>,
    tensors: usize,
    bytes: u64,
}

impl Reader<'_> {
    fn pull(
        &mut self,
        shard: &Shard,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
    ) -> Result<Tensor, RefinerError> {
        let tensor = shard
            .find(name)
            .ok_or_else(|| RefinerError::Missing(name.to_string()))?;
        let want: Vec<u64> = shape.iter().map(|dim| *dim as u64).collect();

        if tensor.dtype != dtype || tensor.shape != want {
            return Err(RefinerError::Shape {
                name: name.to_string(),
                detail: format!(
                    "checkpoint has {} {:?}, the graph needs {} {:?}",
                    tensor.dtype.name(),
                    tensor.shape,
                    dtype.name(),
                    want
                ),
            });
        }

        let elements: u64 = want.iter().product();

        if tensor.elements() != elements || tensor.bytes() != elements * dtype.size() {
            return Err(RefinerError::Shape {
                name: name.to_string(),
                detail: format!(
                    "{} elements in {} bytes do not match {} x {}",
                    tensor.elements(),
                    tensor.bytes(),
                    elements,
                    dtype.size()
                ),
            });
        }

        // One host buffer for the whole load; a per-tensor buffer would keep
        // every block's weights resident twice over.
        self.buffer.clear();
        self.buffer.resize(tensor.bytes() as usize, 0);
        shard
            .read_data(tensor, &mut self.buffer)
            .map_err(|error| RefinerError::Reader(error.token()))?;

        let device = alloc(self.gpu, elements as usize)?;
        self.gpu
            .upload(&device, &self.buffer)
            .map_err(RefinerError::Run)?;

        self.tensors += 1;
        self.bytes += tensor.bytes();

        Ok(device)
    }

    /// One `torch.nn.Linear`: weight `[out, in]`, bias `[out]`.
    fn linear(
        &mut self,
        shard: &Shard,
        name: &str,
        input_dim: usize,
        output_dim: usize,
        bias: Bias,
    ) -> Result<Linear, RefinerError> {
        let weight = self.pull(
            shard,
            &format!("{name}.weight"),
            Dtype::Bf16,
            &[output_dim, input_dim],
        )?;
        let bias = match bias {
            Bias::Present => {
                Some(self.pull(shard, &format!("{name}.bias"), Dtype::Bf16, &[output_dim])?)
            }
            Bias::Absent => None,
        };

        Ok(Linear {
            weight,
            bias,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        })
    }

    /// One AdaLN table. The checkpoint stores it F32 and the reference casts it
    /// to the timestep's BF16 before adding (`transformer.py:195-197`), so it
    /// is rounded once here and the block's add runs on BF16 operands.
    fn table(
        &mut self,
        shard: &Shard,
        name: &str,
        rows: usize,
        width: usize,
    ) -> Result<Tensor, RefinerError> {
        let tensor = shard
            .find(name)
            .ok_or_else(|| RefinerError::Missing(name.to_string()))?;

        if tensor.dtype != Dtype::F32 || tensor.shape != [rows as u64, width as u64] {
            return Err(RefinerError::Shape {
                name: name.to_string(),
                detail: format!(
                    "checkpoint has {} {:?}, the graph needs F32 [{rows}, {width}]",
                    tensor.dtype.name(),
                    tensor.shape
                ),
            });
        }

        let mut raw = vec![0u8; tensor.bytes() as usize];
        shard
            .read_data(tensor, &mut raw)
            .map_err(|error| RefinerError::Reader(error.token()))?;

        let mut bytes = Vec::with_capacity(raw.len() / 2);

        for word in raw.chunks_exact(4) {
            let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            bytes.extend_from_slice(&((bf16_round(value).to_bits() >> 16) as u16).to_le_bytes());
        }

        let device = alloc(self.gpu, rows * width)?;
        self.gpu
            .upload(&device, &bytes)
            .map_err(RefinerError::Run)?;

        self.tensors += 1;
        self.bytes += tensor.bytes();

        Ok(device)
    }

    /// One `Attention` (`attention.py:478-519`).
    fn attention(
        &mut self,
        shard: &Shard,
        name: &str,
        query_dim: usize,
        context_dim: usize,
        shape: (usize, usize),
    ) -> Result<Attention, RefinerError> {
        let (heads, head_dim) = shape;
        let inner = heads * head_dim;

        Ok(Attention {
            name: name.to_string(),
            q_norm: self.pull(
                shard,
                &format!("{name}.q_norm.weight"),
                Dtype::Bf16,
                &[inner],
            )?,
            k_norm: self.pull(
                shard,
                &format!("{name}.k_norm.weight"),
                Dtype::Bf16,
                &[inner],
            )?,
            query: self.linear(
                shard,
                &format!("{name}.to_q"),
                query_dim,
                inner,
                Bias::Present,
            )?,
            key: self.linear(
                shard,
                &format!("{name}.to_k"),
                context_dim,
                inner,
                Bias::Present,
            )?,
            value: self.linear(
                shard,
                &format!("{name}.to_v"),
                context_dim,
                inner,
                Bias::Present,
            )?,
            output: self.linear(
                shard,
                &format!("{name}.to_out.0"),
                inner,
                query_dim,
                Bias::Present,
            )?,
            gate: self.linear(
                shard,
                &format!("{name}.to_gate_logits"),
                query_dim,
                heads,
                Bias::Present,
            )?,
            inner: inner as u32,
            heads: heads as u32,
            head_dim: head_dim as u32,
        })
    }
}

fn alloc(gpu: &Gpu, elements: usize) -> Result<Tensor, RefinerError> {
    gpu.alloc(Device::Bf16, elements).map_err(RefinerError::Run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_geometry_is_the_configs_own() {
        // `model_configurator.py:222-225` against the shard's own header: 48
        // blocks at 4096 and 2048 inner width.
        assert_eq!(HEADS * VIDEO_HEAD_DIM, VIDEO_DIM);
        assert_eq!(HEADS * AUDIO_HEAD_DIM, AUDIO_DIM);
        assert_eq!(LAYERS, 48);
    }

    #[test]
    fn the_av_pair_takes_the_audio_head_width() {
        // `transformer.py:151-177`: both AV modules are built with
        // `heads=audio.heads, dim_head=audio.d_head`, so their cross-RoPE
        // halves are 32 wide on both sides.
        assert_eq!(AUDIO_DIM / HEADS, AUDIO_HEAD_DIM);
        assert_eq!(AUDIO_DIM / 2 / HEADS, 32);
        assert_eq!(VIDEO_DIM / 2 / HEADS, 64);
    }

    #[test]
    fn the_release_geometry_is_the_shards_own() {
        // The latent `(1, 128, 16, 24, 42)` at patch size 1 is 16,128 tokens,
        // and `round(121 / 24 * 16000 / 160 / 4)` is 126 audio frames.
        assert_eq!(VIDEO_TOKENS, 16_128);
        assert_eq!(AUDIO_TOKENS, 126);
        assert_eq!(CONTEXT_ROWS, 1024);
    }

    #[test]
    fn the_cross_rope_max_pos_is_the_releases_own() {
        // `model.py:130`: `max(positional_embedding_max_pos[0]=20,
        // audio_positional_embedding_max_pos[0]=20)`.
        assert_eq!(CROSS_MAX_POS, 20.0);
    }
}
