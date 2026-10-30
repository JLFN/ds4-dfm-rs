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
//! Unit 7 adds the edges the block does not cover: [`Edges::prepare`], the
//! preprocessing that turns the latents, a sigma and the prompt contexts into
//! the [`Signals`] and states the 48 blocks consume
//! (`transformer_args.py:174-410`), and [`Edges::head`], the model-level output
//! head (`model.py:471-489`). Three of the reference's steps are host
//! arithmetic, cited to the line that defines them: the patch positions
//! (`patchifiers.py:71-170`, `tools.py:158-172`), the RoPE tables
//! (`rope.py:86-223`) and the AdaLN sinusoid (`timestep_embedding.py:5-51`).
//!
//! One layout note: `split_freqs_cis` reshapes the RoPE tables to
//! `(b, heads, t, half)` because it rotates a head-major tensor
//! (`rope.py:179-183`); this port keeps the states token-major and writes the
//! tables `[row][head][half]`, the same values in the layout
//! [`Gpu::ltx_rope_pair`] reads.
//!
//! Not this unit: the sampler (`STAGE_2_DISTILLED_SIGMA_VALUES`, the mask
//! blend, the noiser), the Sol-Attn policy, the cache plumbing and the audio
//! unpatchify.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype as Device, Gpu, Tensor};

use crate::minimax_h3::gemma::bf16_round;
use crate::minimax_h3::safetensors::{self, Dtype, Shard};

/// `num_layers` (`model.py:56`, `model_configurator.py:52`).
pub const LAYERS: usize = 48;
/// `num_attention_heads` (`model.py:52`), every attention module.
pub const HEADS: usize = 32;
/// `attention_head_dim` (`model.py:53`).
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
/// `adaln_embedding_coefficient(True)` (`adaln.py:9-16`): 6 base + 3 cross.
pub const COEFFICIENT: usize = 9;
/// `_init_audio_video(num_scale_shift_values=4)` (`model.py:133`): the A2V/V2A
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
/// feed-forward is bias-free while the audio one is not (`model.py:78-79`:
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
pub struct Signals<'a> {
    pub rows: u32,
    pub dim: u32,
    pub modulation: Tensor,
    pub cross_scale_shift: Tensor,
    pub cross_gate: Tensor,
    pub prompt_modulation: Tensor,
    /// Borrowed: `_prepare_context` only reshapes the prompt encoder's own
    /// tensor (`transformer_args.py:186-196`), it does not copy it.
    pub context: &'a Tensor,
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
        video_signals: &Signals<'_>,
        audio_signals: &Signals<'_>,
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
        video_signals: &Signals<'_>,
        audio_signals: &Signals<'_>,
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
        v: &Signals<'_>,
        a: &Signals<'_>,
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
    signals: &Signals<'_>,
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
        signals.context,
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
    signals: &Signals<'_>,
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
    own_signals: &Signals<'_>,
    own_state: &Tensor,
    own: &ModalityScratch,
    other_signals: &Signals<'_>,
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
    signals: &Signals<'_>,
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

// ---------------------------------------------------------------------------
// The edges: the preprocessing (`transformer_args.py:174-410`) and the output
// head (`model.py:471-489`), the 59 tensors outside the 48 blocks.
// ---------------------------------------------------------------------------

/// The release's frame rate (`tools.py:170`, the stage's own `24.0`).
pub const FPS: f32 = 24.0;
/// `SpatioTemporalScaleFactors.default()` (`types.py:31-33`): the VAE's
/// `(time, height, width)` downsampling.
pub const TIME_SCALE: u32 = 8;
pub const SPACE_SCALE: u32 = 32;
/// `AudioPatchifier`'s grid and `AudioLatentShape.from_video_pixel_shape`'s
/// defaults (`patchifiers.py:176-180`, `types.py:183-193`).
pub const AUDIO_CHANNELS: u32 = 8;
pub const AUDIO_BINS: u32 = 16;
pub const AUDIO_SAMPLE_RATE: f32 = 16_000.0;
pub const AUDIO_HOP: f32 = 160.0;
pub const AUDIO_DOWNSAMPLE: u32 = 4;
/// `positional_embedding_max_pos` and `audio_positional_embedding_max_pos`
/// (`model.py:103-107`, `:117-118`; the shard's own config).
pub const VIDEO_MAX_POS: [f32; 3] = [20.0, 2048.0, 2048.0];
pub const AUDIO_MAX_POS: [f32; 1] = [20.0];
/// The AdaLN `Timesteps`'s parameters (`timestep_embedding.py:11-13`,
/// `:131`): 256 channels, `downscale_freq_shift` 0, `max_period` 10000.
pub const SINUSOID_WIDTH: usize = 256;
pub const MAX_PERIOD: f64 = 10_000.0;
/// `timestep_scale_multiplier` (`model.py:62`). The AV gate's own multiplier
/// is the same 1000.0, so `av_ca_factor` (`transformer_args.py:395-396`) is
/// 1.0 and the gate's input is the other modality's sigma times 1000.
pub const TIMESTEP_SCALE: f32 = 1000.0;

/// The release's latent geometry: the video grid at 121 pixel frames and
/// 1344x768 (`VideoLatentShape.from_pixel_shape` at the VAE's scale factors)
/// and the audio frame count (`AudioLatentShape.from_video_pixel_shape`).
#[derive(Clone, Copy)]
pub struct Geometry {
    pub frames: u32,
    pub height: u32,
    pub width: u32,
    pub audio_frames: u32,
}

impl Geometry {
    /// `(16, 24, 42)` video latents and 126 audio frames.
    pub fn release() -> Geometry {
        Geometry {
            frames: 16,
            height: 24,
            width: 42,
            audio_frames: AUDIO_TOKENS,
        }
    }

    /// `VideoLatentPatchifier.get_token_count` (`patchifiers.py:24-26`).
    pub fn video_tokens(&self) -> u32 {
        self.frames * self.height * self.width
    }

    /// The tokens one latent frame covers, the rows `_first_frame_keyframes_mask`
    /// marks (`tools.py:186-205`).
    pub fn frame_tokens(&self) -> u32 {
        self.height * self.width
    }

    /// `channels * mel_bins` (`types.py:183-193`): the audio patch's width.
    pub fn audio_width(&self) -> u32 {
        AUDIO_CHANNELS * AUDIO_BINS
    }
}

/// `create_initial_state`'s positions (`patchifiers.py:71-170`, `tools.py:158-172`):
/// every video patch's `[start, end)` bounds in seconds, `[rows, 3, 2]` over
/// `(time, height, width)` with the token order `(f h w)`.
///
/// The pixel coordinates are `latent * scale` with the causal fix on the
/// temporal axis only, and `tools.py:170` divides that axis by the frame rate;
/// height and width stay in pixels.
pub fn video_positions(geometry: &Geometry) -> Vec<f32> {
    let seconds =
        |frame: u32| ((frame * TIME_SCALE) as f32 + 1.0 - TIME_SCALE as f32).max(0.0) / FPS;
    let mut positions = Vec::with_capacity(geometry.video_tokens() as usize * 6);

    for frame in 0..geometry.frames {
        for height in 0..geometry.height {
            for width in 0..geometry.width {
                positions.push(seconds(frame));
                positions.push(seconds(frame + 1));
                positions.push((height * SPACE_SCALE) as f32);
                positions.push(((height + 1) * SPACE_SCALE) as f32);
                positions.push((width * SPACE_SCALE) as f32);
                positions.push(((width + 1) * SPACE_SCALE) as f32);
            }
        }
    }

    positions
}

/// `AudioPatchifier.get_patch_grid_bounds` (`patchifiers.py:215-283`, causal,
/// no shift): every audio patch's bounds in seconds, `[rows, 1, 2]`. The
/// audio tools never scale these - the patchifier is already in seconds
/// (`tools.py:257-278`).
pub fn audio_positions(geometry: &Geometry) -> Vec<f32> {
    let seconds = |index: u32| {
        let mel = (index * AUDIO_DOWNSAMPLE) as f32;
        let causal = (mel + 1.0 - AUDIO_DOWNSAMPLE as f32).max(0.0);

        causal * AUDIO_HOP / AUDIO_SAMPLE_RATE
    };
    let mut positions = Vec::with_capacity(geometry.audio_frames as usize * 2);

    for index in 0..geometry.audio_frames {
        positions.push(seconds(index));
        positions.push(seconds(index + 1));
    }

    positions
}

/// One axis of a `[rows, 3, 2]` position block: the cross-attention's RoPE is
/// the time axis alone (`transformer_args.py:365-370`).
pub fn time_axis(positions: &[f32]) -> Vec<f32> {
    positions
        .chunks_exact(6)
        .flat_map(|row| row[..2].to_vec())
        .collect()
}

/// `precompute_freqs_cis` with the float64 grid (`rope.py:196-223`, `:86-108`,
/// the config's `frequencies_precision: "float64"`).
///
/// `positions` is `[rows, axes, 2]` - the patch bounds whose midpoint RoPE is
/// evaluated at (`rope.py:149-155`) - and the result is the
/// `[rows, heads, dim/2/heads]` BF16 `(cos, sin)` pair, the padding the
/// reference prepends included (`split_freqs_cis:163-184`). Note the layout:
/// the reference reshapes to `(b, t, heads, -1)` and swaps to head-major
/// (`:179-183`) because it rotates a head-major tensor; this keeps the tokens
/// token-major and folds the same values into `[row][head][half]`, the layout
/// [`Gpu::ltx_rope_pair`] reads.
pub fn rope_tables(
    positions: &[f32],
    axes: usize,
    dim: usize,
    max_pos: &[f32],
    heads: usize,
) -> Result<(Vec<u8>, Vec<u8>), RefinerError> {
    let count = dim / (2 * axes);
    let expected = dim / 2;
    let (current, half) = (count * axes, expected / heads);

    if axes == 0
        || axes != max_pos.len()
        || heads == 0
        || positions.len() % (axes * 2) != 0
        || current > expected
        || expected % heads != 0
    {
        return Err(RefinerError::Shape {
            name: "rope".into(),
            detail: format!(
                "dim {dim} at {axes} axes and {heads} heads does not take the {} positions",
                positions.len()
            ),
        });
    }

    let rows = positions.len() / (axes * 2);
    // `np.linspace(0, 1, count)` is `k / (count - 1)`, so the exponents run from
    // `theta^0` to `theta^1` (`rope.py:96-107`).
    let grid: Vec<f32> = (0..count)
        .map(|k| {
            let unit = if count > 1 {
                k as f64 / (count - 1) as f64
            } else {
                0.0
            };

            (ROPE_BASE.powf(unit) * std::f64::consts::FRAC_PI_2) as f32
        })
        .collect();

    let mut cos = vec![1.0f32; expected];
    let mut sin = vec![0.0f32; expected];
    let mut cos_bytes = Vec::with_capacity(rows * heads * half * 2);
    let mut sin_bytes = Vec::with_capacity(rows * heads * half * 2);

    for row in 0..rows {
        cos[..expected - current].fill(1.0);
        sin[..expected - current].fill(0.0);

        for k in 0..count {
            for axis in 0..axes {
                let start = positions[(row * axes + axis) * 2];
                let end = positions[(row * axes + axis) * 2 + 1];
                let middle = (start + end) / 2.0;
                let fractional = middle / max_pos[axis];
                let angle = grid[k] * (fractional * 2.0 - 1.0);

                cos[expected - current + k * axes + axis] = angle.cos();
                sin[expected - current + k * axes + axis] = angle.sin();
            }
        }

        for value in &cos {
            cos_bytes
                .extend_from_slice(&((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes());
        }
        for value in &sin {
            sin_bytes
                .extend_from_slice(&((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes());
        }
    }

    Ok((cos_bytes, sin_bytes))
}

/// `get_timestep_embedding` at the AdaLN `Timesteps`'s parameters
/// (`timestep_embedding.py:5-51`, `:131`): 256 channels, `flip_sin_to_cos`
/// true, `downscale_freq_shift` 0, `scale` 1. The output is `[cos, sin]`,
/// each half 128 wide.
pub fn sinusoid(timestep: f32) -> [f32; SINUSOID_WIDTH] {
    let half = SINUSOID_WIDTH / 2;
    // `-math.log(max_period) * arange(0, half_dim)` then `/ (half_dim - 0)`,
    // each in float32.
    let base = -(MAX_PERIOD.ln()) as f32;
    let mut out = [0.0f32; SINUSOID_WIDTH];

    for k in 0..half {
        let exponent = base * k as f32 / half as f32;
        let angle = timestep * exponent.exp();

        out[k] = angle.cos();
        out[half + k] = angle.sin();
    }

    out
}

/// One `[rows, width]` BF16 tensor holding `values` repeated on every row.
fn sinusoid_bytes(values: &[f32], rows: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * values.len() * 2);

    for _ in 0..rows {
        for value in values {
            bytes.extend_from_slice(&((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes());
        }
    }

    bytes
}

/// One `AdaLayerNormSingle` (`adaln.py:19-45`): the sinusoid through
/// `linear_1 -> SiLU -> linear_2` is the `embedded_timestep` the head needs,
/// and `SiLU -> linear` on top of it is the `coefficient * dim` modulation the
/// block reads.
struct AdaLn {
    name: String,
    linear_1: Linear,
    linear_2: Linear,
    linear: Linear,
    coefficient: u32,
}

impl AdaLn {
    /// `linear_1` is `256 -> dim`, `linear_2` `dim -> dim`, and `linear`
    /// `dim -> coefficient * dim`, every one biased
    /// (`timestep_embedding.py:69`, `:79`, `adaln.py:35`).
    fn load(
        reader: &mut Reader<'_>,
        shard: &Shard,
        prefix: &str,
        dim: usize,
        coefficient: usize,
    ) -> Result<AdaLn, RefinerError> {
        let embedder = format!("{prefix}.emb.timestep_embedder");

        Ok(AdaLn {
            name: prefix.to_string(),
            linear_1: reader.linear(
                shard,
                &format!("{embedder}.linear_1"),
                SINUSOID_WIDTH,
                dim,
                Bias::Present,
            )?,
            linear_2: reader.linear(
                shard,
                &format!("{embedder}.linear_2"),
                dim,
                dim,
                Bias::Present,
            )?,
            linear: reader.linear(
                shard,
                &format!("{prefix}.linear"),
                dim,
                coefficient * dim,
                Bias::Present,
            )?,
            coefficient: coefficient as u32,
        })
    }

    /// `AdaLayerNormSingle.forward` (`adaln.py:39-45`) verbatim: `linear_2`'s
    /// output is the `embedded_timestep`, `SiLU` of it feeds `linear`, and the
    /// `SiLU` runs on a separate tensor in the reference, so `embedded` is
    /// never clobbered. `hidden` and `embedded` are `[rows, dim]` of the
    /// caller's making; a caller that discards the embedded timestep
    /// (`transformer_args.py:274-279`, `:399-409`) hands a scratch buffer.
    fn apply(
        &self,
        gpu: &Gpu,
        sinusoid: &Tensor,
        rows: u32,
        embedded: &Tensor,
        hidden: &Tensor,
        modulation: &Tensor,
    ) -> Result<(), RefinerError> {
        let named = |detail: String| RefinerError::Run(format!("{}: {detail}", self.name));

        self.linear_1
            .apply(gpu, hidden, sinusoid, rows)
            .map_err(|error| named(error.token()))?;
        gpu.silu_bf16(hidden, hidden).map_err(named)?;
        self.linear_2
            .apply(gpu, embedded, hidden, rows)
            .map_err(|error| named(error.token()))?;
        gpu.silu_bf16(hidden, embedded).map_err(named)?;
        self.linear
            .apply(gpu, modulation, hidden, rows)
            .map_err(|error| named(error.token()))?;

        Ok(())
    }

    /// The modulation's row stride, for a caller sizing it.
    fn width(&self, dim: u32) -> u32 {
        self.coefficient * dim
    }
}

/// The model-level output head (`model.py:471-489`, `:520-536`): the table
/// combined with the embedded timestep - **row 0 the shift, row 1 the scale** -
/// the affine-free LayerNorm, the modulation and `proj_out` to 128 channels.
struct Head {
    video_table: Tensor,
    audio_table: Tensor,
    video_proj: Linear,
    audio_proj: Linear,
}

impl Head {
    /// `scale_shift_table`, `norm_out`, `proj_out` and the audio trio
    /// (`model.py:230-232`, `:260-262`). `norm_out` is
    /// `elementwise_affine=False`, so it has no tensors of its own.
    fn open(reader: &mut Reader<'_>, shard: &Shard) -> Result<Head, RefinerError> {
        let at = "model.diffusion_model";

        Ok(Head {
            video_table: reader.table(shard, &format!("{at}.scale_shift_table"), 2, VIDEO_DIM)?,
            audio_table: reader.table(
                shard,
                &format!("{at}.audio_scale_shift_table"),
                2,
                AUDIO_DIM,
            )?,
            video_proj: reader.linear(
                shard,
                &format!("{at}.proj_out"),
                VIDEO_DIM,
                CHANNELS,
                Bias::Present,
            )?,
            audio_proj: reader.linear(
                shard,
                &format!("{at}.audio_proj_out"),
                AUDIO_DIM,
                CHANNELS,
                Bias::Present,
            )?,
        })
    }
}

/// One modality's head: `_process_output` on `state`, written to
/// `scratch.output` as `[rows, 128]`.
fn process_output(
    gpu: &Gpu,
    table: &Tensor,
    proj: &Linear,
    state: &Tensor,
    embedded: &Tensor,
    scratch: &HeadScratch,
) -> Result<(), RefinerError> {
    let (rows, dim) = (scratch.rows, scratch.dim);

    // `scale_shift_table[None, None].to(dtype) + embedded_timestep[:, :, None]`
    // (`model.py:484-485`), then `shift, scale = [..., 0], [..., 1]`.
    gpu.ada_add(&scratch.shift, table, embedded, rows, dim, 1, 0, 0)
        .map_err(RefinerError::Run)?;
    gpu.ada_add(&scratch.scale, table, embedded, rows, dim, 1, 1, 0)
        .map_err(RefinerError::Run)?;

    // `norm_out` (`model.py:487`), `x * (1 + scale) + shift` (`:488`) with the
    // per-token row, then `proj_out` (`:489`).
    gpu.layer_norm(&scratch.normed, state, rows, dim, RMS_EPSILON)
        .map_err(RefinerError::Run)?;
    gpu.modulate(
        &scratch.normed,
        &scratch.normed,
        &scratch.scale,
        &scratch.shift,
        rows,
        dim,
        rows,
    )
    .map_err(RefinerError::Run)?;
    proj.apply(gpu, &scratch.output, &scratch.normed, rows)
}

/// `VideoLatentPatchifier.unpatchify` at patch size 1 (`patchifiers.py:40-64`):
/// the head's `[tokens, 128]` back to the `(1, 128, f, h, w)` latent, the same
/// transpose the patchify direction uses.
pub fn unpatchify_video(
    gpu: &Gpu,
    output: &Tensor,
    latent: &Tensor,
    rows: u32,
) -> Result<(), RefinerError> {
    gpu.channel_transpose(latent, output, rows, CHANNELS as u32)
        .map_err(RefinerError::Run)
}

/// The head's working set for one modality: the LayerNorm output, the two
/// modulation rows and the `[rows, 128]` projection.
struct HeadScratch {
    rows: u32,
    dim: u32,
    normed: Tensor,
    shift: Tensor,
    scale: Tensor,
    output: Tensor,
}

impl HeadScratch {
    fn new(gpu: &Gpu, rows: u32, dim: u32) -> Result<HeadScratch, RefinerError> {
        let (row, width) = (rows as usize, dim as usize);

        Ok(HeadScratch {
            rows,
            dim,
            normed: alloc(gpu, row * width)?,
            shift: alloc(gpu, row * width)?,
            scale: alloc(gpu, row * width)?,
            output: alloc(gpu, row * CHANNELS)?,
        })
    }
}

/// One modality's AdaLN working set. Two buffers per call - `linear_1`'s output
/// and `linear_2`'s - and the reference allocates a fresh pair every call, so a
/// call whose embedded timestep is discarded (`transformer_args.py:274-279`,
/// `:399-409`) needs its own: `embedded` survives for the head, `wide` takes
/// the per-token discards and the `single` pair the per-request ones.
struct AdaScratch {
    hidden: Tensor,
    embedded: Tensor,
    wide: Tensor,
    single_hidden: Tensor,
    single_embedded: Tensor,
}

impl AdaScratch {
    fn new(gpu: &Gpu, rows: u32, dim: u32) -> Result<AdaScratch, RefinerError> {
        let (row, width) = (rows as usize, dim as usize);

        Ok(AdaScratch {
            hidden: alloc(gpu, row * width)?,
            embedded: alloc(gpu, row * width)?,
            wide: alloc(gpu, row * width)?,
            single_hidden: alloc(gpu, width)?,
            single_embedded: alloc(gpu, width)?,
        })
    }
}

/// The preprocessing's reusable working set. `keyframe_mask` is constant for a
/// geometry - `_first_frame_keyframes_mask` marks the first latent frame
/// unconditionally (`tools.py:186-200`) - and the rest is overwritten on every
/// call.
pub struct EdgeScratch {
    video_latent: Tensor,
    audio_latent: Tensor,
    keyframe_mask: Tensor,
    video_sinusoid: Tensor,
    audio_sinusoid: Tensor,
    single_sinusoid: Tensor,
    video: AdaScratch,
    audio: AdaScratch,
    video_head: HeadScratch,
    audio_head: HeadScratch,
}

impl EdgeScratch {
    /// Allocates for `geometry` alone; the token counts are the geometry's.
    pub fn new(gpu: &Gpu, geometry: &Geometry) -> Result<EdgeScratch, RefinerError> {
        let video_rows = geometry.video_tokens();
        let audio_rows = geometry.audio_frames;
        let mut mask = vec![0.0f32; video_rows as usize];

        mask[..geometry.frame_tokens() as usize].fill(1.0);

        Ok(EdgeScratch {
            video_latent: alloc(gpu, CHANNELS * video_rows as usize)?,
            audio_latent: alloc(gpu, geometry.audio_width() as usize * audio_rows as usize)?,
            keyframe_mask: tensor_of(gpu, &sinusoid_bytes(&mask, 1))?,
            video_sinusoid: alloc(gpu, video_rows as usize * SINUSOID_WIDTH)?,
            audio_sinusoid: alloc(gpu, audio_rows as usize * SINUSOID_WIDTH)?,
            single_sinusoid: alloc(gpu, SINUSOID_WIDTH)?,
            video: AdaScratch::new(gpu, video_rows, VIDEO_DIM as u32)?,
            audio: AdaScratch::new(gpu, audio_rows, AUDIO_DIM as u32)?,
            video_head: HeadScratch::new(gpu, video_rows, VIDEO_DIM as u32)?,
            audio_head: HeadScratch::new(gpu, audio_rows, AUDIO_DIM as u32)?,
        })
    }
}

/// One BF16 tensor from an already-rounded byte block.
fn tensor_of(gpu: &Gpu, bytes: &[u8]) -> Result<Tensor, RefinerError> {
    let tensor = alloc(gpu, bytes.len() / 2)?;
    gpu.upload(&tensor, bytes).map_err(RefinerError::Run)?;

    Ok(tensor)
}

/// One denoising step's inputs: the sigma the schedule gives all of the step's
/// tokens (`denoisers.py:236-246`, `:298-310`), the geometry, the raw latents
/// and the two text contexts.
pub struct Request<'a> {
    pub sigma: f32,
    pub geometry: Geometry,
    pub video: &'a Tensor,
    pub audio: &'a Tensor,
    pub video_context: &'a Tensor,
    pub audio_context: &'a Tensor,
}

/// What the preprocessing produces: the states the 48 blocks move, the signals
/// they read besides their weights, and the embedded timesteps the head takes
/// (`model.py:520-536`).
pub struct Prepared<'a> {
    pub video: Tensor,
    pub audio: Tensor,
    pub video_signals: Signals<'a>,
    pub audio_signals: Signals<'a>,
}

/// The refiner's edges: the preprocessing and the output head, over the 59
/// tensors the 48 blocks do not carry.
pub struct Edges {
    video_patchify: Linear,
    audio_patchify: Linear,
    keyframe: Tensor,
    video_adaln: AdaLn,
    audio_adaln: AdaLn,
    video_prompt: AdaLn,
    audio_prompt: AdaLn,
    video_cross: AdaLn,
    audio_cross: AdaLn,
    a2v_gate: AdaLn,
    v2a_gate: AdaLn,
    head: Head,
    tensors: usize,
    bytes: u64,
}

impl Edges {
    /// Read the 59 tensors outside the blocks: the two patchify projections,
    /// the keyframe embedding, the eight AdaLN singles and the head.
    pub fn open(gpu: &Gpu, path: &Path) -> Result<Edges, RefinerError> {
        let shard =
            safetensors::read_shard(path).map_err(|error| RefinerError::Reader(error.token()))?;
        let at = "model.diffusion_model";
        let mut reader = Reader {
            gpu,
            buffer: Vec::new(),
            tensors: 0,
            bytes: 0,
        };

        let video_patchify = reader.linear(
            &shard,
            &format!("{at}.patchify_proj"),
            CHANNELS,
            VIDEO_DIM,
            Bias::Present,
        )?;
        let audio_patchify = reader.linear(
            &shard,
            &format!("{at}.audio_patchify_proj"),
            CHANNELS,
            AUDIO_DIM,
            Bias::Present,
        )?;
        let keyframe = reader.pull(
            &shard,
            &format!("{at}.keyframes_abs_pos_embedding"),
            Dtype::Bf16,
            &[1, VIDEO_DIM],
        )?;
        // `adaln_single` is the block coefficient, `av_ca_*_scale_shift` the
        // four-value pair, the two gates one each and the prompts two
        // (`model.py:221-227`, `:268-287`, `:248-256`).
        let video_adaln = AdaLn::load(
            &mut reader,
            &shard,
            &format!("{at}.adaln_single"),
            VIDEO_DIM,
            COEFFICIENT,
        )?;
        let audio_adaln = AdaLn::load(
            &mut reader,
            &shard,
            &format!("{at}.audio_adaln_single"),
            AUDIO_DIM,
            COEFFICIENT,
        )?;
        let video_prompt = AdaLn::load(
            &mut reader,
            &shard,
            &format!("{at}.prompt_adaln_single"),
            VIDEO_DIM,
            PROMPT_COEFFICIENT,
        )?;
        let audio_prompt = AdaLn::load(
            &mut reader,
            &shard,
            &format!("{at}.audio_prompt_adaln_single"),
            AUDIO_DIM,
            PROMPT_COEFFICIENT,
        )?;
        let video_cross = AdaLn::load(
            &mut reader,
            &shard,
            &format!("{at}.av_ca_video_scale_shift_adaln_single"),
            VIDEO_DIM,
            CROSS_COEFFICIENT,
        )?;
        let audio_cross = AdaLn::load(
            &mut reader,
            &shard,
            &format!("{at}.av_ca_audio_scale_shift_adaln_single"),
            AUDIO_DIM,
            CROSS_COEFFICIENT,
        )?;
        let a2v_gate = AdaLn::load(
            &mut reader,
            &shard,
            &format!("{at}.av_ca_a2v_gate_adaln_single"),
            VIDEO_DIM,
            GATE_COEFFICIENT,
        )?;
        let v2a_gate = AdaLn::load(
            &mut reader,
            &shard,
            &format!("{at}.av_ca_v2a_gate_adaln_single"),
            AUDIO_DIM,
            GATE_COEFFICIENT,
        )?;
        let head = Head::open(&mut reader, &shard)?;

        Ok(Edges {
            video_patchify,
            audio_patchify,
            keyframe,
            video_adaln,
            audio_adaln,
            video_prompt,
            audio_prompt,
            video_cross,
            audio_cross,
            a2v_gate,
            v2a_gate,
            head,
            tensors: reader.tensors,
            bytes: reader.bytes,
        })
    }

    /// The tensors read, and their bytes, for the gate's record.
    pub fn loaded(&self) -> (usize, u64) {
        (self.tensors, self.bytes)
    }

    /// `TransformerArgsPreprocessor.prepare` (`transformer_args.py:347-410`)
    /// plus `apply_keyframes_absolute_embedding` (`:22-43`), in the
    /// reference's order.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<'a>(
        &self,
        gpu: &Gpu,
        request: &'a Request<'_>,
        scratch: &EdgeScratch,
    ) -> Result<Prepared<'a>, RefinerError> {
        let geometry = request.geometry;
        let (video_rows, audio_rows) = (geometry.video_tokens(), geometry.audio_frames);
        let scaled = request.sigma * TIMESTEP_SCALE;
        let video_time = time_axis(&video_positions(&geometry));

        // The states: patchify (`transformer_args.py:268`), project
        // (`:268`) and mark the keyframe rows (`:269`).
        let video = alloc(gpu, video_rows as usize * VIDEO_DIM)?;
        let audio = alloc(gpu, audio_rows as usize * AUDIO_DIM)?;
        gpu.channel_transpose(
            &scratch.video_latent,
            request.video,
            CHANNELS as u32,
            video_rows,
        )
        .map_err(RefinerError::Run)?;
        gpu.audio_patchify(
            &scratch.audio_latent,
            request.audio,
            1,
            AUDIO_CHANNELS,
            audio_rows,
            AUDIO_BINS,
        )
        .map_err(RefinerError::Run)?;
        self.video_patchify
            .apply(gpu, &video, &scratch.video_latent, video_rows)?;
        gpu.keyframe_add(
            &video,
            &video,
            &scratch.keyframe_mask,
            &self.keyframe,
            video_rows,
            VIDEO_DIM as u32,
        )
        .map_err(RefinerError::Run)?;
        self.audio_patchify
            .apply(gpu, &audio, &scratch.audio_latent, audio_rows)?;

        // The timesteps (`transformer_args.py:174-185`): `timestep * 1000`
        // through the 256-wide sinusoid, then `adaln_single` - `_prepare_timestep`
        // returns the modulation *and* the embedded timestep the head takes.
        let values = sinusoid(scaled);
        let modulation = alloc(
            gpu,
            video_rows as usize * self.video_adaln.width(VIDEO_DIM as u32) as usize,
        )?;
        let audio_modulation = alloc(
            gpu,
            audio_rows as usize * self.audio_adaln.width(AUDIO_DIM as u32) as usize,
        )?;
        let video_prompt = alloc(gpu, self.video_prompt.width(VIDEO_DIM as u32) as usize)?;
        let audio_prompt = alloc(gpu, self.audio_prompt.width(AUDIO_DIM as u32) as usize)?;
        let cross_scale_shift = alloc(
            gpu,
            video_rows as usize * self.video_cross.width(VIDEO_DIM as u32) as usize,
        )?;
        let audio_cross_scale_shift = alloc(
            gpu,
            audio_rows as usize * self.audio_cross.width(AUDIO_DIM as u32) as usize,
        )?;
        let cross_gate = alloc(gpu, self.a2v_gate.width(VIDEO_DIM as u32) as usize)?;
        let audio_cross_gate = alloc(gpu, self.v2a_gate.width(AUDIO_DIM as u32) as usize)?;

        gpu.upload(
            &scratch.video_sinusoid,
            &sinusoid_bytes(&values, video_rows as usize),
        )
        .map_err(RefinerError::Run)?;
        gpu.upload(
            &scratch.audio_sinusoid,
            &sinusoid_bytes(&values, audio_rows as usize),
        )
        .map_err(RefinerError::Run)?;
        gpu.upload(&scratch.single_sinusoid, &sinusoid_bytes(&values, 1))
            .map_err(RefinerError::Run)?;
        self.video_adaln.apply(
            gpu,
            &scratch.video_sinusoid,
            video_rows,
            &scratch.video.embedded,
            &scratch.video.hidden,
            &modulation,
        )?;
        self.audio_adaln.apply(
            gpu,
            &scratch.audio_sinusoid,
            audio_rows,
            &scratch.audio.embedded,
            &scratch.audio.hidden,
            &audio_modulation,
        )?;

        // The prompt AdaLN runs on the raw sigma (`:274-279`) and its embedded
        // timestep is discarded.
        self.video_prompt.apply(
            gpu,
            &scratch.single_sinusoid,
            1,
            &scratch.video.single_embedded,
            &scratch.video.single_hidden,
            &video_prompt,
        )?;
        self.audio_prompt.apply(
            gpu,
            &scratch.single_sinusoid,
            1,
            &scratch.audio.single_embedded,
            &scratch.audio.single_hidden,
            &audio_prompt,
        )?;

        // The A-V cross AdaLN: this modality's per-token timesteps scale and
        // shift its own attention input, and the *other* modality's sigma
        // (`cross_modality_sigma`) gates its output
        // (`transformer_args.py:388-410`, `:207-220`).
        self.video_cross.apply(
            gpu,
            &scratch.video_sinusoid,
            video_rows,
            &scratch.video.wide,
            &scratch.video.hidden,
            &cross_scale_shift,
        )?;
        self.audio_cross.apply(
            gpu,
            &scratch.audio_sinusoid,
            audio_rows,
            &scratch.audio.wide,
            &scratch.audio.hidden,
            &audio_cross_scale_shift,
        )?;
        self.a2v_gate.apply(
            gpu,
            &scratch.single_sinusoid,
            1,
            &scratch.video.single_embedded,
            &scratch.video.single_hidden,
            &cross_gate,
        )?;
        self.v2a_gate.apply(
            gpu,
            &scratch.single_sinusoid,
            1,
            &scratch.audio.single_embedded,
            &scratch.audio.single_hidden,
            &audio_cross_gate,
        )?;

        // The positional tables (`:239-260`, `:365-370`): the self-attention
        // over every axis, the cross over the time axis at
        // `audio_cross_attention_dim`.
        let (self_cos, self_sin) = rope_tables(
            &video_positions(&geometry),
            3,
            VIDEO_DIM,
            &VIDEO_MAX_POS,
            HEADS,
        )?;
        let (audio_self_cos, audio_self_sin) = rope_tables(
            &audio_positions(&geometry),
            1,
            AUDIO_DIM,
            &AUDIO_MAX_POS,
            HEADS,
        )?;
        let (cross_cos, cross_sin) =
            rope_tables(&video_time, 1, AUDIO_CONTEXT_DIM, &[CROSS_MAX_POS], HEADS)?;
        let (audio_cross_cos, audio_cross_sin) = rope_tables(
            &audio_positions(&geometry),
            1,
            AUDIO_CONTEXT_DIM,
            &[CROSS_MAX_POS],
            HEADS,
        )?;

        Ok(Prepared {
            video_signals: Signals {
                rows: video_rows,
                dim: VIDEO_DIM as u32,
                modulation,
                cross_scale_shift,
                cross_gate,
                prompt_modulation: video_prompt,
                context: request.video_context,
                context_rows: CONTEXT_ROWS,
                self_cos: tensor_of(gpu, &self_cos)?,
                self_sin: tensor_of(gpu, &self_sin)?,
                cross_cos: tensor_of(gpu, &cross_cos)?,
                cross_sin: tensor_of(gpu, &cross_sin)?,
            },
            audio_signals: Signals {
                rows: audio_rows,
                dim: AUDIO_DIM as u32,
                modulation: audio_modulation,
                cross_scale_shift: audio_cross_scale_shift,
                cross_gate: audio_cross_gate,
                prompt_modulation: audio_prompt,
                context: request.audio_context,
                context_rows: CONTEXT_ROWS,
                self_cos: tensor_of(gpu, &audio_self_cos)?,
                self_sin: tensor_of(gpu, &audio_self_sin)?,
                cross_cos: tensor_of(gpu, &audio_cross_cos)?,
                cross_sin: tensor_of(gpu, &audio_cross_sin)?,
            },
            video,
            audio,
        })
    }

    /// `LTXModel._process_output` on both modalities (`model.py:520-536`),
    /// after the blocks: the head's `[rows, 128]` output lands in
    /// [`EdgeScratch`]'s head buffer, and `unpatchify_video` takes the video's
    /// back to a latent grid.
    pub fn head(
        &self,
        gpu: &Gpu,
        prepared: &Prepared<'_>,
        scratch: &EdgeScratch,
    ) -> Result<(), RefinerError> {
        process_output(
            gpu,
            &self.head.video_table,
            &self.head.video_proj,
            &prepared.video,
            &scratch.video.embedded,
            &scratch.video_head,
        )?;
        process_output(
            gpu,
            &self.head.audio_table,
            &self.head.audio_proj,
            &prepared.audio,
            &scratch.audio.embedded,
            &scratch.audio_head,
        )
    }

    /// The audio head's `[frames, 128]`, for a caller that dumps or decodes it.
    pub fn audio_output<'a>(&self, scratch: &'a EdgeScratch) -> &'a Tensor {
        &scratch.audio_head.output
    }

    /// The video head's `[tokens, 128]`.
    pub fn video_output<'a>(&self, scratch: &'a EdgeScratch) -> &'a Tensor {
        &scratch.video_head.output
    }

    /// The video AdaLN's `embedded_timestep`: the head's own modulation input
    /// (`model.py:520-527`), kept so a caller can check the head's wiring.
    pub fn video_embedded<'a>(&self, scratch: &'a EdgeScratch) -> &'a Tensor {
        &scratch.video.embedded
    }

    /// The audio AdaLN's `embedded_timestep` (`model.py:528-536`).
    pub fn audio_embedded<'a>(&self, scratch: &'a EdgeScratch) -> &'a Tensor {
        &scratch.audio.embedded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_geometry_is_the_configs_own() {
        // `model.py:56` and `model_configurator.py:52` against the shard's own header: 48
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

    #[test]
    fn the_video_positions_are_the_releases() {
        // `patchifiers.py:113-124` + `get_pixel_coords` (`:137-170`) +
        // `tools.py:170`: frame 0's bounds are `(0*8 + 1 - 8).clamp(min=0)` to
        // `(8 + 1 - 8)` in pixel frames, divided by 24 fps; the spatial axes
        // stay in pixels.
        let positions = video_positions(&Geometry::release());

        assert_eq!(positions.len(), 16_128 * 6);
        assert_eq!(positions[0], 0.0);
        assert_eq!(positions[1], 1.0 / FPS);
        assert_eq!(positions[2], 0.0);
        assert_eq!(positions[3], 32.0);
        assert_eq!(positions[4], 0.0);
        assert_eq!(positions[5], 32.0);
        // The token order is `(f h w)` with `w` fastest, so the second latent
        // frame's bounds sit `height * width` tokens in.
        assert_eq!(positions[1008 * 6], 1.0 / FPS);
        assert_eq!(positions[1008 * 6 + 1], 9.0 / FPS);
    }

    #[test]
    fn the_audio_positions_are_the_releases() {
        // `patchifiers.py:239-248`: mel frame `index * 4` with a causal offset
        // of 1, over `160 / 16000` seconds per mel frame.
        let positions = audio_positions(&Geometry::release());

        assert_eq!(positions.len(), 126 * 2);
        assert_eq!(positions[0], 0.0);
        assert!((positions[1] - 0.01).abs() < 1e-9);
        assert!((positions[2] - 0.01).abs() < 1e-9);
        assert!((positions[3] - 0.05).abs() < 1e-9);
    }

    #[test]
    fn the_video_rope_table_is_padded_and_head_split() {
        // `split_freqs_cis` (`rope.py:163-184`): 682 angles over 3 axes is
        // 2046 values, and the 2 the half-width 2048 is missing are cos 1,
        // sin 0 in head 0; a zero position puts the first angle at
        // `-grid[0] = -pi/2`.
        let (cos, sin) =
            rope_tables(&[0.0; 6], 3, VIDEO_DIM, &VIDEO_MAX_POS, HEADS).expect("rope tables");
        let (cos, sin) = (widen(&cos), widen(&sin));

        assert_eq!(cos.len(), HEADS * 64);
        assert_eq!(cos[0], 1.0);
        assert_eq!(cos[1], 1.0);
        assert_eq!(sin[0], 0.0);
        assert_eq!(sin[1], 0.0);
        assert!(cos[2].abs() < 1e-6);
        assert!((sin[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_audio_rope_table_needs_no_padding() {
        // `dim = 2048` at one axis is 1024 angles, exactly the half-width the
        // head split consumes, so `pad_size` is 0
        // (`transformer_args.py:367-368`, `rope.py:213-219`). A zero position
        // is `2 * 0 - 1 = -1` at `grid[0] = pi/2`.
        let (cos, sin) = rope_tables(&[0.0, 0.0], 1, AUDIO_DIM, &AUDIO_MAX_POS, HEADS)
            .expect("audio rope tables");
        let (cos, sin) = (widen(&cos), widen(&sin));

        assert_eq!(cos.len(), HEADS * 32);
        assert_eq!(sin.len(), HEADS * 32);
        assert!(cos[0].abs() < 1e-6);
        assert!((sin[0] + 1.0).abs() < 1e-6);
    }

    /// A BF16 table widened to the values a comparison reads.
    fn widen(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
            .collect()
    }

    #[test]
    fn the_sinusoid_is_the_releases_own() {
        // `timestep_embedding.py:35-49`: `half_dim = 128`, the exponent
        // `-log(10000) * k / 128` runs from 1 to `10000^(-127/128)`, and
        // `flip_sin_to_cos` puts cosine first.
        let values = sinusoid(1.0);
        let last = 10_000.0f32.powf(-127.0 / 128.0);

        assert!((values[0] - 1.0f32.cos()).abs() < 1e-9);
        assert!((values[128] - 1.0f32.sin()).abs() < 1e-9);
        assert!((values[127] - last.cos()).abs() < 1e-6);
        assert!((values[255] - last.sin()).abs() < 1e-6);
    }
}
