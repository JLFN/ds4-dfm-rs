//! The Gemma4 text tower: the INT8 checkpoint and the 48-layer graph.
//!
//! Unit 2b. The tower's primitives are its own `h3d_gemma_*` set (read
//! first-hand in [`crate::minimax_h3`]'s records, `gemma-tower-primitives-*`)
//! and the INT8 convrot linear is unit 1's route (`gemma-int8-convrot-*`);
//! this module is the graph that calls them, from the forward read in
//! `gemma4-tower-port-map-2026-10-29.md`.
//!
//! Source of truth: HuggingFace `gemma4`/`gemma4_unified` at the release's pin
//! (`transformers` 5.14.1, unpacked read-only at
//! `/data/refs/transformers-5.14.1`). Every step below cites it:
//!
//! - `modeling_gemma4.py:1405-1463` `Gemma4TextDecoderLayer.forward`: the eight
//!   steps of one layer and the closing `*= layer_scalar`.
//! - `:1235-1297` `Gemma4TextAttention.forward`: `q = rope(q_norm(q_proj))`,
//!   `k = rope(k_norm(k_proj))`, `v = v_norm(v_proj)` with no rope - and on a
//!   `k_eq_v` layer `v = v_norm(k_proj)`, the raw projection, because
//!   `value_states = key_states` binds the same tensor before either norm.
//! - `:1088-1090` `Gemma4TextMLP.forward`, `:824-858`
//!   `eager_attention_forward` at `scaling = 1.0` (`:1200`).
//! - `:1653-1742` `Gemma4TextModel.forward`: the scaled embedding, `arange`
//!   positions, the two masks, the 48-layer loop, the closing `norm`.
//! - `:1474-1477` `Gemma4TextScaledWordEmbedding.forward`: the scale is
//!   downcast to the weight's dtype, so it is `bf16(sqrt(3840))` here.
//!
//! The 49 hidden states the extractor stacks (`feature_extractor.py:115-129`,
//! whose aggregate embeds are `[out, 49 * 3840]`) are not a list the forward
//! builds: they come from the output-capture hook.
//! `utils/output_capturing.py:112-118` prepends the first decoder layer's own
//! input - the scaled embedding - and appends every layer's output, and
//! `:270-278` then replaces the last of those with `last_hidden_state`. So
//! state 0 is the embedding, states 1..=48 are layers 0..47, and state 48 is
//! `model.norm`'s output, **not** layer 47's raw output. A port that emitted
//! the raw 48th layer output there would look right and be wrong.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype as Device, Gpu, Int8PlanBuffers, Tensor};

use crate::minimax_h3::safetensors::{self, Dtype, Shard};

/// `num_hidden_layers` (`__metadata__.gemma_config`, text_config).
pub const LAYERS: usize = 48;
/// `hidden_size`.
pub const HIDDEN: usize = 3840;
/// `intermediate_size`; `use_double_wide_mlp` is false, so it never doubles.
pub const INTERMEDIATE: usize = 15360;
/// `num_attention_heads`.
pub const HEADS: usize = 16;
/// `head_dim`, the sliding layers' head width.
pub const HEAD_DIM: usize = 256;
/// `global_head_dim`, the full layers' head width.
pub const GLOBAL_HEAD_DIM: usize = 512;
/// `num_key_value_heads`, the sliding layers'.
pub const KV_HEADS: usize = 8;
/// `num_global_key_value_heads`, the full layers' - one, with `k_eq_v`.
pub const GLOBAL_KV_HEADS: usize = 1;
/// `vocab_size`.
pub const VOCABULARY: usize = 262144;
/// `rms_norm_eps`.
pub const RMS_EPSILON: f32 = 1e-6;
/// `sliding_window`, the sliding mask's reach.
pub const SLIDING_WINDOW: usize = 1024;
/// The convrot group size every `comfy_quant` blob in the checkpoint states
/// (`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`),
/// checked against the artifact in [`Tower::open`].
pub const GROUP: usize = 256;
/// The hidden states the extractor stacks: the embedding plus the 48 layers.
pub const STATES: usize = 49;
/// `layer_types` from the checkpoint's own `gemma_config`, the indices that
/// are `full_attention`; the other 40 are `sliding_attention`.
pub const FULL_LAYERS: [usize; 8] = [5, 11, 17, 23, 29, 35, 41, 47];
/// `rope_parameters.sliding_attention.rope_theta` (`rope_type` default).
pub const SLIDING_BASE: f32 = 10_000.0;
/// `rope_parameters.full_attention.rope_theta` (`rope_type` proportional).
pub const FULL_BASE: f32 = 1_000_000.0;
/// The default table rotates every pair: `head_dim / 2` frequencies.
pub const SLIDING_ANGLES: usize = HEAD_DIM / 2;
/// `int(partial_rotary_factor * global_head_dim // 2)` with 0.25
/// (`modeling_rope_utils.py:234`): 64 rotated frequencies, the rest the
/// identity pair.
pub const FULL_ANGLES: usize = GLOBAL_HEAD_DIM / 8;
/// `self.scaling = 1.0` (`modeling_gemma4.py:1200`), passed explicitly so the
/// eager default never applies.
pub const ATTENTION_SCALE: f32 = 1.0;
/// `H3D_GEMMA_ROWS_MAX` (`native/h3/h3_dit_gpu.h`): one row's score vector is
/// staged in shared memory, so a run cannot exceed this many tokens.
pub const ROWS_MAX: usize = 4096;

/// Is this layer one of the eight `full_attention` layers.
pub fn is_full(layer: usize) -> bool {
    FULL_LAYERS.contains(&layer)
}

/// The layer's head width: `global_head_dim` on a full layer (`:1195`).
pub fn head_dim(layer: usize) -> usize {
    if is_full(layer) {
        GLOBAL_HEAD_DIM
    } else {
        HEAD_DIM
    }
}

/// The layer's key/value head count.
pub fn kv_heads(layer: usize) -> usize {
    if is_full(layer) {
        GLOBAL_KV_HEADS
    } else {
        KV_HEADS
    }
}

/// Round to BF16 and widen, which is what `__float2bfloat16` does: nearest,
/// ties to even, over the low 16 bits.
pub fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();

    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return value;
    }

    let half = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + half) & 0xffff_0000)
}

/// `embed_scale = hidden_size ** 0.5` cast to the weight's dtype
/// (`:1474`, `:1608-1610`): the embedding op takes the already-rounded BF16
/// value widened.
pub fn embed_scale() -> f32 {
    bf16_round((HIDDEN as f32).sqrt())
}

/// Why the tower could not be loaded or run.
#[derive(Debug)]
pub enum GemmaError {
    /// The checkpoint could not be read as a shard.
    Reader(String),
    /// A tensor the graph needs is absent from the checkpoint.
    Missing(String),
    /// A tensor is present with a dtype, shape or byte span the graph cannot use.
    Shape { name: String, detail: String },
    /// The artifact's own `comfy_quant` blob disagrees with [`GROUP`].
    Group(String),
    /// A device call refused, or the arguments do not fit the graph's contract.
    Run(String),
}

impl GemmaError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            Self::Reader(detail) => format!("reader: {detail}"),
            Self::Missing(name) => format!("missing {name}"),
            Self::Shape { name, detail } => format!("shape {name}: {detail}"),
            Self::Group(detail) => format!("convrot group: {detail}"),
            Self::Run(detail) => format!("run: {detail}"),
        }
    }
}

impl fmt::Display for GemmaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for GemmaError {}

/// One quantized linear: the convrot route's two operands, exactly as the
/// checkpoint carries them. The weight is already rotated, so nothing here
/// rotates a weight.
struct Linear {
    weight: Tensor,
    scale: Tensor,
    input_dim: u32,
    output_dim: u32,
}

impl Linear {
    /// `h3d_linear_int8_convrot`, with the plan belonging to this shape.
    fn apply(
        &self,
        gpu: &Gpu,
        plans: &Plans,
        output: &Tensor,
        input: &Tensor,
        rows: u32,
    ) -> Result<(), GemmaError> {
        gpu.linear_int8_convrot(
            output,
            input,
            &self.weight,
            &self.scale,
            None,
            rows,
            self.input_dim,
            self.output_dim,
            GROUP as u32,
            plans.get(self.input_dim, self.output_dim),
        )
        .map_err(GemmaError::Run)
    }
}

/// One layer's tensors.
struct Layer {
    input_norm: Tensor,
    post_attention_norm: Tensor,
    pre_feedforward_norm: Tensor,
    post_feedforward_norm: Tensor,
    layer_scalar: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    query: Linear,
    key: Linear,
    /// Absent on the eight `k_eq_v` layers (`:1224-1228`), where V is the raw
    /// `k_proj` output.
    value: Option<Linear>,
    output: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// The convrot route's workspaces, one plan per `(input_dim, output_dim)`
/// shape the run uses: `h3d_int8_plan`'s own contract is one plan per shape
/// serving every layer.
struct Plans {
    entries: Vec<(u32, u32, Int8PlanBuffers)>,
}

impl Plans {
    fn build(gpu: &Gpu, rows: u32, layers: usize) -> Result<Plans, GemmaError> {
        let mut shapes: Vec<(u32, u32)> = Vec::new();

        for layer in 0..layers {
            let width = head_dim(layer) as u32;
            let kv_width = (kv_heads(layer) * head_dim(layer)) as u32;

            shapes.push((HIDDEN as u32, HEADS as u32 * width));
            shapes.push((HIDDEN as u32, kv_width));
            shapes.push((HEADS as u32 * width, HIDDEN as u32));
            if !is_full(layer) {
                shapes.push((HIDDEN as u32, kv_width));
            }
            shapes.push((HIDDEN as u32, INTERMEDIATE as u32));
            shapes.push((INTERMEDIATE as u32, HIDDEN as u32));
        }

        let mut entries: Vec<(u32, u32, Int8PlanBuffers)> = Vec::new();

        for (input_dim, output_dim) in shapes {
            let known = entries
                .iter()
                .any(|(seen_in, seen_out, _)| *seen_in == input_dim && *seen_out == output_dim);

            if known {
                continue;
            }

            let plan = gpu
                .int8_plan(rows, input_dim, output_dim, GROUP as u32)
                .map_err(GemmaError::Run)?;

            entries.push((input_dim, output_dim, plan));
        }

        Ok(Plans { entries })
    }

    fn get(&self, input_dim: u32, output_dim: u32) -> &Int8PlanBuffers {
        &self
            .entries
            .iter()
            .find(|(seen_in, seen_out, _)| *seen_in == input_dim && *seen_out == output_dim)
            .expect("a plan was built for every shape the run uses")
            .2
    }
}

/// The loaded tower: the embedding, the closing norm and the layers asked for.
pub struct Tower {
    embed_tokens: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    /// The tensors `open` read, for the gate's record.
    tensors: usize,
    bytes: u64,
}

impl Tower {
    /// Read the checkpoint's own header, check every tensor the graph needs
    /// against its descriptor, and upload the first `layers` layers.
    ///
    /// The checkpoint is one file with `model.*` names already in the
    /// flattened HuggingFace layout (`gemma_assets.py:314-331`), so every name
    /// here is the model's own. Membaership in [`FULL_LAYERS`] decides each
    /// layer's shape, and a layer whose `q_proj` disagrees with that reading
    /// is refused rather than reshaped.
    pub fn open(gpu: &Gpu, path: &Path, layers: usize) -> Result<Tower, GemmaError> {
        if layers == 0 || layers > LAYERS {
            return Err(GemmaError::Run(format!(
                "layers must be 1..={LAYERS}, got {layers}"
            )));
        }

        let shard =
            safetensors::read_shard(path).map_err(|error| GemmaError::Reader(error.token()))?;
        check_group(&shard)?;

        let mut reader = Reader {
            gpu,
            shard: &shard,
            buffer: Vec::new(),
            tensors: 0,
            bytes: 0,
        };

        let embed_tokens = reader.pull(
            "model.embed_tokens.weight",
            Dtype::Bf16,
            &[VOCABULARY, HIDDEN],
        )?;
        let norm = reader.pull("model.norm.weight", Dtype::Bf16, &[HIDDEN])?;
        let mut loaded = Vec::with_capacity(layers);

        for layer in 0..layers {
            loaded.push(load_layer(&mut reader, layer)?);
        }

        Ok(Tower {
            embed_tokens,
            norm,
            layers: loaded,
            tensors: reader.tensors,
            bytes: reader.bytes,
        })
    }

    /// The tensors read, and their bytes, for the gate's record.
    pub fn loaded(&self) -> (usize, u64) {
        (self.tensors, self.bytes)
    }

    /// How many layers are resident.
    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// `Gemma4TextModel.forward` (`:1653-1742`) without `pool`: the scaled
    /// embedding, the two masks, the layer loop, and the 49 hidden states the
    /// extractor stacks.
    ///
    /// `ids` are the token ids and `padding` the attention mask over the keys
    /// (1 for a real token, 0 for the left padding), which is what
    /// `encoders/base_encoder.py:56-66` forwards. Positions are the row
    /// indices, `arange(sequence_length)` (`:1689-1691`): the left padding does
    /// not shift them.
    pub fn encode(
        &self,
        gpu: &Gpu,
        ids: &[u32],
        padding: &[u32],
        layers: usize,
    ) -> Result<Vec<Tensor>, GemmaError> {
        let rows = ids.len();

        if rows == 0 || rows != padding.len() {
            return Err(GemmaError::Run(
                "ids and padding must be the same non-zero length".to_string(),
            ));
        }
        if rows > ROWS_MAX {
            return Err(GemmaError::Run(format!(
                "rows must be at most {ROWS_MAX}, got {rows}"
            )));
        }
        if layers == 0 || layers > self.layers.len() {
            return Err(GemmaError::Run(format!(
                "layers must be 1..={}, got {layers}",
                self.layers.len()
            )));
        }

        let rows = rows as u32;
        let plans = Plans::build(gpu, rows, layers)?;

        let device_ids = upload_u32(gpu, ids)?;
        let device_padding = upload_u32(gpu, padding)?;

        // The two masks and the two rope tables are built once for the whole
        // tower, as the reference builds them once per forward
        // (`:1694-1708`, `:1710-1715`).
        let sliding_mask = mask(gpu, rows, SLIDING_WINDOW as u32, &device_padding)?;
        let full_mask = mask(gpu, rows, 0, &device_padding)?;
        let (sliding_cos, sliding_sin) = tables(
            gpu,
            rows,
            HEAD_DIM as u32,
            SLIDING_ANGLES as u32,
            SLIDING_BASE,
        )?;
        let (full_cos, full_sin) = tables(
            gpu,
            rows,
            GLOBAL_HEAD_DIM as u32,
            FULL_ANGLES as u32,
            FULL_BASE,
        )?;

        let mut states = Vec::with_capacity(STATES);
        let embedding = alloc(gpu, Device::Bf16, rows as usize * HIDDEN)?;
        gpu.embedding(
            &embedding,
            &self.embed_tokens,
            &device_ids,
            rows,
            HIDDEN as u32,
            embed_scale(),
        )
        .map_err(GemmaError::Run)?;
        states.push(embedding);

        // Scratch, reused by every layer: the layer's own output is the only
        // thing that has to survive it, and that is `states[layer + 1]`.
        let scratch = Scratch::new(gpu, rows)?;

        for layer in 0..layers {
            let weights = &self.layers[layer];
            let width = head_dim(layer);
            let heads = kv_heads(layer);
            let full = is_full(layer);
            let head = if full {
                &scratch.full
            } else {
                &scratch.sliding
            };
            let input = &states[layer];
            let output = alloc(gpu, Device::Bf16, rows as usize * HIDDEN)?;

            // `input_layernorm` (`:1417`).
            gpu.rms_norm(
                &scratch.normed,
                input,
                Some(&weights.input_norm),
                rows,
                HIDDEN as u32,
                RMS_EPSILON,
            )
            .map_err(GemmaError::Run)?;

            let (cos, sin, mask) = if full {
                (&full_cos, &full_sin, &full_mask)
            } else {
                (&sliding_cos, &sliding_sin, &sliding_mask)
            };

            // `q = rope(q_norm(q_proj(x)))` (`:1249-1252`). `q_norm` is an RMS
            // norm over the head, so its rows are tokens x heads.
            weights
                .query
                .apply(gpu, &plans, &head.query_raw, &scratch.normed, rows)?;
            gpu.rms_norm(
                &head.query,
                &head.query_raw,
                Some(&weights.q_norm),
                rows * HEADS as u32,
                width as u32,
                RMS_EPSILON,
            )
            .map_err(GemmaError::Run)?;

            // `k = rope(k_norm(k_proj(x)))`, and on a `k_eq_v` layer V is that
            // same raw projection through the weightless norm (`:1265-1271`).
            weights
                .key
                .apply(gpu, &plans, &head.key_raw, &scratch.normed, rows)?;
            gpu.rms_norm(
                &head.key,
                &head.key_raw,
                Some(&weights.k_norm),
                rows * heads as u32,
                width as u32,
                RMS_EPSILON,
            )
            .map_err(GemmaError::Run)?;

            let value_source = match &weights.value {
                Some(value) => {
                    value.apply(gpu, &plans, &head.value_raw, &scratch.normed, rows)?;
                    &head.value_raw
                }
                None => &head.key_raw,
            };

            gpu.rms_norm(
                &head.value,
                value_source,
                None,
                rows * heads as u32,
                width as u32,
                RMS_EPSILON,
            )
            .map_err(GemmaError::Run)?;

            gpu.gemma_rope(
                &head.query,
                &head.key,
                cos,
                sin,
                rows,
                HEADS as u32,
                heads as u32,
                width as u32,
            )
            .map_err(GemmaError::Run)?;

            gpu.gemma_attention(
                &head.attention,
                &head.query,
                &head.key,
                &head.value,
                Some(mask),
                rows,
                HEADS as u32,
                heads as u32,
                width as u32,
                ATTENTION_SCALE,
            )
            .map_err(GemmaError::Run)?;

            weights
                .output
                .apply(gpu, &plans, &scratch.projected, &head.attention, rows)?;

            // `post_attention_layernorm`, then the first residual (`:1427-1429`).
            gpu.rms_norm(
                &scratch.attention_normed,
                &scratch.projected,
                Some(&weights.post_attention_norm),
                rows,
                HIDDEN as u32,
                RMS_EPSILON,
            )
            .map_err(GemmaError::Run)?;
            gpu.add(&scratch.residual, input, &scratch.attention_normed)
                .map_err(GemmaError::Run)?;

            // `pre_feedforward_layernorm` and the MLP (`:1432-1433`).
            gpu.rms_norm(
                &scratch.normed,
                &scratch.residual,
                Some(&weights.pre_feedforward_norm),
                rows,
                HIDDEN as u32,
                RMS_EPSILON,
            )
            .map_err(GemmaError::Run)?;
            weights
                .gate
                .apply(gpu, &plans, &scratch.gate, &scratch.normed, rows)?;
            weights
                .up
                .apply(gpu, &plans, &scratch.up, &scratch.normed, rows)?;
            gpu.gelu_tanh(&scratch.gate, &scratch.gate)
                .map_err(GemmaError::Run)?;
            gpu.mul(&scratch.gate, &scratch.gate, &scratch.up)
                .map_err(GemmaError::Run)?;
            weights
                .down
                .apply(gpu, &plans, &scratch.feedforward, &scratch.gate, rows)?;

            // `post_feedforward_layernorm`, the second residual and the layer
            // scalar (`:1449-1462`).
            gpu.rms_norm(
                &scratch.feedforward_normed,
                &scratch.feedforward,
                Some(&weights.post_feedforward_norm),
                rows,
                HIDDEN as u32,
                RMS_EPSILON,
            )
            .map_err(GemmaError::Run)?;
            gpu.add(&output, &scratch.residual, &scratch.feedforward_normed)
                .map_err(GemmaError::Run)?;
            gpu.scale_bf16(&output, &output, &weights.layer_scalar)
                .map_err(GemmaError::Run)?;

            states.push(output);
        }

        // The hook replaces the last collected state with `last_hidden_state`
        // (`output_capturing.py:273-278`), which is `self.norm(hidden_states)`
        // (`:1736`) - so state 48 is the normed output, not layer 47's raw one.
        //
        // That substitution only happens when the whole model ran, which is
        // `layers == LAYERS`. A layer-limited run is a diagnostic, and there
        // the last state stays the layer's own output so a comparison
        // localizes to that layer instead of folding a norm into it.
        if layers == LAYERS {
            let closing = alloc(gpu, Device::Bf16, rows as usize * HIDDEN)?;
            gpu.rms_norm(
                &closing,
                &states[STATES - 1],
                Some(&self.norm),
                rows,
                HIDDEN as u32,
                RMS_EPSILON,
            )
            .map_err(GemmaError::Run)?;
            states[STATES - 1] = closing;
        }

        Ok(states)
    }
}

/// The layer-sized scratch one forward reuses, so a 48-layer run allocates a
/// fixed set of buffers rather than one set per layer.
///
/// The attention buffers are per layer **flavour**, because every native call
/// checks the element count exactly: Q on a sliding layer is
/// `rows x 16 x 256`, on a full layer `rows x 16 x 512`, and an oversized
/// buffer is refused rather than partly used.
struct Scratch {
    normed: Tensor,
    projected: Tensor,
    attention_normed: Tensor,
    residual: Tensor,
    gate: Tensor,
    up: Tensor,
    feedforward: Tensor,
    feedforward_normed: Tensor,
    sliding: Head,
    full: Head,
}

/// One flavour's head-major Q/K/V and output.
struct Head {
    query_raw: Tensor,
    query: Tensor,
    key_raw: Tensor,
    key: Tensor,
    value_raw: Tensor,
    value: Tensor,
    attention: Tensor,
}

impl Head {
    fn new(gpu: &Gpu, rows: u32, kv_heads: usize, width: usize) -> Result<Head, GemmaError> {
        let query = rows as usize * HEADS * width;
        let key = rows as usize * kv_heads * width;

        Ok(Head {
            query_raw: alloc(gpu, Device::Bf16, query)?,
            query: alloc(gpu, Device::Bf16, query)?,
            key_raw: alloc(gpu, Device::Bf16, key)?,
            key: alloc(gpu, Device::Bf16, key)?,
            value_raw: alloc(gpu, Device::Bf16, key)?,
            value: alloc(gpu, Device::Bf16, key)?,
            attention: alloc(gpu, Device::Bf16, query)?,
        })
    }
}

impl Scratch {
    fn new(gpu: &Gpu, rows: u32) -> Result<Scratch, GemmaError> {
        let hidden = rows as usize * HIDDEN;
        let wide = rows as usize * INTERMEDIATE;

        Ok(Scratch {
            normed: alloc(gpu, Device::Bf16, hidden)?,
            projected: alloc(gpu, Device::Bf16, hidden)?,
            attention_normed: alloc(gpu, Device::Bf16, hidden)?,
            residual: alloc(gpu, Device::Bf16, hidden)?,
            gate: alloc(gpu, Device::Bf16, wide)?,
            up: alloc(gpu, Device::Bf16, wide)?,
            feedforward: alloc(gpu, Device::Bf16, hidden)?,
            feedforward_normed: alloc(gpu, Device::Bf16, hidden)?,
            sliding: Head::new(gpu, rows, KV_HEADS, HEAD_DIM)?,
            full: Head::new(gpu, rows, GLOBAL_KV_HEADS, GLOBAL_HEAD_DIM)?,
        })
    }
}

/// The checkpoint's reader: one reusable host buffer, and the counters a gate
/// reports. Every read goes through [`Reader::pull`], so no tensor reaches the
/// device without its descriptor being checked against the graph's own
/// expectation first.
struct Reader<'a> {
    gpu: &'a Gpu,
    shard: &'a Shard,
    buffer: Vec<u8>,
    tensors: usize,
    bytes: u64,
}

impl Reader<'_> {
    /// Read one tensor into the reusable buffer and upload it, checking the
    /// checkpoint's own descriptor first: dtype, every dimension and the byte
    /// span must agree with what the graph will do with it.
    fn pull(&mut self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<Tensor, GemmaError> {
        let tensor = self
            .shard
            .find(name)
            .ok_or_else(|| GemmaError::Missing(name.to_string()))?;
        let want: Vec<u64> = shape.iter().map(|dim| *dim as u64).collect();

        if tensor.dtype != dtype || tensor.shape != want {
            return Err(GemmaError::Shape {
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
            return Err(GemmaError::Shape {
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

        let device = device_dtype(dtype).ok_or_else(|| GemmaError::Shape {
            name: name.to_string(),
            detail: format!("{} has no device type", dtype.name()),
        })?;

        self.buffer.clear();
        self.buffer.resize(tensor.bytes() as usize, 0);
        self.shard
            .read_data(tensor, &mut self.buffer)
            .map_err(|error| GemmaError::Reader(error.token()))?;

        let device = alloc(self.gpu, device, elements as usize)?;
        self.gpu
            .upload(&device, &self.buffer)
            .map_err(GemmaError::Run)?;

        self.tensors += 1;
        self.bytes += tensor.bytes();

        Ok(device)
    }
}

/// One layer's tensors, read and uploaded.
fn load_layer(reader: &mut Reader, layer: usize) -> Result<Layer, GemmaError> {
    let width = head_dim(layer);
    let kv_width = kv_heads(layer) * width;
    let full = is_full(layer);

    let norm = |reader: &mut Reader, suffix: &str| -> Result<Tensor, GemmaError> {
        reader.pull(
            &format!("model.layers.{layer}.{suffix}"),
            Dtype::Bf16,
            &[HIDDEN],
        )
    };
    let head_norm = |reader: &mut Reader, suffix: &str| -> Result<Tensor, GemmaError> {
        reader.pull(
            &format!("model.layers.{layer}.{suffix}"),
            Dtype::Bf16,
            &[width],
        )
    };
    let linear = |reader: &mut Reader,
                  suffix: &str,
                  input_dim: usize,
                  output_dim: usize|
     -> Result<Linear, GemmaError> {
        Ok(Linear {
            weight: reader.pull(
                &format!("model.layers.{layer}.{suffix}.weight"),
                Dtype::I8,
                &[output_dim, input_dim],
            )?,
            scale: reader.pull(
                &format!("model.layers.{layer}.{suffix}.weight_scale"),
                Dtype::F32,
                &[output_dim, 1],
            )?,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        })
    };

    let input_norm = norm(reader, "input_layernorm.weight")?;
    let post_attention_norm = norm(reader, "post_attention_layernorm.weight")?;
    let pre_feedforward_norm = norm(reader, "pre_feedforward_layernorm.weight")?;
    let post_feedforward_norm = norm(reader, "post_feedforward_layernorm.weight")?;
    let layer_scalar = reader.pull(
        &format!("model.layers.{layer}.layer_scalar"),
        Dtype::Bf16,
        &[1],
    )?;
    let q_norm = head_norm(reader, "self_attn.q_norm.weight")?;
    let k_norm = head_norm(reader, "self_attn.k_norm.weight")?;
    let query = linear(reader, "self_attn.q_proj", HIDDEN, HEADS * width)?;
    let key = linear(reader, "self_attn.k_proj", HIDDEN, kv_width)?;
    let value = if full {
        None
    } else {
        Some(linear(reader, "self_attn.v_proj", HIDDEN, kv_width)?)
    };
    let output = linear(reader, "self_attn.o_proj", HEADS * width, HIDDEN)?;
    let gate = linear(reader, "mlp.gate_proj", HIDDEN, INTERMEDIATE)?;
    let up = linear(reader, "mlp.up_proj", HIDDEN, INTERMEDIATE)?;
    let down = linear(reader, "mlp.down_proj", INTERMEDIATE, HIDDEN)?;

    Ok(Layer {
        input_norm,
        post_attention_norm,
        pre_feedforward_norm,
        post_feedforward_norm,
        layer_scalar,
        q_norm,
        k_norm,
        query,
        key,
        value,
        output,
        gate,
        up,
        down,
    })
}

/// Check the artifact's own `comfy_quant` blob against [`GROUP`], so the group
/// size comes from the checkpoint rather than from a comment.
fn check_group(shard: &Shard) -> Result<(), GemmaError> {
    let name = "model.layers.0.self_attn.q_proj.comfy_quant";
    let tensor = shard
        .find(name)
        .ok_or_else(|| GemmaError::Missing(name.to_string()))?;

    if tensor.dtype != Dtype::U8 {
        return Err(GemmaError::Group(format!(
            "{name} is {}, not U8",
            tensor.dtype.name()
        )));
    }

    let mut blob = vec![0u8; tensor.bytes() as usize];
    shard
        .read_data(tensor, &mut blob)
        .map_err(|error| GemmaError::Reader(error.token()))?;
    let text = String::from_utf8_lossy(&blob);
    let wanted = format!("\"convrot_groupsize\": {GROUP}");

    if !text.contains(&wanted) {
        return Err(GemmaError::Group(format!("{name} states {text}")));
    }

    Ok(())
}

fn device_dtype(dtype: Dtype) -> Option<Device> {
    match dtype {
        Dtype::Bf16 => Some(Device::Bf16),
        Dtype::F32 => Some(Device::F32),
        Dtype::U32 => Some(Device::U32),
        Dtype::I8 => Some(Device::I8),
        _ => None,
    }
}

fn alloc(gpu: &Gpu, dtype: Device, elements: usize) -> Result<Tensor, GemmaError> {
    gpu.alloc(dtype, elements).map_err(GemmaError::Run)
}

fn upload_u32(gpu: &Gpu, values: &[u32]) -> Result<Tensor, GemmaError> {
    let tensor = alloc(gpu, Device::U32, values.len())?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload(&tensor, bytes).map_err(GemmaError::Run)?;

    Ok(tensor)
}

fn mask(gpu: &Gpu, rows: u32, window: u32, padding: &Tensor) -> Result<Tensor, GemmaError> {
    let output = alloc(gpu, Device::Bf16, rows as usize * rows as usize)?;
    gpu.gemma_mask(&output, rows, window, Some(padding))
        .map_err(GemmaError::Run)?;

    Ok(output)
}

/// One RoPE flavour's cosine and sine pair tables (`h3d_gemma_rope_table`),
/// separate tensors: the kernel writes both, so one buffer cannot serve for
/// both.
fn tables(
    gpu: &Gpu,
    rows: u32,
    dim: u32,
    angles: u32,
    base: f32,
) -> Result<(Tensor, Tensor), GemmaError> {
    let elements = rows as usize * dim as usize;
    let cos = alloc(gpu, Device::Bf16, elements)?;
    let sin = alloc(gpu, Device::Bf16, elements)?;
    gpu.gemma_rope_table(&cos, &sin, rows, dim, angles, base)
        .map_err(GemmaError::Run)?;

    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_matches_the_hardware_conversion() {
        // `__float2bfloat16`: nearest, ties to even over the low 16 bits.
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(-2.5), -2.5);
        // 1 + 2^-8 is the first value BF16 cannot hold; the tie goes to even.
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 2.0 / 128.0);
        assert_eq!(bf16_round(0.0), 0.0);
    }

    #[test]
    fn the_embed_scale_is_the_reference_downcast() {
        // sqrt(3840) = 61.96773353931867, and BF16 keeps 8 mantissa bits: the
        // binade [32, 64) steps by 0.125, and 61.9677 is 0.0322 above 61.9375
        // but only 0.0323 below 62.0, so the downcast lands on **62.0**. The
        // embedding therefore multiplies by exactly 62, not by its square root;
        // the reference's own comment records the same rounding for Gemma 3
        // (sqrt(3072) = 55.4256 -> 55.5).
        assert_eq!(
            (HIDDEN as f32).sqrt().to_bits(),
            61.967_735_29_f32.to_bits()
        );
        assert_eq!(embed_scale(), 62.0);
        assert_eq!(bf16_round(embed_scale()), embed_scale());
    }

    #[test]
    fn the_layer_flavours_follow_layer_types() {
        assert!(is_full(5) && is_full(47));
        assert!(!is_full(0) && !is_full(4) && !is_full(6));
        assert_eq!(head_dim(5), GLOBAL_HEAD_DIM);
        assert_eq!(head_dim(0), HEAD_DIM);
        assert_eq!(kv_heads(5), GLOBAL_KV_HEADS);
        assert_eq!(kv_heads(0), KV_HEADS);
        assert_eq!(FULL_ANGLES, 64);
        assert_eq!(SLIDING_ANGLES, 128);
    }
}
