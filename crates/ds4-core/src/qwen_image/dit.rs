//! The P1 CPU oracle's stage 2: one complete F32 DiT forward.
//!
//! Stage 1 ([`super::oracle`]) settled the numerics that can be checked without
//! weights — the noise, the layout, the RoPE table, the schedule. This module
//! adds the evaluation itself, driven by the reference's own conditioning dumps,
//! so P2 has a measured target to compare kernels against. It is slow, F32 and
//! single-machine by design; nothing here is a serving path.
//!
//! Ported 1:1 from `stable-diffusion.cpp` at `6dcb5bb`: `qwen_image_2_1.hpp`
//! for the model, `ggml_block.hpp` for the blocks, `ggml_extend.cpp` for the
//! glue and `ggml-cpu/ops.cpp` for the elementwise kernels. Every function
//! names the reference lines it came from.
//!
//! Fidelity class: stated tolerance. The reference ran this fixture on CUDA
//! (`diffusion=cuda0`) with Q6_K weights and its own accumulation order, so this
//! port is expected to land close, not bit-exact. What must be exact is the
//! structure: shapes, layout, segment boundaries, operation order and the RoPE
//! pairing. The pairing is settled by measurement here, not by reading.

use std::path::Path;
use std::thread;

use super::oracle::{apply_rope, build_layout, rope_table, text_mask, Layout, LayoutError, RopePairing};
use super::{
    DIT_AXES_DIM, DIT_CONTEXT_DIM, DIT_HEAD_DIM, DIT_HEADS, DIT_HIDDEN, DIT_IN_CHANNELS,
    DIT_INTERMEDIATE, DIT_LAYERS, DIT_MODULATION, DIT_NORM_EPS, DIT_OUT_CHANNELS, DIT_ROPE_THETA,
    DIT_TIME_EMBED_DIM, TYPE_BF16, TYPE_Q6_K,
};
use crate::gguf::GgufFile;
use crate::tensors::{TensorError, TensorInfo, TensorInventory};

// ---------------------------------------------------------------------------
// Block formats (`ggml.h`, `ds4.c:1167`)
// ---------------------------------------------------------------------------

/// Values per Q6_K super-block (`QK_K`).
pub const QK_K: usize = 256;
/// `ql[128] + qh[64] + scales[16] i8 + d f16`.
pub const Q6K_BLOCK_BYTES: usize = 210;

/// IEEE 754 binary16 -> f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x03ff);

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal half: renormalize into the f32 exponent range. A half
        // subnormal is `mant * 2^-24`, so shifting `mant` up to bit 10 costs
        // one exponent step each.
        let mut shift = 0i32;
        let mut m = mant;
        while m & 0x0400 == 0 {
            m <<= 1;
            shift += 1;
        }
        return f32::from_bits(sign | ((113 - shift) as u32) << 23 | ((m & 0x03ff) << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13))
}

/// BF16 -> f32 is exact: the top half of the f32 bit pattern.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One Q6_K super-block, dequantized in the reference's own order
/// (`ds4.c:4538`: four 32-value groups per 128-half, scales at `is`, `is+2`,
/// `is+4`, `is+6`).
fn dequant_q6k_block(block: &[u8], dst: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    for half in 0..2 {
        let ql = &block[half * 64..];
        let qh = &block[128 + half * 32..];
        let scales = &block[192 + half * 8..];

        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

            let base = half * 128 + l;
            let s = |slot: usize| d * (scales[slot] as i8) as f32;
            dst[base] = s(is) * q1 as f32;
            dst[base + 32] = s(is + 2) * q2 as f32;
            dst[base + 64] = s(is + 4) * q3 as f32;
            dst[base + 96] = s(is + 6) * q4 as f32;
        }
    }
}

/// A pinned weight tensor in the two layouts the artifact carries.
#[derive(Clone, Copy)]
enum Rows<'a> {
    Bf16(&'a [u8]),
    Q6K(&'a [u8]),
}

impl Rows<'_> {
    /// Dequantizes output row `row` (the `k` inputs of one weight row) into
    /// `dst`. Q6_K rows are whole super-blocks; BF16 rows are a shift.
    fn row(&self, k: usize, row: usize, dst: &mut [f32]) {
        match self {
            Rows::Bf16(bytes) => {
                let row_bytes = &bytes[row * k * 2..(row + 1) * k * 2];
                for (out, pair) in dst.iter_mut().zip(row_bytes.chunks_exact(2)) {
                    *out = bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
                }
            }
            Rows::Q6K(bytes) => {
                let per_row = k / QK_K;
                for block in 0..per_row {
                    let at = (row * per_row + block) * Q6K_BLOCK_BYTES;
                    dequant_q6k_block(&bytes[at..at + Q6K_BLOCK_BYTES], &mut dst[block * QK_K..]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// F32 kernels
// ---------------------------------------------------------------------------

/// Eight independent accumulators. LLVM vectorizes this without reassociating
/// F32; a single-accumulator loop is a serial dependency chain and runs several
/// times slower. The horizontal sum order is fixed, so a result never depends
/// on the thread count.
fn dot8(w: &[f32], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let full = w.len() / 8;

    for chunk in 0..full {
        for lane in 0..8 {
            acc[lane] += w[chunk * 8 + lane] * x[chunk * 8 + lane];
        }
    }

    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in full * 8..w.len() {
        sum += w[i] * x[i];
    }
    sum
}

/// `y = W x`: `W` is `[m, k]` row-major (GGML's `ne0` fastest), `x` is `[k, n]`
/// feature-fastest, `y` is `[m, n]` feature-fastest — the same layout every
/// activation buffer in the forward uses.
///
/// Output rows are independent, so the work splits over rows and the quantized
/// weight is streamed once: one dequantized row is 16 KiB and stays in cache
/// while it is dotted against every token. The threads fill a row-major `[m, n]`
/// scratch, which is transposed into the feature-fastest result at the end —
/// 4 MiB of scattered writes against matmuls that take most of a second.
fn matmul_rows(src: &Rows, k: usize, m: usize, x: &[f32], n: usize, out: &mut [f32]) {
    let threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    let per = m.div_ceil(threads.max(1));
    let mut staged = vec![0f32; m * n];

    thread::scope(|scope| {
        for (index, block) in staged.chunks_mut(per * n).enumerate() {
            let src = *src;
            let first = index * per;

            scope.spawn(move || {
                let mut row = vec![0f32; k];
                for (r, row_out) in block.chunks_mut(n).enumerate() {
                    src.row(k, first + r, &mut row);
                    for (t, value) in row_out.iter_mut().enumerate() {
                        *value = dot8(&row, &x[t * k..(t + 1) * k]);
                    }
                }
            });
        }
    });

    for o in 0..m {
        for t in 0..n {
            out[t * m + o] = staged[o * n + t];
        }
    }
}

/// One row of `ggml_ext_layer_norm` with no affine weight (`ggml_norm`):
/// `(x - mean) / sqrt(var + eps)`. The reference accumulates in F32 SIMD; this
/// accumulates in double, a ~1e-7 relative difference that is inside the stated
/// tolerance.
fn layer_norm_row(x: &mut [f32]) {
    let count = x.len() as f64;
    let mean = (x.iter().map(|v| *v as f64).sum::<f64>() / count) as f32;
    let var =
        (x.iter().map(|v| (*v - mean) as f64 * (*v - mean) as f64).sum::<f64>() / count) as f32;
    let scale = 1.0 / (var + DIT_NORM_EPS).sqrt();

    for v in x.iter_mut() {
        *v = (*v - mean) * scale;
    }
}

/// One row of `ggml_compute_forward_rms_norm_f32` (`ops.cpp:3836`): the sum of
/// squares in double, `scale = 1/sqrt(mean + eps)`, then `x*scale*w`. The
/// zero-centered text norm passes `w + 1` (`qwen_image_2_1.hpp:122-133`).
fn rms_norm_row(x: &mut [f32], weight: &[f32]) {
    let sum: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let mean = (sum / x.len() as f64) as f32;
    let scale = 1.0f32 / (mean + DIT_NORM_EPS).sqrt();

    for (v, w) in x.iter_mut().zip(weight) {
        *v = *v * scale * *w;
    }
}

/// `ggml_vec_silu_f32`: `x / (1 + exp(-x))`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `ggml_gelu_f32` (`vec.h:963`), the exact tanh form — not the fp16-table
/// variant `GGML_GELU_FP16` selects.
fn gelu(x: f32) -> f32 {
    const GELU_COEF_A: f32 = 0.044715;
    const SQRT_2_OVER_PI: f32 = 0.79788456;

    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// `ggml_compute_forward_timestep_embedding_f32` (`ops.cpp:8319`): per timestep
/// and `j < dim/2`, `freq = expf(-logf(max_period) * j / half)`, then
/// `cos(t*freq)` in the low half and `sin(t*freq)` in the high half.
fn timestep_embedding(times: &[f32], max_period: f32) -> Vec<f32> {
    let dim = DIT_TIME_EMBED_DIM as usize;
    let half = dim / 2;
    let log_period = max_period.ln();
    let mut out = vec![0f32; dim * times.len()];

    for (i, t) in times.iter().enumerate() {
        for j in 0..half {
            let freq = (-log_period * j as f32 / half as f32).exp();
            let arg = t * freq;
            out[j + dim * i] = arg.cos();
            out[j + half + dim * i] = arg.sin();
        }
    }
    out
}

/// Numerically stable softmax over one row; `-inf` mask entries become zero.
fn softmax_row(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;

    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum DitError {
    Io(std::io::Error),
    Gguf(String),
    Tensors(String),
    Layout(LayoutError),
    /// The artifact does not carry a pinned tensor, or carries it with a shape
    /// or type the contract does not allow.
    Contract { tensor: String, why: String },
    /// The pass asked for something stage 2 does not implement.
    Unsupported(&'static str),
}

impl DitError {
    pub fn token(&self) -> String {
        match self {
            DitError::Io(e) => format!("io {e}"),
            DitError::Gguf(m) => format!("gguf {m}"),
            DitError::Tensors(m) => format!("tensors {m}"),
            DitError::Layout(e) => format!("layout {e}"),
            DitError::Contract { tensor, why } => format!("contract {tensor}: {why}"),
            DitError::Unsupported(what) => format!("unsupported {what}"),
        }
    }
}

impl std::fmt::Display for DitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "qwen-image-2.1 dit: {}", self.token())
    }
}

impl std::error::Error for DitError {}

impl From<std::io::Error> for DitError {
    fn from(e: std::io::Error) -> Self {
        DitError::Io(e)
    }
}

impl From<LayoutError> for DitError {
    fn from(e: LayoutError) -> Self {
        DitError::Layout(e)
    }
}

fn contract<T>(tensor: &str, why: &str) -> Result<T, DitError> {
    Err(DitError::Contract { tensor: tensor.to_string(), why: why.to_string() })
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// The DiT artifact, mmap-backed. The model is 5.6 GiB of Q6_K and about 28 GiB
/// as F32, so nothing is materialized wholesale: a matmul dequantizes one row
/// at a time into a 16 KiB buffer and streams the quantized bytes once.
pub struct DitWeights {
    gguf: GgufFile,
    inventory: TensorInventory,
}

impl DitWeights {
    pub fn open(path: &Path) -> Result<Self, DitError> {
        let gguf = GgufFile::open(path).map_err(|e| DitError::Gguf(e.token()))?;
        let inventory = TensorInventory::from_file(path, &gguf)
            .map_err(|e: TensorError| DitError::Tensors(e.token()))?;
        Ok(Self { gguf, inventory })
    }

    fn info(&self, name: &str) -> Result<&TensorInfo, DitError> {
        self.inventory.find(name).ok_or(DitError::Contract {
            tensor: name.to_string(),
            why: "missing from the artifact".into(),
        })
    }

    fn bytes(&self, t: &TensorInfo) -> Result<&[u8], DitError> {
        let start = t.abs_offset as usize;
        let end = start + t.bytes as usize;
        self.gguf
            .as_bytes()
            .get(start..end)
            .ok_or_else(|| DitError::Contract {
                tensor: t.name.clone(),
                why: "outside the file".into(),
            })
    }

    fn rows<'a>(&'a self, name: &str, k: usize, m: usize) -> Result<Rows<'a>, DitError> {
        let t = self.info(name)?;
        let (tk, tm) = (t.dim[0], t.dim[1]);

        if t.ndim != 2 || tk != k as u64 || tm != m as u64 {
            return contract(
                name,
                &format!(
                    "expected a [{k}, {m}] matrix, the artifact has ne ({tk}, {tm}) rank {}",
                    t.ndim
                ),
            );
        }

        Ok(match t.typ {
            TYPE_Q6_K if k % QK_K == 0 => Rows::Q6K(self.bytes(t)?),
            TYPE_BF16 => Rows::Bf16(self.bytes(t)?),
            other => return contract(name, &format!("unsupported type {other}")),
        })
    }

    /// `y = W x` for one pinned tensor: `W` is `[k, m]` in GGML order, `x` is
    /// `[k, n]` feature-fastest and `y` is `[m, n]`. `Linear::forward` is a bare
    /// `ggml_mul_mat(weight, x)`: this model has no bias anywhere
    /// (`ggml_block.hpp:141`, `qwen_image_2_1.hpp:254-258`).
    pub fn linear(
        &self,
        name: &str,
        k: usize,
        m: usize,
        x: &[f32],
        n: usize,
        y: &mut [f32],
    ) -> Result<(), DitError> {
        if x.len() != k * n {
            return contract(name, &format!("input {} != {k}x{n}", x.len()));
        }
        if y.len() != m * n {
            return contract(name, &format!("output {} != {m}x{n}", y.len()));
        }

        let rows = self.rows(name, k, m)?;
        matmul_rows(&rows, k, m, x, n, y);
        Ok(())
    }

    /// The full F32 contents of one pinned tensor in GGML order.
    ///
    /// The oracle's verification hook: an independent dequantizer (the `gguf`
    /// Python package, whose Q6_K comes from llama.cpp rather than from this
    /// port) is compared against it once, so a row-order or scale-index error in
    /// [`dequant_q6k_block`] cannot hide behind a plausible-looking forward.
    pub fn dequantize_f32(&self, name: &str) -> Result<Vec<f32>, DitError> {
        let info = self.info(name)?;

        let (rows, k) = match info.ndim {
            1 => (1usize, info.dim[0] as usize),
            2 => (info.dim[1] as usize, info.dim[0] as usize),
            _ => return contract(name, "expected a 1-D or 2-D tensor"),
        };

        let src = match info.typ {
            TYPE_Q6_K if k % QK_K == 0 => Rows::Q6K(self.bytes(info)?),
            TYPE_BF16 => Rows::Bf16(self.bytes(info)?),
            other => return contract(name, &format!("unsupported type {other}")),
        };

        let mut out = vec![0f32; rows * k];
        for row in 0..rows {
            src.row(k, row, &mut out[row * k..]);
        }
        Ok(out)
    }

    /// A 1-D BF16 weight as F32 (the head norms and the text norm).
    fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, DitError> {
        let t = self.info(name)?;
        if t.ndim != 1 || t.dim[0] != len as u64 || t.typ != TYPE_BF16 {
            return contract(
                name,
                &format!("expected a BF16 [{len}] vector, got rank {} type {}", t.ndim, t.typ),
            );
        }

        let bytes = self.bytes(t)?;
        Ok(bytes.chunks_exact(2).map(|p| bf16_to_f32(u16::from_le_bytes([p[0], p[1]]))).collect())
    }
}

// ---------------------------------------------------------------------------
// The forward
// ---------------------------------------------------------------------------

/// One DiT evaluation: the latent being denoised, its text conditioning, the
/// flow timestep on the reference's [0, 1000] scale, and the RoPE pairing under
/// test.
pub struct DitPass<'a> {
    /// Latent `[width, height, 64]`, GGML order (`ne0` = width fastest).
    pub latent: &'a [f32],
    pub height: usize,
    pub width: usize,
    /// Text conditioning `[4096, text_length]`, `ne0` fastest.
    pub context: &'a [f32],
    pub text_length: usize,
    pub timestep: f32,
    pub pairing: RopePairing,
}

/// Activations in GGML order: feature fastest, then token. Every buffer in the
/// forward follows this, so the ggml ops port literally.
struct Act {
    data: Vec<f32>,
    dim: usize,
}

impl Act {
    fn zeros(dim: usize, tokens: usize) -> Self {
        Self { data: vec![0f32; dim * tokens], dim }
    }

    fn row(&mut self, token: usize) -> &mut [f32] {
        &mut self.data[token * self.dim..(token + 1) * self.dim]
    }
}

/// `QwenImage21TransformerBlock::modulate` (`qwen_image_2_1.hpp:207-220`): the
/// parameter carries two rows over the timesteps — row 0 (the real timestep)
/// modulates the image tokens, row 1 (the zero timestep) the text prefix. A
/// non-gated row is used as `row + 1`; a gated row is `tanh(row)` and the caller
/// adds the result to the residual.
fn modulate(x: &mut [f32], dim: usize, prefix: usize, param: &[f32], gate: bool) {
    let value = |row: usize, i: usize| {
        let v = param[i + dim * row];
        if gate {
            v.tanh()
        } else {
            v + 1.0
        }
    };

    for token in prefix..x.len() / dim {
        for i in 0..dim {
            x[i + token * dim] *= value(0, i);
        }
    }
    for token in 0..prefix {
        for i in 0..dim {
            x[i + token * dim] *= value(1, i);
        }
    }
}

/// One segment's attention: `q[start..end]` attends to `k[0..end]`, with the
/// causal mask on the text segment (`ggml_extend.cpp:616`, the manual path —
/// the fixture ran with `flash_attn: false`, `refdump/run1.log:71`).
///
/// `q` and `k` are `[head][token][head_dim]`; `v` and `out` are feature-fastest
/// `[d + head_dim*head + hidden*token]`. Queries own disjoint output runs, so
/// the split is over queries and the sum order never changes with the threads.
#[allow(clippy::too_many_arguments)]
fn segment_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    head_dim: usize,
    tokens: usize,
    mask: Option<&[f32]>,
    start: usize,
    end: usize,
    out: &mut [f32],
) {
    let hidden = heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    let per = (end - start).div_ceil(threads.max(1));

    thread::scope(|scope| {
        for (index, block) in out[start * hidden..end * hidden].chunks_mut(per * hidden).enumerate() {
            let first = start + index * per;

            scope.spawn(move || {
                let mut scores = vec![0f32; end];

                for (offset, row) in block.chunks_mut(hidden).enumerate() {
                    let query = first + offset;

                    for head in 0..heads {
                        let q_row = &q[(head * tokens + query) * head_dim..][..head_dim];

                        for (key, score) in scores.iter_mut().enumerate() {
                            let k_row = &k[(head * tokens + key) * head_dim..][..head_dim];
                            *score = dot8(q_row, k_row) * scale;
                        }
                        if let Some(mask) = mask {
                            for key in 0..end {
                                scores[key] += mask[query * end + key];
                            }
                        }
                        softmax_row(&mut scores);

                        for d in 0..head_dim {
                            let mut acc = [0f32; 8];
                            let mut key = 0;
                            while key + 8 <= end {
                                for lane in 0..8 {
                                    acc[lane] += scores[key + lane]
                                        * v[d + head_dim * (head + heads * (key + lane))];
                                }
                                key += 8;
                            }
                            let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                                + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
                            for key in key..end {
                                sum += scores[key] * v[d + head_dim * (head + heads * key)];
                            }
                            row[head * head_dim + d] = sum;
                        }
                    }
                }
            });
        }
    });
}

/// The attention block (`qwen_image_2_1.hpp:153-190`): `to_q`/`to_k`/`to_v`,
/// per-head RMSNorm on q and k, RoPE on q and k only, then one attention per
/// segment concatenated in segment order through `to_out.0`.
fn attention(
    weights: &DitWeights,
    prefix: &str,
    x: &[f32],
    table: &[f32],
    layout: &Layout,
    pairing: RopePairing,
    tokens: usize,
) -> Result<Act, DitError> {
    let hidden = DIT_HIDDEN as usize;
    let heads = DIT_HEADS as usize;
    let head_dim = DIT_HEAD_DIM as usize;

    let mut q = Act::zeros(hidden, tokens);
    let mut k = Act::zeros(hidden, tokens);
    let mut v = Act::zeros(hidden, tokens);
    weights.linear(&format!("{prefix}attn.to_q.weight"), hidden, hidden, x, tokens, &mut q.data)?;
    weights.linear(&format!("{prefix}attn.to_k.weight"), hidden, hidden, x, tokens, &mut k.data)?;
    weights.linear(&format!("{prefix}attn.to_v.weight"), hidden, hidden, x, tokens, &mut v.data)?;

    // q and k move to [head][token][head_dim]: the layout the head norm wants
    // and the layout `Rope::apply_rope` consumes.
    let mut qh = vec![0f32; hidden * tokens];
    let mut kh = vec![0f32; hidden * tokens];
    for head in 0..heads {
        for token in 0..tokens {
            let src = token * hidden + head * head_dim;
            let dst = (head * tokens + token) * head_dim;
            qh[dst..dst + head_dim].copy_from_slice(&q.data[src..src + head_dim]);
            kh[dst..dst + head_dim].copy_from_slice(&k.data[src..src + head_dim]);
        }
    }

    let norm_q = weights.vector(&format!("{prefix}attn.norm_q.weight"), head_dim)?;
    let norm_k = weights.vector(&format!("{prefix}attn.norm_k.weight"), head_dim)?;
    for head in 0..heads {
        for token in 0..tokens {
            let at = (head * tokens + token) * head_dim;
            rms_norm_row(&mut qh[at..at + head_dim], &norm_q);
            rms_norm_row(&mut kh[at..at + head_dim], &norm_k);
        }
    }

    apply_rope(&mut qh, table, tokens, heads, head_dim, pairing);
    apply_rope(&mut kh, table, tokens, heads, head_dim, pairing);

    let mut out = Act::zeros(hidden, tokens);
    for segment in &layout.segments {
        let (start, end) = (segment.start as usize, segment.end as usize);
        let mask = if segment.image_index < 0 {
            // Only the text segment has a mask, and it is the first segment.
            debug_assert_eq!(start, 0, "the masked segment is the text prefix");
            Some(text_mask(end))
        } else {
            None
        };
        segment_attention(&qh, &kh, &v.data, heads, head_dim, tokens, mask.as_deref(), start, end, &mut out.data);
    }

    let mut projected = Act::zeros(hidden, tokens);
    weights.linear(&format!("{prefix}attn.to_out.0.weight"), hidden, hidden, &out.data, tokens, &mut projected.data)?;
    Ok(projected)
}

/// One transformer block (`qwen_image_2_1.hpp:191-243`).
#[allow(clippy::too_many_arguments)]
fn block(
    weights: &DitWeights,
    prefix: &str,
    x: &mut [f32],
    mods: [&[f32]; 4],
    table: &[f32],
    layout: &Layout,
    pairing: RopePairing,
    tokens: usize,
) -> Result<(), DitError> {
    let hidden = DIT_HIDDEN as usize;
    let intermediate = DIT_INTERMEDIATE as usize;
    let text = layout.prefix_length as usize;

    // h = img_norm1(x) -> modulate -> attention -> gated residual
    let mut h = Act { data: x.to_vec(), dim: hidden };
    for token in 0..tokens {
        layer_norm_row(h.row(token));
    }
    modulate(&mut h.data, hidden, text, mods[0], false);

    let mut attn = attention(weights, prefix, &h.data, table, layout, pairing, tokens)?;
    modulate(&mut attn.data, hidden, text, mods[1], true);
    for (dst, add) in x.iter_mut().zip(&attn.data) {
        *dst += add;
    }

    // h = img_norm2(x) -> modulate -> SwiGLU MLP -> gated residual
    let mut h = Act { data: x.to_vec(), dim: hidden };
    for token in 0..tokens {
        layer_norm_row(h.row(token));
    }
    modulate(&mut h.data, hidden, text, mods[2], false);

    let mut gate = Act::zeros(intermediate, tokens);
    let mut up = Act::zeros(intermediate, tokens);
    weights.linear(&format!("{prefix}img_mlp.gate_layer.weight"), hidden, intermediate, &h.data, tokens, &mut gate.data)?;
    weights.linear(&format!("{prefix}img_mlp.proj.weight"), hidden, intermediate, &h.data, tokens, &mut up.data)?;
    for (value, gate) in up.data.iter_mut().zip(&gate.data) {
        *value *= silu(*gate);
    }

    let mut mlp = Act::zeros(hidden, tokens);
    weights.linear(&format!("{prefix}img_mlp.out.weight"), intermediate, hidden, &up.data, tokens, &mut mlp.data)?;
    modulate(&mut mlp.data, hidden, text, mods[3], true);
    for (dst, add) in x.iter_mut().zip(&mlp.data) {
        *dst += add;
    }
    Ok(())
}

/// One complete DiT evaluation. The return is the velocity in the latent's own
/// layout (`[width, height, 64]`, `ne0` = width fastest).
pub fn forward(weights: &DitWeights, pass: &DitPass) -> Result<Vec<f32>, DitError> {
    let hidden = DIT_HIDDEN as usize;
    let in_channels = DIT_IN_CHANNELS as usize;
    let out_channels = DIT_OUT_CHANNELS as usize;
    let modulation = DIT_MODULATION as usize;
    let image = pass.height * pass.width;
    let prefix = pass.text_length;

    if pass.latent.len() != image * in_channels {
        return contract(
            "latent",
            &format!(
                "{} values for a {}x{} grid of {in_channels} channels",
                pass.latent.len(),
                pass.width,
                pass.height
            ),
        );
    }
    if pass.context.len() != prefix * hidden {
        return contract("context", &format!("{} values for {prefix} tokens", pass.context.len()));
    }

    let layout = build_layout(prefix as i64, &[], &[(pass.height as i64, pass.width as i64)])?;
    if layout.segments.len() != 2 || layout.segments[0].image_index >= 0 {
        // Stage 2 covers the text-to-image graph only; reference latents are
        // the img2img feature and arrive with their own gate (plan stage 4).
        return Err(DitError::Unsupported("reference latents in the joint sequence"));
    }
    let tokens = layout.positions.len();
    let table = rope_table(&layout.positions, &DIT_AXES_DIM, DIT_ROPE_THETA);

    // time = concat(t, 0) under the sinusoidal embedding: column 0 is the real
    // timestep, column 1 the zero one that modulates the text prefix.
    let embedded = timestep_embedding(&[pass.timestep, 0.0], 10000.0);
    let mut time = vec![0f32; hidden * 2];
    weights.linear(
        "time_text_embed.timestep_embedder.linear_1.weight",
        DIT_TIME_EMBED_DIM as usize,
        hidden,
        &embedded,
        2,
        &mut time,
    )?;
    time.iter_mut().for_each(|v| *v = silu(*v));
    let mut time2 = vec![0f32; hidden * 2];
    weights.linear("time_text_embed.timestep_embedder.linear_2.weight", hidden, hidden, &time, 2, &mut time2)?;
    time2.iter_mut().for_each(|v| *v = silu(*v));

    let mut params = vec![0f32; modulation * 2];
    weights.linear("modulation.1.weight", hidden, modulation, &time2, 2, &mut params)?;
    // chunk(params, 4, dim 0) splits the 16384-wide feature axis, so the four
    // chunks are strided views over both timestep columns, not contiguous
    // quarters. Repack each as its own [hidden, 2] block, which is what
    // `modulate` walks.
    let mut chunks = vec![vec![0f32; hidden * 2]; 4];
    for (c, chunk) in chunks.iter_mut().enumerate() {
        for col in 0..2 {
            for f in 0..hidden {
                chunk[f + hidden * col] = params[c * hidden + f + modulation * col];
            }
        }
    }
    let mods: [&[f32]; 4] = [&chunks[0], &chunks[1], &chunks[2], &chunks[3]];

    // text: the zero-centered norm turns the artifact's weight into `weight + 1`.
    let text_norm = weights.vector("txt_in.text_norm.weight", DIT_CONTEXT_DIM as usize)?;
    let text_norm: Vec<f32> = text_norm.iter().map(|w| w + 1.0).collect();
    let mut text = Act { data: pass.context.to_vec(), dim: hidden };
    for token in 0..prefix {
        rms_norm_row(text.row(token), &text_norm);
    }
    let mut projected = Act::zeros(hidden, prefix);
    weights.linear("txt_in.in_layer.weight", hidden, hidden, &text.data, prefix, &mut projected.data)?;
    projected.data.iter_mut().for_each(|v| *v = gelu(*v));
    let mut text = Act::zeros(hidden, prefix);
    weights.linear("txt_in.out_layer.weight", hidden, hidden, &projected.data, prefix, &mut text.data)?;

    // image: `DiT::patchify(image, 1, 1)` is the reshape that makes the channel
    // fastest and the token `h*width + w` — the layout's own token order. The
    // latent is channel-slowest (`ne0` is the width), so the copy gathers one
    // channel at a time.
    let mut patches = vec![0f32; in_channels * image];
    for c in 0..in_channels {
        for index in 0..image {
            patches[c + in_channels * index] = pass.latent[index + image * c];
        }
    }
    let mut image_act = Act::zeros(hidden, image);
    weights.linear("img_in.weight", in_channels, hidden, &patches, image, &mut image_act.data)?;

    // The joint sequence: the text prefix, then the image grid.
    let mut joint = Act::zeros(hidden, tokens);
    joint.data[..hidden * prefix].copy_from_slice(&text.data);
    joint.data[hidden * prefix..].copy_from_slice(&image_act.data);


    for il in 0..DIT_LAYERS {
        block(weights, &format!("transformer_blocks.{il}."), &mut joint.data, mods, &table, &layout, pass.pairing, tokens)?;
    }

    // norm_out: only the image tokens survive, scaled by the real-timestep row
    // of the embedding as `scale + 1`, then projected back to the channels.
    let mut out_tokens = Act { data: joint.data[hidden * prefix..].to_vec(), dim: hidden };
    for token in 0..image {
        layer_norm_row(out_tokens.row(token));
    }
    let mut scale = vec![0f32; hidden];
    weights.linear("norm_out.linear.weight", hidden, hidden, &time2[..hidden], 1, &mut scale)?;
    for token in 0..image {
        for (value, s) in out_tokens.row(token).iter_mut().zip(&scale) {
            *value *= s + 1.0;
        }
    }
    let mut projected = Act::zeros(out_channels, image);
    weights.linear("proj_out.weight", hidden, out_channels, &out_tokens.data, image, &mut projected.data)?;

    // unpatchify: back to the layout the step's input was dumped in, which is
    // channel-slowest.
    let mut velocity = vec![0f32; image * out_channels];
    for index in 0..image {
        for c in 0..out_channels {
            velocity[index + image * c] = projected.data[c + out_channels * index];
        }
    }
    Ok(velocity)
}

/// The reference's classifier-free guidance combination (`guidance.cpp:171`):
/// `uncond + scale * (cond - uncond)`, elementwise in that order.
pub fn cfg_combine(cond: &[f32], uncond: &[f32], scale: f32) -> Vec<f32> {
    uncond.iter().zip(cond).map(|(u, c)| u + scale * (c - u)).collect()
}

/// What a predicted velocity measures against the reference's own dump.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parity {
    /// Pearson correlation over the whole tensor.
    pub correlation: f32,
    /// `||ours - reference|| / ||reference||`.
    pub relative_rms: f32,
    /// Largest single-element absolute difference.
    pub max_abs: f32,
}

/// Compares a predicted velocity with a dumped one.
pub fn parity(ours: &[f32], reference: &[f32]) -> Parity {
    let n = ours.len() as f64;
    let (mut sum_a, mut sum_b, mut sum_aa, mut sum_bb, mut sum_ab) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let (mut diff_sq, mut ref_sq, mut max_abs) = (0f64, 0f64, 0f32);

    for (a, b) in ours.iter().zip(reference) {
        let (a, b) = (*a as f64, *b as f64);
        sum_a += a;
        sum_b += b;
        sum_aa += a * a;
        sum_bb += b * b;
        sum_ab += a * b;
        let d = a - b;
        diff_sq += d * d;
        ref_sq += b * b;
        max_abs = max_abs.max(d.abs() as f32);
    }

    let cov = sum_ab - sum_a * sum_b / n;
    let var_a = sum_aa - sum_a * sum_a / n;
    let var_b = sum_bb - sum_b * sum_b / n;

    Parity {
        correlation: (cov / (var_a.sqrt() * var_b.sqrt())) as f32,
        relative_rms: (diff_sq / ref_sq).sqrt() as f32,
        max_abs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built Q6_K block: `d = 1`, `scales = [1..8]` per half, and every
    /// `ql`/`qh` byte chosen so the four groups read 0 after the -32 offset.
    /// The four groups take `scales[is]`, `is+2`, `is+4`, `is+6` (`ds4.c:4538`).
    #[test]
    fn q6k_dequant_follows_the_reference_group_order() {
        let mut block = [0u8; Q6K_BLOCK_BYTES];
        block[209] = 0x3c; // d = 1.0
        for (i, byte) in block[192..208].iter_mut().enumerate() {
            *byte = (i as i8 + 1) as u8;
        }

        let mut out = [0f32; QK_K];
        block[200] = 1; // the second half's scales[0]
        dequant_q6k_block(&block, &mut out);
        assert_eq!(out[0], -32.0); // l < 16, group 1, scales[0]
        assert_eq!(out[16], -64.0); // l >= 16, group 1, scales[1]
        assert_eq!(out[32], -96.0); // group 2, scales[2]
        assert_eq!(out[48], -128.0); // group 2, scales[3]
        assert_eq!(out[64], -160.0); // group 3, scales[4]
        assert_eq!(out[96], -224.0); // group 4, scales[6]
        assert_eq!(out[128], -32.0); // the second 128-half restarts at its own scales[0]

        // The second half advances ql/qh/scales by 64/32/8 bytes.
        block[200] = 2;
        dequant_q6k_block(&block, &mut out);
        assert_eq!(out[128], -64.0);
        assert_eq!(out[0], -32.0);
    }

    /// `ql` carries the low four bits of a value and `qh` the top two; the -32
    /// offset makes the largest six-bit code 31 and the smallest -32.
    #[test]
    fn q6k_dequant_places_ql_and_qh_bits() {
        let mut block = [0u8; Q6K_BLOCK_BYTES];
        block[209] = 0x3c; // d = 1.0
        block[192] = 1; // scales[0]
        block[0] = 0xff; // ql[0]: both nibbles full
        block[128] = 0x33; // qh[0]: both bit pairs full
        block[192 + 4] = 1; // scales[4], the group the high nibble feeds

        let mut out = [0f32; QK_K];
        dequant_q6k_block(&block, &mut out);
        assert_eq!(out[0], 31.0);
        assert_eq!(out[64], 31.0);

        // Without the high bits the same nibble reads 15 - 32.
        block[128] = 0;
        dequant_q6k_block(&block, &mut out);
        assert_eq!(out[0], -17.0);
    }

    #[test]
    fn half_and_bfloat16_decode_known_patterns() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0x0001), 5.9604645e-8);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
    }

    /// The matmul is `y[o, t] = sum_i W[o, i] * x[i, t]` with rows streamed, so
    /// the result must not depend on how many rows a thread takes.
    #[test]
    fn matmul_matches_a_scalar_reference_regardless_of_row_split() {
        let (k, m, n) = (8usize, 64usize, 3usize);
        // Quarter steps are exact in BF16, so the encoding loses nothing.
        let w: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.25 - 1.5).collect();
        let x: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
        let w_bytes: Vec<u8> =
            w.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();

        let mut out = vec![0f32; m * n];
        matmul_rows(&Rows::Bf16(&w_bytes), k, m, &x, n, &mut out);

        // The result is feature-fastest, like every activation buffer.
        for o in 0..m {
            for t in 0..n {
                let expect: f32 = (0..k).map(|i| w[o * k + i] * x[i + k * t]).sum();
                assert!((out[t * m + o] - expect).abs() < 1e-4, "row {o} token {t}");
            }
        }
    }

    /// `dot8` splits the reduction into eight lanes; the tail is added in order.
    #[test]
    fn dot8_covers_a_length_that_is_not_a_multiple_of_eight() {
        let w: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let x: Vec<f32> = (0..11).map(|i| 1.0 / (i + 1) as f32).collect();

        let expect: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((dot8(&w, &x) - expect).abs() < 1e-4);
    }

    /// The zero-centered text norm is the artifact weight plus one, and the
    /// gated modulation rows are `tanh(row)` while the others are `row + 1`.
    #[test]
    fn modulation_applies_row_one_to_the_prefix_and_row_zero_to_the_image() {
        let dim = 2;
        let param = [2.0f32, 3.0, 0.5, -0.5]; // [dim, 2]
        let mut x = vec![1.0f32; dim * 3];

        modulate(&mut x, dim, 1, &param, false);
        // Tokens 1 and 2 (the image) take row 0 as `row + 1`.
        assert_eq!(&x[2..4], &[3.0, 4.0]);
        // Token 0 (the prefix) takes row 1 the same way.
        assert_eq!(&x[0..2], &[1.5, 0.5]);

        let mut x = vec![1.0f32; dim * 2];
        modulate(&mut x, dim, 0, &param, true);
        assert_eq!(&x[0..2], &[2.0f32.tanh(), 3.0f32.tanh()]);
    }

    /// The softmax ignores masked (`-inf`) entries and sums to one.
    #[test]
    fn softmax_row_handles_a_negative_infinity_mask() {
        let mut row = [1.0f32, f32::NEG_INFINITY, 2.0];
        softmax_row(&mut row);

        assert_eq!(row[1], 0.0);
        assert!((row[0] + row[1] + row[2] - 1.0).abs() < 1e-6);
        assert!(row[2] > row[0]);
    }

    /// The CFG combination is the reference's order: `uncond + s*(cond - uncond)`.
    #[test]
    fn cfg_combination_uses_the_reference_operation_order() {
        let cond = [1.0f32, -2.0];
        let uncond = [0.5f32, -0.5];

        assert_eq!(cfg_combine(&cond, &uncond, 6.0), [3.5, -9.5]);
        assert_eq!(cfg_combine(&cond, &cond, 6.0), cond);
    }

    /// A perfect match correlates exactly and has no residual.
    #[test]
    fn parity_is_exact_for_identical_tensors() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let p = parity(&values, &values);

        assert!((p.correlation - 1.0).abs() < 1e-6);
        assert_eq!(p.relative_rms, 0.0);
        assert_eq!(p.max_abs, 0.0);
    }
}
