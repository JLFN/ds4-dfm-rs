//! The P1 CPU oracle's stage 3: the decode-only VAE, one frame, F32.
//!
//! Stage 1 settled the numerics and stage 2 the DiT forward; this module closes
//! P1 with the decoder that turns the final latent into pixels, so the
//! reference's own PNG becomes reproducible end to end. Slow, F32, no FFI and
//! no GPU by design; nothing here is a serving path.
//!
//! Ported 1:1 from `stable-diffusion.cpp` at `6dcb5bb`: `wan_vae.hpp` for the
//! decoder, `vae.hpp` for the output scaling and the latent statistics,
//! `ggml_extend.cpp` for the conv/attention glue, and
//! `image.cpp`/`util.cpp`/`preprocessing.hpp` for the pixel conversion. Every
//! item names the lines it came from.
//!
//! Three properties of the reference this artifact forces, verified there
//! rather than assumed:
//!
//! 1. Singleton temporal kernel. Every Conv3d weight in this export has
//!    `ne[2] == 1` (kT == 1), and the reference collapses `kernel_size[0]` and
//!    `padding[0]` for precisely that case (`wan_vae.hpp:30-34`). So each conv
//!    here is a spatial 3x3 (or 1x1) conv: no temporal kernel, no temporal
//!    padding.
//! 2. Single-frame decode. The fixture latent is one frame, so `WanVAE::decode`'s
//!    frame loop (`wan_vae.hpp:1231-1247`) runs once with `chunk_idx == 0`, and
//!    `Resample::forward`'s temporal branch is literally "pass" at chunk 0
//!    (`wan_vae.hpp:203-206`): `time_conv` is never applied and the frame count
//!    stays 1. Every `feat_cache` read on that single chunk is a nullptr
//!    (`wan_vae.hpp:198-199`), so the streaming cache path does not exist here;
//!    any other temporal length is refused by name.
//! 3. F16 conv operands. `ggml_conv_2d`/`ggml_conv_3d` write the im2col patches
//!    in the weight's type (`ggml.c:4758`, `:4840`), this artifact's conv
//!    weights are F16 (`ggml_block.hpp:444-445`, `wan_vae.hpp:36-40`), and all
//!    of its convs run through those two functions, so every conv operand is
//!    rounded to binary16 first (see [`f16_round`]). That is not cosmetic here:
//!    this latent pushes the last two up levels past F16's 65504, the reference
//!    saturates those pixels to infinity and then to NaN, and only the same
//!    contract reproduces its image.
//!
//! The decode is therefore a per-frame 2D pipeline over `(C, H, W)` planes:
//!
//! ```text
//!   z [64, 16, 16]
//!     conv2            1x1, 64 -> 64            (`wan_vae.hpp:1100-1105`, `:1220-1228`)
//!     decoder.conv1    3x3, 64 -> 1152
//!     middle.0         ResidualBlock 1152 -> 1152
//!     middle.1         AttentionBlock 1152      (one head over the h*w tokens)
//!     middle.2         ResidualBlock 1152 -> 1152
//!     upsamples.0..4   3 residual blocks each; levels 0..3 upsample 2x
//!                      spatially and add a DupUp3D of the block input
//!     head             RMS_norm, SiLU, 3x3 conv 144 -> 4
//! ```
//!
//! Activations are stored in the reference's own GGML order `ne = [W, H, C, 1]`
//! (`wan_vae.hpp:62-63`, "x: [N*IC, ID, IH, IW]"), so
//! `index(w, h, c) = w + W * (h + H * c)`: a pixel row is contiguous and the
//! channel axis is the slowest one.

use std::path::Path;
use std::thread;

use super::{
    vae_decoder_dims, VAE_LATENT_MEAN, VAE_LATENT_STD, VAE_NORM_EPS, VAE_OUT_CHANNELS,
    VAE_SCALE_FACTOR, VAE_TEMPORAL_UPSAMPLE, VAE_Z_DIM, TYPE_BF16,
};
use crate::gguf::GgufFile;
use crate::tensors::{TensorError, TensorInfo, TensorInventory};

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// One frame of activations: GGML's `ne = [W, H, C, 1]` with `ne0` fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    data: Vec<f32>,
    width: usize,
    height: usize,
    channels: usize,
}

impl Plane {
    pub fn new(width: usize, height: usize, channels: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            width * height * channels,
            "plane {width}x{height}x{channels} carries {} values",
            data.len()
        );
        Self { data, width, height, channels }
    }

    pub fn zeros(width: usize, height: usize, channels: usize) -> Self {
        Self { data: vec![0f32; width * height * channels], width, height, channels }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The raw values in GGML order (`ne0` fastest).
    pub fn values(&self) -> &[f32] {
        &self.data
    }

    fn index(&self, w: usize, h: usize, c: usize) -> usize {
        w + self.width * (h + self.height * c)
    }

    fn get(&self, w: usize, h: usize, c: usize) -> f32 {
        self.data[self.index(w, h, c)]
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum VaeError {
    Io(std::io::Error),
    Gguf(String),
    Tensors(String),
    /// The artifact does not carry a pinned tensor, or carries it with a shape
    /// or type the contract does not allow.
    Contract { tensor: String, why: String },
    /// Something the single-frame F32 oracle deliberately does not implement.
    Unsupported(&'static str),
    Image(String),
}

impl VaeError {
    pub fn token(&self) -> String {
        match self {
            VaeError::Io(e) => format!("io {e}"),
            VaeError::Gguf(m) => format!("gguf {m}"),
            VaeError::Tensors(m) => format!("tensors {m}"),
            VaeError::Contract { tensor, why } => format!("contract {tensor}: {why}"),
            VaeError::Unsupported(what) => format!("unsupported {what}"),
            VaeError::Image(m) => format!("image {m}"),
        }
    }
}

impl std::fmt::Display for VaeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "qwen-image-2.1 vae: {}", self.token())
    }
}

impl std::error::Error for VaeError {}

impl From<std::io::Error> for VaeError {
    fn from(e: std::io::Error) -> Self {
        VaeError::Io(e)
    }
}

fn contract<T>(tensor: &str, why: &str) -> Result<T, VaeError> {
    Err(VaeError::Contract { tensor: tensor.to_string(), why: why.to_string() })
}

// ---------------------------------------------------------------------------
// Latent statistics (`wan_vae.hpp:1332-1405`)
// ---------------------------------------------------------------------------

/// `diffusion -> vae`: `latents * std / scale_factor + mean`
/// (`WanVAERunner::diffusion_to_vae_latents`, `wan_vae.hpp:1393-1396`), with
/// the 64-channel statistics of `wan_vae.hpp:1368-1375`.
pub fn diffusion_to_vae(latent: &Plane) -> Result<Plane, VaeError> {
    if latent.channels != VAE_LATENT_MEAN.len() {
        return contract(
            "latent",
            &format!(
                "{} channels: the pinned statistics cover {}",
                latent.channels,
                VAE_LATENT_MEAN.len()
            ),
        );
    }
    let mut out = latent.clone();
    convert_channels(&mut out, |c, v| {
        (v * VAE_LATENT_STD[c]) / VAE_SCALE_FACTOR + VAE_LATENT_MEAN[c]
    });
    Ok(out)
}

/// `vae -> diffusion`: `(latents - mean) * scale_factor / std`
/// (`wan_vae.hpp:1398-1401`). The exact inverse of [`diffusion_to_vae`] up to
/// F32 rounding.
pub fn vae_to_diffusion(latent: &Plane) -> Result<Plane, VaeError> {
    if latent.channels != VAE_LATENT_MEAN.len() {
        return contract(
            "latent",
            &format!(
                "{} channels: the pinned statistics cover {}",
                latent.channels,
                VAE_LATENT_MEAN.len()
            ),
        );
    }
    let mut out = latent.clone();
    convert_channels(&mut out, |c, v| {
        ((v - VAE_LATENT_MEAN[c]) * VAE_SCALE_FACTOR) / VAE_LATENT_STD[c]
    });
    Ok(out)
}

/// Applies one per-channel map over a whole plane.
fn convert_channels(plane: &mut Plane, map: impl Fn(usize, f32) -> f32) {
    let spatial = plane.width * plane.height;
    for c in 0..plane.channels {
        for s in 0..spatial {
            let at = s + spatial * c;
            plane.data[at] = map(c, plane.data[at]);
        }
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Spatial taps of a 3x3 kernel.
const TAP_SIDE: usize = 3;
const TAPS: usize = TAP_SIDE * TAP_SIDE;
/// Resample/DupUp3D spatial factor, always 2 on the decode path
/// (`wan_vae.hpp:543`, `:891`).
const DUP_FACTOR_S: usize = 2;
/// `num_res_blocks + 1` residual blocks per up level (`wan_vae.hpp:889`).
const UP_RESIDUALS: usize = 3;
/// Decoder levels: `dims.len() - 1` for `dims = vae_decoder_dims()`.
const DECODER_LEVELS: usize = 5;

/// The decode-only VAE artifact, mmap-backed. Every weight is BF16 in this
/// export, so a tensor is dequantized to F32 on demand (one layer at a time,
/// the decoded activations are larger than any single one of them).
pub struct VaeWeights {
    gguf: GgufFile,
    inventory: TensorInventory,
}

impl VaeWeights {
    pub fn open(path: &Path) -> Result<Self, VaeError> {
        let gguf = GgufFile::open(path).map_err(|e| VaeError::Gguf(e.token()))?;
        let inventory = TensorInventory::from_file(path, &gguf)
            .map_err(|e: TensorError| VaeError::Tensors(e.token()))?;
        Ok(Self { gguf, inventory })
    }

    fn info(&self, name: &str) -> Result<&TensorInfo, VaeError> {
        self.inventory.find(name).ok_or(VaeError::Contract {
            tensor: name.to_string(),
            why: "missing from the artifact".into(),
        })
    }

    /// One tensor as F32 in GGML order (`ne0` fastest). Everything the decoder
    /// consumes comes through here or the typed readers below.
    fn dequantize_f32(&self, name: &str) -> Result<Vec<f32>, VaeError> {
        let t = self.info(name)?;
        let start = t.abs_offset as usize;
        let end = start + t.bytes as usize;
        let bytes = self.gguf.as_bytes().get(start..end).ok_or_else(|| VaeError::Contract {
            tensor: name.to_string(),
            why: "outside the file".into(),
        })?;

        if t.typ != TYPE_BF16 {
            return contract(name, &format!("type {} is not BF16", t.typ));
        }
        Ok(bytes.chunks_exact(2).map(|p| bf16_to_f32(u16::from_le_bytes([p[0], p[1]]))).collect())
    }

    /// A rank-1 weight: the RMS_norm gammas and the conv biases.
    fn vector(&self, name: &str, len: usize) -> Result<Vec<f32>, VaeError> {
        let t = self.info(name)?;
        if t.ndim != 1 || t.dim[0] != len as u64 {
            return contract(
                name,
                &format!("expected a rank-1 [{len}] tensor, got rank {} dims {:?}", t.ndim, &t.dim[..t.ndim as usize]),
            );
        }
        self.dequantize_f32(name)
    }

    fn gamma(&self, name: &str, channels: usize) -> Result<Vec<f32>, VaeError> {
        self.vector(name, channels)
    }

    fn bias(&self, name: &str, channels: usize) -> Result<Vec<f32>, VaeError> {
        self.vector(name, channels)
    }

    /// A 1x1 conv weight `[1, 1, IC, OC]` (a 4-D tensor whose slowest axis is
    /// `IC*OC`), repacked to `[IC, OC]` so `w[ic * OC + oc]` is one reduction
    /// row. The artifact's flat order is `ic + IC*oc` (`ggml_extend.cpp:452-475`
    /// derives `OC = ne[3] / IC`), which is not the same index.
    fn matrix(&self, name: &str, ci: usize, co: usize) -> Result<Vec<f32>, VaeError> {
        let t = self.info(name)?;
        let elements = (t.dim[..t.ndim as usize].iter().product::<u64>()) as usize;
        if t.ndim != 4 || t.dim[0] != 1 || t.dim[1] != 1 || elements != ci * co {
            return contract(
                name,
                &format!("expected a [1, 1, {ci}, {co}] matrix, got dims {:?}", &t.dim[..t.ndim as usize]),
            );
        }
        let flat = self.dequantize_f32(name)?;
        let mut out = vec![0f32; flat.len()];
        for ic in 0..ci {
            for oc in 0..co {
                out[ic * co + oc] = flat[ic + ci * oc];
            }
        }
        Ok(out)
    }

    /// A 3x3 conv weight `[3, 3, IC, OC]`/`[3, 3, 1, IC*OC]`, repacked into nine
    /// contiguous `[IC, OC]` blocks: tap `t = kh*3 + kw` at `(t*IC + ic)*OC + oc`.
    ///
    /// The artifact layout is the GGML one (`wan_vae.hpp:36-40`,
    /// `qwen_image.rs::conv3d_dims`): flat index `kw + 3*kh + 9*(ic + IC*oc)`,
    /// i.e. exactly `t + 9*(ic + IC*oc)` for the singleton temporal kernel.
    fn taps(&self, name: &str, ci: usize, co: usize) -> Result<Vec<f32>, VaeError> {
        let info = self.info(name)?;
        let elements = (info.dim[..info.ndim as usize].iter().product::<u64>()) as usize;
        if info.ndim != 4 || info.dim[0] != TAP_SIDE as u64 || info.dim[1] != TAP_SIDE as u64 || elements != TAPS * ci * co {
            return contract(
                name,
                &format!("expected a [3, 3, ..., {ci}x{co}] kernel, got dims {:?}", &info.dim[..info.ndim as usize]),
            );
        }
        let flat = self.dequantize_f32(name)?;
        let mut out = vec![0f32; flat.len()];

        for tap in 0..TAPS {
            for ic in 0..ci {
                for oc in 0..co {
                    out[(tap * ci + ic) * co + oc] = flat[tap + TAPS * (ic + ci * oc)];
                }
            }
        }
        Ok(out)
    }
}

/// BF16 -> F32 is exact: the top half of the f32 bit pattern.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The reference's conv operand contract: every activation that enters a conv
/// product is first rounded to F16.
///
/// `ggml_conv_2d`/`ggml_conv_3d` write the im2col patches in the WEIGHT's type
/// (`ggml.c:4758`, `:4840`: `a->type == GGML_TYPE_BF16 ? F32 : F16`), and this
/// artifact's conv weights are loaded as F16 (`Conv2d::init_params`,
/// `ggml_block.hpp:444-445`), so the `mul_mat` that follows multiplies F16
/// activations. The consequence is not a small rounding: F16's largest finite
/// value is 65504, and this decoder's activations pass that in the last two up
/// levels, so the reference saturates them to infinity (and then to NaN) —
/// reproducing the reference's own image requires the same.
///
/// Round to nearest, ties to even; overflow to infinity; NaN stays NaN.
fn f16_round(value: f32) -> f32 {
    const SIGN: u32 = 0x8000_0000;
    const INF: u32 = 0x7f80_0000;
    const NAN: u32 = 0x7fc0_0000;
    const MANTISSA: u32 = 0x007f_ffff;

    let bits = value.to_bits();
    let sign = bits & SIGN;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & MANTISSA;

    if bits & 0x7fff_ffff >= INF {
        // Infinity or NaN (keep NaN quiet, as a cast does).
        return f32::from_bits(if mantissa == 0 { sign | INF } else { sign | NAN });
    }
    if exponent > 15 {
        return f32::from_bits(sign | INF);
    }
    if exponent >= -14 {
        // Normal in F16: 10 mantissa bits, ties to even.
        let keep = mantissa >> 13;
        let rest = mantissa & 0x1fff;
        let mut half = keep;
        if rest > 0x1000 || (rest == 0x1000 && keep & 1 == 1) {
            half += 1;
        }
        if half == 0x400 {
            // The rounding carried into the exponent.
            if exponent == 15 {
                return f32::from_bits(sign | INF);
            }
            return f32::from_bits(sign | (((exponent + 128) as u32) << 23));
        }
        return f32::from_bits(sign | (((exponent + 127) as u32) << 23) | (half << 13));
    }
    // Subnormal in F16 (or zero): the grid step is 2^-24. The magnitude is
    // what rounds; `as u32` saturates a negative float, so using the signed
    // value here would flush every negative subnormal to -0.0.
    if exponent < -25 {
        return f32::from_bits(sign);
    }
    let steps = (value.abs() * 16_777_216.0).round_ties_even() as u32;
    if steps == 0 {
        return f32::from_bits(sign);
    }
    // Re-encode `steps * 2^-24` as a normal F32.
    let leading = 31 - steps.leading_zeros();
    let biased = leading + 127 - 24;
    f32::from_bits(sign | (biased << 23) | ((steps - (1 << leading)) << (23 - leading)))
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// `ggml_vec_silu_f32`: `x / (1 + exp(-x))`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_in_place(x: &mut Plane) {
    for v in x.data.iter_mut() {
        *v = silu(*v);
    }
}

/// `RMS_norm::forward` (`wan_vae.hpp:110-122`): channel-wise RMS with eps
/// `1e-12`, then the gamma multiply. The reference permutes the channels
/// fastest for `ggml_rms_norm` and back; that is the same per-pixel reduction
/// over C. The sum runs in double here (dit.rs's convention), a ~1e-7 relative
/// difference inside the stated tolerance.
fn rms_norm(x: &Plane, gamma: &[f32]) -> Plane {
    debug_assert_eq!(gamma.len(), x.channels);
    let spatial = x.width * x.height;
    let mut out = x.clone();

    for s in 0..spatial {
        let sum: f64 = (0..x.channels).map(|c| (x.data[s + spatial * c] as f64).powi(2)).sum();
        let mean = (sum / x.channels as f64) as f32;
        let scale = 1.0f32 / (mean + VAE_NORM_EPS).sqrt();

        for c in 0..x.channels {
            let at = s + spatial * c;
            out.data[at] = x.data[at] * scale * gamma[c];
        }
    }
    out
}

/// `CausalConv3d` at kT == 1 (`wan_vae.hpp:18-92`): a spatial 3x3, stride 1,
/// zero pad 1 on both sides, then bias. The temporal kernel and its causal
/// padding collapse away (`wan_vae.hpp:30-34`), and the single frame's cache
/// read is nullptr, so `forward` reduces to pad + conv.
fn conv3x3(x: &Plane, w: &[f32], bias: &[f32], cout: usize) -> Plane {
    let (wi, hi, ci) = (x.width, x.height, x.channels);
    debug_assert_eq!(w.len(), TAPS * ci * cout);
    let mut out = Plane::zeros(wi, hi, cout);
    let threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(1).min(hi.max(1));
    let per = hi.div_ceil(threads);
    // Rows are staged pixel-major so a thread's rows stay contiguous, then
    // scattered into the plane's channel-slowest layout.
    let mut staged = vec![0f32; wi * hi * cout];

    thread::scope(|scope| {
        for (index, block) in staged.chunks_mut(per * wi * cout).enumerate() {
            let first = index * per;

            scope.spawn(move || {
                let mut acc = vec![0f32; wi * cout];

                for (r, row_out) in block.chunks_mut(wi * cout).enumerate() {
                    let h = first + r;
                    acc.iter_mut().for_each(|v| *v = 0.0);

                    // One tap at a time; inside a tap the output channels of one
                    // kernel row stay in L1 while the pixels stream past. The
                    // plane is channel-slowest, so a run of W pixels at a fixed
                    // (h, ic) is contiguous and a channel step is H*W.
                    for tap in 0..TAPS {
                        let (kh, kw) = (tap / TAP_SIDE, tap % TAP_SIDE);
                        let ih = h as isize + kh as isize - 1;
                        if ih < 0 || ih >= hi as isize {
                            continue;
                        }
                        let ih = ih as usize;
                        let wt = &w[tap * ci * cout..(tap + 1) * ci * cout];

                        for ic in 0..ci {
                            let wrow = &x.data[wi * (ih + hi * ic)..wi * (ih + hi * ic + 1)];
                            let b = &wt[ic * cout..(ic + 1) * cout];

                            for p in 0..wi {
                                let iw = p as isize + kw as isize - 1;
                                if iw < 0 || iw >= wi as isize {
                                    continue;
                                }
                                let a = f16_round(wrow[iw as usize]);
                                let dst = &mut acc[p * cout..(p + 1) * cout];
                                for oc in 0..cout {
                                    dst[oc] += a * b[oc];
                                }
                            }
                        }
                    }

                    for (p, row) in row_out.chunks_mut(cout).enumerate() {
                        for (oc, value) in row.iter_mut().enumerate() {
                            *value = acc[p * cout + oc] + bias[oc];
                        }
                    }
                }
            });
        }
    });

    for oc in 0..cout {
        for h in 0..hi {
            for p in 0..wi {
                out.data[p + wi * (h + hi * oc)] = staged[(h * wi + p) * cout + oc];
            }
        }
    }
    out
}

/// A 1x1 conv (`Conv2d` with a `{1,1}` kernel, `wan_vae.hpp:124-141`, and
/// `CausalConv3d` with `{1,1,1}`/no padding, `wan_vae.hpp:1100-1105`): one
/// unchanged position per pixel. The plane is repacked pixel-major so the
/// reduction over `IC` is contiguous, then the weight streams once per pixel
/// block.
fn conv1x1(x: &Plane, w: &[f32], bias: &[f32], cout: usize) -> Plane {
    let (wi, hi, ci) = (x.width, x.height, x.channels);
    debug_assert_eq!(w.len(), ci * cout);
    let spatial = wi * hi;
    let mut pixels = vec![0f32; spatial * ci];
    for s in 0..spatial {
        for c in 0..ci {
            pixels[s * ci + c] = x.data[s + spatial * c];
        }
    }

    let mut out = Plane::zeros(wi, hi, cout);
    const PIXEL_BLOCK: usize = 16;
    for block in 0..spatial.div_ceil(PIXEL_BLOCK) {
        let first = block * PIXEL_BLOCK;
        let count = PIXEL_BLOCK.min(spatial - first);
        let mut acc = vec![0f32; count * cout];

        for ic in 0..ci {
            let b = &w[ic * cout..(ic + 1) * cout];

            for p in 0..count {
                let a = f16_round(pixels[(first + p) * ci + ic]);
                let dst = &mut acc[p * cout..(p + 1) * cout];
                for oc in 0..cout {
                    dst[oc] += a * b[oc];
                }
            }
        }

        for p in 0..count {
            let s = first + p;
            for oc in 0..cout {
                out.data[s + spatial * oc] = acc[p * cout + oc] + bias[oc];
            }
        }
    }
    out
}

/// `ggml_upscale(..., 2, GGML_SCALE_MODE_NEAREST)` (`wan_vae.hpp:240`): each
/// pixel becomes a 2x2 block.
fn nearest_up2(x: &Plane) -> Plane {
    let mut out = Plane::zeros(x.width * 2, x.height * 2, x.channels);

    for h in 0..out.height {
        for w in 0..out.width {
            for c in 0..x.channels {
                let at = out.index(w, h, c);
                out.data[at] = x.get(w / 2, h / 2, c);
            }
        }
    }
    out
}

/// `DupUp3D::forward` at one frame with `first_chunk` (`wan_vae.hpp:322-370`).
///
/// The block concatenates `repeats = OC*factor_t*factor_s^2 / IC` copies of the
/// single frame along the temporal axis, then reshapes/permutes that axis back
/// into the channel and sub-pixel axes, and finally slices the last of the
/// `factor_t` frames. Every one of those ops is a pure permutation, and the
/// closed form of the whole chain is
///
/// ```text
///   m     = fs^2 * (factor_t - 1 + factor_t*oc) + fs*(y % fs) + (x % fs)
///   out[oc][y][x] = in[m / repeats][y / fs][x / fs]
/// ```
///
/// which the test module checks against a literal replay of the op chain. The
/// copy index inside `m` only selects which of the identical duplicates is
/// read, so the op is a gather: a nearest 2x spatial upsample whose channel and
/// parity grouping is set by the temporal factor (the reference calls this out
/// itself at `wan_vae.hpp:1084-1086`).
fn dup_up3d(x: &Plane, cout: usize, factor_t: usize) -> Result<Plane, VaeError> {
    let fs = DUP_FACTOR_S;
    let factor = factor_t * fs * fs;
    let cin = x.channels;

    if (cout * factor) % cin != 0 {
        return contract(
            "decoder.upsamples.avg_shortcut",
            &format!("DupUp3D asserts OC*factor % IC == 0, got {cout}*{factor} % {cin}"),
        );
    }
    let repeats = cout * factor / cin;
    let mut out = Plane::zeros(x.width * fs, x.height * fs, cout);

    for oc in 0..cout {
        for y in 0..out.height {
            for px in 0..out.width {
                let m = fs * fs * (factor_t - 1 + factor_t * oc) + fs * (y % fs) + (px % fs);
                let c_in = m / repeats;

                if c_in >= cin {
                    return contract("decoder.upsamples.avg_shortcut", "DupUp3D gather out of range");
                }
                let at = out.index(px, y, oc);
                out.data[at] = x.get(px / fs, y / fs, c_in);
            }
        }
    }
    Ok(out)
}

/// `Resample::forward` for `upsample2d`/`upsample3d` at `chunk_idx == 0`
/// (`wan_vae.hpp:184-274`): the temporal branch is a no-op and both modes
/// reduce to nearest 2x then the 3x3 conv. Which of the two modes it is only
/// changes DupUp3D's temporal factor, applied by the caller.
fn resample_up2(x: &Plane, w: &[f32], bias: &[f32], cout: usize) -> Plane {
    conv3x3(&nearest_up2(x), w, bias, cout)
}

/// `AttentionBlock::forward` (`wan_vae.hpp:588-648`) with one head
/// (`ggml_ext_attention_ext` is called with `n_head = 1`, no mask, non-causal).
///
/// The body is RMS_norm, `to_qkv`, the attention, `proj`, and the block adds
/// the pre-norm input back (`identity`, `wan_vae.hpp:609`, `:644`). The permutes
/// in the reference move C to `ne2` for the 1x1 convs and back; the attention
/// itself is a plain softmax over all `h*w` tokens per head, so here it is
/// written directly in token order. `to_qkv` is a 1x1 conv (one matvec per
/// pixel, so it goes through [`conv1x1`] and its F16 operand contract),
/// `q`/`k`/`v` are the three contiguous channel thirds (`split_image_qkv`,
/// `ggml_extend.cpp:539-555`), and the token index is `w + W*h` — the flattened
/// spatial index, which is what the reference's reshape to `[t, h*w, c]`
/// produces.
fn attention(x: &Plane, norm: &[f32], wqkv: &[f32], bqkv: &[f32], wproj: &[f32], bproj: &[f32]) -> Plane {
    let (wi, hi, c) = (x.width, x.height, x.channels);
    let tokens = wi * hi;
    let normed = rms_norm(x, norm);
    // to_qkv is one 1x1 conv per pixel, so it carries the same F16 operand
    // contract as every other conv in the reference.
    let qkv = conv1x1(&normed, wqkv, bqkv, 3 * c);

    // q/k/v are the three contiguous channel thirds, each [token][c] with the
    // token fastest — the layout the reference's `[t, h*w, c]` reshape builds.
    let mut q = vec![0f32; tokens * c];
    let mut k = vec![0f32; tokens * c];
    let mut v = vec![0f32; tokens * c];
    for t in 0..tokens {
        for ch in 0..c {
            q[t * c + ch] = qkv.data[t + tokens * ch];
            k[t * c + ch] = qkv.data[t + tokens * (c + ch)];
            v[t * c + ch] = qkv.data[t + tokens * (2 * c + ch)];
        }
    }

    let scale = 1.0f32 / (c as f32).sqrt();
    let mut scores = vec![0f32; tokens];
    let mut attended = vec![0f32; tokens * c];

    for qt in 0..tokens {
        for kt in 0..tokens {
            let mut dot = 0f32;
            for i in 0..c {
                dot += q[qt * c + i] * k[kt * c + i];
            }
            scores[kt] = dot * scale;
        }
        softmax(&mut scores);

        for i in 0..c {
            let mut acc = 0f32;
            for kt in 0..tokens {
                acc += scores[kt] * v[kt * c + i];
            }
            attended[qt * c + i] = acc;
        }
    }

    // The projection is the block's second 1x1 conv; back to the plane's
    // channel-slowest order for it. The block then adds its pre-norm input.
    let mut plane = Plane::zeros(wi, hi, c);
    for t in 0..tokens {
        for ch in 0..c {
            plane.data[t + tokens * ch] = attended[t * c + ch];
        }
    }
    add(&conv1x1(&plane, wproj, bproj, c), x)
}

/// `ggml_soft_max` (`ops.cpp`): max-subtracted, F32.
fn softmax(x: &mut [f32]) {
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

/// `ggml_add`: the residual sums, elementwise.
fn add(a: &Plane, b: &Plane) -> Plane {
    debug_assert_eq!((a.width, a.height, a.channels), (b.width, b.height, b.channels));
    let mut out = a.clone();
    for (v, other) in out.data.iter_mut().zip(&b.data) {
        *v += other;
    }
    out
}

// ---------------------------------------------------------------------------
// The decoder (`Decoder3d::forward`, `wan_vae.hpp:832-1021`)
// ---------------------------------------------------------------------------

/// One `ResidualBlock` (`wan_vae.hpp:373-464`): shortcut only when the widths
/// differ, body order RMS_norm, SiLU, conv, RMS_norm, SiLU, conv, add.
///
/// `prefix` is the block itself (`...middle.0`); its children are the
/// `residual.{0,2,3,6}` params and the sibling `shortcut` conv, which is why
/// the artifact names read `...upsamples.0.shortcut.weight` beside
/// `...upsamples.0.residual.2.weight`.
fn residual(weights: &VaeWeights, prefix: &str, x: &Plane, cin: usize, cout: usize) -> Result<Plane, VaeError> {
    let base = if cin != cout {
        conv1x1(
            x,
            &weights.matrix(&format!("{prefix}.shortcut.weight"), cin, cout)?,
            &weights.bias(&format!("{prefix}.shortcut.bias"), cout)?,
            cout,
        )
    } else {
        x.clone()
    };

    let mut h = rms_norm(x, &weights.gamma(&format!("{prefix}.residual.0.gamma"), cin)?);
    silu_in_place(&mut h);
    h = conv3x3(
        &h,
        &weights.taps(&format!("{prefix}.residual.2.weight"), cin, cout)?,
        &weights.bias(&format!("{prefix}.residual.2.bias"), cout)?,
        cout,
    );
    h = rms_norm(&h, &weights.gamma(&format!("{prefix}.residual.3.gamma"), cout)?);
    silu_in_place(&mut h);
    h = conv3x3(
        &h,
        &weights.taps(&format!("{prefix}.residual.6.weight"), cout, cout)?,
        &weights.bias(&format!("{prefix}.residual.6.bias"), cout)?,
        cout,
    );
    Ok(add(&h, &base))
}

/// One `Up_ResidualBlock` (`wan_vae.hpp:526-587`): three residual blocks, then
/// on every level but the last a 2x spatial resample and a DupUp3D of the
/// block input added to it.
fn up_block(weights: &VaeWeights, level: usize, x: &Plane, cin: usize, cout: usize) -> Result<Plane, VaeError> {
    let lvl = format!("decoder.upsamples.{level}");
    let shortcut_src = x.clone();
    let mut h = x.clone();

    for j in 0..UP_RESIDUALS {
        let (a, b) = if j == 0 { (cin, cout) } else { (cout, cout) };
        h = residual(weights, &format!("{lvl}.upsamples.{j}"), &h, a, b)?;
    }

    // `up_flag` is false on the last level only (`wan_vae.hpp:891`).
    if level >= DECODER_LEVELS - 1 {
        return Ok(h);
    }

    let t_up = level < VAE_TEMPORAL_UPSAMPLE.len() && VAE_TEMPORAL_UPSAMPLE[level];
    h = resample_up2(
        &h,
        &weights.taps(&format!("{lvl}.upsamples.3.resample.1.weight"), cout, cout)?,
        &weights.bias(&format!("{lvl}.upsamples.3.resample.1.bias"), cout)?,
        cout,
    );

    // The shortcut is the block input, duplicated (`wan_vae.hpp:576-580`); its
    // temporal factor is the one thing the upsample3d/upsample2d choice changes
    // at chunk 0.
    let shortcut = dup_up3d(&shortcut_src, cout, if t_up { 2 } else { 1 })?;
    Ok(add(&h, &shortcut))
}

/// The full decode (`WanVAE::decode`, `wan_vae.hpp:1207-1255`): `conv2`, the
/// decoder stack, then `unpatchify`, which for `patch_size == 1` is the
/// identity (`wan_vae.hpp:1140-1142`).
pub fn decode(weights: &VaeWeights, latent: &Plane, frames: usize) -> Result<Plane, VaeError> {
    check_frames(frames)?;

    if latent.channels != VAE_Z_DIM as usize {
        return contract(
            "latent",
            &format!("{} channels, the decoder takes {}", latent.channels, VAE_Z_DIM),
        );
    }

    let dims = vae_decoder_dims();
    let dims: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
    let d0 = dims[0];

    let mut x = conv1x1(
        latent,
        &weights.matrix("conv2.weight", latent.channels, latent.channels)?,
        &weights.bias("conv2.bias", latent.channels)?,
        latent.channels,
    );
    x = conv3x3(
        &x,
        &weights.taps("decoder.conv1.weight", latent.channels, d0)?,
        &weights.bias("decoder.conv1.bias", d0)?,
        d0,
    );

    x = residual(weights, "decoder.middle.0", &x, d0, d0)?;
    x = attention(
        &x,
        &weights.gamma("decoder.middle.1.norm.gamma", d0)?,
        &weights.matrix("decoder.middle.1.to_qkv.weight", d0, 3 * d0)?,
        &weights.bias("decoder.middle.1.to_qkv.bias", 3 * d0)?,
        &weights.matrix("decoder.middle.1.proj.weight", d0, d0)?,
        &weights.bias("decoder.middle.1.proj.bias", d0)?,
    );
    x = residual(weights, "decoder.middle.2", &x, d0, d0)?;

    for level in 0..DECODER_LEVELS {
        x = up_block(weights, level, &x, dims[level], dims[level + 1])?;
    }

    let last = dims[DECODER_LEVELS];
    let mut out = rms_norm(&x, &weights.gamma("decoder.head.0.gamma", last)?);
    silu_in_place(&mut out);
    Ok(conv3x3(
        &out,
        &weights.taps("decoder.head.2.weight", last, VAE_OUT_CHANNELS as usize)?,
        &weights.bias("decoder.head.2.bias", VAE_OUT_CHANNELS as usize)?,
        VAE_OUT_CHANNELS as usize,
    ))
}

/// The single-frame restriction, stated by name.
fn check_frames(frames: usize) -> Result<(), VaeError> {
    if frames == 1 {
        return Ok(());
    }
    Err(VaeError::Unsupported(
        "a latent with a temporal length other than 1: the reference's streaming feat-cache decode \
         (`wan_vae.hpp:1231-1247`) is not ported",
    ))
}

// ---------------------------------------------------------------------------
// Pixels
// ---------------------------------------------------------------------------

/// The reference's tensor-to-pixels path, in its own order: `scale_tensor_to_0_1`
/// (`vae.hpp:107-113`, applied because `scale_input` is true and no runner
/// overrides it) then `preprocessing_float_to_u8` (`preprocessing.hpp:27-35`).
/// Channel `c` becomes byte `c` (`tensor_to_sd_image`, `util.cpp:742-757`:
/// the decoded head is 4 channels wide, so the PNG is RGBA).
///
/// The clamp is spelled with the reference's own comparison order:
/// `std::max(0.0f, std::min(1.0f, value))` turns a NaN into 1.0, which is how
/// this decoder's overflowed pixels become white instead of black.
pub fn to_rgba8(image: &Plane) -> Vec<u8> {
    let mut out = vec![0u8; image.width * image.height * image.channels];

    for h in 0..image.height {
        for w in 0..image.width {
            for c in 0..image.channels {
                let scaled = (image.get(w, h, c) + 1.0) * 0.5;
                let scaled = if scaled < 1.0 { scaled } else { 1.0 };
                let scaled = if 0.0 < scaled { scaled } else { 0.0 };
                out[(h * image.width + w) * image.channels + c] =
                    if scaled >= 1.0 { 255 } else { (scaled * 255.0 + 0.5) as u8 };
            }
        }
    }
    out
}

/// Writes the decoded frame as RGBA PNG, the format the reference's own
/// `stbi_write_png` call writes for a 4-channel image (`media_io.cpp:767`).
pub fn write_png(path: &Path, image: &Plane) -> Result<(), VaeError> {
    use image::ImageEncoder;

    if image.channels != VAE_OUT_CHANNELS as usize {
        return contract(
            "decode output",
            &format!("{} channels: the PNG path writes the 4 the decoder produces", image.channels),
        );
    }
    let pixels = to_rgba8(image);
    let file = std::fs::File::create(path)?;
    image::codecs::png::PngEncoder::new(std::io::BufWriter::new(file))
        .write_image(&pixels, image.width as u32, image.height as u32, image::ExtendedColorType::Rgba8)
        .map_err(|e| VaeError::Image(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GGML DupUp3D chain (`wan_vae.hpp:344-370`) replayed literally over a
    /// flat index space: concat, reshape (a shape change on the same buffer),
    /// permute + cont, ..., slice. It is the independent check of
    /// [`dup_up3d`]'s closed form.
    fn dup_up3d_ggml_chain(x: &[f32], shape: [usize; 4], cout: usize, ft: usize) -> Vec<f32> {
        let fs = DUP_FACTOR_S;
        let cin = shape[3];
        let repeats = cout * ft * fs * fs / cin;
        let flat = |shape: &[usize; 4], i: &[usize; 4]| {
            i[0] + shape[0] * (i[1] + shape[1] * (i[2] + shape[2] * i[3]))
        };
        // `ggml_permute` then cont: the result's axis i is the source's `axes[i]`.
        let permute = |data: &[f32], from: [usize; 4], axes: [usize; 4]| {
            let to = [from[axes[0]], from[axes[1]], from[axes[2]], from[axes[3]]];
            let mut out = vec![0f32; data.len()];
            for i0 in 0..to[0] {
                for i1 in 0..to[1] {
                    for i2 in 0..to[2] {
                        for i3 in 0..to[3] {
                            let dst = [i0, i1, i2, i3];
                            let mut src = [0usize; 4];
                            for (k, axis) in axes.iter().enumerate() {
                                src[*axis] = dst[k];
                            }
                            out[flat(&to, &dst)] = data[flat(&from, &src)];
                        }
                    }
                }
            }
            (out, to)
        };
        let reshape = |from: [usize; 4], to: [usize; 4]| {
            assert_eq!(from.iter().product::<usize>(), to.iter().product::<usize>(), "reshape {from:?} -> {to:?}");
            to
        };

        // concat the single frame `repeats` times along ggml dim 2. The copy
        // index sits between H and C in the flat order, so each channel's
        // WxH block is repeated, not the whole buffer.
        let wh = shape[0] * shape[1];
        let mut data = Vec::with_capacity(x.len() * repeats);
        for c in 0..cin {
            for _ in 0..repeats {
                data.extend_from_slice(&x[c * wh..(c + 1) * wh]);
            }
        }
        let from = [shape[0], shape[1], repeats, cin];
        // reshape to [W, H*T, factor_s, factor_s*factor_t*C] with T stale at 1.
        let from = reshape(from, [shape[0], shape[1] * shape[2], fs, fs * ft * cout]);
        let (permuted, to) = permute(&data, from, [2, 0, 1, 3]);
        data = permuted;
        // reshape to [factor_s*W, H*T, factor_s, factor_t*C].
        let from = reshape(to, [to[0] * to[1], to[2] * shape[2], fs, ft * cout]);
        let (permuted, to) = permute(&data, from, [0, 2, 1, 3]);
        data = permuted;
        // reshape to [factor_s*W*factor_s*H, T, factor_t, C].
        let from = reshape(to, [to[0] * to[1] * to[2], shape[2], ft, cout]);
        let (permuted, to) = permute(&data, from, [0, 2, 1, 3]);
        data = permuted;
        // reshape to [factor_s*W, factor_s*H, factor_t*T, C] and slice the last
        // temporal frame (first_chunk at chunk_idx == 0).
        let from = reshape(to, [fs * shape[0], fs * shape[1], ft * shape[2], cout]);

        let mut out = Vec::with_capacity(from[0] * from[1] * from[3]);
        // Read out in plane order: x fastest, then y, then the channel.
        for oc in 0..from[3] {
            for y in 0..from[1] {
                for px in 0..from[0] {
                    out.push(data[flat(&from, &[px, y, ft - 1, oc])]);
                }
            }
        }
        out
    }

    #[test]
    fn dup_up3d_matches_the_reference_op_chain() {
        // in = 6, out = 3 covers a channel-grouping ratio; the others use the
        // ratios the decoder itself carries.
        for (cin, cout, ft) in [(6usize, 3usize, 1usize), (6, 3, 2), (8, 8, 2), (8, 4, 2), (4, 4, 1)] {
            let (w, h) = (3usize, 2usize);
            let values: Vec<f32> = (0..w * h * cin).map(|i| i as f32 * 0.25 - 3.0).collect();
            let plane = Plane::new(w, h, cin, values.clone());
            let ours = dup_up3d(&plane, cout, ft).expect("dup_up3d");

            let reference = dup_up3d_ggml_chain(&values, [w, h, 1, cin], cout, ft);
            // The chain returns [x, y, c] with x fastest, like the plane.
            assert_eq!(ours.values(), reference.as_slice(), "in={cin} out={cout} ft={ft}");
        }
    }

    /// The channel/parity grouping the closed form predicts: when `repeats` is
    /// the full `factor`, both duplicates land on the same channel (a plain
    /// nearest 2x upsample), and the two-halving ratios pick the channels the
    /// reference's grouping leaves.
    #[test]
    fn dup_up3d_channel_groups_follow_the_temporal_factor() {
        let values: Vec<f32> = (0..2 * 2 * 8).map(|i| i as f32).collect();
        let plane = Plane::new(2, 2, 8, values);

        // 8 -> 8, factor 8: channel oc reads channel oc.
        let same = dup_up3d(&plane, 8, 2).expect("8->8");
        for y in 0..4 {
            for x in 0..4 {
                for oc in 0..8 {
                    assert_eq!(same.values()[x + 4 * (y + 4 * oc)], plane.values()[x / 2 + 2 * (y / 2 + 2 * oc)]);
                }
            }
        }

        // 8 -> 4, factor 8, repeats 4: the sliced frame reads channel 2*oc + 1
        // (`b / factor_s` lands on the last temporal frame, `ft - 1`).
        let halved = dup_up3d(&plane, 4, 2).expect("8->4");
        for y in 0..4 {
            for x in 0..4 {
                for oc in 0..4 {
                    assert_eq!(halved.values()[x + 4 * (y + 4 * oc)], plane.values()[x / 2 + 2 * (y / 2 + 2 * (2 * oc + 1))]);
                }
            }
        }
    }

    #[test]
    fn conv3x3_zero_pads_and_sums_the_taps() {
        // A 3x3 averaging kernel over a 2x2 input: every output is the mean of
        // the in-bounds taps, and the output size is unchanged.
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let x = Plane::new(2, 2, 1, values);
        let taps: Vec<f32> = (0..9).map(|_| 1.0 / 9.0).collect();
        let out = conv3x3(&x, &taps, &[0.0], 1);

        assert_eq!((out.width(), out.height()), (2, 2));
        assert!((out.get(0, 0, 0) - 10.0 / 9.0).abs() < 1e-6);
        assert!((out.get(1, 1, 0) - 10.0 / 9.0).abs() < 1e-6);

        // A delta kernel at the center is the identity.
        let mut delta = vec![0f32; 9];
        delta[4] = 1.0;
        let out = conv3x3(&x, &delta, &[0.0], 1);
        assert_eq!(out.values(), x.values());
    }

    /// A multi-channel kernel against a direct summation: the channel axis is
    /// the slowest one in the plane, which a one-channel fixture cannot catch.
    #[test]
    fn conv3x3_matches_a_direct_sum_over_the_plane_layout() {
        let (wi, hi, ci, cout) = (3usize, 2usize, 2usize, 2usize);
        let x = Plane::new(wi, hi, ci, (0..wi * hi * ci).map(|i| i as f32 * 0.5 - 1.0).collect());
        let taps: Vec<f32> = (0..TAPS * ci * cout).map(|i| (i % 11) as f32 * 0.125 - 0.5).collect();
        let bias = [0.25f32, -0.75];
        let out = conv3x3(&x, &taps, &bias, cout);

        for oc in 0..cout {
            for h in 0..hi {
                for p in 0..wi {
                    let mut expect = bias[oc];
                    for tap in 0..TAPS {
                        let (kh, kw) = (tap / TAP_SIDE, tap % TAP_SIDE);
                        for ic in 0..ci {
                            let ih = h as isize + kh as isize - 1;
                            let iw = p as isize + kw as isize - 1;
                            if ih < 0 || ih >= hi as isize || iw < 0 || iw >= wi as isize {
                                continue;
                            }
                            expect += x.get(iw as usize, ih as usize, ic)
                                * taps[(tap * ci + ic) * cout + oc];
                        }
                    }
                    assert!((out.get(p, h, oc) - expect).abs() < 1e-5, "({p},{h},{oc})");
                }
            }
        }
    }

    #[test]
    fn conv1x1_is_a_per_pixel_matrix() {
        // Channel-slowest: pixel 0 is (1, 3), pixel 1 is (2, 4).
        let x = Plane::new(2, 1, 2, vec![1.0, 2.0, 3.0, 4.0]);
        // [[1, 1], [0, 1]] maps (a, b) -> (a, a + b).
        let w = vec![1.0, 1.0, 0.0, 1.0];
        let out = conv1x1(&x, &w, &[0.5, -0.5], 2);

        assert_eq!(out.values(), &[1.5, 2.5, 3.5, 5.5]);
    }

    #[test]
    fn nearest_up2_duplicates_each_pixel_into_a_block() {
        let x = Plane::new(2, 1, 1, vec![1.0, 2.0]);
        let out = nearest_up2(&x);

        assert_eq!((out.width(), out.height()), (4, 2));
        assert_eq!(out.values(), &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn rms_norm_scales_to_unit_rms_and_applies_gamma() {
        let x = Plane::new(1, 1, 4, vec![3.0, 4.0, 0.0, 0.0]);
        let out = rms_norm(&x, &[1.0, 1.0, 1.0, 1.0]);

        // mean square is 25/4, so the scale is 2/5.
        assert!((out.values()[0] - 1.2).abs() < 1e-6);
        assert!((out.values()[1] - 1.6).abs() < 1e-6);

        let out = rms_norm(&x, &[2.0, 0.0, 0.0, 0.0]);
        assert!((out.values()[0] - 2.4).abs() < 1e-6);
        assert_eq!(out.values()[1], 0.0);
    }

    #[test]
    fn silu_is_x_over_one_plus_exp_minus_x() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-6);
        assert!((silu(-1.0) + 0.26894143).abs() < 1e-6);
    }

    #[test]
    fn latent_conversions_are_inverse_round_trips() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.125).collect();
        let diffusion = Plane::new(1, 1, 64, values.clone());
        let vae = diffusion_to_vae(&diffusion).expect("to vae");
        let back = vae_to_diffusion(&vae).expect("to diffusion");

        for ((a, b), raw) in back.values().iter().zip(&values).zip(0..) {
            assert!((a - b).abs() < 1e-5, "channel {raw}: {a} vs {b}");
        }

        // The map itself is `x*std + mean` with scale_factor 1.
        assert!((vae.values()[0] - (values[0] * VAE_LATENT_STD[0] + VAE_LATENT_MEAN[0])).abs() < 1e-6);
        assert!((vae.values()[63] - (values[63] * VAE_LATENT_STD[63] + VAE_LATENT_MEAN[63])).abs() < 1e-6);

        // Any other channel count is refused, like the reference's GGML_ABORT.
        let wrong = Plane::new(1, 1, 16, vec![0.0; 16]);
        assert!(diffusion_to_vae(&wrong).is_err());
    }

    #[test]
    fn to_rgba8_follows_the_reference_range_map() {
        // (x + 1) * 0.5, clamped: -1 -> 0, 0 -> 128, 1 -> 255, and channel c
        // lands on byte c of the RGBA pixel.
        let x = Plane::new(1, 1, 4, vec![-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(to_rgba8(&x), vec![0u8, 128, 255, 255]);

        // Two pixels, the first all-black, the second all-white, in the
        // channel-slowest layout.
        let x = Plane::new(2, 1, 4, vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
        assert_eq!(to_rgba8(&x), vec![0, 0, 0, 0, 255, 255, 255, 255]);
    }

    #[test]
    fn decode_refuses_a_multi_frame_latent_by_name() {
        let err = check_frames(2).expect_err("two frames");
        match err {
            VaeError::Unsupported(what) => assert!(what.contains("temporal length"), "{what}"),
            other => panic!("unexpected {other:?}"),
        }
        assert!(check_frames(1).is_ok());
    }

    #[test]
    fn bf16_decodes_exactly() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    /// The PNG writer round-trips the RGBA bytes the reference's own writer
    /// would have produced.
    #[test]
    fn write_png_round_trips_the_pixels() {
        let image = Plane::new(2, 2, 4, vec![
            -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        let expected = to_rgba8(&image);

        let path = std::env::temp_dir().join("ds4_vae_write_png_test.png");
        write_png(&path, &image).expect("write");
        let read = image::ImageReader::open(&path)
            .expect("open")
            .decode()
            .expect("decode")
            .to_rgba8();
        let _ = std::fs::remove_file(&path);

        assert_eq!((read.width(), read.height()), (2, 2));
        assert_eq!(read.into_raw(), expected);
    }

    /// The conv operand cast: the F16 grid, ties to even, overflow to
    /// infinity, and the subnormal range.
    #[test]
    fn f16_round_follows_binary16() {
        // Exact values pass through.
        for value in [0.0f32, 1.0, -2.0, 0.5, 2048.0, -65504.0, 6.1035156e-5] {
            assert_eq!(f16_round(value), value, "{value}");
        }
        // 65504 is the largest finite F16; the next grid point overflows.
        assert_eq!(f16_round(65504.0), 65504.0);
        assert!(f16_round(65520.0).is_infinite() && f16_round(65520.0) > 0.0);
        assert!(f16_round(-70000.0).is_infinite() && f16_round(-70000.0) < 0.0);
        // Rounding to the nearest representable value (0.1 is not on the grid).
        assert_eq!(f16_round(0.1), f32::from_bits(0x3dcc_c000));
        assert_eq!(f16_round(1.0 + 1.0 / 2048.0), 1.0);
        assert_eq!(f16_round(1.0 + 3.0 / 2048.0), 1.0 + 1.0 / 512.0);
        // Subnormals: the step is 2^-24, and 2^-25 ties to even (zero).
        assert_eq!(f16_round(2f32.powi(-24)), 2f32.powi(-24));
        assert_eq!(f16_round(2f32.powi(-25)), 0.0);
        assert_eq!(f16_round(0.75 * 2f32.powi(-24)), 2f32.powi(-24));
        assert_eq!(f16_round(2f32.powi(-30)), 0.0);
        // Negative subnormals round by magnitude and keep their sign; the
        // reference's own conversion returns -168 * 2^-24 and -1007 * 2^-24
        // for these two, not a signed zero.
        assert_eq!(f16_round(-2f32.powi(-24)), -2f32.powi(-24));
        assert_eq!(f16_round(-0.75 * 2f32.powi(-24)), -2f32.powi(-24));
        assert_eq!(f16_round(-1e-5), -168.0 * 2f32.powi(-24));
        assert_eq!(f16_round(-6e-5), -1007.0 * 2f32.powi(-24));
        assert!(f16_round(-1e-5).is_sign_negative());
        // NaN and infinity keep their kind.
        assert!(f16_round(f32::NAN).is_nan());
        assert_eq!(f16_round(f32::INFINITY), f32::INFINITY);
        assert_eq!(f16_round(f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    /// The convs really do transport the F16 range limit: a value past 65504
    /// reaches the accumulation as an infinity.
    #[test]
    fn conv3x3_saturates_its_input_at_the_f16_range() {
        let x = Plane::new(1, 1, 1, vec![70_000.0]);
        let mut center = vec![0f32; 9];
        center[4] = 1.0;
        let out = conv3x3(&x, &center, &[0.0], 1);
        assert!(out.values()[0].is_infinite(), "{}", out.values()[0]);

        // Just below the limit the value survives, rounded to the F16 grid
        // (32 is the spacing in this binade, so 65000 lands on 64992).
        let x = Plane::new(1, 1, 1, vec![65_000.0]);
        let out = conv3x3(&x, &center, &[0.0], 1);
        assert_eq!(out.values()[0], 64_992.0);
    }
}
