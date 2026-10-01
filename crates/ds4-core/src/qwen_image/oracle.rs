//! The P1 CPU oracle: exact numerics for the Qwen-Image-2.1 image engine.
//!
//! Slow, single-threaded, F32, no FFI and no GPU. This module exists so every
//! numeric question is settled before a kernel exists; P2 measures kernels
//! against it, so a quiet error here would poison every later phase. Every item
//! is a 1:1 port of `stable-diffusion.cpp` at `6dcb5bb` and names the reference
//! lines it came from.
//!
//! Exactness class of each item (plan section 12):
//!
//! | item                                    | class               |
//! |-----------------------------------------|---------------------|
//! | Philox initial noise, positions, masks  | exact by construction |
//! | RoPE table inputs, segment boundaries   | exact by construction |
//! | the DiT evaluation, the VAE decode      | stated tolerance    |
//!
//! The reference's own intermediates are the fixtures: `SD_DUMP_IDS`,
//! `SD_DUMP_COND` and `SD_DUMP_STEPS` write `dims <n> <d0> <d1> ...\n` followed
//! by raw little-endian F32 in GGML order (ne0 fastest). They are generated on
//! the host with `/data/imagegen/bin/sd-cli` and never committed.

use super::{DIT_AXES_DIM, DIT_ROPE_THETA};

// ---------------------------------------------------------------------------
// Philox RNG — the initial noise (`rng_philox.hpp`)
// ---------------------------------------------------------------------------

/// `philox_m` (`rng_philox.hpp:20`).
const PHILOX_M: [u32; 2] = [0xD2511F53, 0xCD9E8D57];
/// `philox_w` (`rng_philox.hpp:21`).
const PHILOX_W: [u32; 2] = [0x9E3779B9, 0xBB67AE85];
/// `two_pow32_inv` (`rng_philox.hpp:22`).
const TWO_POW32_INV: f32 = 2.3283064e-10;
/// `two_pow32_inv_2pi` (`rng_philox.hpp:23`). Kept as a product so the constant
/// carries the reference's own float rounding.
const TWO_POW32_INV_2PI: f32 = TWO_POW32_INV * 6.2831855;

/// Rounds of the Philox 4x32 mix (`rng_philox.hpp:56`).
const PHILOX_ROUNDS: usize = 10;

/// Philox 4x32, the CUDA/`sd-webui` RNG of the reference: `--rng cuda`.
///
/// The counter is the tensor element: lane 0 is the call's offset, lane 2 the
/// element index. A second `randn` call therefore continues the stream, which
/// is why the offset is state and not an argument.
#[derive(Clone, Debug)]
pub struct Philox {
    seed: u64,
    offset: u32,
}

impl Philox {
    pub fn new(seed: u64) -> Self {
        Self { seed, offset: 0 }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// One round of the mix (`rng_philox.hpp:26-38`). The 32x32 products are
    /// 64-bit: the high halves feed the cross lanes.
    fn round(lanes: &mut [[u32; 4]], key0: u32, key1: u32) {
        for lane in lanes.iter_mut() {
            let v1 = (lane[0] as u64) * (PHILOX_M[0] as u64);
            let v2 = (lane[2] as u64) * (PHILOX_M[1] as u64);

            lane[0] = ((v2 >> 32) as u32) ^ lane[1] ^ key0;
            lane[1] = v2 as u32;
            lane[2] = ((v1 >> 32) as u32) ^ lane[3] ^ key1;
            lane[3] = v1 as u32;
        }
    }

    /// `philox4_32` (`rng_philox.hpp:48-58`): ten rounds, the key advanced by
    /// `philox_w` between them.
    fn mix(lanes: &mut [[u32; 4]], mut key0: u32, mut key1: u32) {
        for _ in 0..PHILOX_ROUNDS {
            Self::round(lanes, key0, key1);
            key0 = key0.wrapping_add(PHILOX_W[0]);
            key1 = key1.wrapping_add(PHILOX_W[1]);
        }
    }

    /// `box_muller` (`rng_philox.hpp:60-69`): the Box-Muller transform on two
    /// uniform lanes, returning only the sine branch.
    ///
    /// The reference writes `float u = ...; float s = sqrt(-2.0f * log(u));` with
    /// `<cmath>` included, and in that build `log(u)` and `sqrt(...)` resolve to
    /// the DOUBLE C functions: the float argument is promoted, and only the
    /// assignment to `float s` rounds. `s * sin(v)` is a double product for the
    /// same reason. Measured against the reference's own noise dump: an all-F32
    /// chain matches 11242 of 16384 samples bit-exactly, double-throughout
    /// matches 12136, and this mixed form matches all 16384. Mirroring it is what
    /// makes the initial noise byte-identical instead of merely close.
    fn box_muller(x: f32, y: f32) -> f32 {
        let u = x * TWO_POW32_INV + TWO_POW32_INV / 2.0;
        let v = y * TWO_POW32_INV_2PI + TWO_POW32_INV_2PI / 2.0;

        let s = ((-2.0f64) * (u as f64).ln()).sqrt() as f32;
        (s as f64 * (v as f64).sin()) as f32
    }

    /// `randn` (`rng_philox.hpp:77-96`): `n` standard normals, advancing the
    /// counter offset by one.
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        let key0 = self.seed as u32;
        let key1 = (self.seed >> 32) as u32;

        let mut lanes = vec![[self.offset, 0u32, 0u32, 0u32]; n];
        for (i, lane) in lanes.iter_mut().enumerate() {
            lane[2] = i as u32;
        }
        self.offset = self.offset.wrapping_add(1);

        Self::mix(&mut lanes, key0, key1);

        lanes.iter().map(|lane| Self::box_muller(lane[0] as f32, lane[1] as f32)).collect()
    }
}

/// The initial noise for a latent of `elements` with a pinned seed: the same
/// call the reference makes for a batch of one (`--rng cuda`).
pub fn initial_noise(elements: usize, seed: u64) -> Vec<f32> {
    Philox::new(seed).randn(elements)
}

// ---------------------------------------------------------------------------
// Layout: segments, positions, prefix (`qwen_image_2_1.hpp:64-121`)
// ---------------------------------------------------------------------------

/// One attention segment. `image_index` is -1 for text, else the reference
/// image's index; the last image uses the latent being denoised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub start: i64,
    pub end: i64,
    pub context_start: i64,
    pub image_index: i64,
}

/// The joint sequence: text first, then the image token grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub segments: Vec<Segment>,
    /// One `(t, h, w)` triple per joint token, in sequence order.
    pub positions: Vec<[f32; 3]>,
    /// Text tokens in the joint sequence; the image is everything after it.
    pub prefix_length: i64,
}

/// Why a layout could not be built (`qwen_image_2_1.hpp:78-119`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// No image shape, or a vision-slot vector that does not span the prompt.
    Invalid,
    /// A vision slot run whose size is not the matching reference latent's.
    SlotShapeMismatch,
    /// The slot runs do not cover every reference latent.
    MissingRefSlots,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            LayoutError::Invalid => "invalid image token layout",
            LayoutError::SlotShapeMismatch => {
                "vision slots and reference latents must have matching sizes"
            }
            LayoutError::MissingRefSlots => "missing reference image slots",
        };
        write!(f, "qwen-image-2.1: {text}")
    }
}

impl std::error::Error for LayoutError {}

/// Builds the joint layout (`QwenImage21Layout::build`).
///
/// `image_slots` is the vision placeholder id per prompt token, empty for
/// text-to-image; `shapes` is `(height, width)` in tokens, one per reference
/// latent plus the latent being denoised, in that order.
pub fn build_layout(
    text_length: i64,
    image_slots: &[i32],
    shapes: &[(i64, i64)],
) -> Result<Layout, LayoutError> {
    if shapes.is_empty() || (!image_slots.is_empty() && image_slots.len() as i64 != text_length) {
        return Err(LayoutError::Invalid);
    }

    let mut layout = Layout { segments: Vec::new(), positions: Vec::new(), prefix_length: 0 };
    let mut position: i64 = 0;
    let mut next_image = 0usize;

    let mut i: i64 = 0;
    while i < text_length {
        let tag = image_slots.get(i as usize).copied().unwrap_or(0);
        let begin = i;
        i += 1;
        while i < text_length && image_slots.get(i as usize).copied().unwrap_or(0) == tag {
            i += 1;
        }

        if tag != 0 {
            let matches = tag == next_image as i32 + 1
                && next_image + 1 < shapes.len()
                && (i - begin) * 4 == shapes[next_image].0 * shapes[next_image].1;
            if !matches {
                return Err(LayoutError::SlotShapeMismatch);
            }
            append_image(&mut layout, &mut position, shapes, next_image, begin);
            next_image += 1;
            continue;
        }

        let start = layout.positions.len() as i64;
        layout.segments.push(Segment { start, end: start + i - begin, context_start: begin, image_index: -1 });
        for _ in begin..i {
            layout.positions.push([position as f32, position as f32, position as f32]);
            position += 1;
        }
    }

    if next_image + 1 != shapes.len() {
        return Err(LayoutError::MissingRefSlots);
    }
    layout.prefix_length = layout.positions.len() as i64;
    append_image(&mut layout, &mut position, shapes, next_image, text_length);
    Ok(layout)
}

/// Adds one image grid to the joint sequence: row-major tokens with a centered
/// spatial id on both axes, so a crop keeps its place in the whole image
/// (`qwen_image_2_1.hpp:96-107`). The temporal id is the running position, which
/// then advances by the grid's longer side.
fn append_image(
    layout: &mut Layout,
    position: &mut i64,
    shapes: &[(i64, i64)],
    index: usize,
    context_start: i64,
) {
    let (height, width) = shapes[index];
    let start = layout.positions.len() as i64;

    layout.segments.push(Segment {
        start,
        end: start + height * width,
        context_start,
        image_index: index as i64,
    });

    for h in 0..height {
        for w in 0..width {
            layout.positions.push([
                *position as f32,
                (h - (height - height / 2)) as f32,
                (w - (width - width / 2)) as f32,
            ]);
        }
    }
    *position += height.max(width);
}

// ---------------------------------------------------------------------------
// RoPE (`rope.hpp:26-110`, `rope.hpp:191-250`, `rope.hpp:1110-1151`)
// ---------------------------------------------------------------------------

/// `linspace` in the reference's own precision (`rope.hpp:26-38`).
fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
    if num == 1 {
        return vec![start];
    }
    let step = (end - start) / (num - 1) as f32;
    (0..num).map(|i| start + i as f32 * step).collect()
}

/// `omega[j] = 1 / theta^scale[j]` for one axis (`rope.hpp:66-70`).
fn rope_omega(dim: u64, theta: f32) -> Vec<f32> {
    let half = (dim / 2) as usize;
    let scale = linspace(0.0, (dim as f32 - 2.0) / dim as f32, half);
    scale.iter().map(|s| 1.0 / theta.powf(*s)).collect()
}

/// The RoPE table: `[2, 2, head_dim/2, L]` holding `[[cos, -sin], [sin, cos]]`
/// per pair (`Rope::embed_nd` at `rope.hpp:191-250` and the tensor it fills at
/// `qwen_image_2_1.hpp:335`).
///
/// One rope is applied per axis with that axis's width; the axes concatenate on
/// the pair axis, so a pair index selects its axis by position.
pub fn rope_table(positions: &[[f32; 3]], axes_dim: &[u64; 3], theta: f32) -> Vec<f32> {
    let pairs: usize = axes_dim.iter().map(|d| (d / 2) as usize).sum();
    let mut table = vec![0.0f32; 4 * pairs * positions.len()];

    for (axis, &dim) in axes_dim.iter().enumerate() {
        let omega = rope_omega(dim, theta);
        let pair_offset: usize = axes_dim[..axis].iter().map(|d| (d / 2) as usize).sum();
        for (pos, ids) in positions.iter().enumerate() {
            let block = 4 * pairs * pos;
            for (j, w) in omega.iter().enumerate() {
                let angle = ids[axis] * w;
                let (cos, sin) = (angle.cos(), angle.sin());
                let base = block + 4 * (pair_offset + j);
                // (i0, i1): cos at (0,0), -sin at (1,0), sin at (0,1), cos at (1,1).
                table[base] = cos;
                table[base + 1] = -sin;
                table[base + 2] = sin;
                table[base + 3] = cos;
            }
        }
    }
    table
}

/// How the head dimension's pairs are formed.
///
/// `rope_interleaved` defaults to true in the reference (`rope.hpp:1110`),
/// which pairs adjacent elements. Reading cannot settle which one these weights
/// were trained with (recipe section 4): the DiT evaluation is the probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopePairing {
    /// Adjacent pairs: `(2j, 2j+1)`. The reference's default.
    Interleaved,
    /// Halves: `(j, j + head_dim/2)`, the non-interleaved branch.
    HalfSplit,
}

/// Applies the RoPE table in place to one tensor of `[heads, positions, dim_head]`
/// (`Rope::apply_rope`, `rope.hpp:1110-1151`): `out0 = x0*cos - x1*sin`,
/// `out1 = x0*sin + x1*cos` per pair.
pub fn apply_rope(
    x: &mut [f32],
    table: &[f32],
    positions: usize,
    heads: usize,
    dim_head: usize,
    pairing: RopePairing,
) {
    let pair_count = dim_head / 2;
    let half = dim_head / 2;

    for head in 0..heads {
        for pos in 0..positions {
            let block = head * positions * dim_head + pos * dim_head;
            let table_block = 4 * pair_count * pos;

            for j in 0..pair_count {
                let (d0, d1) = match pairing {
                    RopePairing::Interleaved => (2 * j, 2 * j + 1),
                    RopePairing::HalfSplit => (j, j + half),
                };
                let cos = table[table_block + 4 * j];
                let sin = table[table_block + 4 * j + 2];

                let x0 = x[block + d0];
                let x1 = x[block + d1];
                x[block + d0] = x0 * cos - x1 * sin;
                x[block + d1] = x0 * sin + x1 * cos;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Masks (`qwen_image_2_1.hpp:340-357`)
// ---------------------------------------------------------------------------

/// The text segment's causal mask, `[query][key]` row-major, -inf where a
/// query cannot see a key. The image segment has no mask: it attends to the
/// whole prefix and to every image token.
pub fn text_mask(length: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; length * length];
    for q in 0..length {
        for k in (q + 1)..length {
            mask[q * length + k] = f32::NEG_INFINITY;
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// Flow schedule (`denoiser.hpp:725-790`)
// ---------------------------------------------------------------------------

/// `FluxScheduler` shift anchors (`denoiser.hpp:732-733`, `:755-756`).
pub const FLUX_BASE_SHIFT: f32 = 0.5;
pub const FLUX_MAX_SHIFT: f32 = 1.15;
const FLUX_BASE_ANCHOR: f32 = 256.0;
const FLUX_MAX_ANCHOR: f32 = 4096.0;

/// `DiscreteFlowDenoiser` shift for Qwen-Image (`diffusion_engine.cpp:1372`).
pub const FLOW_SHIFT: f32 = 3.0;

/// `flux_time_shift` (`denoiser.hpp:725-727`).
pub fn flux_time_shift(mu: f32, sigma: f32, t: f32) -> f32 {
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

/// `FluxScheduler::compute_mu` (`denoiser.hpp:754-759`): a line through
/// (256 tokens, base_shift) and (4096 tokens, max_shift).
pub fn flux_mu(image_seq_len: i64) -> f32 {
    let m = (FLUX_MAX_SHIFT - FLUX_BASE_SHIFT) / (FLUX_MAX_ANCHOR - FLUX_BASE_ANCHOR);
    let b = FLUX_BASE_SHIFT - m * FLUX_BASE_ANCHOR;
    image_seq_len as f32 * m + b
}

/// `FluxScheduler::get_sigmas` (`denoiser.hpp:761-787`): `t = 1 - i/n` under
/// the flow shift, with the terminal sigma pinned to zero.
pub fn flux_sigmas(image_seq_len: i64, steps: usize) -> Vec<f32> {
    let mu = flux_mu(image_seq_len);
    let mut sigmas = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let t = 1.0 - i as f32 / steps as f32;
        if t <= 0.0 {
            sigmas.push(0.0);
        } else {
            sigmas.push(flux_time_shift(mu, 1.0, t));
        }
    }
    sigmas[steps] = 0.0;
    sigmas
}

/// The DiT's own time input: `DiscreteFlowDenoiser::sigma_to_t` is `sigma*1000`
/// (`denoiser.hpp:1316-1318`) and `prepare_sample_timesteps` passes it through
/// (`diffusion_engine.cpp:2105-2110`).
pub fn flow_timestep(sigma: f32) -> f32 {
    sigma * 1000.0
}

/// `DiscreteFlowDenoiser::noise_scaling` (`denoiser.hpp:1341-1345`). For a
/// text-to-image run the init latent is empty, so the sampled latent is the
/// noise times the first sigma, which the schedule pins to 1.
pub fn noise_scaling(sigma: f32, noise: &[f32], latent: Option<&[f32]>) -> Vec<f32> {
    match latent {
        Some(l) => l.iter().zip(noise).map(|(l, n)| l * (1.0 - sigma) + n * sigma).collect(),
        None => noise.iter().map(|n| n * sigma).collect(),
    }
}

/// One Euler step (`sample_euler`, `denoiser.hpp:1765-1780`) in the reference's
/// own operation order.
///
/// The order is not cosmetic. The reference turns the model output into the
/// denoised prediction first and takes the velocity from that, which with
/// `c_skip = 1` and `c_out = -sigma` reads `denoised = x - sigma*pred`,
/// `d = (x - denoised)/sigma`, `x += d*(sigma_next - sigma)`. Rebuilt that way
/// from the reference's own dumps, the second step's latent input comes back bit
/// for bit (16384 of 16384 values); the algebraically equal one-expression form
/// `x + pred*(sigma_next - sigma)` differs in 692 of them, because
/// `x - (x - sigma*pred)` is not exactly `sigma*pred` in F32.
pub fn euler_step(x: &[f32], pred: &[f32], sigma: f32, sigma_next: f32) -> Vec<f32> {
    x.iter()
        .zip(pred)
        .map(|(x, p)| {
            let denoised = x - sigma * p;
            let d = (x - denoised) / sigma;
            x + d * (sigma_next - sigma)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schedule the reference logged for this fixture: `image_seq_len=256,
    /// steps=2, mu=0.500` (`refdump/run1.log`). The second sigma is pinned to its
    /// exact F32, which is the value the reference's own F32 schedule produces
    /// and the one the dumped step is consistent with.
    #[test]
    fn flux_schedule_matches_the_reference_log() {
        assert!((flux_mu(256) - 0.5).abs() < 1e-6, "mu {}", flux_mu(256));

        let sigmas = flux_sigmas(256, 2);
        assert_eq!(sigmas.len(), 3);
        assert_eq!(sigmas[0], 1.0);
        assert_eq!(sigmas[1], f32::from_bits(0x3f1f_597f), "sigma1 {}", sigmas[1]);
        assert_eq!(sigmas[1] - sigmas[0], f32::from_bits(0xbec1_4d02));
        assert_eq!(sigmas[2], 0.0);
    }

    /// At 4096 image tokens the anchors put mu at max_shift.
    #[test]
    fn flux_mu_hits_both_anchors() {
        assert!((flux_mu(256) - FLUX_BASE_SHIFT).abs() < 1e-6);
        assert!((flux_mu(4096) - FLUX_MAX_SHIFT).abs() < 1e-6);
    }

    /// The DiT's time input is the sigma on the [0, 1000] scale
    /// (`qwen_image_2_1.hpp:268`).
    #[test]
    fn flow_timestep_is_sigma_on_the_thousand_scale() {
        assert_eq!(flow_timestep(1.0), 1000.0);
        assert!((flow_timestep(0.62245933) - 622.45933).abs() < 1e-3);
    }

    /// A text-to-image layout: one text segment, then the image grid. The last
    /// image id carries the temporal position past the prompt, and the spatial
    /// ids are centered.
    #[test]
    fn layout_places_text_then_the_centered_image_grid() {
        let layout = build_layout(3, &[], &[(2, 2)]).expect("layout");

        assert_eq!(layout.prefix_length, 3);
        assert_eq!(layout.segments.len(), 2);
        assert_eq!(layout.segments[0], Segment { start: 0, end: 3, context_start: 0, image_index: -1 });
        assert_eq!(layout.segments[1], Segment { start: 3, end: 7, context_start: 3, image_index: 0 });

        assert_eq!(layout.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(layout.positions[2], [2.0, 2.0, 2.0]);
        // h - (h - h/2) with h=2 is h-1, so the ids run -1..0.
        assert_eq!(layout.positions[3], [3.0, -1.0, -1.0]);
        assert_eq!(layout.positions[6], [3.0, 0.0, 0.0]);
        assert_eq!(layout.positions.len(), 7);
    }

    /// An image is rejected unless a latent of the matching grid follows every
    /// vision-slot run (`qwen_image_2_1.hpp:95-99`). A slot run with no latent
    /// left to claim fails the size check, not the coverage check, because the
    /// reference tests the grid before it indexes it.
    #[test]
    fn layout_rejects_unmatched_slots() {
        assert_eq!(build_layout(3, &[], &[]), Err(LayoutError::Invalid));
        assert_eq!(build_layout(3, &[], &[(2, 2), (2, 2)]), Err(LayoutError::MissingRefSlots));
        assert_eq!(build_layout(3, &[0, 1, 0], &[(2, 2)]), Err(LayoutError::SlotShapeMismatch));
        assert_eq!(build_layout(4, &[0, 1, 1, 1], &[(2, 2)]), Err(LayoutError::SlotShapeMismatch));
    }

    /// A vision-slot run claims the reference latent whose grid fits it, four
    /// image tokens per slot (`qwen_image_2_1.hpp:92-99`).
    #[test]
    fn layout_accepts_a_matched_reference_grid() {
        let slots = [1, 1];
        let layout = build_layout(2, &slots, &[(2, 4), (2, 2)]).expect("layout");

        // The slots are the whole prompt, so there is no text prefix and the
        // reference grid starts the sequence.
        assert_eq!(layout.prefix_length, 8);
        assert_eq!(layout.segments.len(), 2);
        assert_eq!(layout.segments[0], Segment { start: 0, end: 8, context_start: 0, image_index: 0 });
        assert_eq!(layout.segments[1], Segment { start: 8, end: 12, context_start: 2, image_index: 1 });
        assert_eq!(layout.positions[0], [0.0, -1.0, -2.0]);
        // The latent being denoised continues the temporal id past max(2, 4).
        assert_eq!(layout.positions[8], [4.0, -1.0, -1.0]);
    }

    /// The causal mask covers the lower triangle; the image segment has none.
    #[test]
    fn text_mask_is_lower_triangular() {
        let mask = text_mask(3);
        let visible = |q: usize, k: usize| mask[q * 3 + k].is_finite();

        assert!(visible(0, 0));
        assert!(!visible(0, 1));
        assert!(visible(2, 1));
        assert!(visible(2, 2));
    }

    /// A rope pair is a rotation: the pair's norm survives it.
    #[test]
    fn apply_rope_preserves_the_pair_norm() {
        let positions = vec![[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]];
        let table = rope_table(&positions, &DIT_AXES_DIM, DIT_ROPE_THETA);
        let dim_head = 128usize;
        let x: Vec<f32> = (0..2 * dim_head).map(|i| (i % 7) as f32 - 3.0).collect();

        for pairing in [RopePairing::Interleaved, RopePairing::HalfSplit] {
            let mut rotated = x.clone();
            apply_rope(&mut rotated, &table, 2, 1, dim_head, pairing);

            for pos in 0..2 {
                for j in 0..(dim_head / 2) {
                    let (d0, d1) = match pairing {
                        RopePairing::Interleaved => (2 * j, 2 * j + 1),
                        RopePairing::HalfSplit => (j, j + dim_head / 2),
                    };
                    let before = [x[pos * dim_head + d0], x[pos * dim_head + d1]];
                    let after = [rotated[pos * dim_head + d0], rotated[pos * dim_head + d1]];
                    let norm = |v: [f32; 2]| (v[0] * v[0] + v[1] * v[1]).sqrt();
                    assert!((norm(before) - norm(after)).abs() < 1e-5, "pair {j} not norm preserving");
                }
            }
        }
    }

    /// The two pairings differ, so the DiT probe can tell them apart.
    #[test]
    fn the_pairings_are_distinguishable() {
        let positions = vec![[0.5, 0.25, 0.75]];
        let table = rope_table(&positions, &DIT_AXES_DIM, DIT_ROPE_THETA);
        let x: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();

        let mut interleaved = x.clone();
        let mut half_split = x.clone();
        apply_rope(&mut interleaved, &table, 1, 1, 128, RopePairing::Interleaved);
        apply_rope(&mut half_split, &table, 1, 1, 128, RopePairing::HalfSplit);

        assert!(interleaved.iter().zip(&half_split).any(|(a, b)| (a - b).abs() > 1e-3));
    }

    /// The omega ladder is `theta^-scale`, so pair 0 is unrotated and the
    /// fastest pair turns once per `theta` positions.
    #[test]
    fn rope_omega_follows_the_theta_ladder() {
        let omega = rope_omega(16, 10000.0);

        assert_eq!(omega.len(), 8);
        assert_eq!(omega[0], 1.0);
        assert!((omega[7] - 1.0 / 10000f32.powf(14.0 / 16.0)).abs() < 1e-12);
    }

    /// The Euler step keeps the reference's operation order. At the terminal
    /// sigma that is visible without a fixture: the step lands on the denoised
    /// prediction itself, because `sigma_next - sigma` is `-sigma` and
    /// `d * -sigma` cancels the `x - denoised`.
    #[test]
    fn euler_step_lands_on_the_denoised_prediction_at_the_terminal_sigma() {
        let x = [0.5f32, -1.25, 3.0, 0.125];
        let pred = [0.25f32, 0.75, -2.0, 0.0625];
        let sigma = 1.0f32;

        let stepped = euler_step(&x, &pred, sigma, 0.0);
        let denoised: Vec<f32> = x.iter().zip(&pred).map(|(x, p)| x - sigma * p).collect();

        assert_eq!(stepped, denoised);
    }

    /// The first call's offset is zero, and a second call advances it, so the
    /// stream is a sequence and not a function of the index alone.
    #[test]
    fn philox_stream_advances_by_call() {
        let mut rng = Philox::new(42);
        let first = rng.randn(4);
        let second = rng.randn(4);

        assert_eq!(rng.seed(), 42);
        assert_ne!(first, second);
        assert_eq!(first, initial_noise(4, 42));
        assert!(first.iter().all(|v| v.is_finite() && v.abs() < 6.0));
    }
}
