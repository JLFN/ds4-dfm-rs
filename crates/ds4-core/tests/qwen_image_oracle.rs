//! P1 oracle parity against the reference's own intermediates.
//!
//! The reference binary can dump what it computes: `SD_DUMP_IDS` prints the
//! token ids, `SD_DUMP_COND=<prefix>` writes `<prefix>.<n>.bin` for the
//! conditioning, and `SD_DUMP_STEPS=<prefix>` writes `<prefix>.step<n>.in.bin`
//! and `<prefix>.step<n>.pred.bin` per sampling step. Each file is
//! `dims <n> <d0> <d1> ...\n` followed by raw little-endian F32 in GGML order
//! (ne0 fastest).
//!
//! The dumps are fixtures on this host, never committed:
//!
//!     DS4_QWEN_IMAGE_ORACLE=<prefix> cargo test -p ds4-core --test qwen_image_oracle
//!
//! Stage 2 (the DiT forward) additionally needs the artifact, and supports two
//! diagnostics: `DS4_QWEN_IMAGE_DIT_OUT=<prefix>` writes the per-pass velocities
//! in the reference's dump format, and `DS4_QWEN_IMAGE_EXPORT=<tensor>` with
//! `DS4_QWEN_IMAGE_EXPORT_OUT=<file>` writes one dequantized tensor so an
//! independent dequantizer can be diffed against it.
//!
//!     DS4_QWEN_IMAGE_ORACLE=<prefix> DS4_QWEN_IMAGE_DIT=<gguf> \
//!         cargo test -p ds4-core --release --test qwen_image_oracle
//!
//! With the variables unset every test returns early, so the suite stays
//! model-free by default, in the style the other families use.

use std::path::PathBuf;

use ds4_core::qwen_image::dit::{cfg_combine, forward, parity, DitPass, DitWeights};
use ds4_core::qwen_image::oracle::{
    apply_rope, build_layout, euler_step, flux_sigmas, flow_timestep, initial_noise,
    noise_scaling, rope_table, text_mask, Philox, RopePairing,
};
use ds4_core::qwen_image::vae::{decode, diffusion_to_vae, to_rgba8, Plane, VaeWeights};
use ds4_core::qwen_image::DIT_AXES_DIM;

/// A dumped tensor: its dims on the GGML axes (ne0 first) and its data.
struct Dump {
    dims: Vec<u64>,
    data: Vec<f32>,
}

impl Dump {
    fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// Reads one `<prefix>...bin` dump. Panics with the path, so a wrong fixture
/// says which file was wrong.
fn read_dump(path: &PathBuf) -> Dump {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let newline = raw.iter().position(|b| *b == b'\n').expect("dims header");
    let header = std::str::from_utf8(&raw[..newline]).expect("utf8 header");

    let fields: Vec<&str> = header.split(' ').collect();
    assert_eq!(fields[0], "dims", "{}: not a dump file", path.display());

    let dims: Vec<u64> = fields[2..].iter().map(|f| f.parse().expect("dim")).collect();
    assert_eq!(fields[1].parse::<usize>().expect("rank"), dims.len(), "rank");

    let body = &raw[newline + 1..];
    assert_eq!(body.len() % 4, 0, "{}: not F32", path.display());
    let data: Vec<f32> = body
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let dump = Dump { dims, data };
    assert_eq!(dump.data.len() as u64, dump.elements(), "{}: payload size", path.display());
    dump
}

fn fixture(prefix: &str, suffix: &str) -> Option<(PathBuf, Dump)> {
    let ok = std::env::var("DS4_QWEN_IMAGE_ORACLE").ok()?;
    let path = PathBuf::from(format!("{ok}{suffix}"));
    let dump = read_dump(&path);
    assert!(prefix.is_empty() || path.to_string_lossy().contains(prefix), "fixture path");
    Some((path, dump))
}

/// The reference's initial noise is the oracle's, bit for bit.
///
/// `--rng cuda` selects Philox (`rng_philox.hpp`); the first `randn` call uses
/// counter offset 0, so the latent's flat element order is the counter's lane-2
/// index. `step1.in.bin` is `x * c_in` with `c_in = 1` at the first sigma, so
/// the dump is that noise unscaled.
#[test]
fn initial_noise_matches_the_reference_bit_for_bit() {
    let Some((path, reference)) = fixture("", ".step1.in.bin") else {
        return;
    };
    let noise = initial_noise(reference.data.len(), 42);

    let differing = noise.iter().zip(&reference.data).filter(|(a, b)| a != b).count();
    let worst = noise
        .iter()
        .zip(&reference.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert_eq!(
        differing, 0,
        "{}: {differing}/{} differ from the reference, worst {worst:e}",
        path.display(),
        reference.data.len()
    );
    assert_eq!(reference.dims, vec![16, 16, 64, 1], "fixture latent");
}

/// The schedule and the Euler step reproduce the reference's second step input
/// from its first, bit for bit.
///
/// The reference's order matters: it denoises first and takes the velocity from
/// that (`denoised = x - sigma*pred`, `d = (x - denoised)/sigma`,
/// `x += d*(sigma_next - sigma)`), which is not the same F32 as the
/// one-expression `x + pred*(sigma_next - sigma)`. Mirroring it makes the step
/// exact rather than close; the test asserts both, so a future simplification
/// cannot quietly cost the exactness.
#[test]
fn the_flow_schedule_and_euler_step_reproduce_the_reference() {
    let Some((_, x0)) = fixture("", ".step1.in.bin") else {
        return;
    };
    let Some((_, pred0)) = fixture("", ".step1.pred.bin") else {
        return;
    };
    let Some((path, x1)) = fixture("", ".step2.in.bin") else {
        return;
    };

    let sigmas = flux_sigmas(16 * 16, 2);
    assert_eq!(sigmas[0], 1.0, "the first sigma is 1, so c_in is 1");

    let stepped = euler_step(&x0.data, &pred0.data, sigmas[0], sigmas[1]);
    let differing = stepped.iter().zip(&x1.data).filter(|(a, b)| a != b).count();

    let simplified: Vec<f32> = x0
        .data
        .iter()
        .zip(&pred0.data)
        .map(|(x, d)| x + d * (sigmas[1] - sigmas[0]))
        .collect();
    let simplified_differing = simplified.iter().zip(&x1.data).filter(|(a, b)| a != b).count();

    assert_eq!(
        differing, 0,
        "{}: {differing}/{} values differ from the reference's next latent",
        path.display(),
        x1.data.len()
    );
    assert!(
        simplified_differing > 0,
        "the one-expression form matched too: the reference's order is then not what this test claims"
    );

    assert_eq!(flow_timestep(sigmas[0]), 1000.0);
    assert_eq!(flow_timestep(sigmas[1]), sigmas[1] * 1000.0);
}

/// The conditioning dump is `[4096, text_length]`, and the layout built from it
/// is the reference's: a text prefix, then a 16x16 image grid.
#[test]
fn conditioning_and_layout_match_the_reference_shapes() {
    let Some((_, cond)) = fixture("", ".cond.0.bin") else {
        return;
    };
    let Some((_, uncond)) = fixture("", ".cond.1.bin") else {
        return;
    };

    assert_eq!(cond.dims, vec![4096, 15]);
    assert_eq!(uncond.dims, vec![4096, 9]);

    let latent = read_dump(&PathBuf::from(format!(
        "{}.step1.in.bin",
        std::env::var("DS4_QWEN_IMAGE_ORACLE").unwrap()
    )));
    let (height, width) = (latent.dims[1], latent.dims[0]);

    let layout = build_layout(cond.dims[1] as i64, &[], &[(height as i64, width as i64)])
        .expect("layout");
    assert_eq!(layout.prefix_length, 15);
    assert_eq!(layout.positions.len() as i64, 15 + (height * width) as i64);
    assert_eq!(layout.segments[1].end - layout.segments[1].start, (height * width) as i64);

    // The first image token carries the prompt's length as its temporal id.
    assert_eq!(layout.positions[15], [15.0, -8.0, -8.0]);
}

/// The uncond pass uses the shorter prefix, so the pair of passes differ in
/// sequence length: the reference builds one layout per pass
/// (`qwen_image_2_1.hpp:340-357`).
#[test]
fn both_conditioning_lengths_build_valid_layouts() {
    let Some((_, cond)) = fixture("", ".cond.0.bin") else {
        return;
    };
    let Some((_, uncond)) = fixture("", ".cond.1.bin") else {
        return;
    };

    for (name, text) in [("cond", cond.dims[1]), ("uncond", uncond.dims[1])] {
        let layout = build_layout(text as i64, &[], &[(16, 16)]).expect(name);
        assert_eq!(layout.prefix_length, text as i64);

        let mask = text_mask(text as usize);
        assert_eq!(mask.len(), (text * text) as usize);
        assert!(mask[0].is_finite());
    }
}

/// The rope table's cos/sin are the reference's own frequencies, and the
/// interleaved application is a rotation of adjacent pairs.
#[test]
fn rope_table_matches_the_reference_frequencies() {
    if std::env::var("DS4_QWEN_IMAGE_ORACLE").is_err() {
        return;
    }

    let positions = vec![[0.0f32, 0.0, 0.0], [1.0, 2.0, 3.0]];
    let table = rope_table(&positions, &DIT_AXES_DIM, 10000.0);

    let pairs = (DIT_AXES_DIM.iter().map(|d| d / 2).sum::<u64>()) as usize;
    assert_eq!(table.len(), 4 * pairs * positions.len());

    // Position 0 is unrotated: every pair is [1, 0, 0, 1].
    for j in 0..pairs {
        assert_eq!(table[4 * j], 1.0);
        assert_eq!(table[4 * j + 1], 0.0);
        assert_eq!(table[4 * j + 2], 0.0);
        assert_eq!(table[4 * j + 3], 1.0);
    }

    // The first pair of axis 0 has omega 1, so its angle is the temporal id.
    let block = 4 * pairs;
    assert!((table[block] - 1.0f32.cos()).abs() < 1e-7);
    assert!((table[block + 2] - 1.0f32.sin()).abs() < 1e-7);

    // Pair 0 is (1, 1) at the temporal id 1, so the rotation gives
    // (cos - sin, sin + cos).
    // Position 0 is unrotated, so the first head is untouched; position 1 has
    // the temporal id 1 and pair 0 of axis 0 turns by exactly that.
    let angle = 1.0f32;
    let mut x = vec![1.0f32; 2 * 128];
    apply_rope(&mut x, &table, 2, 1, 128, RopePairing::Interleaved);
    assert_eq!(&x[..128], vec![1.0f32; 128].as_slice());
    assert!((x[128] - (angle.cos() - angle.sin())).abs() < 1e-6);
    assert!((x[129] - (angle.sin() + angle.cos())).abs() < 1e-6);
}

/// The noise scaling at sigma 1 is the identity and at the last sigma is a
/// plain scale, which is what makes the first dump the raw noise.
#[test]
fn noise_scaling_is_identity_at_the_first_sigma() {
    let noise: Vec<f32> = (0..16).map(|i| i as f32 * 0.25 - 2.0).collect();

    assert_eq!(noise_scaling(1.0, &noise, None), noise);

    let last: Vec<f32> = noise.iter().map(|n| n * 0.25).collect();
    assert_eq!(noise_scaling(0.25, &noise, None), last);
}

/// A Philox stream is a sequence: two seeds and two offsets give four distinct
/// draws, so the fixture cannot be matched by a different call pattern.
#[test]
fn philox_is_seed_and_call_position_dependent() {
    let a = Philox::new(42).randn(8);
    let b = Philox::new(43).randn(8);
    let mut rng = Philox::new(42);
    let first = rng.randn(8);
    let second = rng.randn(8);

    assert_ne!(a, b);
    assert_eq!(a, first);
    assert_ne!(first, second);
}

// ---------------------------------------------------------------------------
// Stage 2: the DiT forward
// ---------------------------------------------------------------------------

/// The artifact the stage-2 gate evaluates, from `DS4_QWEN_IMAGE_DIT`.
fn dit_weights() -> Option<DitWeights> {
    let path = std::env::var("DS4_QWEN_IMAGE_DIT").ok()?;
    Some(DitWeights::open(&PathBuf::from(path)).expect("open the DiT artifact"))
}

/// Writes one tensor in the reference dump's own format, so a scratch check can
/// diff it without this crate knowing how to read the format back.
fn write_dump(path: &str, dims: &[u64], values: &[f32]) {
    let header: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    let mut bytes = format!("dims {} {}\n", dims.len(), header.join(" ")).into_bytes();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write dump");
}

/// One CFG-combined velocity for a pairing: both guidance passes, then the
/// reference's own combination (`guidance.cpp:171`, `--cfg-scale 6.0`).
fn cfg_velocity(
    weights: &DitWeights,
    cond: &Dump,
    uncond: &Dump,
    latent: &Dump,
    timestep: f32,
    pairing: RopePairing,
) -> Vec<f32> {
    let (height, width) = (latent.dims[1] as usize, latent.dims[0] as usize);
    let evaluate = |context: &Dump| {
        forward(
            weights,
            &DitPass {
                latent: &latent.data,
                height,
                width,
                context: &context.data,
                text_length: context.dims[1] as usize,
                timestep,
                pairing,
            },
        )
        .expect("the DiT forward")
    };

    let cond_pred = evaluate(cond);
    let uncond_pred = evaluate(uncond);
    assert_eq!(cond_pred.len(), latent.data.len());

    let combined = cfg_combine(&cond_pred, &uncond_pred, 6.0);

    if let Ok(prefix) = std::env::var("DS4_QWEN_IMAGE_DIT_OUT") {
        let name = match pairing {
            RopePairing::Interleaved => "interleaved",
            RopePairing::HalfSplit => "halfsplit",
        };
        for (part, values) in
            [("cond", &cond_pred), ("uncond", &uncond_pred), ("cfg", &combined)]
        {
            write_dump(&format!("{prefix}.{name}.{part}.bin"), &latent.dims, values);
        }
    }
    combined
}

/// Stage 2's gate: the reference's own step-1 velocity, and the probe that
/// settles the RoPE pairing by measurement rather than by reading
/// (`rope_interleaved` defaults to true but these weights were trained
/// elsewhere).
///
/// `pred` is the model output itself: `DiscreteFlowDenoiser` has `c_skip = 1`
/// and `c_out = -sigma`, so `denoised = x - sigma*model_out` and the Euler
/// velocity `d = (x - denoised)/sigma` is `model_out` — which is why the CFG
/// combination of the two passes is what the dump holds.
#[test]
fn dit_forward_reproduces_the_reference_step_velocity() {
    if std::env::var("DS4_QWEN_IMAGE_ORACLE").is_err() {
        return;
    }
    let Some(weights) = dit_weights() else {
        return;
    };

    let (_, cond) = fixture("", ".cond.0.bin").expect("cond dump");
    let (_, uncond) = fixture("", ".cond.1.bin").expect("uncond dump");
    let (_, latent) = fixture("", ".step1.in.bin").expect("latent dump");
    let (path, reference) = fixture("", ".step1.pred.bin").expect("pred dump");

    let image_tokens = (latent.dims[0] * latent.dims[1]) as i64;
    let sigmas = flux_sigmas(image_tokens, 2);
    let timestep = flow_timestep(sigmas[0]);
    assert_eq!(timestep, 1000.0, "the first sigma is 1, so the DiT sees 1000");

    let mut measured = Vec::new();
    for pairing in [RopePairing::Interleaved, RopePairing::HalfSplit] {
        let start = std::time::Instant::now();
        let velocity = cfg_velocity(&weights, &cond, &uncond, &latent, timestep, pairing);
        let elapsed = start.elapsed().as_secs_f32();
        let result = parity(&velocity, &reference.data);

        println!(
            "{}: correlation {:.6}, relative RMS {:.4e}, max |diff| {:.4e}, {:.1}s",
            match pairing {
                RopePairing::Interleaved => "interleaved",
                RopePairing::HalfSplit => "half-split",
            },
            result.correlation,
            result.relative_rms,
            result.max_abs,
            elapsed
        );
        measured.push((pairing, result));
    }

    let (winner, best) =
        measured.iter().copied().max_by(|a, b| a.1.correlation.total_cmp(&b.1.correlation)).expect("two pairings");
    let (loser, worst) = measured.iter().copied().find(|(p, _)| *p != winner).expect("the other pairing");

    assert_eq!(winner, RopePairing::Interleaved, "the pairing this port applies");
    assert!(
        best.correlation > 0.999,
        "{}: correlation {:.6} against {}",
        path.display(),
        best.correlation,
        best.relative_rms
    );
    assert!(best.relative_rms < 0.02, "{}: relative RMS {:.4e}", path.display(), best.relative_rms);
    assert!(
        best.correlation - worst.correlation > 0.05,
        "the pairings are not distinguishable: {winner:?} {:.6} vs {loser:?} {:.6}",
        best.correlation,
        worst.correlation
    );
}

/// Scratch: writes one tensor's dequantized F32 for the independent
/// cross-check against the `gguf` Python dequantizer. Gated, like the rest.
#[test]
fn export_tensor_for_crosscheck() {
    let (Ok(path), Ok(name), Ok(out)) = (
        std::env::var("DS4_QWEN_IMAGE_DIT"),
        std::env::var("DS4_QWEN_IMAGE_EXPORT"),
        std::env::var("DS4_QWEN_IMAGE_EXPORT_OUT"),
    ) else {
        return;
    };

    let weights = DitWeights::open(&PathBuf::from(path)).expect("open");
    let values = weights.dequantize_f32(&name).expect("dequantize");

    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(out, bytes).expect("write");
}

// ---------------------------------------------------------------------------
// Stage 3: the VAE decode
// ---------------------------------------------------------------------------

/// The stage-3 gate: the reference's own PNG, rebuilt from its dumps.
///
/// The chain is the reference's: the second Euler step lands on the final latent
/// (its next sigma is the terminal zero, so the step is the denoised
/// prediction), the 64-channel statistics move it into the VAE's own space
/// (`diffusion_to_vae_latents`, `wan_vae.hpp:1393-1396`), the decoder runs, and
/// the result is scaled to [0,1] and quantized the way `VAE::decode` and
/// `tensor_to_sd_image` do it.
///
/// Fidelity class: stated tolerance. The reference ran its VAE on the CPU
/// (`vae=cpu`, `run1.log:62`) with F16 parameters and F16 conv operands summed
/// in ggml's own order, so this decode lands close, not byte-exact. The adopted
/// bound is a 6/255 per-channel maximum with PSNR above 55 dB, one final
/// quantization step plus the F32 accumulation-order slack that this decoder's
/// own unstable cascade amplifies: the latent drives the reference past F16's
/// range, so ~6% of its pixels overflow to infinity and NaN (both
/// implementations write those white) and the rest agree to a byte or two.
#[test]
fn vae_decode_reproduces_the_reference_image() {
    if std::env::var("DS4_QWEN_IMAGE_ORACLE").is_err() {
        return;
    }
    let Some(vae_path) = std::env::var("DS4_QWEN_IMAGE_VAE").ok() else {
        return;
    };

    let (_, step1_in) = fixture("", ".step1.in.bin").expect("step1 input dump");
    let (_, step1_pred) = fixture("", ".step1.pred.bin").expect("step1 velocity dump");
    let (_, step2_in) = fixture("", ".step2.in.bin").expect("step2 input dump");
    let (_, step2_pred) = fixture("", ".step2.pred.bin").expect("step2 velocity dump");

    let image_tokens = (step2_in.dims[0] * step2_in.dims[1]) as i64;
    let sigmas = flux_sigmas(image_tokens, 2);

    // The stage-1 parity, restated because everything downstream stands on the
    // latent it produces.
    let rebuilt = euler_step(&step1_in.data, &step1_pred.data, sigmas[0], sigmas[1]);
    assert_eq!(rebuilt, step2_in.data, "the first Euler step");

    let final_latent = euler_step(&step2_in.data, &step2_pred.data, sigmas[1], sigmas[2]);
    let latent = Plane::new(
        step2_in.dims[0] as usize,
        step2_in.dims[1] as usize,
        step2_in.dims[2] as usize,
        final_latent,
    );

    let weights = VaeWeights::open(&PathBuf::from(vae_path)).expect("open the VAE artifact");
    let vae_latent = diffusion_to_vae(&latent).expect("latent statistics");

    let start = std::time::Instant::now();
    let decoded = decode(&weights, &vae_latent, 1).expect("the VAE decode");
    println!(
        "vae decode: {}x{}x{} in {:.1}s",
        decoded.width(),
        decoded.height(),
        decoded.channels(),
        start.elapsed().as_secs_f32()
    );

    let ours = to_rgba8(&decoded);
    let reference_path = PathBuf::from(format!(
        "{}.png",
        std::env::var("DS4_QWEN_IMAGE_ORACLE").unwrap()
    ));
    let reference = image::ImageReader::open(&reference_path)
        .expect("open the reference PNG")
        .decode()
        .expect("decode the reference PNG")
        .to_rgba8();

    assert_eq!(
        (reference.width(), reference.height()),
        (decoded.width() as u32, decoded.height() as u32),
        "the reference PNG's geometry"
    );
    let reference = reference.into_raw();
    assert_eq!(ours.len(), reference.len());

    // Per-channel maxima and a whole-image PSNR, all on the [0,1] scale the
    // reference's own u8 conversion uses.
    let mut max_diff = [0f32; 4];
    let mut above_one = [0usize; 4];
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    for (index, (a, b)) in ours.iter().zip(&reference).enumerate() {
        let channel = index % 4;
        let diff = (*a as f32 - *b as f32).abs() / 255.0;
        max_diff[channel] = max_diff[channel].max(diff);
        if *a != *b {
            above_one[channel] += 1;
        }
        diff_sq += (diff as f64) * (diff as f64);
        ref_sq += (*b as f64 / 255.0) * (*b as f64 / 255.0);
    }
    let mse = diff_sq / ours.len() as f64;
    let psnr = 10.0 * (1.0 / mse).log10();
    let relative_rms = (diff_sq / ref_sq).sqrt();

    println!(
        "decode vs reference: max |diff| r {:.5} g {:.5} b {:.5} a {:.5}, PSNR {:.1} dB, relative RMS {:.4e}, \
         differing bytes r {} g {} b {} a {} of {}",
        max_diff[0], max_diff[1], max_diff[2], max_diff[3], psnr, relative_rms,
        above_one[0], above_one[1], above_one[2], above_one[3], ours.len() / 4
    );

    let worst = max_diff.iter().copied().fold(0f32, f32::max);
    assert!(
        worst <= 6.0 / 255.0,
        "{}: worst channel deviation {worst:.5} ({:.2} of 255)",
        reference_path.display(),
        worst * 255.0
    );
    assert!(psnr >= 55.0, "{}: PSNR {psnr:.1} dB", reference_path.display());
}
