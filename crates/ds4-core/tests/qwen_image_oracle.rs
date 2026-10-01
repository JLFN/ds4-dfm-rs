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
//! With the variable unset every test returns early, so the suite stays
//! model-free by default, in the style the other families use.

use std::path::PathBuf;

use ds4_core::qwen_image::oracle::{
    apply_rope, build_layout, euler_step, flux_sigmas, flow_timestep, initial_noise,
    noise_scaling, rope_table, text_mask, Philox, RopePairing,
};
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
