//! Unit 12's gate: the LTX-2.5 video VAE decoder.
//!
//! Usage:
//!   h3_ltx_vae_gate --probe [--seed S]
//!   h3_ltx_vae_gate --vae PATH --decode-shape F,H,W [--latent FILE]
//!                   [--frame FILE] [--seed S] [--expect HEX]
//!   h3_ltx_vae_gate --shapes
//!
//! Three modes:
//!
//! - `--probe`: each new device kernel against a host transcription of the
//!   reference's own expression, element by element. The transcriptions
//!   reproduce the kernel's rounding chain, so `denormalize`, `pixel_norm`,
//!   `depth_to_space`, `unpatchify` and `to_rgb` must come back **bit
//!   identical** and are reported as exact; the convolution's transcription
//!   cannot be (its reduction order inside a tap is cuBLASLt's) and is
//!   reported as the residue it is, with the number. Two controls prove the
//!   convolution probe can fail: a zero-padded transcription and one that
//!   reuses the first plan's input must both differ from the device's result.
//! - default (`--vae`): the whole decoder over the artifact's own weights, on
//!   a synthetic latent or one read from `--latent` (`[channel][frame]
//!   [height][width]` BF16, the file `h3_ltx_adapter_gate` writes). It prints
//!   the plan's stage shapes, the working set the untiled decode costs, the
//!   decode's hash and its first and last values, and writes the first frame
//!   as a PNG when `--frame` names one.
//! - `--shapes`: the geometry contract alone, no device: the artifact config's
//!   block list against the port's table, the scale factors, and the
//!   release's `(121, 768, 1344)` against its `(16, 24, 42)`.
//!
//! There is no Python in this campaign and the pinned reference does not run
//! on either host, so this gate is the same shape as the refiner's: a second
//! implementation of the reference's own expressions, cited line by line to
//! `ltx-core/src/ltx_core/model/video_vae/`.
//!
//! Exit 3 when a probe or a comparison fails.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::ltx_vae::{
    latent_shape, pixel_shape, release_latent, scale_factors, scratch_plan, Block, LtxVae, Shape,
    DECODER_BLOCKS, LATENT_CHANNELS, PIXEL_NORM_EPSILON, RELEASE_LATENT, RELEASE_PIXELS,
};
use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

/// What a transcribed convolution may differ by: same inputs, f32 accumulation
/// in both, only the order inside a tap differs.
const CONV_RELATIVE: f64 = 1e-3;
/// The ceiling a *zero-padded* transcription is expected to exceed, so a probe
/// that cannot tell the two halo policies apart fails loudly.
const HALO_SEPARATION: f64 = 1e-2;

struct Options {
    probe: bool,
    shapes: bool,
    impulse: bool,
    vae: Option<PathBuf>,
    latent: Option<PathBuf>,
    frame: Option<PathBuf>,
    decode_shape: Option<[u32; 3]>,
    seed: u64,
    expect: Option<u64>,
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();

    match run(&arguments) {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(1)
        }
    }
}

fn run(arguments: &[String]) -> Result<ExitCode, String> {
    let options = split(arguments)?;

    if options.shapes {
        return shapes();
    }

    let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;

    if options.probe {
        let failed = probe(&gpu, options.seed)?;

        return Ok(ExitCode::from(if failed { 3 } else { 0 }));
    }

    let vae = options
        .vae
        .clone()
        .ok_or_else(|| usage("--vae is required"))?;
    let decode = options
        .decode_shape
        .ok_or_else(|| usage("--decode-shape is required"))?;

    decode_vae(&gpu, &vae, decode, &options)
}

/// The geometry contract, cited: the artifact config's block list (`:80-94` of
/// the configurator), the scale factors the blocks imply (`types.py:41-57`),
/// and the release's own pixel target (`models.py:20-22`).
fn shapes() -> Result<ExitCode, String> {
    let factors = scale_factors();
    println!(
        "SCALE time={} height={} width={}",
        factors[0], factors[1], factors[2]
    );

    for (index, block) in DECODER_BLOCKS.iter().enumerate() {
        match block {
            Block::Res { layers } => println!("BLOCK {index} res_x layers={layers}"),
            Block::Upsample { stride, reduction } => println!(
                "BLOCK {index} compress stride={}x{}x{} reduction={reduction}",
                stride[0], stride[1], stride[2]
            ),
        }
    }

    let latent = latent_shape(RELEASE_PIXELS);
    let pixel = pixel_shape(RELEASE_LATENT);
    println!(
        "GEOMETRY pixels={}x{}x{} latent={}x{}x{} roundtrip={}x{}x{}",
        RELEASE_PIXELS[0],
        RELEASE_PIXELS[1],
        RELEASE_PIXELS[2],
        latent[0],
        latent[1],
        latent[2],
        pixel[0],
        pixel[1],
        pixel[2]
    );

    if latent != RELEASE_LATENT || pixel != RELEASE_PIXELS {
        println!("GATE FAIL the release geometry does not round-trip");
        return Ok(ExitCode::from(3));
    }

    // The untiled decode's own footprint at that geometry, which is the one
    // number this host cannot measure (15 GiB of scratch against a 12 GiB
    // card) and the Spark can.
    let plan = scratch_plan(release_latent());
    let total: u64 = plan.iter().map(|(_, bytes)| bytes).sum();
    for (name, bytes) in plan {
        println!("SCRATCH {name} bytes={bytes}");
    }
    println!("SCRATCH total bytes={total}");

    Ok(ExitCode::from(0))
}

/// The whole decoder: load the artifact, decode, report.
fn decode_vae(
    gpu: &Gpu,
    path: &Path,
    decode: [u32; 3],
    options: &Options,
) -> Result<ExitCode, String> {
    let shape = Shape {
        channels: LATENT_CHANNELS,
        frames: decode[0],
        height: decode[1],
        width: decode[2],
    };

    let started = Instant::now();
    let vae = LtxVae::open(gpu, path, shape).map_err(|error| error.to_string())?;
    let load = started.elapsed().as_secs_f64();
    let (tensors, bytes) = vae.loaded();

    println!(
        "VAE tensors={tensors} bytes={bytes} load={load:.3} scratch={}",
        vae.scratch_bytes()
    );
    println!(
        "LATENT channels={} frames={} height={} width={} source={}",
        shape.channels,
        shape.frames,
        shape.height,
        shape.width,
        if options.latent.is_some() {
            "file"
        } else {
            "synthetic"
        }
    );

    let latent = match &options.latent {
        Some(file) => read_latent(gpu, file, shape)?,
        None => {
            let mut state = options.seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            draw(gpu, shape.elements(), &mut state)?
        }
    };

    let started = Instant::now();
    let rgb = vae
        .decode(gpu, &latent, shape)
        .map_err(|error| error.to_string())?;
    let seconds = started.elapsed().as_secs_f64();

    let pixel = pixel_shape([shape.frames, shape.height, shape.width]);
    let values = read_bf16(gpu, rgb)?;
    let hash = fnv(&values);
    println!(
        "DECODE frames={} height={} width={} values={} seconds={seconds:.3} fnv={hash:016x} \
         first={:.9} last={:.9}",
        pixel[0],
        pixel[1],
        pixel[2],
        values.len(),
        values[0],
        values[values.len() - 1]
    );

    // The reference clamps to `[0, 1]`, so a value outside it means the fold
    // is not the reference's (`conv_video_decoder.py:499`).
    let mut outside = 0usize;
    for value in &values {
        if !(0.0..=1.0).contains(value) {
            outside += 1;
        }
    }
    println!("RANGE outside={outside}");

    let mut failed = outside != 0;

    if let Some(expect) = options.expect {
        let pass = expect == hash;
        println!(
            "EXPECT fnv={hash:016x} want={expect:016x} {}",
            if pass { "pass" } else { "FAIL" }
        );
        failed |= !pass;
    }

    if let Some(file) = &options.frame {
        write_frame(file, &values, pixel)?;
    }

    if options.impulse {
        impulse_check(gpu, &vae, shape, pixel)?;
    }

    Ok(ExitCode::from(if failed { 3 } else { 0 }))
}

/// Where a single latent cell lands in the frame, as a measurement rather than
/// a gate. A latent that is zero everywhere but one channel at one spatial
/// cell, at every frame, gives an impulse response whose centroid stands at
/// that cell's own block; moving the cell one step along an axis should move
/// the centroid by the scale factor, 32 pixels = the patch size times the three
/// spatial doublings, and the cross axes should not move.
///
/// It does not, and the measurement is kept because of what that says: the
/// response's centroid moves only ~0.7 of a cell per cell (section 3 of the
/// record). The observation is that this decoder's receptive field is
/// essentially global - a pixel-scale stack of twelve 3x3x3 convolutions
/// reaches a hundred pixels, and the four res groups before it reach several
/// hundred - so most of an impulse's response is a diffuse field centred on the
/// frame and the centroid is damped towards it. A centroid is therefore not a
/// valid geometric guard here and this is reporting only; the axis order and
/// the upsampling geometry are pinned by the exact depth-to-space and
/// unpatchify probes instead.
fn impulse_check(gpu: &Gpu, vae: &LtxVae, shape: Shape, pixel: [u32; 3]) -> Result<(), String> {
    let latent_height = shape.height as usize;
    let latent_width = shape.width as usize;
    // Deep enough into the frame that every response's own halo stays inside
    // it, and asymmetric between the axes so a swap of the two cannot cancel.
    let (cell_h, cell_w) = (latent_height / 2, latent_width - 3);

    let origin = impulse_centroid(gpu, vae, shape, pixel, cell_h, cell_w)?;
    let down = impulse_centroid(gpu, vae, shape, pixel, cell_h + 1, cell_w)?;
    let right = impulse_centroid(gpu, vae, shape, pixel, cell_h, cell_w + 1)?;

    let scale = f64::from(scale_factors()[1]);
    println!(
        "IMPULSE cell=({cell_h},{cell_w}) step_down=({:.2},{:.2}) step_right=({:.2},{:.2}) \
         scale={scale:.0} spread=({:.1},{:.1}) (reporting only)",
        down.1 - origin.1,
        down.0 - origin.0,
        right.0 - origin.0,
        right.1 - origin.1,
        origin.2,
        down.2,
    );

    Ok(())
}

/// The centroid of one cell's impulse response, in pixels, against the zero
/// latent's own decode.
fn impulse_centroid(
    gpu: &Gpu,
    vae: &LtxVae,
    shape: Shape,
    pixel: [u32; 3],
    cell_h: usize,
    cell_w: usize,
) -> Result<(f64, f64, f64), String> {
    let elements = shape.elements();
    let latent_plane = shape.plane();
    let latent_width = shape.width as usize;
    let (height, width) = (pixel[1] as usize, pixel[2] as usize);

    let zeros = vec![0u8; elements * 2];
    let mut impulse = zeros.clone();
    let value = ((bf16(5.0).to_bits() >> 16) as u16).to_le_bytes();
    for frame in 0..shape.frames as usize {
        let at = (frame * latent_plane + cell_h * latent_width + cell_w) * 2;
        impulse[at..at + 2].copy_from_slice(&value);
    }

    let baseline = {
        let latent = upload_bf16(gpu, &zeros)?;
        let rgb = vae
            .decode(gpu, &latent, shape)
            .map_err(|error| error.to_string())?;
        read_bf16(gpu, rgb)?
    };
    let latent = upload_bf16(gpu, &impulse)?;
    let rgb = vae
        .decode(gpu, &latent, shape)
        .map_err(|error| error.to_string())?;
    let response = read_bf16(gpu, rgb)?;

    let mut mass = 0.0f64;
    let (mut sum_x, mut sum_y) = (0.0f64, 0.0f64);
    let (mut square_x, mut square_y) = (0.0f64, 0.0f64);
    for y in 0..height {
        for x in 0..width {
            let at = (y * width + x) * 3;
            for channel in 0..3 {
                let difference = (response[at + channel] - baseline[at + channel]).abs() as f64;
                mass += difference;
                sum_x += difference * x as f64;
                sum_y += difference * y as f64;
                square_x += difference * (x as f64).powi(2);
                square_y += difference * (y as f64).powi(2);
            }
        }
    }

    if mass <= 1e-6 {
        return Err(format!(
            "the impulse at ({cell_h}, {cell_w}) moved nothing: mass {mass:.3e}"
        ));
    }

    let (centroid_x, centroid_y) = (sum_x / mass, sum_y / mass);
    let spread_x = (square_x / mass - centroid_x * centroid_x).max(0.0).sqrt();
    let spread_y = (square_y / mass - centroid_y * centroid_y).max(0.0).sqrt();

    Ok((centroid_x, centroid_y, spread_x.max(spread_y)))
}

/// Every new device op against a host transcription of the reference's own
/// expression.
fn probe(gpu: &Gpu, seed: u64) -> Result<bool, String> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut failed = false;

    failed |= probe_denormalize(gpu, &mut state)?;
    failed |= probe_pixel_norm(gpu, &mut state)?;
    failed |= probe_depth_to_space(gpu, &mut state)?;
    failed |= probe_unpatchify(gpu, &mut state)?;
    failed |= probe_to_rgb(gpu, &mut state)?;
    failed |= probe_conv(gpu, &mut state)?;

    Ok(failed)
}

/// `PerChannelStatistics.un_normalize` (`ops.py:74-78`): `x * std` then
/// `+ mean`, each bf16, each rounding once.
fn probe_denormalize(gpu: &Gpu, state: &mut u64) -> Result<bool, String> {
    let (channels, plane) = (4usize, 6usize);
    let input = draw(gpu, channels * plane, state)?;
    let mean = draw(gpu, channels, state)?;
    let scale = draw(gpu, channels, state)?;
    let values = read_bf16(gpu, &input)?;
    let (means, scales) = (read_bf16(gpu, &mean)?, read_bf16(gpu, &scale)?);

    let output = alloc(gpu, Dtype::Bf16, channels * plane)?;
    gpu.video_vae_denormalize(
        &output,
        &input,
        &mean,
        &scale,
        Dtype::Bf16,
        channels as u32,
        plane,
    )
    .map_err(|detail| format!("video_vae_denormalize: {detail}"))?;

    let want: Vec<f32> = (0..channels * plane)
        .map(|index| {
            let channel = index / plane;
            bf16(bf16(values[index] * scales[channel]) + means[channel])
        })
        .collect();
    let got = read_bf16(gpu, &output)?;

    // A control that cannot be the reference's: one rounding instead of two.
    let fused: Vec<f32> = (0..channels * plane)
        .map(|index| {
            let channel = index / plane;
            bf16(values[index] * scales[channel] + means[channel])
        })
        .collect();
    let (fused_differ, _) = compare(&got, &fused);

    let mut failed = report("denormalize", &got, &want, Verdict::Exact);
    if fused_differ == 0 {
        println!(
            "PROBE denormalize-control differ=0 verdict=FAIL the two roundings are not observable"
        );
        failed = true;
    } else {
        println!("PROBE denormalize-control differ={fused_differ} verdict=pass");
    }

    Ok(failed)
}

/// `PixelNorm` (`normalization.py:31-38`): five ops, five roundings.
fn probe_pixel_norm(gpu: &Gpu, state: &mut u64) -> Result<bool, String> {
    let (channels, plane) = (5usize, 7usize);
    let input = draw(gpu, channels * plane, state)?;
    let values = read_bf16(gpu, &input)?;

    let output = alloc(gpu, Dtype::Bf16, channels * plane)?;
    gpu.video_vae_pixel_norm(
        &output,
        &input,
        Dtype::Bf16,
        channels as u32,
        plane,
        PIXEL_NORM_EPSILON,
    )
    .map_err(|detail| format!("video_vae_pixel_norm: {detail}"))?;

    let mut want = vec![0.0f32; channels * plane];
    for position in 0..plane {
        let mut total = 0.0f32;
        for channel in 0..channels {
            let value = values[channel * plane + position];
            total += bf16(value * value);
        }
        let mean = bf16(total / channels as f32);
        let rms = bf16(bf16(mean + PIXEL_NORM_EPSILON).sqrt());
        for channel in 0..channels {
            want[channel * plane + position] = bf16(values[channel * plane + position] / rms);
        }
    }

    let got = read_bf16(gpu, &output)?;

    Ok(report("pixel_norm", &got, &want, Verdict::Exact))
}

/// `DepthToSpaceUpsample` (`sampling.py:66-124`) for all three strides the
/// artifact uses, so the temporal drop and the channel order are both read.
fn probe_depth_to_space(gpu: &Gpu, state: &mut u64) -> Result<bool, String> {
    let (depth, height, width) = (2usize, 3usize, 4usize);
    let mut failed = false;

    for (stride, reduction) in [([2u32, 2, 2], 2u32), ([2, 1, 1], 2), ([1, 2, 2], 2)] {
        let channels = reduction as usize * 8;
        let input = draw(gpu, channels * depth * height * width, state)?;
        let values = read_bf16(gpu, &input)?;

        let out_depth = depth * stride[0] as usize - usize::from(stride[0] == 2);
        let out_height = height * stride[1] as usize;
        let out_width = width * stride[2] as usize;
        let out_channels = channels / (stride[0] * stride[1] * stride[2]) as usize;
        let output = alloc(
            gpu,
            Dtype::Bf16,
            out_channels * out_depth * out_height * out_width,
        )?;
        gpu.video_vae_depth_to_space(
            &output,
            &input,
            Dtype::Bf16,
            channels as u32,
            depth as u32,
            height as u32,
            width as u32,
            stride[0],
            stride[1],
            stride[2],
            u32::from(stride[0] == 2),
        )
        .map_err(|detail| format!("video_vae_depth_to_space: {detail}"))?;

        // The rearrange `(c p1 p2 p3) -> c (d p1) (h p2) (w p3)`, with the
        // first output frame cut when the temporal axis doubles.
        let mut want = vec![0.0f32; out_channels * out_depth * out_height * out_width];
        for channel in 0..out_channels {
            for t in 0..out_depth {
                for y in 0..out_height {
                    for x in 0..out_width {
                        let (p1, p2, p3) =
                            (stride[0] as usize, stride[1] as usize, stride[2] as usize);
                        let (a, source_t) = (
                            (t + usize::from(p1 == 2)) % p1,
                            (t + usize::from(p1 == 2)) / p1,
                        );
                        let source_channel = ((channel * p1 + a) * p2 + y % p2) * p3 + x % p3;
                        want[((channel * out_depth + t) * out_height + y) * out_width + x] = values
                            [((source_channel * depth + source_t) * height + y / p2) * width
                                + x / p3];
                    }
                }
            }
        }

        let got = read_bf16(gpu, &output)?;
        failed |= report(
            &format!("depth_to_space_{}x{}x{}", stride[0], stride[1], stride[2]),
            &got,
            &want,
            Verdict::Exact,
        );
    }

    Ok(failed)
}

/// `unpatchify` at patch size 4 (`ops.py:34-60`).
fn probe_unpatchify(gpu: &Gpu, state: &mut u64) -> Result<bool, String> {
    let (channels, depth, height, width, patch) = (3usize, 2usize, 2usize, 3usize, 4usize);
    let input = draw(
        gpu,
        channels * patch * patch * depth * height * width,
        state,
    )?;
    let values = read_bf16(gpu, &input)?;

    let output = alloc(
        gpu,
        Dtype::Bf16,
        channels * depth * height * patch * width * patch,
    )?;
    gpu.video_vae_unpatchify(
        &output,
        &input,
        Dtype::Bf16,
        channels as u32,
        patch as u32,
        depth as u32,
        height as u32,
        width as u32,
    )
    .map_err(|detail| format!("video_vae_unpatchify: {detail}"))?;

    // `(c p r q) f h w -> c (f p) (h q) (w r)`: source channel
    // `c p^2 + r p + q` for row `h p + q` and column `w p + r`, `p = 1`.
    let mut want = vec![0.0f32; channels * depth * height * patch * width * patch];
    for channel in 0..channels {
        for f in 0..depth {
            for y in 0..height * patch {
                for x in 0..width * patch {
                    let (q, h) = (y % patch, y / patch);
                    let (r, w) = (x % patch, x / patch);
                    let source = (channel * patch + r) * patch + q;
                    want[((channel * depth + f) * height * patch + y) * width * patch + x] =
                        values[((source * depth + f) * height + h) * width + w];
                }
            }
        }
    }

    let got = read_bf16(gpu, &output)?;

    Ok(report("unpatchify", &got, &want, Verdict::Exact))
}

/// `to_rgb` (`conv_video_decoder.py:497-499`): `(x + 1) / 2` clamped, two
/// roundings, channel-major to interleaved.
fn probe_to_rgb(gpu: &Gpu, state: &mut u64) -> Result<bool, String> {
    let (depth, height, width) = (2usize, 3usize, 4usize);
    let plane = depth * height * width;
    let input = draw(gpu, 3 * plane, state)?;
    let values = read_bf16(gpu, &input)?;

    let output = alloc(gpu, Dtype::Bf16, 3 * plane)?;
    gpu.video_vae_to_rgb(
        &output,
        &input,
        Dtype::Bf16,
        depth as u32,
        height as u32,
        width as u32,
    )
    .map_err(|detail| format!("video_vae_to_rgb: {detail}"))?;

    let want: Vec<f32> = (0..3 * plane)
        .map(|index| {
            let channel = index % 3;
            let point = index / 3;
            let shifted = bf16(values[channel * plane + point] + 1.0);
            bf16(0.5 * shifted).clamp(0.0, 1.0)
        })
        .collect();

    let got = read_bf16(gpu, &output)?;

    Ok(report("to_rgb", &got, &want, Verdict::Exact))
}

/// The convolution: `h3b_planar` plus one dense matmul per `(kt, ky)` tap,
/// with the artifact's own padding. The transcription is deliberately the
/// naive one - one f32 sum in `(kt, ky, kx, c)` order per output element - so
/// the two agree to the reduction order and no further.
fn probe_conv(gpu: &Gpu, state: &mut u64) -> Result<bool, String> {
    let (in_channels, out_channels) = (3usize, 4usize);
    let (frames, height, width) = (2usize, 3usize, 4usize);
    let input = draw(gpu, in_channels * frames * height * width, state)?;
    let weight = draw(gpu, out_channels * in_channels * 27, state)?;
    let bias = draw(gpu, out_channels, state)?;
    let values = read_bf16(gpu, &input)?;
    let weights = read_bf16(gpu, &weight)?;
    let biases = read_bf16(gpu, &bias)?;

    let output = alloc(gpu, Dtype::Bf16, out_channels * frames * height * width)?;

    // The whole convolution through the same kernels the decoder uses: the
    // planar buffer, one matmul per tap, the epilogue with the bias.
    let planar = alloc(
        gpu,
        Dtype::Bf16,
        planar_elements(in_channels, frames, height, width),
    )?;
    gpu.latent_planar(
        &planar,
        &input,
        Dtype::Bf16,
        in_channels as u32,
        frames as u32,
        height as u32,
        width as u32,
        3,
        1,
        1,
        1,
        width as u32,
        true,
    )
    .map_err(|detail| format!("latent_planar: {detail}"))?;

    let packed = alloc(gpu, Dtype::Bf16, weight.elements())?;
    gpu.latent_weight_pack(
        &packed,
        &weight,
        Dtype::Bf16,
        out_channels as u32,
        in_channels as u32,
        27,
        3,
    )
    .map_err(|detail| format!("latent_weight_pack: {detail}"))?;

    let temp = alloc(gpu, Dtype::F32, out_channels * height * width)?;
    let columns = in_channels * 3;
    let padded_height = height + 2;
    for frame in 0..frames {
        for tap_t in 0..3 {
            for tap_h in 0..3 {
                let pass = tap_t * 3 + tap_h;
                let offset = ((frame + tap_t) * padded_height + tap_h) * width * columns;
                gpu.latent_matmul_planar(
                    &temp,
                    &planar,
                    offset,
                    &packed,
                    pass * out_channels * columns,
                    0,
                    (height * width) as u32,
                    columns as u32,
                    out_channels as u32,
                    tap_t != 0 || tap_h != 0,
                    Dtype::Bf16,
                )
                .map_err(|detail| format!("latent_matmul_planar: {detail}"))?;
            }
        }
        gpu.latent_epilogue(
            &output,
            &temp,
            Some(&bias),
            Dtype::Bf16,
            out_channels as u32,
            (height * width) as u32,
            frames * height * width,
            frame * height * width,
        )
        .map_err(|detail| format!("latent_epilogue: {detail}"))?;
    }

    let got = read_bf16(gpu, &output)?;

    // `CausalConv3d` with `causal=False` (`convolution.py:302-312`): one
    // replicated frame each side in time, one zero frame each side in space.
    let want = transcribe_conv(
        &values,
        &weights,
        &biases,
        in_channels,
        out_channels,
        frames,
        height,
        width,
        Halo::Replicate,
    );
    let zeroed = transcribe_conv(
        &values,
        &weights,
        &biases,
        in_channels,
        out_channels,
        frames,
        height,
        width,
        Halo::Zero,
    );
    let (zero_differ, zero_relative) = compare(&got, &zeroed);

    let mut failed = report("conv3d", &got, &want, Verdict::Residual);
    if zero_relative < HALO_SEPARATION {
        println!(
            "PROBE conv3d-control differ={zero_differ} rel_rms={zero_relative:.3e} verdict=FAIL \
             a zero-padded halo is not distinguishable"
        );
        failed = true;
    } else {
        println!(
            "PROBE conv3d-control differ={zero_differ} rel_rms={zero_relative:.3e} verdict=pass"
        );
    }

    Ok(failed)
}

/// Which temporal halo a transcription applies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Halo {
    Replicate,
    Zero,
}

#[allow(clippy::too_many_arguments)]
fn transcribe_conv(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_channels: usize,
    out_channels: usize,
    frames: usize,
    height: usize,
    width: usize,
    halo: Halo,
) -> Vec<f32> {
    let mut want = vec![0.0f32; out_channels * frames * height * width];

    for o in 0..out_channels {
        for t in 0..frames {
            for y in 0..height {
                for x in 0..width {
                    let mut total = 0.0f32;
                    for c in 0..in_channels {
                        for kt in 0..3 {
                            let source_t = t as i64 + kt as i64 - 1;
                            let source_t = if source_t < 0 || source_t >= frames as i64 {
                                match halo {
                                    Halo::Zero => continue,
                                    Halo::Replicate => source_t.clamp(0, frames as i64 - 1),
                                }
                            } else {
                                source_t
                            } as usize;
                            for ky in 0..3 {
                                let source_y = y as i64 + ky as i64 - 1;
                                if source_y < 0 || source_y >= height as i64 {
                                    continue;
                                }
                                for kx in 0..3 {
                                    let source_x = x as i64 + kx as i64 - 1;
                                    if source_x < 0 || source_x >= width as i64 {
                                        continue;
                                    }
                                    let at = ((c * frames + source_t) * height + source_y as usize)
                                        * width
                                        + source_x as usize;
                                    let tap = ((c * 3 + kt) * 3 + ky) * 3 + kx;
                                    total += input[at] * weight[o * in_channels * 27 + tap];
                                }
                            }
                        }
                    }
                    want[((o * frames + t) * height + y) * width + x] = bf16(total + bias[o]);
                }
            }
        }
    }

    want
}

/// The planar buffer the convolution reads: the padded activation in
/// channel-last form, `kw` copies wide.
fn planar_elements(channels: usize, frames: usize, height: usize, width: usize) -> usize {
    channels * (frames + 2) * (height + 2) * width * 3
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Exact,
    Residual,
}

fn report(name: &str, got: &[f32], want: &[f32], verdict: Verdict) -> bool {
    let (differ, rel_rms) = compare(got, want);
    let pass = match verdict {
        Verdict::Exact => differ == 0,
        Verdict::Residual => rel_rms <= CONV_RELATIVE,
    };

    println!(
        "PROBE {name} values={} differ={differ} rel_rms={rel_rms:.3e} verdict={} {}",
        got.len(),
        if verdict == Verdict::Exact {
            "exact"
        } else {
            "residual"
        },
        if pass { "pass" } else { "FAIL" }
    );

    !pass
}

fn compare(got: &[f32], want: &[f32]) -> (usize, f64) {
    let mut differ = 0usize;
    let mut square = 0.0f64;
    let mut reference = 0.0f64;

    for (left, right) in got.iter().zip(want.iter()) {
        if left.to_bits() != right.to_bits() {
            differ += 1;
        }
        square += ((*left as f64) - (*right as f64)).powi(2);
        reference += (*right as f64).powi(2);
    }

    (differ, (square / reference.max(1e-30)).sqrt())
}

/// Round to BF16 the way the device's `__float2bfloat16` does: nearest even.
fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);

    f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
}

/// FNV-1a over the BF16 bytes, the family's own state hash.
fn fnv(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;

    for value in values {
        let bits = (value.to_bits() >> 16) as u16;
        for byte in bits.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    hash
}

fn read_latent(gpu: &Gpu, path: &Path, shape: Shape) -> Result<Tensor, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let want = shape.elements() * 2;

    if bytes.len() != want {
        return Err(format!(
            "{} holds {} bytes, a {}x{}x{}x{} latent is {want}",
            path.display(),
            bytes.len(),
            shape.channels,
            shape.frames,
            shape.height,
            shape.width
        ));
    }

    upload_bf16(gpu, &bytes)
}

/// The first frame as a PNG, so the end of the chain can be looked at.
fn write_frame(path: &Path, values: &[f32], pixel: [u32; 3]) -> Result<(), String> {
    let plane = (pixel[1] * pixel[2]) as usize;
    let mut rgb = Vec::with_capacity(plane * 3);

    for index in 0..plane * 3 {
        let value = values[index].clamp(0.0, 1.0);
        rgb.push((value * 255.0).round() as u8);
    }

    let image = image::RgbImage::from_raw(pixel[2], pixel[1], rgb)
        .ok_or_else(|| "the first frame does not match its dimensions".to_string())?;
    image
        .save(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;

    println!(
        "FRAME path={} frames={} height={} width={}",
        path.display(),
        pixel[0],
        pixel[1],
        pixel[2]
    );

    Ok(())
}

fn alloc(gpu: &Gpu, dtype: Dtype, elements: usize) -> Result<Tensor, String> {
    gpu.alloc(dtype, elements)
}

fn upload_bf16(gpu: &Gpu, bytes: &[u8]) -> Result<Tensor, String> {
    let tensor = alloc(gpu, Dtype::Bf16, bytes.len() / 2)?;
    gpu.upload(&tensor, bytes)?;

    Ok(tensor)
}

/// A draw in `[-1, 1)` exactly representable in BF16, so the probes' inputs
/// are the same values on the host and the device.
fn draw(gpu: &Gpu, elements: usize, state: &mut u64) -> Result<Tensor, String> {
    let mut bytes = Vec::with_capacity(elements * 2);

    for _ in 0..elements {
        let value = bf16(next(state));
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }

    upload_bf16(gpu, &bytes)
}

fn next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);

    (((*state >> 40) as f32) / 8_388_608.0) - 1.0
}

fn read_bf16(gpu: &Gpu, tensor: &Tensor) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; tensor.bytes()];
    gpu.read(tensor, &mut bytes)?;

    Ok(bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
        .collect())
}

fn split(arguments: &[String]) -> Result<Options, String> {
    let mut options = Options {
        probe: false,
        shapes: false,
        impulse: false,
        vae: None,
        latent: None,
        frame: None,
        decode_shape: None,
        seed: 1,
        expect: None,
    };
    let mut index = 0usize;

    while index < arguments.len() {
        let argument = arguments[index].as_str();
        index += 1;

        match argument {
            "--probe" => options.probe = true,
            "--shapes" => options.shapes = true,
            "--impulse-check" => options.impulse = true,
            "--vae" => options.vae = Some(PathBuf::from(value(arguments, &mut index, argument)?)),
            "--latent" => {
                options.latent = Some(PathBuf::from(value(arguments, &mut index, argument)?))
            }
            "--frame" => {
                options.frame = Some(PathBuf::from(value(arguments, &mut index, argument)?))
            }
            "--decode-shape" => {
                let text = value(arguments, &mut index, argument)?;
                let parts: Vec<&str> = text.split(',').collect();
                if parts.len() != 3 {
                    return Err(usage("--decode-shape takes F,H,W"));
                }
                let number = |index: usize| -> Result<u32, String> {
                    parts[index]
                        .parse::<u32>()
                        .map_err(|_| usage("--decode-shape takes F,H,W"))
                };
                options.decode_shape = Some([number(0)?, number(1)?, number(2)?]);
            }
            "--seed" => {
                options.seed = value(arguments, &mut index, argument)?
                    .parse()
                    .map_err(|_| usage("--seed takes a number"))?;
            }
            "--expect" => {
                let text = value(arguments, &mut index, argument)?;
                options.expect = Some(
                    u64::from_str_radix(text.trim_start_matches("0x"), 16)
                        .map_err(|_| usage("--expect takes a hex hash"))?,
                );
            }
            other => return Err(usage(&format!("unknown argument {other}"))),
        }
    }

    Ok(options)
}

fn value<'a>(arguments: &'a [String], index: &mut usize, name: &str) -> Result<&'a str, String> {
    let value = arguments
        .get(*index)
        .ok_or_else(|| usage(&format!("{name} needs a value")))?;
    *index += 1;

    Ok(value)
}

fn usage(detail: &str) -> String {
    format!("{detail}\n\nusage:\n  h3_ltx_vae_gate --probe [--seed S]\n  h3_ltx_vae_gate --shapes\n  h3_ltx_vae_gate --vae PATH --decode-shape F,H,W [--latent FILE] [--frame FILE] [--seed S] [--expect HEX]")
}
