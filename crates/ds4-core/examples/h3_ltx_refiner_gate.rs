//! The unit-6 gate: the LTX-2.5 refiner's transformer block.
//!
//! Usage:
//!   h3_ltx_refiner_gate --probe [--rows R] [--seed S]
//!   h3_ltx_refiner_gate --transformer PATH [--blocks K] [--rows R]
//!                       [--audio-rows A] [--seed S] [--dump DIR]
//!                       [--residual DIR] [--expect NAME=HEX ...]
//!   h3_ltx_refiner_gate --transformer PATH --identity FILE [--blocks K] ...
//!
//! Four modes:
//!
//! - `--probe`: each new device kernel against a host transcription of the
//!   same expression, element by element. The transcriptions reproduce the
//!   kernel's own rounding chain, so a kernel whose reference has no libm call
//!   must come back **bit-identical**; the two that end in `rsqrtf` are
//!   reported as the residues they are, with the number.
//! - default: the block (or the first `--blocks` of them) over the shard's own
//!   weights, on synthetic modulation and positional tables - one FNV-1a line
//!   per state, so a golden from the reference is a straight comparison - and
//!   `--dump DIR` writes every block's state as BF16 for that diff.
//! - `--identity FILE`: the residual structure, exactly. It writes a shard
//!   whose six `to_out` projections and two feed-forward outputs are zero, so
//!   every added term is exactly zero and the block must return its input
//!   **bit for bit**. A mis-wired residual (a term written into the wrong
//!   buffer, a missing add, a doubled branch) fails it.
//! - `--residual DIR`: the same run compared against a previous `--dump DIR`
//!   as an element-wise residual, which is the cross-host check: `h3d_linear`
//!   is cuBLASLt and not bit-stable across architectures, so the two hosts'
//!   states differ at the BF16 noise floor rather than by hash.
//!
//! There is no Python in this campaign and the pinned reference does not run on
//! either host, so this gate is the same shape as the tower's and the
//! connector's: a second implementation of the reference's own expressions,
//! cited to `model/transformer/` line by line.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::ltx_refiner::{
    Cross, EdgeScratch, Edges, Geometry, Refiner, RefinerScratch, Request, Signals, AUDIO_BINS,
    AUDIO_CHANNELS, AUDIO_CONTEXT_DIM, AUDIO_DIM, AUDIO_HEAD_DIM, AUDIO_TOKENS, CHANNELS,
    CONTEXT_ROWS, HEADS, LAYERS, VIDEO_CONTEXT_DIM, VIDEO_DIM, VIDEO_HEAD_DIM, VIDEO_TOKENS,
};
use ds4_core::minimax_h3::ltx_stage2::{Generator, Stage2, NOISE_SCALE, SIGMAS, STEPS};
use ds4_core::minimax_h3::safetensors::{self, Entry, Shard, ShardWriter};
use ds4_core::minimax_h3::text_forward::fnv1a;
use ds4_sys::h3_dit::{Dtype as Device, Gpu, Tensor};

/// The ceiling a probe whose reference ends in a libm call must stay under.
/// BF16 holds eight mantissa bits, so one ulp of `rsqrtf` moves a rounded value
/// by a few tenths of a percent; a wiring fault moves every element, by two
/// orders more.
const RESIDUAL_CEILING: f64 = 2e-2;

/// The per-block tensors whose zeroing makes every added term exactly zero:
/// the six attention output projections and the two feed-forward outputs.
const RESIDUAL_SUFFIXES: [&str; 15] = [
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "audio_attn1.to_out.0.weight",
    "audio_attn1.to_out.0.bias",
    "audio_attn2.to_out.0.weight",
    "audio_attn2.to_out.0.bias",
    "audio_to_video_attn.to_out.0.weight",
    "audio_to_video_attn.to_out.0.bias",
    "video_to_audio_attn.to_out.0.weight",
    "video_to_audio_attn.to_out.0.bias",
    "ff.net.2.weight",
    "audio_ff.net.2.weight",
    "audio_ff.net.2.bias",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match run(&args) {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let options = split(args)?;
    let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;

    if options.probe {
        let failed = probe(&gpu, options.rows.unwrap_or(8), options.seed.unwrap_or(1))?;

        return Ok(ExitCode::from(if failed { 3 } else { 0 }));
    }

    let transformer = options
        .transformer
        .clone()
        .ok_or_else(|| usage("--transformer is required"))?;

    if let Some(target) = &options.identity {
        return identity(&gpu, &transformer, target, &options);
    }

    if options.edges {
        return run_edges(&gpu, &transformer, &options);
    }

    if options.stage {
        return run_stage(&gpu, &transformer, &options);
    }

    run_blocks(&gpu, &transformer, &options)
}

/// The whole stage-2 evaluation (`minimax_h3::ltx_stage2`): one noised state,
/// the release's three intervals, and the final latent, with a hash per step
/// for the golden diff.
///
/// The two inputs are the route's own and are not this gate's to produce. The
/// initial latent is the H3 latent through the adapter and the latent
/// upsampler (`runtime/stage2_ops/h3_upscale.py`), which is not ported, so it
/// is synthetic unless `--latent` names a dumped one; the prompt context is the
/// Gemma4 tower plus the LTX connector's output, so it is synthetic unless
/// `--context` names the connector gate's dump. A synthetic pair still
/// exercises every kernel on the release's shapes and prices the step; it says
/// nothing about the video.
fn run_stage(gpu: &Gpu, path: &Path, options: &Options) -> Result<ExitCode, String> {
    let geometry = Geometry::release();
    let rows = geometry.video_tokens();
    let layers: Vec<usize> = (0..options.layers.unwrap_or(LAYERS)).collect();
    let seed = options.seed.unwrap_or(1);

    let started = Instant::now();
    let mut stage =
        Stage2::open(gpu, path, path, geometry, &layers).map_err(|error| error.token())?;
    let load = started.elapsed().as_secs_f64();

    let mut lcg = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let latent = match &options.latent {
        Some(file) => read_tensor_file(gpu, file, CHANNELS * rows as usize)?,
        None => draw(gpu, CHANNELS as u32, rows as usize, &mut lcg)?,
    };
    let context = match &options.prompt {
        Some(file) => read_tensor_file(gpu, file, CONTEXT_ROWS as usize * VIDEO_CONTEXT_DIM)?,
        None => draw(gpu, CONTEXT_ROWS, VIDEO_CONTEXT_DIM, &mut lcg)?,
    };

    println!(
        "STAGE layers={} rows={rows} seed={seed} noise_scale={NOISE_SCALE} schedule={SIGMAS:?} \
         latent={} context={} load={load:.3}",
        layers.len(),
        if options.latent.is_some() {
            "file"
        } else {
            "synthetic"
        },
        if options.prompt.is_some() {
            "file"
        } else {
            "synthetic"
        },
    );

    stage.seed(seed as u64);

    let mut failed = false;
    let mut expect = options.expect.clone();
    let started = Instant::now();
    stage
        .noised_state(gpu, &latent)
        .map_err(|error| error.token())?;
    println!(
        "STAGE-TIMING prepare={:.3}",
        started.elapsed().as_secs_f64()
    );

    // The noise is the only place the RNG enters, so its hash is the one that
    // a fixed seed makes comparable across hosts.
    let generator = stage.generator();
    println!(
        "STAGE-GENERATOR seed={} offset={}",
        generator.seed(),
        generator.offset()
    );
    failed |= report_state(
        gpu,
        "stage-noise",
        stage.noise(),
        &mut expect,
        options.dump.as_deref(),
        None,
        0.0,
    )?;
    failed |= report_state(
        gpu,
        "stage-state0",
        stage.latent(),
        &mut expect,
        options.dump.as_deref(),
        options.residual.as_deref(),
        0.0,
    )?;

    let mut steps = [0.0f64; STEPS];
    let started = Instant::now();

    for index in 0..STEPS {
        let step_started = Instant::now();
        stage
            .step(gpu, index, &context)
            .map_err(|error| error.token())?;
        steps[index] = step_started.elapsed().as_secs_f64();

        failed |= report_state(
            gpu,
            &format!("stage-step{index}"),
            stage.latent(),
            &mut expect,
            options.dump.as_deref(),
            options.residual.as_deref(),
            steps[index],
        )?;
    }

    let denoise = started.elapsed().as_secs_f64();
    failed |= report_state(
        gpu,
        "stage-latent",
        stage.latent_grid(gpu).map_err(|error| error.token())?,
        &mut expect,
        options.dump.as_deref(),
        options.residual.as_deref(),
        0.0,
    )?;

    for name in expect.keys() {
        println!("STAGE {name} expected but never produced");
        failed = true;
    }

    println!("STAGE-TIMING steps={steps:?} denoise={denoise:.3} load={load:.3}");

    if failed {
        return Ok(ExitCode::from(3));
    }

    println!("LTX-REFINER-STAGE-OK");

    Ok(ExitCode::SUCCESS)
}

/// The first `blocks` blocks over the shard's weights, with synthetic
/// modulation and positional tables.
fn run_blocks(gpu: &Gpu, path: &Path, options: &Options) -> Result<ExitCode, String> {
    let blocks = options.blocks.unwrap_or(1);
    let layers: Vec<usize> = (0..blocks).collect();
    let rows = options.rows.unwrap_or(8) as u32;
    let audio_rows = options.audio_rows.unwrap_or(4) as u32;
    let seed = options.seed.unwrap_or(1);

    let started = Instant::now();
    let refiner = Refiner::open(gpu, path, &layers).map_err(|error| error.token())?;
    let load = started.elapsed().as_secs_f64();
    let (tensors, bytes) = refiner.loaded();

    let scratch =
        RefinerScratch::new(gpu, rows, audio_rows, CONTEXT_ROWS).map_err(|error| error.token())?;
    let mut lcg = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let (video, _) = synthetic(gpu, rows, VIDEO_DIM, &mut lcg)?;
    let (audio, _) = synthetic(gpu, audio_rows, AUDIO_DIM, &mut lcg)?;
    let v_context = context(gpu, VIDEO_DIM, &mut lcg)?;
    let v = signals(gpu, rows, VIDEO_DIM, VIDEO_HEAD_DIM, &v_context, &mut lcg)?;
    let a_context = context(gpu, AUDIO_DIM, &mut lcg)?;
    let a = signals(
        gpu,
        audio_rows,
        AUDIO_DIM,
        AUDIO_HEAD_DIM,
        &a_context,
        &mut lcg,
    )?;

    println!(
        "REFINER blocks={} video_rows={} audio_rows={} tensors={} bytes={} load={load:.3}",
        refiner.blocks(),
        rows,
        audio_rows,
        tensors,
        bytes
    );

    let mut failed = false;
    let mut expect = options.expect.clone();

    for block in 0..refiner.blocks() {
        let started = Instant::now();
        refiner
            .apply_one(block, gpu, &video, &v, Some((&audio, &a)), &scratch)
            .map_err(|error| error.token())?;
        let seconds = started.elapsed().as_secs_f64();

        failed |= report_state(
            gpu,
            &format!("block{block}-video"),
            &video,
            &mut expect,
            options.dump.as_deref(),
            options.residual.as_deref(),
            seconds,
        )?;
        failed |= report_state(
            gpu,
            &format!("block{block}-audio"),
            &audio,
            &mut expect,
            options.dump.as_deref(),
            options.residual.as_deref(),
            seconds,
        )?;
    }

    for name in expect.keys() {
        println!("STAGE {name} expected but never produced");
        failed = true;
    }

    if failed {
        return Ok(ExitCode::from(3));
    }

    println!("LTX-REFINER-OK");

    Ok(ExitCode::SUCCESS)
}

/// The residual structure: zero every term the block adds and require the
/// state back unchanged, bit for bit.
///
/// `x + y * gate` with `y = to_out(0) = 0` is `x` exactly, and the same holds
/// for the feed-forward's `down(0)`. The only way the block can move the state
/// is by the residual adds themselves, so this pins their wiring without
/// needing the reference.
fn identity(gpu: &Gpu, path: &Path, target: &Path, options: &Options) -> Result<ExitCode, String> {
    let blocks = options.blocks.unwrap_or(1);
    let layers: Vec<usize> = (0..blocks).collect();
    let rows = options.rows.unwrap_or(8) as u32;
    let audio_rows = options.audio_rows.unwrap_or(4) as u32;
    let seed = options.seed.unwrap_or(1);

    let zeroed = zero_shard(path, target, blocks)?;
    println!("IDENTITY shard={} zeroed={zeroed}", target.display());

    let refiner = Refiner::open(gpu, target, &layers).map_err(|error| error.token())?;
    let scratch =
        RefinerScratch::new(gpu, rows, audio_rows, CONTEXT_ROWS).map_err(|error| error.token())?;

    let mut lcg = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let (video, video_values) = synthetic(gpu, rows, VIDEO_DIM, &mut lcg)?;
    let (audio, audio_values) = synthetic(gpu, audio_rows, AUDIO_DIM, &mut lcg)?;
    let v_context = context(gpu, VIDEO_DIM, &mut lcg)?;
    let v = signals(gpu, rows, VIDEO_DIM, VIDEO_HEAD_DIM, &v_context, &mut lcg)?;
    let a_context = context(gpu, AUDIO_DIM, &mut lcg)?;
    let a = signals(
        gpu,
        audio_rows,
        AUDIO_DIM,
        AUDIO_HEAD_DIM,
        &a_context,
        &mut lcg,
    )?;

    refiner
        .apply(gpu, &video, &v, Some((&audio, &a)), &scratch)
        .map_err(|error| error.token())?;

    let got_video = read_bf16(gpu, &video)?;
    let got_audio = read_bf16(gpu, &audio)?;
    let mut failed = false;

    for (name, got, want) in [
        ("identity-video", &got_video, &video_values),
        ("identity-audio", &got_audio, &audio_values),
    ] {
        let (differ, rel_rms) = compare(got, want);
        let pass = differ == 0;
        println!(
            "IDENTITY {name} values={} differ={differ} rel_rms={rel_rms:.3e} {}",
            got.len(),
            if pass { "pass" } else { "FAIL" }
        );
        failed |= !pass;
    }

    if failed {
        return Ok(ExitCode::from(3));
    }

    println!("LTX-REFINER-IDENTITY-OK");

    Ok(ExitCode::SUCCESS)
}

/// Write `path`'s block tensors to `target` with the residual projections
/// zeroed, and return how many tensors were zeroed.
fn zero_shard(path: &Path, target: &Path, blocks: usize) -> Result<usize, String> {
    let shard = safetensors::read_shard(path).map_err(|error| error.token())?;
    let prefix = |block: usize| format!("model.diffusion_model.transformer_blocks.{block}.");

    let mut entries = Vec::new();
    let mut selected = Vec::new();

    for block in 0..blocks {
        let at = prefix(block);

        for tensor in &shard.tensors {
            if !tensor.name.starts_with(&at) {
                continue;
            }

            entries.push(Entry {
                name: tensor.name.clone(),
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
            });
            selected.push(tensor);
        }
    }

    if selected.is_empty() {
        return Err(format!(
            "no transformer_blocks tensors for {} blocks in {}",
            blocks,
            path.display()
        ));
    }

    let metadata = shard.metadata.clone();
    let writer = ShardWriter::create(target, &entries, &metadata).map_err(|error| error.token())?;
    let mut zeroed = 0;

    for (index, tensor) in selected.iter().enumerate() {
        let zero = RESIDUAL_SUFFIXES
            .iter()
            .any(|suffix| tensor.name.ends_with(suffix));
        let mut bytes = vec![0u8; tensor.bytes() as usize];

        if !zero {
            shard
                .read_data(tensor, &mut bytes)
                .map_err(|error| error.token())?;
        } else {
            zeroed += 1;
        }

        writer.write(index, &bytes).map_err(|error| error.token())?;
    }

    Ok(zeroed)
}

/// `STAGE2_SIGMAS[0]`, the release's own first step
/// (`runtime/stage2_ops/models.py:19`).
const SIGMA: f32 = 0.909375;

/// The whole per-step graph: `prepare` over synthetic latents, the blocks, and
/// the head. The preprocessing's own buffers are reported under `--signals`
/// (they are the largest tensors in the graph) and dumped like the states, so a
/// golden from the reference is a diff and a cross host is a residual.
fn run_edges(gpu: &Gpu, path: &Path, options: &Options) -> Result<ExitCode, String> {
    let rows = options.rows.unwrap_or(VIDEO_TOKENS as usize) as u32;
    let audio_rows = options.audio_rows.unwrap_or(AUDIO_TOKENS as usize) as u32;
    let seed = options.seed.unwrap_or(1);
    let sigma = options.sigma.unwrap_or(SIGMA);
    let geometry = geometry_for(rows, audio_rows)?;

    let started = Instant::now();
    let edges = Edges::open(gpu, path).map_err(|error| error.token())?;
    let (tensors, bytes) = edges.loaded();
    let load = started.elapsed().as_secs_f64();
    let scratch = EdgeScratch::new(gpu, &geometry).map_err(|error| error.token())?;

    let mut lcg = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let video = draw(gpu, geometry.video_tokens(), CHANNELS, &mut lcg)?;
    let audio = draw(
        gpu,
        audio_rows,
        AUDIO_CHANNELS as usize * AUDIO_BINS as usize,
        &mut lcg,
    )?;
    let v_context = context(gpu, VIDEO_DIM, &mut lcg)?;
    let a_context = context(gpu, AUDIO_DIM, &mut lcg)?;

    println!(
        "EDGES tensors={tensors} bytes={bytes} video_rows={} audio_rows={audio_rows} sigma={sigma} load={load:.3}",
        geometry.video_tokens()
    );

    let request = Request {
        sigma,
        geometry,
        video: &video,
        audio: Some(&audio),
        video_context: &v_context,
        audio_context: Some(&a_context),
    };

    let started = Instant::now();
    let prepared = edges
        .prepare(gpu, &request, &scratch)
        .map_err(|error| error.token())?;
    let prepare_seconds = started.elapsed().as_secs_f64();
    let audio_state = prepared
        .audio
        .as_ref()
        .ok_or_else(|| "the edges run asked for an audio state".to_string())?;
    let audio_signals = prepared
        .audio_signals
        .as_ref()
        .ok_or_else(|| "the edges run asked for audio signals".to_string())?;
    let video_cross = prepared
        .video_signals
        .cross
        .as_ref()
        .ok_or_else(|| "the edges run asked for the video cross group".to_string())?;
    let audio_cross = audio_signals
        .cross
        .as_ref()
        .ok_or_else(|| "the edges run asked for the audio cross group".to_string())?;

    let mut failed = false;
    let mut expect = options.expect.clone();

    if let Some(zeroed) = &options.keyframe {
        let count = edge_shard(path, zeroed)?;
        println!("KEYFRAME shard={} zeroed={count}", zeroed.display());
        failed |= keyframe(&gpu, path, zeroed, &request, &prepared.video)?;
    }

    if options.signals {
        for (name, tensor) in [
            ("edges-video-state", &prepared.video),
            ("edges-audio-state", audio_state),
            ("edges-video-modulation", &prepared.video_signals.modulation),
            ("edges-video-cross-scale-shift", &video_cross.scale_shift),
            ("edges-video-cross-gate", &video_cross.gate),
            (
                "edges-video-prompt-modulation",
                &prepared.video_signals.prompt_modulation,
            ),
            ("edges-video-self-cos", &prepared.video_signals.self_cos),
            ("edges-video-self-sin", &prepared.video_signals.self_sin),
            ("edges-video-cross-cos", &video_cross.cos),
            ("edges-audio-modulation", &audio_signals.modulation),
            ("edges-audio-self-cos", &audio_signals.self_cos),
            ("edges-audio-cross-scale-shift", &audio_cross.scale_shift),
        ] {
            failed |= report_state(
                gpu,
                name,
                tensor,
                &mut expect,
                options.dump.as_deref(),
                options.residual.as_deref(),
                0.0,
            )?;
        }
    }

    // The blocks over the prepared states, one report per block.
    let blocks = options.blocks.unwrap_or(1);
    let refiner = Refiner::open(gpu, path, &(0..blocks).collect::<Vec<usize>>())
        .map_err(|error| error.token())?;
    let refiner_scratch =
        RefinerScratch::new(gpu, rows, audio_rows, CONTEXT_ROWS).map_err(|error| error.token())?;
    let started = Instant::now();

    for block in 0..refiner.blocks() {
        let started = Instant::now();
        refiner
            .apply_one(
                block,
                gpu,
                &prepared.video,
                &prepared.video_signals,
                Some((audio_state, audio_signals)),
                &refiner_scratch,
            )
            .map_err(|error| error.token())?;
        let seconds = started.elapsed().as_secs_f64();

        failed |= report_state(
            gpu,
            &format!("edges-block{block}-video"),
            &prepared.video,
            &mut expect,
            options.dump.as_deref(),
            options.residual.as_deref(),
            seconds,
        )?;
    }

    let blocks_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    edges
        .head(gpu, &prepared, &scratch)
        .map_err(|error| error.token())?;
    let head_seconds = started.elapsed().as_secs_f64();

    failed |= report_state(
        gpu,
        "edges-head-video",
        edges.video_output(&scratch),
        &mut expect,
        options.dump.as_deref(),
        options.residual.as_deref(),
        head_seconds,
    )?;
    failed |= report_state(
        gpu,
        "edges-head-audio",
        edges.audio_output(&scratch),
        &mut expect,
        options.dump.as_deref(),
        options.residual.as_deref(),
        0.0,
    )?;

    for name in expect.keys() {
        println!("STAGE {name} expected but never produced");
        failed = true;
    }

    // The head's wiring, against a host transcription of the same expression.
    // Only at the small geometries: the transcription reads 1.19 GB back at the
    // release's own.
    if rows < VIDEO_TOKENS {
        for (name, dim, table, proj, state, embedded, output) in [
            (
                "video",
                VIDEO_DIM,
                "scale_shift_table",
                "proj_out",
                &prepared.video,
                edges.video_embedded(&scratch),
                edges.video_output(&scratch),
            ),
            (
                "audio",
                AUDIO_DIM,
                "audio_scale_shift_table",
                "audio_proj_out",
                audio_state,
                edges.audio_embedded(&scratch),
                edges.audio_output(&scratch),
            ),
        ] {
            failed |= head_check(gpu, path, name, dim, table, proj, state, embedded, output)?;
        }
    }

    println!(
        "EDGES-TIMING prepare={prepare_seconds:.3} blocks={blocks_seconds:.3} head={head_seconds:.3} load={load:.3}"
    );

    if failed {
        return Ok(ExitCode::from(3));
    }

    println!("LTX-REFINER-EDGES-OK");

    Ok(ExitCode::SUCCESS)
}

/// One modality's head against a host transcription of `_process_output`
/// (`model.py:484-489`): the table and the embedded timestep, the LayerNorm,
/// the modulation and the projection.
///
/// The transcription stops before `proj_out` and hands its own modulated values
/// to the *device's* linear with the same weights, so both sides run one
/// `h3d_linear` and the comparison is bit-level apart from the LayerNorm's
/// `rsqrtf`. That is what makes a wiring fault visible: a swapped table row, a
/// projection fed the pre-modulation tensor or a missing norm moves the output
/// by order 1, two orders above the residue.
#[allow(clippy::too_many_arguments)]
fn head_check(
    gpu: &Gpu,
    path: &Path,
    name: &str,
    dim: usize,
    table_name: &str,
    proj_name: &str,
    state: &Tensor,
    embedded: &Tensor,
    output: &Tensor,
) -> Result<bool, String> {
    let shard = safetensors::read_shard(path).map_err(|error| error.token())?;
    let at = "model.diffusion_model";
    let table = read_f32(&shard, &format!("{at}.{table_name}"))?;
    let weight = read_tensor(&shard, &format!("{at}.{proj_name}.weight"))?;
    let bias = read_tensor(&shard, &format!("{at}.{proj_name}.bias"))?;
    let state = read_bf16(gpu, state)?;
    let embedded = read_bf16(gpu, embedded)?;
    let rows = state.len() / dim;
    let mut modulated = Vec::with_capacity(state.len());

    for row in 0..rows {
        let (x, e) = (&state[row * dim..][..dim], &embedded[row * dim..][..dim]);
        let mean = x.iter().sum::<f32>() / dim as f32;
        let variance = x
            .iter()
            .map(|value| (value - mean).mul_add(value - mean, 0.0))
            .sum::<f32>()
            / dim as f32;
        let inverse = (variance + 1e-6).sqrt().recip();

        for column in 0..dim {
            let shift = bf16(table[column] + e[column]);
            let scale = bf16(table[dim + column] + e[column]);
            let normed = bf16((x[column] - mean) * inverse);

            modulated.push(bf16(bf16(normed * bf16(1.0 + scale)) + shift));
        }
    }

    let input = upload_bf16(gpu, &pack_bf16(&modulated))?;
    let want = alloc(gpu, Device::Bf16, rows * CHANNELS)?;
    gpu.linear(
        &want,
        &input,
        &upload_bf16(gpu, &weight)?,
        Some(&upload_bf16(gpu, &bias)?),
        rows as u32,
        dim as u32,
        CHANNELS as u32,
        Device::Bf16,
    )
    .map_err(|detail| format!("head linear: {detail}"))?;

    let want = read_bf16(gpu, &want)?;

    Ok(report(
        &format!("head_{name}"),
        &read_bf16(gpu, output)?,
        &want,
        Verdict::Residual,
    ))
}

/// One tensor's bytes from the shard, by name.
fn read_tensor(shard: &Shard, name: &str) -> Result<Vec<u8>, String> {
    let tensor = shard
        .find(name)
        .ok_or_else(|| format!("the shard carries no {name}"))?;
    let mut bytes = vec![0u8; tensor.bytes() as usize];

    shard
        .read_data(tensor, &mut bytes)
        .map_err(|error| error.token())?;

    Ok(bytes)
}

/// One F32 table from the shard, by name.
fn read_f32(shard: &Shard, name: &str) -> Result<Vec<f32>, String> {
    Ok(read_tensor(shard, name)?
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

/// Write `path`'s tensors outside the 48 blocks to `target` with the keyframe
/// embedding zeroed, and return how many tensors were zeroed.
fn edge_shard(path: &Path, target: &Path) -> Result<usize, String> {
    let shard = safetensors::read_shard(path).map_err(|error| error.token())?;
    let mut entries = Vec::new();
    let mut selected = Vec::new();

    for tensor in &shard.tensors {
        // The connector's own towers are another unit's; only the refiner's
        // edges belong in this shard.
        if tensor
            .name
            .starts_with("model.diffusion_model.transformer_blocks.")
            || tensor.name.contains("embeddings_connector")
        {
            continue;
        }

        entries.push(Entry {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
        });
        selected.push(tensor);
    }

    if selected.is_empty() {
        return Err(format!("no edge tensors in {}", path.display()));
    }

    let metadata = shard.metadata.clone();
    let writer = ShardWriter::create(target, &entries, &metadata).map_err(|error| error.token())?;
    let mut zeroed = 0;

    for (index, tensor) in selected.iter().enumerate() {
        let zero = tensor.name.ends_with("keyframes_abs_pos_embedding");
        let mut bytes = vec![0u8; tensor.bytes() as usize];

        if !zero {
            shard
                .read_data(tensor, &mut bytes)
                .map_err(|error| error.token())?;
        } else {
            zeroed += 1;
        }

        writer.write(index, &bytes).map_err(|error| error.token())?;
    }

    Ok(zeroed)
}

/// The keyframe term, isolated and oracle-free: against a shard whose
/// `keyframes_abs_pos_embedding` is zero the state *is* the projected latent,
/// so the real state must differ from it by exactly the embedding row, on
/// exactly the rows `_first_frame_keyframes_mask` marks and nowhere else.
fn keyframe(
    gpu: &Gpu,
    path: &Path,
    zeroed: &Path,
    request: &Request<'_>,
    real: &Tensor,
) -> Result<bool, String> {
    let edges = Edges::open(gpu, zeroed).map_err(|error| error.token())?;
    let scratch = EdgeScratch::new(gpu, &request.geometry).map_err(|error| error.token())?;
    let plain = edges
        .prepare(gpu, request, &scratch)
        .map_err(|error| error.token())?;

    let shard = safetensors::read_shard(path).map_err(|error| error.token())?;
    let entry = shard
        .find("model.diffusion_model.keyframes_abs_pos_embedding")
        .ok_or_else(|| "the shard carries no keyframe embedding".to_string())?;
    let mut bytes = vec![0u8; entry.bytes() as usize];

    shard
        .read_data(entry, &mut bytes)
        .map_err(|error| error.token())?;

    let embedding: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
        .collect();
    let marked = request.geometry.frame_tokens() as usize;
    let got = read_bf16(gpu, real)?;
    let base = read_bf16(gpu, &plain.video)?;
    let mut differ = 0usize;

    for (row, (got_row, base_row)) in got
        .chunks_exact(VIDEO_DIM)
        .zip(base.chunks_exact(VIDEO_DIM))
        .enumerate()
    {
        for column in 0..VIDEO_DIM {
            let want = if row < marked {
                bf16(base_row[column] + embedding[column])
            } else {
                base_row[column]
            };

            if got_row[column].to_bits() != want.to_bits() {
                differ += 1;
            }
        }
    }

    println!(
        "KEYFRAME marked_rows={marked} rows={} differ={differ} {}",
        got.len() / VIDEO_DIM,
        if differ == 0 { "pass" } else { "FAIL" }
    );

    Ok(differ != 0)
}

/// The stage-2 sampler's device steps (`minimax_h3::ltx_stage2`), each against
/// a host transcription of its reference expression.
///
/// The noise draw has no transcription and is not given one:
/// `curand_normal4` is `sqrtf`, `logf` and `__sincosf` on the device's own
/// units, so a host reimplementation of Box-Muller would disagree in the last
/// bits of every value - which is the whole reason the port runs the draw on
/// the device. What is checkable here is the three things a wiring fault would
/// break: every element is written, the same generator state draws the same
/// values, and a draw at the advanced offset draws different ones.
///
/// The four steps whose reference reads `a + b*c` (`noise_lerp`,
/// `to_denoised`, `euler_step`, `post_latent`) are reported as **residues**:
/// nvcc contracts to FMA by default and the host transcription does not, so a
/// one-ulp difference is expected, and a swapped operand or a dropped mask
/// moves the value by orders more. The two whose reference has no such product
/// (`timesteps`, `to_velocity`) must be bit-identical.
fn probe_sampler(gpu: &Gpu, state: &mut u32) -> Result<bool, String> {
    let rows = 5u32;
    let width = 8usize;
    let elements = rows as usize * width;
    let sigma = SIGMAS[0];
    let delta = SIGMAS[1] - SIGMAS[0];
    let mut failed = false;

    let noise = draw(gpu, rows, width, state)?;
    let clean = draw(gpu, rows, width, state)?;
    let latent = draw(gpu, rows, width, state)?;
    let velocity = draw(gpu, rows, width, state)?;
    let denoised = draw(gpu, rows, width, state)?;
    let (noise_values, clean_values, latent_values) = (
        read_bf16(gpu, &noise)?,
        read_bf16(gpu, &clean)?,
        read_bf16(gpu, &latent)?,
    );
    let (velocity_values, denoised_values) =
        (read_bf16(gpu, &velocity)?, read_bf16(gpu, &denoised)?);

    // The mask: the release's is all ones, but a probe that only ever ran it
    // at 1 could not tell a transcribed mask from a dropped one.
    let mask_values: Vec<f32> = (0..rows)
        .map(|row| match row {
            0 => 1.0,
            1 => 0.0,
            2 => 0.5,
            3 => 0.25,
            _ => 0.75,
        })
        .collect();
    let mask = upload_f32(gpu, &mask_values)?;

    // `timesteps_from_mask` (`helpers.py:494-501`).
    let timestep = alloc(gpu, Device::F32, rows as usize)?;
    gpu.timesteps(&timestep, &mask, rows, sigma)
        .map_err(|detail| format!("timesteps: {detail}"))?;
    let got = read_f32_tensor(gpu, &timestep)?;
    let timestep_values: Vec<f32> = mask_values.iter().map(|value| value * sigma).collect();
    failed |= report("timesteps", &got, &timestep_values, Verdict::Exact);

    // `to_velocity` (`utils.py:21-35`): `((sample - denoised)/sigma)` in F32,
    // one BF16 round.
    let output = alloc(gpu, Device::Bf16, elements)?;
    gpu.to_velocity(&output, &latent, &denoised, sigma)
        .map_err(|detail| format!("to_velocity: {detail}"))?;
    let got = read_bf16(gpu, &output)?;
    let want: Vec<f32> = latent_values
        .iter()
        .zip(denoised_values.iter())
        .map(|(sample, denoised)| bf16((sample - denoised) / sigma))
        .collect();
    failed |= report("to_velocity", &got, &want, Verdict::Exact);

    // `to_denoised` (`utils.py:39-52`) at the per-token timestep.
    let output = alloc(gpu, Device::Bf16, elements)?;
    gpu.to_denoised(&output, &latent, &velocity, &timestep, rows, width as u32)
        .map_err(|detail| format!("to_denoised: {detail}"))?;
    let got = read_bf16(gpu, &output)?;
    let want: Vec<f32> = (0..elements)
        .map(|index| {
            bf16(latent_values[index] - velocity_values[index] * timestep_values[index / width])
        })
        .collect();
    failed |= report("to_denoised", &got, &want, Verdict::Residual);

    // `EulerDiffusionStep.step` (`diffusion_steps.py:31-40`).
    let output = alloc(gpu, Device::Bf16, elements)?;
    gpu.euler_step(&output, &latent, &velocity, delta)
        .map_err(|detail| format!("euler_step: {detail}"))?;
    let got = read_bf16(gpu, &output)?;
    let want: Vec<f32> = latent_values
        .iter()
        .zip(velocity_values.iter())
        .map(|(sample, velocity)| bf16(sample + velocity * delta))
        .collect();
    failed |= report("euler_step", &got, &want, Verdict::Residual);

    // `post_process_latent` (`helpers.py:461-464`).
    let output = alloc(gpu, Device::Bf16, elements)?;
    gpu.post_latent(&output, &denoised, &mask, &clean, rows, width as u32)
        .map_err(|detail| format!("post_latent: {detail}"))?;
    let got = read_bf16(gpu, &output)?;
    let want: Vec<f32> = (0..elements)
        .map(|index| {
            let row = mask_values[index / width];
            bf16(denoised_values[index] * row + clean_values[index] * (1.0 - row))
        })
        .collect();
    failed |= report("post_latent", &got, &want, Verdict::Residual);

    // `GaussianNoiser.__call__` (`noisers.py:30-37`) at `noise_scale`.
    let output = alloc(gpu, Device::Bf16, elements)?;
    gpu.noise_lerp(
        &output,
        &latent,
        &noise,
        &clean,
        &mask,
        rows,
        width as u32,
        NOISE_SCALE,
    )
    .map_err(|detail| format!("noise_lerp: {detail}"))?;
    let got = read_bf16(gpu, &output)?;
    let want: Vec<f32> = (0..elements)
        .map(|index| {
            let weight = mask_values[index / width];
            let noisy = lerp(latent_values[index], noise_values[index], NOISE_SCALE);
            bf16(lerp(clean_values[index], noisy, weight))
        })
        .collect();
    failed |= report("noise_lerp", &got, &want, Verdict::Residual);

    failed |= probe_randn(gpu, state)?;

    Ok(failed)
}

/// torch's own CUDA normal draw: it has no host transcription, so the probe
/// pins the invariants instead - a full fill, determinism at one generator
/// state, and a different draw once the offset has advanced.
fn probe_randn(gpu: &Gpu, state: &mut u32) -> Result<bool, String> {
    let elements = 4_097usize;
    let sentinel = 1.0e30f32;
    let mut failed = false;

    // An unfilled element would keep the sentinel: `elements` is deliberately
    // not a multiple of the 256 * grid * 4 grid-stride width.
    let output = upload_bf16(
        gpu,
        &vec![(bf16(sentinel).to_bits() >> 16) as u8; elements * 2],
    )?;
    let mut generator = Generator::seeded(20_251_025);
    generator
        .randn(gpu, &output)
        .map_err(|error| error.token())?;

    let first = read_bf16(gpu, &output)?;
    let unfilled = first
        .iter()
        .filter(|value| **value == bf16(sentinel))
        .count();
    let moment = first.iter().map(|value| *value as f64).sum::<f64>() / elements as f64;
    let spread = (first
        .iter()
        .map(|value| (*value as f64 - moment).powi(2))
        .sum::<f64>()
        / elements as f64)
        .sqrt();

    // A 4k sample of a standard normal: the mean is within 5 standard errors
    // and the deviation within a few percent.
    let shaped = unfilled == 0
        && moment.abs() < 5.0 / (elements as f64).sqrt()
        && (spread - 1.0).abs() < 0.05;
    println!(
        "PROBE randn values={elements} unfilled={unfilled} mean={moment:.4} sd={spread:.4} verdict=device {}",
        if shaped { "pass" } else { "FAIL" }
    );
    failed |= !shaped;

    // The offset after one draw is torch's `calc_execution_policy` increment:
    // positive and a multiple of four.
    let increment = generator.offset();
    let advanced = increment > 0 && increment % 4 == 0;
    println!(
        "PROBE randn-offset increment={increment} verdict=multiple-of-four {}",
        if advanced { "pass" } else { "FAIL" }
    );
    failed |= !advanced;

    // The same generator state draws the same values; the advanced offset
    // draws different ones.
    let repeat = alloc(gpu, Device::Bf16, elements)?;
    let mut again = Generator::seeded(20_251_025);
    again.randn(gpu, &repeat).map_err(|error| error.token())?;
    let same = read_bf16(gpu, &repeat)?;
    let identical = same
        .iter()
        .zip(first.iter())
        .filter(|(left, right)| left.to_bits() == right.to_bits())
        .count();
    println!(
        "PROBE randn-repeat identical={identical}/{elements} verdict=deterministic {}",
        if identical == elements {
            "pass"
        } else {
            "FAIL"
        }
    );
    failed |= identical != elements;

    generator
        .randn(gpu, &output)
        .map_err(|error| error.token())?;
    let second = read_bf16(gpu, &output)?;
    let overlaps = second
        .iter()
        .zip(first.iter())
        .filter(|(left, right)| left.to_bits() == right.to_bits())
        .count();
    println!(
        "PROBE randn-advance identical={overlaps}/{elements} verdict=offset-advances {}",
        if overlaps < elements / 100 {
            "pass"
        } else {
            "FAIL"
        }
    );
    failed |= overlaps >= elements / 100;

    // Keep the lcg moving for the callers after this probe.
    *state = state.wrapping_add(elements as u32);

    Ok(failed)
}

/// `aten/src/ATen/native/Lerp.h:22-33` in F32: the arm depends on `|w| < 0.5`.
fn lerp(self_: f32, end: f32, weight: f32) -> f32 {
    if weight.abs() < 0.5 {
        self_ + weight * (end - self_)
    } else {
        end - (end - self_) * (1.0 - weight)
    }
}

/// The gate's geometry: the release's own at its token count, and a small
/// synthetic one otherwise, `--rows` video tokens split into two-token frames
/// so the keyframe mask covers a strict prefix.
fn geometry_for(rows: u32, audio_rows: u32) -> Result<Geometry, String> {
    let mut geometry = if rows == VIDEO_TOKENS {
        Geometry::release()
    } else {
        if rows % 2 != 0 {
            return Err(usage("--rows must be even off the release geometry"));
        }

        Geometry {
            frames: rows / 2,
            height: 1,
            width: 2,
            audio_frames: audio_rows,
        }
    };

    geometry.audio_frames = audio_rows;

    Ok(geometry)
}

/// Read one state back, hash it, and check it against `--expect` and a
/// previous dump.
#[allow(clippy::too_many_arguments)]
fn report_state(
    gpu: &Gpu,
    name: &str,
    tensor: &Tensor,
    expect: &mut HashMap<String, u64>,
    dump: Option<&Path>,
    residual: Option<&Path>,
    seconds: f64,
) -> Result<bool, String> {
    let values = read_bf16(gpu, tensor)?;
    let mut payload = vec![0u8; tensor.bytes()];
    gpu.read(tensor, &mut payload)
        .map_err(|detail| format!("read {name}: {detail}"))?;
    let hash = fnv1a(&payload);
    let (high, low) = extremes(&values);

    println!(
        "STAGE {name} values={} fnv={hash:016x} high={high:.6} low={low:.6} run={seconds:.3}",
        values.len()
    );

    let mut failed = false;

    if let Some(wanted) = expect.remove(name) {
        if wanted != hash {
            println!("STAGE {name} expected {wanted:016x}");
            failed = true;
        }
    }

    if let Some(directory) = dump {
        std::fs::create_dir_all(directory)
            .map_err(|error| format!("--dump {}: {error}", directory.display()))?;
        std::fs::write(directory.join(format!("{name}.bf16")), &payload)
            .map_err(|error| format!("--dump {name}: {error}"))?;
    }

    if let Some(directory) = residual {
        let file = directory.join(format!("{name}.bf16"));
        let reference = std::fs::read(&file)
            .map_err(|error| format!("--residual {}: {error}", file.display()))?;

        if reference.len() != payload.len() {
            return Err(format!(
                "--residual {name}: {} bytes against {}",
                reference.len(),
                payload.len()
            ));
        }

        let previous: Vec<f32> = reference
            .chunks_exact(2)
            .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
            .collect();
        let (differ, rel_rms) = compare(&values, &previous);

        println!(
            "RESIDUAL {name} values={} differ={differ} rel_rms={rel_rms:.3e} ceiling={RESIDUAL_CEILING:.1e}",
            values.len()
        );
        failed |= rel_rms > RESIDUAL_CEILING;
    }

    Ok(failed)
}

/// The probes: every new kernel against a host transcription of the same
/// expression. Each ends in `report`, which returns whether the probe failed.
fn probe(gpu: &Gpu, rows: usize, seed: u32) -> Result<bool, String> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let width = 64usize;
    let coefficient = 9usize;
    let mut failed = false;

    // `get_ada_values` (`transformer.py:190-200`): one table row plus one
    // coefficient row, one BF16 addition.
    let (table_values, table_bytes) = seq(coefficient * width, &mut state);
    let (modulation_values, modulation_bytes) = seq(rows * coefficient * width, &mut state);
    let table = upload_bf16(gpu, &table_bytes)?;
    let modulation = upload_bf16(gpu, &modulation_bytes)?;
    let output = alloc(gpu, Device::Bf16, rows * width)?;
    let table_row = coefficient - 1;
    let value_row = coefficient - 2;

    gpu.ada_add(
        &output,
        &table,
        &modulation,
        rows as u32,
        width as u32,
        coefficient as u32,
        table_row as u32,
        value_row as u32,
    )
    .map_err(|detail| format!("ada add: {detail}"))?;

    let got = read_bf16(gpu, &output)?;
    let want: Vec<f32> = (0..rows * width)
        .map(|index| {
            let row = index / width;
            let column = index % width;

            bf16(
                table_values[table_row * width + column]
                    + modulation_values[(row * coefficient + value_row) * width + column],
            )
        })
        .collect();

    failed |= report("ada_add", &got, &want, Verdict::Exact);

    // `x * (1 + scale) + shift` (`transformer.py:446`, `model.py:487-488`):
    // three ops, three roundings, at both the per-token and the per-request
    // broadcast.
    for scale_rows in [rows, 1] {
        let (input_values, input_bytes) = seq(rows * width, &mut state);
        let (scale_values, scale_bytes) = seq(scale_rows * width, &mut state);
        let (shift_values, shift_bytes) = seq(scale_rows * width, &mut state);
        let input = upload_bf16(gpu, &input_bytes)?;
        let scale = upload_bf16(gpu, &scale_bytes)?;
        let shift = upload_bf16(gpu, &shift_bytes)?;
        let output = alloc(gpu, Device::Bf16, rows * width)?;

        gpu.modulate(
            &output,
            &input,
            &scale,
            &shift,
            rows as u32,
            width as u32,
            scale_rows as u32,
        )
        .map_err(|detail| format!("modulate: {detail}"))?;

        let got = read_bf16(gpu, &output)?;
        let want: Vec<f32> = (0..rows * width)
            .map(|index| {
                let row = index / width;
                let column = index % width;
                let source = if scale_rows == 1 { 0 } else { row };
                let factor = bf16(1.0 + scale_values[source * width + column]);
                let product = bf16(input_values[index] * factor);

                bf16(product + shift_values[source * width + column])
            })
            .collect();

        failed |= report(
            &format!("modulate rows={scale_rows}"),
            &got,
            &want,
            Verdict::Exact,
        );
    }

    // `attn_out * gate_out` (`transformer.py:362`): one rounding, at both
    // broadcasts.
    for right_rows in [rows, 1] {
        let (left_values, left_bytes) = seq(rows * width, &mut state);
        let (right_values, right_bytes) = seq(right_rows * width, &mut state);
        let left = upload_bf16(gpu, &left_bytes)?;
        let right = upload_bf16(gpu, &right_bytes)?;
        let output = alloc(gpu, Device::Bf16, rows * width)?;

        gpu.mul_rows(
            &output,
            &left,
            &right,
            rows as u32,
            width as u32,
            right_rows as u32,
        )
        .map_err(|detail| format!("mul rows: {detail}"))?;

        let got = read_bf16(gpu, &output)?;
        let want: Vec<f32> = (0..rows * width)
            .map(|index| {
                let row = index / width;
                let column = index % width;
                let source = if right_rows == 1 { 0 } else { row };

                bf16(left_values[index] * right_values[source * width + column])
            })
            .collect();

        failed |= report(
            &format!("mul_rows rows={right_rows}"),
            &got,
            &want,
            Verdict::Exact,
        );
    }

    // `PytorchAdaZeroFunction` (`ops.py:50-58`) and the output head's
    // LayerNorm (`model.py:232`): both end in `rsqrtf`, so these are residues.
    let (input_values, input_bytes) = seq(rows * width, &mut state);
    let (scale_values, scale_bytes) = seq(rows * width, &mut state);
    let (shift_values, shift_bytes) = seq(rows * width, &mut state);
    let input = upload_bf16(gpu, &input_bytes)?;
    let scale = upload_bf16(gpu, &scale_bytes)?;
    let shift = upload_bf16(gpu, &shift_bytes)?;
    let output = alloc(gpu, Device::Bf16, rows * width)?;

    gpu.ada_zero(
        &output,
        &input,
        &scale,
        &shift,
        rows as u32,
        width as u32,
        rows as u32,
        1e-6,
    )
    .map_err(|detail| format!("ada zero: {detail}"))?;

    let got = read_bf16(gpu, &output)?;
    let mut want = vec![0.0f32; rows * width];

    for row in 0..rows {
        let slice = &input_values[row * width..][..width];
        let inverse = (slice.iter().map(|value| value * value).sum::<f32>() / width as f32 + 1e-6)
            .sqrt()
            .recip();

        for column in 0..width {
            let index = row * width + column;
            let normalized = bf16(input_values[index] * inverse);
            let factor = bf16(1.0 + scale_values[index]);

            want[index] = bf16(bf16(normalized * factor) + shift_values[index]);
        }
    }

    failed |= report("ada_zero", &got, &want, Verdict::Residual);

    let output = alloc(gpu, Device::Bf16, rows * width)?;
    gpu.layer_norm(&output, &input, rows as u32, width as u32, 1e-6)
        .map_err(|detail| format!("layer norm: {detail}"))?;

    let got = read_bf16(gpu, &output)?;
    let mut want = vec![0.0f32; rows * width];

    for row in 0..rows {
        let slice = &input_values[row * width..][..width];
        let mean = slice.iter().sum::<f32>() / width as f32;
        let variance = slice
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / width as f32;
        let inverse = (variance + 1e-6).sqrt().recip();

        for column in 0..width {
            let index = row * width + column;
            want[index] = bf16((input_values[index] - mean) * inverse);
        }
    }

    failed |= report("layer_norm", &got, &want, Verdict::Residual);

    // `apply_split_rotary_emb` (`rope.py:42-85`) with two tables and two row
    // counts, the AV pair's `pe`/`k_pe` (`transformer.py:359-361`).
    failed |= probe_rope_pair(gpu, &mut state)?;

    // The patchifier's transpose (`components/patchifiers.py:27-64`) at patch
    // size 1: a pure permutation, so it must be exact at sizes that straddle
    // the tile.
    failed |= probe_transpose(gpu, &mut state)?;

    // The audio patchifier's own gather (`:293-304`), also a permutation.
    failed |= probe_audio_patchify(gpu, &mut state)?;

    // `apply_keyframes_absolute_embedding` (`transformer_args.py:36-43`): one
    // BF16 add of the `[width]` embedding on the rows the mask marks.
    failed |= probe_keyframe(gpu, &mut state)?;

    // The stage-2 sampler's seven steps.
    failed |= probe_sampler(gpu, &mut state)?;

    Ok(failed)
}

fn probe_audio_patchify(gpu: &Gpu, state: &mut u32) -> Result<bool, String> {
    let (channels, frames, bins) = (2usize, 3usize, 5usize);
    let (values, bytes) = seq(channels * frames * bins, state);
    let input = upload_bf16(gpu, &bytes)?;
    let output = alloc(gpu, Device::Bf16, frames * channels * bins)?;

    gpu.audio_patchify(
        &output,
        &input,
        1,
        channels as u32,
        frames as u32,
        bins as u32,
    )
    .map_err(|detail| format!("audio patchify: {detail}"))?;

    let got = read_bf16(gpu, &output)?;
    let mut want = Vec::with_capacity(got.len());

    for frame in 0..frames {
        for channel in 0..channels {
            for bin in 0..bins {
                want.push(values[(channel * frames + frame) * bins + bin]);
            }
        }
    }

    Ok(report("audio_patchify", &got, &want, Verdict::Exact))
}

fn probe_keyframe(gpu: &Gpu, state: &mut u32) -> Result<bool, String> {
    let (rows, width) = (6usize, 4usize);
    let (input_values, input_bytes) = seq(rows * width, state);
    let (embedding_values, embedding_bytes) = seq(width, state);
    let mut mask_values = vec![0.0f32; rows];

    mask_values[..3].fill(1.0);

    let input = upload_bf16(gpu, &input_bytes)?;
    let embedding = upload_bf16(gpu, &embedding_bytes)?;
    let mask = upload_bf16(gpu, &pack_bf16(&mask_values))?;
    let output = alloc(gpu, Device::Bf16, rows * width)?;

    gpu.keyframe_add(
        &output,
        &input,
        &mask,
        &embedding,
        rows as u32,
        width as u32,
    )
    .map_err(|detail| format!("keyframe add: {detail}"))?;

    let got = read_bf16(gpu, &output)?;
    let want: Vec<f32> = (0..rows * width)
        .map(|index| {
            let (row, column) = (index / width, index % width);
            let marked = if mask_values[row] != 0.0 {
                embedding_values[column]
            } else {
                0.0
            };

            bf16(input_values[index] + marked)
        })
        .collect();

    Ok(report("keyframe_add", &got, &want, Verdict::Exact))
}

/// A BF16 byte block from values the caller already rounded.
fn pack_bf16(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);

    for value in values {
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }

    bytes
}

fn probe_rope_pair(gpu: &Gpu, state: &mut u32) -> Result<bool, String> {
    let (query_rows, key_rows, heads, head_dim) = (8usize, 4usize, 2usize, 8usize);
    let half = head_dim / 2;
    let (query_values, query_bytes) = seq(query_rows * heads * head_dim, state);
    let (key_values, key_bytes) = seq(key_rows * heads * head_dim, state);
    let (q_cos_values, q_cos_bytes) = seq(query_rows * heads * half, state);
    let (q_sin_values, q_sin_bytes) = seq(query_rows * heads * half, state);
    let (k_cos_values, k_cos_bytes) = seq(key_rows * heads * half, state);
    let (k_sin_values, k_sin_bytes) = seq(key_rows * heads * half, state);
    let query = upload_bf16(gpu, &query_bytes)?;
    let key = upload_bf16(gpu, &key_bytes)?;
    let q_cos = upload_bf16(gpu, &q_cos_bytes)?;
    let q_sin = upload_bf16(gpu, &q_sin_bytes)?;
    let k_cos = upload_bf16(gpu, &k_cos_bytes)?;
    let k_sin = upload_bf16(gpu, &k_sin_bytes)?;

    gpu.ltx_rope_pair(
        &query,
        &key,
        &q_cos,
        &q_sin,
        &k_cos,
        &k_sin,
        query_rows as u32,
        key_rows as u32,
        heads as u32,
        head_dim as u32,
    )
    .map_err(|detail| format!("rope pair: {detail}"))?;

    let got_query = read_bf16(gpu, &query)?;
    let got_key = read_bf16(gpu, &key)?;
    let mut want_query = query_values.clone();
    let mut want_key = key_values.clone();

    rotate(
        &mut want_query,
        &q_cos_values,
        &q_sin_values,
        query_rows,
        heads,
        head_dim,
    );
    rotate(
        &mut want_key,
        &k_cos_values,
        &k_sin_values,
        key_rows,
        heads,
        head_dim,
    );

    let mut failed = report("rope_pair query", &got_query, &want_query, Verdict::Exact);
    failed |= report("rope_pair key", &got_key, &want_key, Verdict::Exact);

    Ok(failed)
}

/// `apply_split_rotary_emb`'s half swap with `addcmul_` (`rope.py:66-77`).
fn rotate(
    values: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;

    for row in 0..rows {
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            let table = (row * heads + head) * half;

            for d in 0..half {
                let first = values[base + d];
                let second = values[base + half + d];
                let c = cos[table + d];
                let s = sin[table + d];
                let turned_first = bf16(first * c);
                let turned_second = bf16(second * c);

                values[base + d] = bf16((-s).mul_add(second, turned_first));
                values[base + half + d] = bf16(s.mul_add(first, turned_second));
            }
        }
    }
}

fn probe_transpose(gpu: &Gpu, state: &mut u32) -> Result<bool, String> {
    let mut failed = false;

    for (rows, width) in [(5usize, 3usize), (33, 35)] {
        let (input_values, input_bytes) = seq(rows * width, state);
        let input = upload_bf16(gpu, &input_bytes)?;
        let output = alloc(gpu, Device::Bf16, rows * width)?;

        gpu.channel_transpose(&output, &input, rows as u32, width as u32)
            .map_err(|detail| format!("channel transpose: {detail}"))?;

        let got = read_bf16(gpu, &output)?;
        let mut want = Vec::with_capacity(rows * width);

        for column in 0..width {
            for row in 0..rows {
                want.push(input_values[row * width + column]);
            }
        }

        failed |= report(
            &format!("channel_transpose {rows}x{width}"),
            &got,
            &want,
            Verdict::Exact,
        );
    }

    Ok(failed)
}

/// A modality's synthetic modulation and positional tables: the shapes the
/// preprocessing produces, drawn from the gate's LCG. The values are not the
/// reference's; the point is that the block consumes them exactly as the
/// reference's own `TransformerArgs` does. The context is the caller's, since
/// the signals borrow it - `prepare` copies nothing.
fn signals<'a>(
    gpu: &Gpu,
    rows: u32,
    dim: usize,
    head_dim: usize,
    context: &'a Tensor,
    state: &mut u32,
) -> Result<Signals<'a>, String> {
    let cross_half = AUDIO_DIM / HEADS / 2;

    let modulation = draw(gpu, rows, dim * 9, state)?;
    let cross_scale_shift = draw(gpu, rows, dim * 4, state)?;
    let cross_gate = draw(gpu, 1, dim, state)?;
    let prompt_modulation = draw(gpu, 1, dim * 2, state)?;
    let self_cos = draw(gpu, rows * HEADS as u32, head_dim / 2, state)?;
    let self_sin = draw(gpu, rows * HEADS as u32, head_dim / 2, state)?;
    let cross_cos = draw(gpu, rows * HEADS as u32, cross_half, state)?;
    let cross_sin = draw(gpu, rows * HEADS as u32, cross_half, state)?;

    Ok(Signals {
        rows,
        dim: dim as u32,
        modulation,
        cross: Some(Cross {
            scale_shift: cross_scale_shift,
            gate: cross_gate,
            cos: cross_cos,
            sin: cross_sin,
        }),
        prompt_modulation,
        context,
        context_rows: CONTEXT_ROWS,
        self_cos,
        self_sin,
    })
}

/// The prompt context a modality's text cross-attention reads, at the
/// modality's own width.
fn context(gpu: &Gpu, dim: usize, state: &mut u32) -> Result<Tensor, String> {
    let width = if dim == VIDEO_DIM {
        VIDEO_CONTEXT_DIM
    } else {
        AUDIO_CONTEXT_DIM
    };

    draw(gpu, CONTEXT_ROWS, width, state)
}

/// Whether a probe's reference expression contains a libm call.
#[derive(Clone, Copy, PartialEq)]
enum Verdict {
    Exact,
    Residual,
}

fn report(name: &str, got: &[f32], want: &[f32], verdict: Verdict) -> bool {
    let (differ, rel_rms) = compare(got, want);
    let pass = match verdict {
        Verdict::Exact => differ == 0,
        Verdict::Residual => rel_rms <= RESIDUAL_CEILING,
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

fn extremes(values: &[f32]) -> (f32, f32) {
    let mut high = f32::NEG_INFINITY;
    let mut low = f32::INFINITY;

    for value in values {
        high = high.max(*value);
        low = low.min(*value);
    }

    (high, low)
}

/// A BF16 value widened to f32, so both sides of a comparison see the same
/// input.
fn bf16(value: f32) -> f32 {
    f32::from_bits(
        value
            .to_bits()
            .wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1))
            & 0xffff_0000,
    )
}

/// One `[rows, width]` BF16 tensor from a seeded LCG, with its values.
fn synthetic(
    gpu: &Gpu,
    rows: u32,
    width: usize,
    state: &mut u32,
) -> Result<(Tensor, Vec<f32>), String> {
    let (values, bytes) = seq(rows as usize * width, state);
    let tensor = upload_bf16(gpu, &bytes)?;

    Ok((tensor, values))
}

/// The same draw without keeping the values: the real geometry's modulation is
/// 594 M elements, and the gate does not need them back.
fn draw(gpu: &Gpu, rows: u32, width: usize, state: &mut u32) -> Result<Tensor, String> {
    let count = rows as usize * width;
    let mut bytes = Vec::with_capacity(count * 2);

    for _ in 0..count {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (*state >> 8) as f32 / (1u32 << 24) as f32;
        let value = bf16((unit * 2.0 - 1.0) * 2.0);
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }

    upload_bf16(gpu, &bytes)
}

/// `n` BF16 values from a seeded LCG with occasional outliers, so an absmax
/// is not uniform.
fn seq(count: usize, state: &mut u32) -> (Vec<f32>, Vec<u8>) {
    let mut values = Vec::with_capacity(count);
    let mut bytes = Vec::with_capacity(count * 2);

    for _ in 0..count {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (*state >> 8) as f32 / (1u32 << 24) as f32;
        let value = bf16((unit * 2.0 - 1.0) * 2.0);
        values.push(value);
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }

    (values, bytes)
}

fn upload_bf16(gpu: &Gpu, bytes: &[u8]) -> Result<Tensor, String> {
    let tensor = gpu
        .alloc(Device::Bf16, bytes.len() / 2)
        .map_err(|detail| format!("alloc bf16: {detail}"))?;
    gpu.upload(&tensor, bytes)
        .map_err(|detail| format!("upload bf16: {detail}"))?;

    Ok(tensor)
}

fn read_bf16(gpu: &Gpu, tensor: &Tensor) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; tensor.bytes()];
    gpu.read(tensor, &mut bytes)
        .map_err(|detail| format!("read bf16: {detail}"))?;

    Ok(bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16))
        .collect())
}

fn upload_f32(gpu: &Gpu, values: &[f32]) -> Result<Tensor, String> {
    let tensor = alloc(gpu, Device::F32, values.len())?;
    let mut bytes = Vec::with_capacity(values.len() * 4);

    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    gpu.upload(&tensor, &bytes)
        .map_err(|detail| format!("upload f32: {detail}"))?;

    Ok(tensor)
}

fn read_f32_tensor(gpu: &Gpu, tensor: &Tensor) -> Result<Vec<f32>, String> {
    let mut values = vec![0.0f32; tensor.elements()];

    gpu.read_f32(tensor, &mut values)
        .map_err(|detail| format!("read f32: {detail}"))?;

    Ok(values)
}

/// One BF16 file dumped by another gate, as a tensor of `rows * width`
/// elements: the stage mode's real inputs come from the connector's dump.
fn read_tensor_file(gpu: &Gpu, path: &Path, elements: usize) -> Result<Tensor, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;

    if bytes.len() != elements * 2 {
        return Err(format!(
            "{} holds {} bytes, expected {}",
            path.display(),
            bytes.len(),
            elements * 2
        ));
    }

    upload_bf16(gpu, &bytes)
}

fn alloc(gpu: &Gpu, dtype: Device, elements: usize) -> Result<Tensor, String> {
    gpu.alloc(dtype, elements)
        .map_err(|detail| format!("alloc: {detail}"))
}

fn usage(detail: &str) -> String {
    format!(
        "{detail}\nusage: h3_ltx_refiner_gate --probe [--rows R] [--seed S]\n       \
         h3_ltx_refiner_gate --transformer PATH [--blocks K] [--rows R] [--audio-rows A]\n\
         \x20                            [--seed S] [--dump DIR] [--residual DIR]\n\
         \x20                            [--expect NAME=HEX ...]\n       \
         h3_ltx_refiner_gate --transformer PATH --identity FILE [--blocks K]\n       \
         h3_ltx_refiner_gate --edges --transformer PATH [--blocks K] [--rows R]\n\
         \x20                            [--audio-rows A] [--sigma S] [--signals]\n\
         \x20                            [--keyframe-zeroed PATH] [--dump DIR]\n\
         \x20                            [--residual DIR] [--expect NAME=HEX ...]\n       \
         h3_ltx_refiner_gate --stage --transformer PATH [--layers K] [--seed S]\n\
         \x20                            [--latent FILE] [--context FILE] [--dump DIR]\n\
         \x20                            [--residual DIR] [--expect NAME=HEX ...]"
    )
}

struct Options {
    transformer: Option<PathBuf>,
    identity: Option<PathBuf>,
    edges: bool,
    stage: bool,
    latent: Option<PathBuf>,
    prompt: Option<PathBuf>,
    layers: Option<usize>,
    keyframe: Option<PathBuf>,
    signals: bool,
    sigma: Option<f32>,
    blocks: Option<usize>,
    rows: Option<usize>,
    audio_rows: Option<usize>,
    seed: Option<u32>,
    probe: bool,
    dump: Option<PathBuf>,
    residual: Option<PathBuf>,
    expect: HashMap<String, u64>,
}

fn split(args: &[String]) -> Result<Options, String> {
    let mut transformer = None;
    let mut identity = None;
    let mut edges = false;
    let mut stage = false;
    let mut latent = None;
    let mut prompt = None;
    let mut layers = None;
    let mut keyframe = None;
    let mut signals = false;
    let mut sigma = None;
    let mut blocks = None;
    let mut rows = None;
    let mut audio_rows = None;
    let mut seed = None;
    let mut probe = false;
    let mut dump = None;
    let mut residual = None;
    let mut expect = HashMap::new();
    let mut index = 0usize;

    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;

        match flag {
            "--probe" => probe = true,
            "--edges" => edges = true,
            "--stage" => stage = true,
            "--signals" => signals = true,
            "--latent" => latent = Some(PathBuf::from(take(args, &mut index)?)),
            "--context" => prompt = Some(PathBuf::from(take(args, &mut index)?)),
            "--layers" => layers = Some(number(&take(args, &mut index)?)?),
            "--keyframe-zeroed" => keyframe = Some(PathBuf::from(take(args, &mut index)?)),
            "--sigma" => {
                sigma = Some(
                    take(args, &mut index)?
                        .parse::<f32>()
                        .map_err(|_| usage("--sigma is not a number"))?,
                )
            }
            "--transformer" => transformer = Some(PathBuf::from(take(args, &mut index)?)),
            "--identity" => identity = Some(PathBuf::from(take(args, &mut index)?)),
            "--blocks" => blocks = Some(number(&take(args, &mut index)?)?),
            "--rows" => rows = Some(number(&take(args, &mut index)?)?),
            "--audio-rows" => audio_rows = Some(number(&take(args, &mut index)?)?),
            "--seed" => seed = Some(number(&take(args, &mut index)?)? as u32),
            "--dump" => dump = Some(PathBuf::from(take(args, &mut index)?)),
            "--residual" => residual = Some(PathBuf::from(take(args, &mut index)?)),
            "--expect" => {
                let value = take(args, &mut index)?;
                let (name, hash) = value
                    .split_once('=')
                    .ok_or_else(|| usage("--expect wants NAME=HEX"))?;
                let hash = u64::from_str_radix(hash.trim_start_matches("0x"), 16)
                    .map_err(|_| usage("--expect wants NAME=HEX"))?;
                expect.insert(name.to_string(), hash);
            }
            other => return Err(usage(&format!("unknown flag {other}"))),
        }
    }

    if let Some(blocks) = blocks {
        if blocks == 0 {
            return Err(usage("--blocks must be at least 1"));
        }
    }

    if let Some(layers) = layers {
        if layers == 0 {
            return Err(usage("--layers must be at least 1"));
        }
    }

    Ok(Options {
        transformer,
        identity,
        edges,
        stage,
        latent,
        prompt,
        layers,
        keyframe,
        signals,
        sigma,
        blocks,
        rows,
        audio_rows,
        seed,
        probe,
        dump,
        residual,
        expect,
    })
}

fn take(args: &[String], index: &mut usize) -> Result<String, String> {
    let value = args
        .get(*index)
        .ok_or_else(|| usage("a flag is missing its value"))?
        .clone();
    *index += 1;

    Ok(value)
}

fn number(text: &str) -> Result<usize, String> {
    text.parse::<usize>()
        .map_err(|_| usage(&format!("{text} is not a number")))
}
