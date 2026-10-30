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
    Refiner, RefinerScratch, Signals, AUDIO_CONTEXT_DIM, AUDIO_DIM, AUDIO_HEAD_DIM, CONTEXT_ROWS,
    HEADS, VIDEO_CONTEXT_DIM, VIDEO_DIM, VIDEO_HEAD_DIM,
};
use ds4_core::minimax_h3::safetensors::{self, Entry, ShardWriter};
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

    run_blocks(&gpu, &transformer, &options)
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
    let v = signals(gpu, rows, VIDEO_DIM, VIDEO_HEAD_DIM, &mut lcg)?;
    let a = signals(gpu, audio_rows, AUDIO_DIM, AUDIO_HEAD_DIM, &mut lcg)?;

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
            .apply_one(block, gpu, &video, &audio, &v, &a, &scratch)
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
    let v = signals(gpu, rows, VIDEO_DIM, VIDEO_HEAD_DIM, &mut lcg)?;
    let a = signals(gpu, audio_rows, AUDIO_DIM, AUDIO_HEAD_DIM, &mut lcg)?;

    refiner
        .apply(gpu, &video, &audio, &v, &a, &scratch)
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

    Ok(failed)
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
/// preprocessing will produce, drawn from the gate's LCG. The values are not
/// the reference's; the point is that the block consumes them exactly as the
/// reference's own `TransformerArgs` does.
fn signals(
    gpu: &Gpu,
    rows: u32,
    dim: usize,
    head_dim: usize,
    state: &mut u32,
) -> Result<Signals, String> {
    let context_dim = if dim == VIDEO_DIM {
        VIDEO_CONTEXT_DIM
    } else {
        AUDIO_CONTEXT_DIM
    };
    let cross_half = AUDIO_DIM / HEADS / 2;

    let modulation = draw(gpu, rows, dim * 9, state)?;
    let cross_scale_shift = draw(gpu, rows, dim * 4, state)?;
    let cross_gate = draw(gpu, 1, dim, state)?;
    let prompt_modulation = draw(gpu, 1, dim * 2, state)?;
    let context = draw(gpu, CONTEXT_ROWS, context_dim, state)?;
    let self_cos = draw(gpu, rows * HEADS as u32, head_dim / 2, state)?;
    let self_sin = draw(gpu, rows * HEADS as u32, head_dim / 2, state)?;
    let cross_cos = draw(gpu, rows * HEADS as u32, cross_half, state)?;
    let cross_sin = draw(gpu, rows * HEADS as u32, cross_half, state)?;

    Ok(Signals {
        rows,
        dim: dim as u32,
        modulation,
        cross_scale_shift,
        cross_gate,
        prompt_modulation,
        context,
        context_rows: CONTEXT_ROWS,
        self_cos,
        self_sin,
        cross_cos,
        cross_sin,
    })
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
         h3_ltx_refiner_gate --transformer PATH --identity FILE [--blocks K]"
    )
}

struct Options {
    transformer: Option<PathBuf>,
    identity: Option<PathBuf>,
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

    Ok(Options {
        transformer,
        identity,
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
