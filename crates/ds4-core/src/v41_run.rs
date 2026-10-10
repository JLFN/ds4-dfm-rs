//! V4.1 (ds41) run surface (P5): the Rust host's side of the engine's
//! single-request generate.
//!
//! The engine's server path IS the one-shot generate with two callbacks
//! (`server_generate_v41.c:412-414` sets the prefill progress hook and calls
//! `ds4_engine_v41_generate_argmax` with `v41_emit`), so the port keeps that
//! shape: one bridge entry, an emit closure per produced token (returning
//! false stops, the CLI's EOS convention), a progress closure per prefill
//! chunk (returning false aborts).
//!
//! Engram rows are host work by design (P4-4): the hash and the shard preads
//! live here (`engram.rs`, verified against the golden erows), the pinned
//! buffers stay native, and the native hands the provider the exact block the
//! forward is about to run (`ds41_forward.h`).

use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use ds4_sys::{
    ds4_bridge_v41_generate, ds4_bridge_v41_set_dspark, ds4_bridge_v41_set_emit_trace,
    ds4_bridge_v41_set_graph, ds4_bridge_v41_set_prof,
};

use crate::engram::{EngramHash, EngramShard};
use crate::{identify_file, Error, GgufFile, Model, Result, TensorInventory, V41Wire};

/// The per-run switches (the engine's CLI globals, `cli_diag.c:58`).
#[derive(Clone, Debug)]
pub struct V41RunOptions<'a> {
    /// `--engram-dir`: the converter wrote the converting machine's absolute
    /// shard paths; the directory is swapped, the file name kept
    /// (`core_validate_v41.c:11-13`).
    pub engram_dir: Option<&'a str>,
    /// `--v41-no-engram` (diagnostic): skip the engram layers entirely.
    pub no_engram: bool,
    /// `--dspark` = Some(2), `--no-dspark` = Some(0); None leaves the native
    /// default (1, or the DS41_NO_DSPARK env fallback).
    pub dspark: Option<i32>,
    /// `--no-graph` = Some(false); None leaves the native default.
    pub graph: Option<bool>,
    /// `--dspark-verify K`: pin the round's k; 0 = the confidence scheduler.
    pub verify_k: i32,
    /// `--emit-trace`: the per-round `[dspark]` lines.
    pub emit_trace: bool,
    /// `--v41-prof`: per-layer ms and the full `[dspark]` lines.
    pub prof: bool,
}

impl Default for V41RunOptions<'_> {
    fn default() -> Self {
        Self {
            engram_dir: None,
            no_engram: false,
            dspark: None,
            graph: None,
            verify_k: 0,
            emit_trace: false,
            prof: false,
        }
    }
}

/// The engram rows provider: the hash tables and one open shard per engram
/// layer, plus the token history the n-gram window reads.
pub struct V41Feed {
    hash: EngramHash,
    shards: Vec<EngramShard>,
    weight_off: Vec<u64>,
    scale_off: Vec<u64>,
    head_dim: u32,
    cols: u32,
    stride: u32,
    hist: Vec<i32>,
}

impl V41Feed {
    /// Open the feed from the artifact: the wire (with the `--engram-dir`
    /// rewrite), the four hash-constant tensors, and the shards.  None when
    /// the model carries no engram layers (the forward then takes no feed).
    pub fn open(path: &Path, engram_dir: Option<&str>) -> Result<Option<Self>> {
        let g = GgufFile::open(path).map_err(asset_error)?;
        let inv = TensorInventory::open(path).map_err(asset_error)?;
        let identified = identify_file(&g).map_err(asset_error)?;
        let mut wire = V41Wire::load(&g, &identified.shape).map_err(asset_error)?;
        if let Some(dir) = engram_dir {
            wire.apply_engram_dir(dir);
        }
        let Some(hash) = EngramHash::load(&g, &inv, &wire, identified.shape.n_vocab)
            .map_err(asset_error)?
        else {
            return Ok(None);
        };
        let mut shards = Vec::with_capacity(wire.engram_layers.len());
        for path in &wire.engram_table_path {
            let shard = EngramShard::open(Path::new(path)).map_err(asset_error)?;
            shards.push(shard);
        }
        let head_dim = wire.engram_head_dim;
        let stride = head_dim + head_dim / 32;
        Ok(Some(Self {
            hash,
            shards,
            weight_off: wire.engram_weight_off.clone(),
            scale_off: wire.engram_scale_off.clone(),
            head_dim,
            cols: (wire.engram_max_ngram - 1) * wire.engram_heads,
            stride,
            hist: Vec::new(),
        }))
    }

    /// Fill the pinned rows for the block at `pos0..pos0+tokens.len()`.
    ///
    /// The forward hands over the exact block it is about to run -- including
    /// draft tokens a verify batch may later reject; the next block's write
    /// overwrites those slots before any read, the engine's hist memcpy rule
    /// (`core_v41_forward.c:377`).  `dst[k]` is engram layer k's pinned
    /// buffer, `[n][cols][stride]`: per row the e4m3 plane then the ue8m0
    /// tail, the on-disk row form.
    pub fn prepare(&mut self, pos0: u32, tokens: &[i32], dst: &[*mut u8]) -> std::result::Result<(), String> {
        let start = pos0 as usize;
        let end = start + tokens.len();
        if self.hist.len() < end {
            self.hist.resize(end, 0);
        }
        self.hist[start..end].copy_from_slice(tokens);
        for (k, &base) in dst.iter().enumerate() {
            let ei = k as u32;
            for (i, p) in (start..end).enumerate() {
                let rows = self.hash.rows(&self.hist, p as i64, ei);
                for (c, &row_id) in rows.iter().enumerate() {
                    let row = self.shards[k]
                        .row(self.weight_off[k], self.scale_off[k], row_id as u64, self.head_dim)
                        .map_err(|e| e.to_string())?;
                    let slot = (i * self.cols as usize + c) * self.stride as usize;
                    unsafe {
                        let dst = base.add(slot);
                        std::ptr::copy_nonoverlapping(row.weights.as_ptr(), dst, row.weights.len());
                        std::ptr::copy_nonoverlapping(
                            row.scale.as_ptr(),
                            dst.add(row.weights.len()),
                            row.scale.len(),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// The pinned row stride the native buffer uses (for the native buffer
    /// sizing test / diagnostics).
    pub fn stride(&self) -> u32 {
        self.stride
    }
}

struct FeedTramp<'a> {
    feed: &'a mut V41Feed,
    n_layers: usize,
}

unsafe extern "C" fn rows_tramp(
    pos0: u32,
    tokens: *const i32,
    n: i32,
    dst: *const *mut u8,
    ud: *mut c_void,
) -> i32 {
    if ud.is_null() || dst.is_null() || (n > 0 && tokens.is_null()) || n < 0 {
        return 1;
    }
    let t = &mut *(ud as *mut FeedTramp<'_>);
    let tokens = std::slice::from_raw_parts(tokens, n as usize);
    let dst = std::slice::from_raw_parts(dst, t.n_layers);
    match catch_unwind(AssertUnwindSafe(|| t.feed.prepare(pos0, tokens, dst))) {
        Ok(Ok(())) => 0,
        // A provider failure (or panic) refuses the block; the forward fails
        // loud instead of running on stale rows.
        _ => 1,
    }
}

struct EmitTramp<'a> {
    emit: &'a mut dyn FnMut(i32) -> bool,
}

unsafe extern "C" fn emit_tramp(token: i32, ud: *mut c_void) -> i32 {
    if ud.is_null() {
        return 1;
    }
    let t = &mut *(ud as *mut EmitTramp<'_>);
    match catch_unwind(AssertUnwindSafe(|| (t.emit)(token))) {
        Ok(true) => 0,
        // false = stop, panic = stop: never keep generating past a dead sink.
        _ => 1,
    }
}

struct ProgressTramp<'a> {
    progress: &'a mut dyn FnMut(&str, i32, i32) -> bool,
}

unsafe extern "C" fn progress_tramp(ud: *mut c_void, event: *const c_char, current: i32, total: i32) -> i32 {
    if ud.is_null() {
        return 1;
    }
    let t = &mut *(ud as *mut ProgressTramp<'_>);
    let event = if event.is_null() {
        ""
    } else {
        std::ffi::CStr::from_ptr(event).to_str().unwrap_or("")
    };
    match catch_unwind(AssertUnwindSafe(|| (t.progress)(event, current, total))) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => 1,
    }
}

fn asset_error(e: impl std::fmt::Display) -> Error {
    Error {
        code: 1,
        message: e.to_string(),
    }
}

impl Model {
    /// Run the engine's one-shot V4.1 generate
    /// (`ds4_engine_v41_generate_argmax`, `core_v41_api.c:426-470` + the
    /// round drive `:175-425`) with the host feed.
    ///
    /// `emit` is called per produced token in order; returning false stops
    /// the run (the CLI's EOS convention -- EOS itself also stops inside the
    /// native).  `progress` is called once per prefill chunk; returning false
    /// aborts the prefill (the server's client-gone check).  Both run on the
    /// calling thread.  `path` is the model file the feed re-reads its
    /// metadata and shard paths from (the model handle itself keeps no path).
    pub fn v41_generate(
        &self,
        path: &Path,
        prompt: &[i32],
        n_predict: i32,
        opts: &V41RunOptions<'_>,
        emit: &mut dyn FnMut(i32) -> bool,
        progress: &mut dyn FnMut(&str, i32, i32) -> bool,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(asset_error("V4.1 prompt is empty"));
        }
        if n_predict < 0 {
            return Err(asset_error("V4.1 n_predict is negative"));
        }
        // The switches are process globals in the native (the engine's CLI
        // sets them before the run); set them here so every entry point lands
        // on the same knob.  None leaves the native default untouched.
        unsafe {
            if let Some(mode) = opts.dspark {
                ds4_bridge_v41_set_dspark(mode);
            }
            if let Some(on) = opts.graph {
                ds4_bridge_v41_set_graph(i32::from(on));
            }
            ds4_bridge_v41_set_emit_trace(i32::from(opts.emit_trace));
            ds4_bridge_v41_set_prof(i32::from(opts.prof));
        }
        let mut feed = if opts.no_engram {
            None
        } else {
            V41Feed::open(path, opts.engram_dir)?
        };
        // The tramp borrows the feed for the duration of the call only; the
        // native never retains the pointer past it (ds4_bridge.h).
        let mut feed_tramp = feed.as_mut().map(|f| FeedTramp {
            n_layers: f.shards.len(),
            feed: f,
        });
        let (rows_fn, rows_ud): (ds4_sys::ds4_bridge_v41_rows_fn, *mut c_void) =
            match feed_tramp.as_mut() {
                Some(t) => (Some(rows_tramp), t as *mut FeedTramp<'_> as *mut c_void),
                None => (None, std::ptr::null_mut()),
            };
        let mut emit_tramp_state = EmitTramp { emit };
        let mut progress_tramp_state = ProgressTramp { progress };
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_v41_generate(
                self.raw_ptr(),
                prompt.as_ptr(),
                prompt.len() as i32,
                n_predict,
                i32::from(opts.no_engram),
                opts.verify_k,
                rows_fn,
                rows_ud,
                Some(emit_tramp),
                &mut emit_tramp_state as *mut EmitTramp<'_> as *mut c_void,
                Some(progress_tramp),
                &mut progress_tramp_state as *mut ProgressTramp<'_> as *mut c_void,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(crate::fail(rc, &err));
        }
        Ok(())
    }
}
