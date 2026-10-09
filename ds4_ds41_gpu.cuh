/* ds4_ds41_gpu.cuh — DeepSeek V4.1 (ds41) native kernel aggregate root, the
 * launcher layer for the VQ decode family. Ported from the C engine at
 * /data/YoungAi (commit 3946dbc); the per-file provenance is in each include.
 *
 * Include order is a hard contract, mirroring the engine's ds4_cuda.cu:78-92:
 * primitives -> row (layout consumer) -> probe (watchdogs) -> decode (gateup/
 * down/reduce/tail + the worker) -> group (multi-token; used by the worker) ->
 * persist (v3 n=1 / verify-batch persistent kernels; used by the worker) ->
 * launch (the (version, width) dispatcher, instantiates the worker).
 *
 * Scope of this unit (P4-1): the V4.1 VQ expert decode path, n=1..8 tokens.
 * The prefill GEMM family, the backward files and the retired/probe-only paths
 * stay behind until their units. The launcher here is the raw entry the native
 * test drives; the tensor-level entry (model-range pointer resolution, the
 * gain-override store) lands with the P4-2 forward wiring.
 */
#pragma once

/* V4.1 diagnostics flags: the engine's g_ds4_v41_prof (--v41-prof, core_v41_api.c:14)
 * and g_ds4_v41_vq_group (--no-vq-group, :21; 1 = grouped path on). The CLI
 * setters land with the serving unit. */
int g_ds4_v41_prof = 0;
int g_ds4_v41_vq_group = 1;

#include "cuda/ds41_primitives.cuh"
#include "cuda/ds41_vq_row.cuh"
#include "cuda/ds41_vq_probe.cuh"
#include "cuda/ds41_vq_decode.cuh"
#include "cuda/ds41_vq_group.cuh"
#include "cuda/ds41_vq_persist.cuh"
#include "cuda/ds41_vq_launch.cuh"

/* Raw decode entry for tests and the P4-2 forward wiring: all pointers are
 * device pointers; `nc` and `ver` come from the blob header (ds4vq_blob_nexp /
 * ds4vq_blob_ver). Returns 1 on success. */
extern "C" int ds4_gpu_v41_vq_decode_raw(float *out, const uint8_t *blob, uint32_t IN, uint32_t MID, uint32_t OUT,
                                         const int32_t *sel, const float *w, uint32_t K, float clamp, const float *x,
                                         uint32_t n_tok, uint32_t nc, const float *gr, uint32_t ver) {
    return v41_vq_fused_moe(out, blob, IN, MID, OUT, sel, w, K, clamp, x, n_tok, nc, gr, ver);
}
