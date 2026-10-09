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

/* Row probe for the P4-1 gate (tests/test_ds41_vq.cu): dot n consecutive rows
 * of matrix (e, which), starting at `row`, against ONE activation x and write
 * the n raw row_dot results (gain included, no bf16 rounding).
 *
 * A one-hot x at column c makes row_dot exactly one nonzero term, so
 * out[i] == codebook[idx(row+i, c/8)][c%8] * gain[row+i] == the decoded value
 * at (row+i, c): the probe compares the device arithmetic against a host
 * oracle bit-for-bit without replicating the kernel's accumulation order.
 *
 * One block, one warp; the carry chain across the n rows mirrors the decode
 * worker's own multi-row arm (ds41_vq_decode.cuh gateup, cb_shared=0 path).
 * The codebook is read from global memory: for v3 the codeword read is
 * identical either way, for v2 it changes the load shape, not the values.
 *
 * Dispatch mirrors v41_vq_fused_moe (ds41_vq_launch.cuh): v3 13-bit ->
 * <13,1,1>, v3 12-bit -> <12,1,0>, v2 12/11-bit -> <12,0,0>/<11,0,0>;
 * anything else is refused, never decoded under the wrong layout.
 * A slot that fails to open (magic/shape/flags) makes the kernel write NaN,
 * a loud failure rather than zeros that could pass a compare.
 * Returns 1 on success (stream synchronized), 0 on refusal or CUDA error. */
static v41_scratch g_v41_rowprobe_x;

template <int NBIT, int V3, int EXT>
__global__ static void v41_vq_row_probe_kernel(float *out, const uint8_t *blob, int e, int which,
                                               uint32_t row, uint32_t rows, uint32_t cols,
                                               const uint32_t *xs, uint32_t n) {
    const v41_vq_mat m = v41_vq_open<V3>(blob, e, which, rows, cols, NULL);
    const uint32_t lane = threadIdx.x & 31u;
    if (!m.ok) {
        if (lane == 0u) { for (uint32_t i = 0; i < n; i++) out[i] = __uint_as_float(0x7FC00000u); }   /* quiet NaN */
        return;
    }
    v41_vq_blk carry = v41_vq_row_first_blk<NBIT, V3, EXT>(m, row);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t r = row + i;
        /* Prefetch the next row's first block as this row's carry-out, exactly
         * like the worker; EXT=1 must pair next with nextex or the next row's
         * 13th bits read as zero (ds41_vq_row.cuh row_dot contract). */
        const uint32_t *next = (i + 1u < n) ? v41_vq_row_ptr<NBIT, V3>(m, r + 1u) : NULL;
        const uint32_t *nextex = (EXT && i + 1u < n) ? v41_vq_ext_ptr(m, r + 1u) : NULL;
        const float v = v41_vq_row_dot<NBIT, V3, EXT>(m, r, xs, m.cb, 0, &carry, next, nextex);
        if (lane == 0u) out[i] = v;
    }
}

extern "C" int ds4_gpu_v41_vq_row_probe(float *out, const uint8_t *blob, uint32_t ver, uint32_t nc,
                                        int32_t e, int which, uint32_t row, uint32_t rows, uint32_t cols,
                                        const float *x, uint32_t n) {
    if (!out || !blob || !x || n == 0u || e < 0 || which < 0 || which > 2) {
        fprintf(stderr, "ds4: [ds41] row probe: bad arguments\n");
        return 0;
    }
    if (cols % 256u || row >= rows || n > rows - row) {
        fprintf(stderr, "ds4: [ds41] row probe: need cols %% 256 == 0 and [row, row+n) within rows (cols %u row %u n %u rows %u)\n",
                cols, row, n, rows);
        return 0;
    }
    const cudaStream_t st = ds4_cuda_moe_stream();
    /* Order the probe after the caller's uploads. The moe stream is created
     * cudaStreamNonBlocking, so it does not inherit the legacy default
     * stream's ordering, and a pageable H2D cudaMemcpy returns once staged —
     * its device-side DMA may still be in flight (CUDA runtime API contract).
     * Without this event the xpack can read the previous probe's activation;
     * measured on the Spark: 2/35 probes returned the previous probe's x
     * bit-for-bit before the ordering was added. */
    static cudaEvent_t ev = NULL;
    if (!ev) (void)cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (ev && cudaEventRecord(ev, 0) == cudaSuccess) (void)cudaStreamWaitEvent(st, ev, 0);
    uint16_t *xb = (uint16_t *)v41_grow(&g_v41_rowprobe_x, (uint64_t)cols * 2u, "v41 row probe x");
    if (!xb) return 0;
    v41_vq_xpack_kernel<<<(unsigned)((cols + 255u) / 256u), 256, 0, st>>>(xb, x, cols);
    const uint32_t *xs = (const uint32_t *)xb;
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    if (ver == 3u && nbit == 13u) v41_vq_row_probe_kernel<13, 1, 1><<<1, 32, 0, st>>>(out, blob, e, which, row, rows, cols, xs, n);
    else if (ver == 3u && nbit == 12u) v41_vq_row_probe_kernel<12, 1, 0><<<1, 32, 0, st>>>(out, blob, e, which, row, rows, cols, xs, n);
    else if (ver == 2u && nbit == 12u) v41_vq_row_probe_kernel<12, 0, 0><<<1, 32, 0, st>>>(out, blob, e, which, row, rows, cols, xs, n);
    else if (ver == 2u && nbit == 11u) v41_vq_row_probe_kernel<11, 0, 0><<<1, 32, 0, st>>>(out, blob, e, which, row, rows, cols, xs, n);
    else {
        fprintf(stderr, "ds4: [ds41] row probe: no decode instance for ver %u nc %u\n", ver, nc);
        return 0;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        fprintf(stderr, "ds4: [ds41] row probe: CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    return 1;
}
