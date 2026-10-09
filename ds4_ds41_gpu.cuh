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
 * stay behind until their units. P4-2 adds the tensor-level entry
 * (ds4_gpu_v41_routed_moe_tensor: host blob-header read + range-resolved
 * device pointer) below the row probe.  P4-3 adds the fp8_32x32 decode family
 * (cuda/ds41_fp8blk.cuh: the tower/engram-wkv matmuls and the engram row
 * dequant).  Every launch in this family runs on
 * ds4_current_stream() — the engine's g_cur_stream convention (stream 0
 * outside capture, the capture stream inside).
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
#include "cuda/ds41_fp8blk.cuh"   /* P4-3: fp8_32x32 decode (towers + engram wkv) */
#include "cuda/ds41_engram.cuh"   /* P4-3: the engram gate + read-path device pieces */
#include "cuda/ds41_q4k.cuh"      /* P4-4: the q4_K skeleton GEMV + embedding */
#include "cuda/ds41_dense.cuh"    /* P4-4: f32/bf16 GEMVs, rms_norm, add, expand_hc */
#include "cuda/ds41_hc.cuh"       /* P4-4: the hyper-connection family (mHC) */
#include "cuda/ds41_attn.cuh"     /* P4-4: rope, act_quant, KV pack, sparse attn, window ring */

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
    const cudaStream_t st = ds4_current_stream();
    /* Ordering with the caller's uploads: the whole ds41 family launches on
     * ds4_current_stream() — stream 0 outside capture, the capture stream
     * inside — which is the engine's g_cur_stream convention.  The P4-1
     * fix (84301f8) treated the symptom with a legacy-stream event; its root
     * cause was this port's original ds4_cuda_moe_stream() choice, a separate
     * non-blocking stream the eager path never ordered against.  Fixed here;
     * the event is gone. */
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

/* Tensor-level decode entry for the forward (mirror of the engine's
 * ds4_gpu_v41_routed_moe_tensor, src/cuda/cuda_v41_3.inc.cu:156-230; the
 * prefill GEMM arm and the per-layer streaming registration are not ported:
 * n_tok > V41_GEMV_MAX_TOK refuses by name, and the device pointer comes from
 * the native range resolver, which owns this tree's memory policy).
 *
 * ver and nc are read from the HOST-side blob header — a wrong version is a
 * load-time error, never a guess (cuda_v41_3.inc.cu:164-175).  A bad header
 * is fatal: a wrong layout decodes silently and only shows as garbage.
 * gr (the zchain per-expert gain override, the engine's g_v41_gr) is NULL
 * until the sidecar gain store lands; named deviation. */
extern "C" int ds4_gpu_v41_routed_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                             uint64_t blob_offset, uint64_t blob_bytes,
                                             uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                                             const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
                                             uint32_t n_total_expert, uint32_t n_expert_used, float clamp,
                                             const ds4_gpu_tensor *x, uint32_t layer, uint32_t n_tok) {
    (void)n_total_expert;
    (void)layer;   /* the gr slot indexes by layer; the sidecar store lands with serving */
    if (!selected || !weights || !x || n_tok == 0) return 0;
    if (n_tok > V41_GEMV_MAX_TOK) {
        fprintf(stderr, "ds4: [ds41] MoE prefill (n_tok %u > %u) is not ported yet\n", n_tok, V41_GEMV_MAX_TOK);
        return 0;
    }
    if (blob_offset > model_size || blob_bytes > model_size - blob_offset) return 0;
    const uint8_t *bh = (const uint8_t *)model_map + blob_offset;
    if (!ds4vq_blob_ok(bh, (size_t)blob_bytes)) {
        fprintf(stderr, "ds4: [ds41] expert blob header is invalid (magic/version/expert count); "
                        "this engine accepts DQVL v%u..v%u\n", DS4VQ_BLOB_VER_MIN, DS4VQ_BLOB_VER_MAX);
        exit(1);
    }
    const uint32_t ver = ds4vq_blob_ver(bh);
    uint32_t nc = 0;
    uint64_t off0 = 0;
    memcpy(&off0, bh + 16, 8);
    if (off0 && off0 + 8 <= blob_bytes) { uint16_t n16; memcpy(&n16, bh + off0 + 6, 2); nc = n16; }
    const uint8_t *blob = (const uint8_t *)cuda_model_range_ptr(model_map, blob_offset, blob_bytes, "v41 vq blob");
    if (!blob) return 0;
    const int rc = ds4_gpu_v41_vq_decode_raw(out ? (float *)out->ptr : NULL, blob, in_dim, mid_dim, out_dim,
                                     (const int32_t *)selected->ptr, (const float *)weights->ptr,
                                     n_expert_used, clamp, (const float *)x->ptr, n_tok, nc, NULL, ver);
    /* Diagnostic switch, kept (P4-2): DS41_MOE_DUMP_MID=1 writes the worker's
     * mid scratch (bf16, np x mid_dim) and the down partials (f32, np x
     * out_dim) to /tmp so the persist path's values can be compared against
     * the emulation's dumps (DS41_MOE_REF_DUMP=1 in ds41_moe_ref.rs). This is
     * what measured the P4-2 acceptance: mid 0/13824 bit-diffs, partials
     * 12/30720 at <=1 bf16 ulp on the real layer-0 payload. */
    if (getenv("DS41_MOE_DUMP_MID") && g_v41_vq_h.p) {
        (void)cudaDeviceSynchronize();
        const uint64_t bytes = (uint64_t)n_tok * n_expert_used * mid_dim * 2u;
        void *tmp = malloc(bytes);
        if (tmp && cudaMemcpy(tmp, g_v41_vq_h.p, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
            FILE *g = fopen("/tmp/ds41_mid_dump.bin", "wb");
            if (g) { fwrite(tmp, 1, bytes, g); fclose(g); }
        }
        free(tmp);
        const uint64_t pbytes = (uint64_t)n_tok * n_expert_used * out_dim * 4u;
        void *ptmp = malloc(pbytes);
        if (ptmp && g_v41_vq_part.p && cudaMemcpy(ptmp, g_v41_vq_part.p, pbytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
            FILE *g = fopen("/tmp/ds41_part_dump.bin", "wb");
            if (g) { fwrite(ptmp, 1, pbytes, g); fclose(g); }
        }
        free(ptmp);
    }
    return rc;
}
