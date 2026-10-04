/* Qwen-Image-2.1 DiT position and attention path (recipe sections 4 and 7).
 * Included by the CUDA backend through ds4_qwen_image_gpu.cuh, where the
 * ds4_gpu tensor entry points validate and launch the kernels.
 *
 * Layouts follow the CPU oracle (crates/ds4-core/src/qwen_image): dit.rs
 * reshapes q and k to head-major [head][token][head_dim] for the per-head
 * norm and rope, and leaves v feature-fastest [hidden, token].  For head h,
 * token t the head_dim run sits at
 *
 *   q, k:   (h * tokens + t) * head_dim
 *   v, out: t * hidden + h * head_dim          hidden = heads * head_dim
 *
 *   rope3d_rows
 *       applies a prebuilt 3-axis rope table in place, adjacent (interleaved)
 *       pairs: out0 = x0*cos - x1*sin, out1 = x0*sin + x1*cos.  pe is
 *       [tokens, head_dim/2, 2, 2] holding [[cos, -sin], [sin, cos]] per pair
 *       -- the [L, 64, 2, 2] tensor the reference's apply_rope consumes,
 *       filled by oracle::rope_table (3 axes, widths 16/56/56, theta 10000).
 *       The table is built once per layout on the host: --use_fast_math (the
 *       tree's NVCCFLAGS) lowers device cosf to __cosf, measured 5.5e-4 max
 *       abs over the rope's [0, 4224] rad on sm_89, five times the 1e-4
 *       absolute gate; the host build sidesteps it with libm's cosf.
 *
 *   attn_segment_rows
 *       one segment's attention: queries [start, end) attend to keys
 *       [0, end).  The text prefix carries the causal mask (key > query is
 *       masked); an image segment is unmasked, so it attends to the whole
 *       text prefix and to every image token.  The oracle's math, in order:
 *
 *           score  = dot8(q, k) * 1/sqrt(head_dim)   (+ -inf when masked)
 *           p      = softmax(score) over keys in [0, end)
 *           out[d] = sum over keys of p * v[key][d]
 *
 *       Scores for the whole key span are staged in shared memory: 4224
 *       keys is 17 KiB, inside the 48 KiB default dynamic limit.  The
 *       reduction idioms (pairwise shared memory) mirror the primitives
 *       header, and q/k/v/output order matches the oracle element by
 *       element so the only drift is the thread count of the two softmax
 *       reductions and FMA contraction.
 *
 *       One block owns one (query, head) row, so an image segment launches
 *       (end - start) * heads blocks and each re-reads the key span: this is
 *       the correctness-proving shape of the phase ("each kernel proven
 *       against the oracle in isolation"); query tiling or online softmax is
 *       a P6 performance question, not a P2 one. */

#pragma once

#include "qwen_image_primitives.cuh"

namespace qwen_image_cuda {

/* Key span one attention block can stage: 8192 * 4 bytes of scores plus the
 * 256-float reduction scratch stay inside the 48 KiB default dynamic shared
 * limit.  The DiT's real span is 4224 (128 text + 4096 image); the entry
 * point refuses a longer segment instead of launching a failing grid. */
constexpr uint32_t kMaxSegmentKeys = 8192u;

/* Block-wide max, the fmaxf twin of the primitives header's block_sum.
 * shared must hold blockDim.x floats; the trailing barrier matches
 * block_sum's, so the buffer can be reused by the next reduction. */
__device__ __forceinline__ float block_max(float value, float *shared) {
    const uint32_t tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (uint32_t span = blockDim.x >> 1; span > 0; span >>= 1) {
        if (tid < span) { shared[tid] = fmaxf(shared[tid], shared[tid + span]); }
        __syncthreads();
    }
    const float total = shared[0];
    __syncthreads();
    return total;
}

/* The oracle's dot8 (dit.rs): eight independent accumulators, a fixed
 * horizontal sum, then a scalar tail.  Mirrored so a score differs only by
 * the compiler's contraction, not by an accumulation order. */
__device__ __forceinline__ float dot8(const float *a, const float *b, uint32_t n) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t full = n / 8u;

    for (uint32_t chunk = 0; chunk < full; chunk++) {
#pragma unroll
        for (uint32_t lane = 0; lane < 8u; lane++) {
            acc[lane] += a[chunk * 8u + lane] * b[chunk * 8u + lane];
        }
    }

    float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (uint32_t i = full * 8u; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* One thread per (row, pair): row = head * tokens + token is index 0 of the
 * pair, so the token -- and with it the pe block -- is row % tokens.  The
 * grid is flat like rope_tail_kernel's; rows * pairs threads cover the whole
 * tensor. */
__global__ void rope3d_rows(
        float *x, const float *pe, uint32_t rows, uint32_t tokens,
        uint32_t pair_count) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= rows * pair_count) { return; }

    const uint32_t pair = gid % pair_count;
    const uint32_t token = (gid / pair_count) % tokens;
    const float *block = pe + ((uint64_t) token * pair_count + pair) * 4u;
    const float cos = block[0];
    const float sin = block[2];

    float *at = x + (uint64_t) gid * 2u;
    const float x0 = at[0];
    const float x1 = at[1];
    at[0] = x0 * cos - x1 * sin;
    at[1] = x0 * sin + x1 * cos;
}

/* blockIdx.x = query offset inside the segment, blockIdx.y = head, blockDim.x
 * = kThreads.  Dynamic shared: end scores followed by kThreads reduction
 * floats. */
__global__ void attn_segment_rows(
        float *out, const float *q, const float *k, const float *v,
        uint32_t tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t start, uint32_t end, uint32_t causal) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *partial = shared + end;

    const uint32_t query = start + blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t hidden = n_head * head_dim;
    const float scale = rsqrtf((float) head_dim);
    const float *q_row = q + ((uint64_t) head * tokens + query) * head_dim;

    float local_max = -INFINITY;
    for (uint32_t key = threadIdx.x; key < end; key += blockDim.x) {
        const float *k_row = k + ((uint64_t) head * tokens + key) * head_dim;
        float score = dot8(q_row, k_row, head_dim) * scale;
        if (causal && key > query) { score = -INFINITY; }
        scores[key] = score;
        local_max = fmaxf(local_max, score);
    }
    const float max_score = block_max(local_max, partial);

    float local_sum = 0.0f;
    for (uint32_t key = threadIdx.x; key < end; key += blockDim.x) {
        const float score = scores[key];
        const float p = isfinite(score) ? expf(score - max_score) : 0.0f;
        scores[key] = p;
        local_sum += p;
    }
    const float denom = block_sum(local_sum, partial);

    for (uint32_t key = threadIdx.x; key < end; key += blockDim.x) {
        scores[key] /= denom;
    }
    __syncthreads();

    /* The oracle's PV order: eight accumulators over key strided by eight,
     * one fixed horizontal sum, then the tail -- so only the score bits can
     * move the result, never this loop's shape. */
    const float *v_head = v + (uint64_t) head * head_dim;
    float *out_head = out + (uint64_t) query * hidden + head * head_dim;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t key = 0;

        for (; key + 8u <= end; key += 8u) {
#pragma unroll
            for (uint32_t lane = 0; lane < 8u; lane++) {
                acc[lane] += scores[key + lane] * v_head[(uint64_t)(key + lane) * hidden + d];
            }
        }

        float total = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; key < end; key++) {
            total += scores[key] * v_head[(uint64_t) key * hidden + d];
        }

        out_head[d] = total;
    }
}

} // namespace qwen_image_cuda
