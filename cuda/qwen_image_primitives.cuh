/* Qwen-Image-2.1 DiT primitives with no upstream ggml-cuda equivalent.
 * Included by the CUDA backend through ds4_qwen_image_gpu.cuh, where the
 * ds4_gpu tensor entry points validate and launch them.
 *
 * Layout: every activation is feature-fastest (ggml ne0 is the feature axis),
 * so element (feature i, token t) of a [n, rows] tensor sits at i + t*n.  One
 * token row is n consecutive floats, which is what the LayerNorm reduction
 * walks and what the elementwise kernels index with a 2-D grid.
 *
 *   layernorm_rows
 *       affine-free LayerNorm over the feature axis, eps 1e-6: per row,
 *       mean = sum(x)/n, var = sum((x-mean)^2)/n, y = (x - mean)/sqrt(var+eps).
 *       This is ggml_norm with no weight, the img_norm1/img_norm2/norm_out
 *       norm of every DiT block (recipe section 7).
 *
 *   modulate_rows
 *       out = residual + x * factor, factor taken from the [hidden, 2]
 *       parameter: row 0 (the real timestep) multiplies the image tokens at
 *       token >= prefix, row 1 (the zero timestep) the text prefix, and the
 *       factor is tanh(p) when gated else p + 1 (dit.rs::modulate).
 *
 *           param [hidden, 2]        x [hidden, tokens]
 *           row 0 -> image tokens    token >= prefix
 *           row 1 -> text  tokens    token <  prefix
 *
 *       The gated call sites add the product to the residual stream
 *       (x = x + modulate(h, mod1)); the plain ones overwrite h.  Passing
 *       residual is optional: NULL leaves out = x * factor.
 *
 *   mlp_gated_rows / mlp_gated_fused_rows
 *       out = up * silu(gate), the DiT's gated MLP middle.  Two layouts: gate
 *       and up as separate [n, rows] projections (img_mlp.gate_layer + .proj),
 *       or one fused [2*n, rows] projection (img_mlp.gate_up) whose feature
 *       axis chunks into gate [0, n) and up [n, 2n).  The Linears on both
 *       sides are the tree's existing GEMM ops, so only the gate is here.
 *
 * tanh goes through expf, not tanhf: under --use_fast_math nvcc lowers tanhf
 * to MUFU.TANH (tanh.approx.f32), measured 7.8e-6 max abs error against libm
 * over [-8, 8] on this sm_89 box, while the expf form's gated cases measured
 * 4.8e-7 against the oracle.  Both fit the 1e-4 absolute gate; the near-exact
 * F32 parity is worth the extra MUFU.EX2. */

#pragma once

#include <stdint.h>

namespace qwen_image_cuda {

/* DiT norm eps (recipe section 3.1/7).  The VAE's channel RMSNorm uses 1e-12
 * and is a different op with its own kernel in a later slice. */
constexpr float kLayerNormEps = 1e-6f;

/* Launch width of every kernel here.  The 4096-wide feature axis then costs
 * 16 blocks per token row and each thread walks 16 elements, which keeps the
 * reductions and the elementwise walks memory-bound. */
constexpr uint32_t kThreads = 256u;

/* ggml_vec_silu_f32 (vec.h), the exact formula the CPU oracle uses. */
__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

/* 1 - 2/(e^2x + 1), algebraically (e^2x - 1)/(e^2x + 1).  Written this way so
 * a large |x| saturates to +/-1 through 2/(huge + 1) instead of overflowing;
 * see the header note for why tanhf is avoided. */
__device__ __forceinline__ float tanh_exp(float value) {
    return 1.0f - 2.0f / (expf(2.0f * value) + 1.0f);
}

/* Block-wide sum, one value per thread, pairwise through shared memory.
 * shared must hold blockDim.x floats.  Every thread reads shared[0] before
 * the next reduction can reuse the buffer, which the trailing barrier
 * guarantees (same shape as mimo2's m2_block_sum). */
__device__ __forceinline__ float block_sum(float value, float *shared) {
    const uint32_t tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (uint32_t span = blockDim.x >> 1; span > 0; span >>= 1) {
        if (tid < span) { shared[tid] += shared[tid + span]; }
        __syncthreads();
    }
    const float total = shared[0];
    __syncthreads();
    return total;
}

/* One block per token row, blockDim.x = kThreads, dynamic shared kThreads
 * floats.  In-place safe (out == x): every thread reads a row element before
 * its own write and the two reduction passes only read.  The variance pass
 * centers on the mean first, so a row with a large offset does not lose
 * precision to E[x^2] - mean^2 cancellation (the oracle's order). */
__global__ void layernorm_rows(float *out, const float *x, uint32_t dim) {
    extern __shared__ float shared[];
    const uint64_t base = (uint64_t) blockIdx.x * dim;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += x[base + i];
    }
    const float mean = block_sum(sum, shared) / (float) dim;

    float var = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = x[base + i] - mean;
        var += d * d;
    }
    const float inv = rsqrtf(block_sum(var, shared) / (float) dim + kLayerNormEps);

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out[base + i] = (x[base + i] - mean) * inv;
    }
}

/* token = blockIdx.x, feature = blockIdx.y * kThreads + threadIdx.x,
 * blockDim.x = kThreads.  Keeping the token on x leaves blockIdx.y for the
 * feature blocks, so a 4224-token prefill launches 4224 * 16 blocks without
 * hitting the 65535 blockIdx.y limit.  The parameter's second axis picks the
 * row; the token's side of the prefix decides which one applies. */
__global__ void modulate_rows(
        float *x, const float *param, const float *residual,
        uint32_t hidden, uint32_t prefix, uint32_t gated) {
    const uint32_t feature = blockIdx.y * kThreads + threadIdx.x;
    if (feature >= hidden) { return; }
    const uint32_t token = blockIdx.x;
    const uint64_t index = (uint64_t) token * hidden + feature;

    const uint32_t row = (token < prefix) ? 1u : 0u;
    const float p = param[(uint64_t) row * hidden + feature];
    const float factor = gated ? tanh_exp(p) : p + 1.0f;
    const float acc = residual ? residual[index] : 0.0f;
    x[index] = acc + x[index] * factor;
}

/* Unfused gate/up pair, both [n, rows]; same grid mapping as modulate_rows
 * (blockDim.x = kThreads). */
__global__ void mlp_gated_rows(
        float *out, const float *gate, const float *up, uint32_t n) {
    const uint32_t feature = blockIdx.y * kThreads + threadIdx.x;
    if (feature >= n) { return; }
    const uint64_t index = (uint64_t) blockIdx.x * n + feature;
    out[index] = up[index] * silu(gate[index]);
}

/* Fused [2*n, rows] projection: chunk 0 of the feature axis is gate, chunk 1
 * is up, so the up element sits exactly n floats after its gate element. */
__global__ void mlp_gated_fused_rows(
        float *out, const float *fused, uint32_t n) {
    const uint32_t feature = blockIdx.y * kThreads + threadIdx.x;
    if (feature >= n) { return; }
    const uint64_t base = (uint64_t) blockIdx.x * 2u * n;
    out[(uint64_t) blockIdx.x * n + feature] =
            fused[base + n + feature] * silu(fused[base + feature]);
}

} // namespace qwen_image_cuda
