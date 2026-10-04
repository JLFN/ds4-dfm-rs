/* Qwen-Image-2.1 VAE decoder primitives with no upstream ggml-cuda
 * equivalent.  Included by the CUDA backend through ds4_qwen_image_gpu.cuh,
 * where the ds4_gpu tensor entry points validate and launch them.
 *
 * Layout: the decoder works on PLANES in the reference's own GGML order
 * ne = [W, H, C, 1] (crates/ds4-core/src/qwen_image/vae.rs::Plane, wan_vae.hpp
 * "x: [N*IC, ID, IH, IW]"), so (pixel p, channel c) sits at p + pixels*c: a
 * pixel row is contiguous and the channel axis is the slowest.  The tree's
 * GEMM instead wants its token matrix feature-fastest, (feature i, token t) at
 * i + n*t, so every conv op transposes between the two with the 1x1 patch
 * permutations already in qwen_image_primitives.cuh.
 *
 * The convs are NOT kernels here: conv3x3 = im2col3x3 + ds4_gpu_matmul_f16_tensor
 * + bias_add + the two transposes, and conv1x1 = the transposes + the same GEMM
 * + bias_add.  The composition is left to the caller (the P4 decode graph and
 * the parity test) because ds4_gpu_matmul_f16_tensor runs its own F16 activation
 * mirror through cuda_tmp_alloc, and a caller that parked the im2col buffer in
 * that same sticky scratch would alias it.
 *
 * Operand contract (vae.rs:360-370): the reference writes its im2col patches in
 * the weight's F16 type and then mul_mat's them (`ggml.c:4758`, `:4840`), so
 * every activation entering a conv product is first rounded to binary16.  The
 * tree's GEMM does exactly that conversion (f32_to_f16_kernel, ds4_cuda.cu), and
 * its __float2half is bit-for-bit vae.rs::f16_round over the F16 grid, subnormal
 * range, overflow-to-infinity and ties-to-even (verified against the oracle's
 * own vectors; the only divergence is the NaN payload, which stays NaN).
 *
 *   im2col3x3_kernel
 *       3x3, stride 1, zero pad 1 on both sides -- CausalConv3d with its
 *       singleton temporal kernel collapsed (wan_vae.hpp:30-34) and the
 *       reference's im2col (im2col.cu:7-40,114-152).  Output is the patch
 *       matrix [9*IC, pixels] the GEMM consumes, feature j = tap + 9*ic with
 *       tap = kh*3 + kw.  That is the weight's own flat reduction order: the
 *       artifact stores [kW, kH, kT=1, IC*OC] (qwen_image.rs::conv3d_dims), so
 *       j = kw + 3*kh + 9*ic, and the [out, in] GEMM matrix is read straight
 *       from the map with no repacking (vae.rs::taps uses the same index).
 *
 *   bias_add_rows
 *       per-output-channel add at the real decoder shapes.  The oracle adds the
 *       bias after the accumulation (`acc[oc] + bias[oc]`, vae.rs:517-520), so
 *       it cannot fold into a GEMM epilogue; no generic add exists in the ABI.
 *
 *   vae_rms_norm
 *       RMS_norm (wan_vae.hpp:110-122): per pixel, mean = sum(x^2)/C, scale =
 *       1/sqrt(mean + 1e-12), then x*scale*gamma[c].  Eps 1e-12 and the
 *       sum-then-scale-then-gamma order are the oracle's (vae.rs:436-457); the
 *       sum is block-parallel, so only its accumulation order differs from the
 *       oracle's sequential double sum.
 *
 *   nearest_up2_gather
 *       ggml_upscale(x, 2, NEAREST) (wan_vae.hpp:240, upscale.cu:3-26): each
 *       pixel becomes a 2x2 block.
 *
 *   dup_up3d_gather
 *       DupUp3D::forward at one frame (wan_vae.hpp:322-370), the closed form
 *       vae.rs::dup_up3d derives from the reference's concat/reshape/permute
 *       chain.  Its temporal factor only re-partitions the input channels
 *       (wan_vae.hpp:1084-1086); the entry refuses out_channels * factor % in
 *       != 0 exactly where the reference asserts it (wan_vae.hpp:336).
 *
 *   vae_attn_head
 *       one head of ggml_ext_attention_ext (wan_vae.hpp:588-648 called with
 *       n_head = 1, no mask, non-causal; ggml_soft_max, ops.cpp): scores =
 *       dot(q, k)/sqrt(C), softmax over all tokens, out = sum(p * v).  q, k, v
 *       are the three contiguous channel thirds of to_qkv's [3*C, tokens] output
 *       (split_image_qkv, ggml_extend.cpp:539-555); the token index is the
 *       flattened w + W*h the reference's reshape to [t, h*w, c] produces.
 *
 * The score vector for one query is staged in shared memory, so the pixel count
 * above which a launch is refused is kMaxAttentionPixels: (8192 + 256) * 4 bytes
 * stays inside the 48 KiB default dynamic limit, while the reference's P1
 * attention runs at 256 tokens (16x16 latent, qwen-image-2.1-p1.md:71) and a
 * 128x128 latent would be 16384. */

#pragma once

#include "qwen_image_attn.cuh"

namespace qwen_image_cuda {

/* RMS_norm's eps (wan_vae.hpp:118), not the DiT LayerNorm's 1e-6. */
constexpr float kChannelNormEps = 1e-12f;

/* 3x3 conv geometry: nine taps, tap = kh*3 + kw. */
constexpr uint32_t kConvTaps = 9u;
constexpr uint32_t kConvSide = 3u;

/* Resample/DupUp3D spatial factor, always 2 on the decode path
 * (wan_vae.hpp:543, :891). */
constexpr uint32_t kUpFactorS = 2u;

/* Largest token count whose score vector fits the shared staging: 8192 floats
 * plus kThreads reduction scratch is 33 KiB, inside the 48 KiB default dynamic
 * shared limit.  The entry point refuses a wider latent instead of launching a
 * failing grid; 16384 (a 128x128 latent) is the first size over. */
constexpr uint32_t kMaxAttentionPixels = 8192u;

/* One thread per output patch element, grid flat over 9*IC*pixels.  gid maps to
 * pixel * in_dim + j so consecutive threads write consecutive features and the
 * patch store is coalesced; src is a plane, so its nine taps of one channel are
 * a one-cache-line read even though the tap walks the channel stride. */
__global__ void im2col3x3_kernel(
        float *dst, const float *src,
        uint32_t channels, uint32_t width, uint32_t height) {
    const uint32_t in_dim = kConvTaps * channels;
    const uint64_t total = (uint64_t) in_dim * width * height;
    const uint64_t gid = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) { return; }

    const uint32_t pixels = width * height;
    const uint32_t pixel = (uint32_t)(gid / in_dim);
    const uint32_t j = (uint32_t)(gid % in_dim);
    const uint32_t ic = j / kConvTaps;
    const uint32_t tap = j % kConvTaps;
    const int iy = (int)(pixel / width) + (int)(tap / kConvSide) - 1;
    const int ix = (int)(pixel % width) + (int)(tap % kConvSide) - 1;

    /* Zero padding: the reference pads with zeros (wan_vae.hpp:74-81) and
     * ggml_cuda_op_im2col writes 0 for an out-of-bounds tap (im2col.cu:33-38). */
    float value = 0.0f;
    if (ix >= 0 && ix < (int) width && iy >= 0 && iy < (int) height) {
        value = src[(uint32_t)(ix + width * iy) + pixels * ic];
    }
    dst[j + (uint64_t) in_dim * pixel] = value;
}

/* token = blockIdx.x, feature = blockIdx.y * kThreads + threadIdx.x, matching
 * the DiT entries' grid mapping (blockDim.x = kThreads). */
__global__ void bias_add_rows(float *out, const float *bias, uint32_t dim) {
    const uint32_t feature = blockIdx.y * kThreads + threadIdx.x;
    if (feature >= dim) { return; }
    out[(uint64_t) blockIdx.x * dim + feature] += bias[feature];
}

/* One block per pixel, blockDim.x = kThreads, dynamic shared kThreads floats.
 * In place: every thread reads its channel before its own write and the
 * reduction only reads.  The plane is channel-slowest, so channel c of pixel p
 * sits at p + pixels*c.  The sum is over x^2 (not centered), then the same
 * divide, sqrt and gamma multiply the oracle applies (vae.rs:444-455). */
__global__ void vae_rms_norm(
        float *x, const float *gamma, uint32_t channels, uint32_t pixels) {
    extern __shared__ float shared[];
    const uint32_t pixel = blockIdx.x;

    float square = 0.0f;
    for (uint32_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float v = x[(uint64_t) c * pixels + pixel];
        square += v * v;
    }
    const float mean = block_sum(square, shared) / (float) channels;
    const float scale = rsqrtf(mean + kChannelNormEps);

    for (uint32_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const uint64_t at = (uint64_t) c * pixels + pixel;
        x[at] = x[at] * scale * gamma[c];
    }
}

/* x = blockIdx.y channel, pixel = blockIdx.x*kThreads + threadIdx.x over the
 * upsampled grid; out is channel-slowest like the input plane. */
__global__ void nearest_up2_gather(
        float *dst, const float *src, uint32_t width, uint32_t height) {
    const uint32_t out_width = kUpFactorS * width;
    const uint32_t out_pixels = out_width * (kUpFactorS * height);
    const uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= out_pixels) { return; }

    const uint32_t ox = pixel % out_width;
    const uint32_t oy = pixel / out_width;
    const uint32_t in_pixels = width * height;
    dst[pixel + (uint64_t) out_pixels * blockIdx.y] =
            src[(ox / kUpFactorS + width * (oy / kUpFactorS)) + in_pixels * blockIdx.y];
}

/* DupUp3D closed form (vae.rs:597-643).  One thread per output element; the
 * output channel is blockIdx.y and repeats = cout*factor/cin is precomputed by
 * the entry (which also performs the reference's divisibility refusal). */
__global__ void dup_up3d_gather(
        float *dst, const float *src,
        uint32_t width, uint32_t height,
        uint32_t factor_t, uint32_t repeats) {
    const uint32_t out_width = kUpFactorS * width;
    const uint32_t out_pixels = out_width * (kUpFactorS * height);
    const uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= out_pixels) { return; }

    const uint32_t oc = blockIdx.y;
    const uint32_t ox = pixel % out_width;
    const uint32_t oy = pixel / out_width;

    /* The sub-pixel parity and the temporal offset select which duplicated
     * input channel feeds this output channel; the copy index inside the
     * duplicated frame is a no-op, so the whole op is a gather. */
    const uint32_t m = kUpFactorS * kUpFactorS * (factor_t - 1u + factor_t * oc)
                     + kUpFactorS * (oy % kUpFactorS) + (ox % kUpFactorS);
    const uint32_t c_in = m / repeats;
    const uint32_t in_pixels = width * height;
    dst[pixel + (uint64_t) out_pixels * oc] =
            src[(ox / kUpFactorS + width * (oy / kUpFactorS)) + in_pixels * c_in];
}

/* blockIdx.x = query token, blockDim.x = kThreads, dynamic shared
 * (tokens + kThreads) floats: scores[tokens] then the reduction scratch.
 * qkv is feature-fastest [3*C, tokens]: q at feature f, k at C + f, v at 2*C + f.
 *
 * One thread owns a whole q.k dot and one thread owns a whole p.v feature, each
 * looping its reduction in index order, so the per-scalar sums keep the oracle's
 * order (vae.rs:692-708) and only the max and denominator reductions are
 * block-wide.  A lower-triangular mask would be a different op: this attention
 * is bidirectional over all tokens (wan_vae.hpp:644, no mask). */
__global__ void vae_attn_head(
        float *out, const float *qkv, uint32_t channels, uint32_t tokens) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *scratch = shared + tokens;

    const uint32_t query = blockIdx.x;
    const float *q = qkv + (uint64_t) 3u * channels * query;
    const float scale = rsqrtf((float) channels);

    float local_max = -INFINITY;
    for (uint32_t key = threadIdx.x; key < tokens; key += blockDim.x) {
        const float *k = qkv + (uint64_t) 3u * channels * key + channels;
        float dot = 0.0f;
        for (uint32_t f = 0; f < channels; f++) {
            dot += q[f] * k[f];
        }
        scores[key] = dot * scale;
        local_max = fmaxf(local_max, scores[key]);
    }
    const float max_score = block_max(local_max, scratch);

    float local_sum = 0.0f;
    for (uint32_t key = threadIdx.x; key < tokens; key += blockDim.x) {
        const float p = expf(scores[key] - max_score);
        scores[key] = p;
        local_sum += p;
    }
    const float denom = block_sum(local_sum, scratch);

    for (uint32_t key = threadIdx.x; key < tokens; key += blockDim.x) {
        scores[key] /= denom;
    }
    __syncthreads();

    for (uint32_t f = threadIdx.x; f < channels; f += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t key = 0; key < tokens; key++) {
            acc += scores[key] * qkv[(uint64_t)(2u * channels + f) + (uint64_t) 3u * channels * key];
        }
        out[(uint64_t) channels * query + f] = acc;
    }
}

} // namespace qwen_image_cuda
