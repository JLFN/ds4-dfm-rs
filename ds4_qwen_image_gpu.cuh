/* Included by the CUDA backend.  Qwen-Image-2.1 DiT entry points: the three
 * primitives in cuda/qwen_image_primitives.cuh, validated and launched on the
 * ds4_gpu tensor surface.  The rest of the DiT graph (RoPE, segment
 * attention, patch and timestep glue) lands with later P2 slices. */

#include "cuda/qwen_image_primitives.cuh"

namespace qwen_image_gpu {

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static int launched(const char *what) {
    return cuda_ok(cudaGetLastError(), what);
}

/* tokens on x, feature blocks on y: matches the kernel indexing and keeps a
 * prefill's thousands of token rows off the 65535 blockIdx.y limit. */
static dim3 grid(uint32_t tokens, uint32_t features) {
    return dim3(tokens, (features + qwen_image_cuda::kThreads - 1u) /
                                qwen_image_cuda::kThreads);
}

} // namespace qwen_image_gpu

/* In-place affine-free LayerNorm over the feature axis, eps 1e-6. */
extern "C" int ds4_gpu_qwen_image_layernorm_tensor(
        ds4_gpu_tensor *x, uint32_t dim, uint32_t rows) {
    if (dim == 0 || rows == 0) return 0;
    if (!qwen_image_gpu::tensor(x, (uint64_t) dim * rows * sizeof(float))) {
        return 0;
    }
    float *buf = (float *) x->ptr;
    qwen_image_cuda::layernorm_rows<<<rows, qwen_image_cuda::kThreads,
            qwen_image_cuda::kThreads * sizeof(float), cuda_decode_stream()>>>(
            buf, buf, dim);
    return qwen_image_gpu::launched("Qwen-Image LayerNorm launch");
}

/* out = residual + x * (gated ? tanh(param[row]) : param[row] + 1), with the
 * parameter row chosen by the token's side of the prefix.  residual == NULL
 * skips the add; the result overwrites x.  param is [hidden, 2]. */
extern "C" int ds4_gpu_qwen_image_modulate_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *param,
        ds4_gpu_tensor       *residual,
        uint32_t              hidden,
        uint32_t              tokens,
        uint32_t              prefix,
        uint32_t              gated) {
    if (hidden == 0 || tokens == 0 || prefix > tokens || gated > 1) return 0;
    const uint64_t count = (uint64_t) hidden * tokens;
    if (!qwen_image_gpu::tensor(x, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(param, (uint64_t) hidden * 2u * sizeof(float))) {
        return 0;
    }
    if (residual && !qwen_image_gpu::tensor(residual, count * sizeof(float))) {
        return 0;
    }
    qwen_image_cuda::modulate_rows<<<qwen_image_gpu::grid(tokens, hidden),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) x->ptr, (const float *) param->ptr,
            residual ? (const float *) residual->ptr : NULL, hidden, prefix,
            gated);
    return qwen_image_gpu::launched("Qwen-Image modulate launch");
}

/* out = up * silu(gate) for separate [n, rows] projections. */
extern "C" int ds4_gpu_qwen_image_mlp_gated_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t              n,
        uint32_t              rows) {
    if (n == 0 || rows == 0) return 0;
    const uint64_t count = (uint64_t) n * rows;
    if (!qwen_image_gpu::tensor(out, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(gate, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(up, count * sizeof(float))) return 0;
    qwen_image_cuda::mlp_gated_rows<<<qwen_image_gpu::grid(rows, n),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) out->ptr, (const float *) gate->ptr,
            (const float *) up->ptr, n);
    return qwen_image_gpu::launched("Qwen-Image gated MLP launch");
}

/* Same op from the fused [2n, rows] projection: gate in feature chunk 0, up
 * in chunk 1. */
extern "C" int ds4_gpu_qwen_image_mlp_gated_fused_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *fused,
        uint32_t              n,
        uint32_t              rows) {
    if (n == 0 || rows == 0) return 0;
    const uint64_t count = (uint64_t) n * rows;
    if (!qwen_image_gpu::tensor(out, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(fused, 2u * count * sizeof(float))) return 0;
    qwen_image_cuda::mlp_gated_fused_rows<<<qwen_image_gpu::grid(rows, n),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) out->ptr, (const float *) fused->ptr, n);
    return qwen_image_gpu::launched("Qwen-Image fused gated MLP launch");
}
