/* ds41_dense.cuh — DeepSeek V4.1 (ds41) dense small-batch GEMVs and the tiny
 * elementwise kernels, ported from the C engine at /data/YoungAi (commit
 * 3946dbc):
 *   - the f32 / bf16 GEMVs and launchers: src/cuda/cuda_v41_gemv_highprec.inc.cu:19-191
 *     (the file the engine marks as transitional: f32 for the mHC fn pair,
 *     bf16 for the router gate / compressor / indexer projections);
 *   - their entries: src/cuda/cuda_v41_1.inc.cu:275-322;
 *   - the embedding row fetch (fp4x32 arm): :365-391;
 *   - rms_norm (:394-411), add (:413-421) and expand_hc (:423-433).
 *
 * Numerics: both GEMVs accumulate f32 with the same warp/ksplit shape as the
 * engine's (a lane's float4/uint4 weights times the activations, xor-shuffle
 * tree, then the ksplit parts summed in k order); bf16 weights go up to f32
 * without touching the bf16 grid, so "the same weights stored f32" differ
 * only in the accumulation order — the engine's gate for those is the main
 * NLL/PPL scale, not cmp (cuda_v41_gemv_highprec.inc.cu:11-14).
 *
 * Scope (P4-4): n <= 8 (the GEMV arms).  The n > 8 cuBLAS arms of the two
 * matmul entries refuse by name (P4-5).  The fp4x32 skeleton arm of the
 * embedding is ported for completeness; the artifact carries none. */
#pragma once

/* ---- f32 weights GEMV (clear.md C2 (3): the decode path's cublasSgemm) ----
 * Consumers: the router gate (f32 form), the compressor kv/gate and the
 * indexer wk/proj — the matrices still f32 in the engine's precision table.
 * in_dim must be a multiple of 128 (one warp step). */
template <uint32_t NT>
__global__ static void v41_f32_gemv_kernel(float *out, const float *w, const float *x, uint32_t in_dim,
                                           uint32_t out_dim, uint32_t x_stride, uint32_t out_stride, uint32_t ksplit) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    if (r < out_dim) {
        const float *wr = w + (uint64_t)r * in_dim;
        for (uint32_t c = kpart * 128u + lane * 4u; c < in_dim; c += ksplit * 128u) {
            const float4 wv = *(const float4 *)(wr + c);
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                const float4 xv = *(const float4 *)(x + (uint64_t)t * x_stride + c);
                acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
        }
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
        }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[(uint64_t)t * out_stride + r] = v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) out[(uint64_t)t * out_stride + r] = acc[t];
    }
}
static int v41_f32_gemv(const float *w, uint64_t in_dim, uint64_t out_dim, const float *x, float *out,
                        uint32_t n_tok, const char *what) {
    if ((in_dim % 128u) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK) return 0;
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit < 32768u) ksplit <<= 1;
    const uint32_t nseg = (uint32_t)(in_dim / 128u);   /* K split unit = 128 elements */
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rpb = 8u / ksplit;
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), 1);
    #define V41_F32GEMV_LAUNCH(NT) v41_f32_gemv_kernel<NT><<<grid, 256, 0, ds4_current_stream()>>>( \
        out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)out_dim, ksplit)
    switch (n_tok) {
        case 1: V41_F32GEMV_LAUNCH(1u); break;  case 2: V41_F32GEMV_LAUNCH(2u); break;
        case 3: V41_F32GEMV_LAUNCH(3u); break;  case 4: V41_F32GEMV_LAUNCH(4u); break;
        case 5: V41_F32GEMV_LAUNCH(5u); break;  case 6: V41_F32GEMV_LAUNCH(6u); break;
        case 7: V41_F32GEMV_LAUNCH(7u); break;  default: V41_F32GEMV_LAUNCH(8u); break;
    }
    #undef V41_F32GEMV_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}

/* ---- bf16 weights GEMV (clear.md C1: router gate / compressor / indexer) ----
 * One 128-bit read takes 8 elements, so a warp step is 256 elements and
 * in_dim must be a multiple of 256 (5120 / 512 in the engine's callers).
 * The skip pointer is the markov-bias-cache exit (NULL everywhere until the
 * sidecar store lands; the port keeps the parameter shape). */
template <uint32_t NT>
__global__ static void v41_bf16_gemv_kernel(float *out, const __nv_bfloat16 *w, const float *x, uint32_t in_dim,
                                            uint32_t out_dim, uint32_t x_stride, uint32_t out_stride, uint32_t ksplit, const int32_t *skip) {
    if (skip && *skip >= 0) return;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t rows_per_block = 8u / ksplit, rloc = warp / ksplit, kpart = warp % ksplit;
    const uint32_t r = blockIdx.x * rows_per_block + rloc;
    __shared__ float red[8][NT];
    float acc[NT];
    #pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.f;
    /* Weights first, upstream second (PDL, 2026-09-23): this lane's first PF
     * weight segments (constants) land in registers before v41_pdl_wait, and
     * only then x is touched. Segment order and multiply-add form unchanged
     * -> bit-identical. */
    constexpr uint32_t PF = 4u;
    const __nv_bfloat16 *wr = w + (uint64_t)r * in_dim;
    const uint32_t c0 = kpart * 256u + lane * 8u, cst = ksplit * 256u;
    uint4 wpre[PF];
    #pragma unroll
    for (uint32_t j = 0; j < PF; j++) {
        wpre[j] = make_uint4(0u, 0u, 0u, 0u);
        if (r < out_dim && c0 + j * cst < in_dim) wpre[j] = *(const uint4 *)(wr + c0 + j * cst);
    }
    v41_pdl_wait();
    if (r < out_dim) {
        auto seg = [&](const uint4 raw, uint32_t c) {   /* one 8-element segment: the original loop body verbatim */
            __nv_bfloat162 wb[4];
            memcpy(wb, &raw, 16);
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                const float4 x0 = *(const float4 *)(x + (uint64_t)t * x_stride + c);
                const float4 x1 = *(const float4 *)(x + (uint64_t)t * x_stride + c + 4u);
                const float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
                const float2 w2 = __bfloat1622float2(wb[2]), w3 = __bfloat1622float2(wb[3]);
                acc[t] += w0.x * x0.x + w0.y * x0.y + w1.x * x0.z + w1.y * x0.w
                        + w2.x * x1.x + w2.y * x1.y + w3.x * x1.z + w3.y * x1.w;
            }
        };
        #pragma unroll
        for (uint32_t k = 0; k < PF; k++) if (c0 + k * cst < in_dim) seg(wpre[k], c0 + k * cst);
        for (uint32_t c = c0 + PF * cst; c < in_dim; c += cst) seg(*(const uint4 *)(wr + c), c);
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t] = v;
        }
    }
    if (ksplit > 1u) {
        if (lane == 0) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) red[warp][t] = acc[t];
        }
        __syncthreads();
        if (kpart == 0 && lane == 0 && r < out_dim) {
            #pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float v = 0.f;
                for (uint32_t k = 0; k < ksplit; k++) v += red[rloc * ksplit + k][t];
                out[(uint64_t)t * out_stride + r] = v;
            }
        }
    } else if (lane == 0 && r < out_dim) {
        #pragma unroll
        for (uint32_t t = 0; t < NT; t++) out[(uint64_t)t * out_stride + r] = acc[t];
    }
}
static int v41_bf16_gemv(const __nv_bfloat16 *w, uint64_t in_dim, uint64_t out_dim, const float *x, float *out,
                         uint32_t n_tok, const char *what, const int32_t *skip) {
    if ((in_dim % 256u) != 0u || n_tok == 0 || n_tok > V41_GEMV_MAX_TOK) return 0;
    uint32_t ksplit = 1;
    while (ksplit < 8u && out_dim * ksplit < 32768u) ksplit <<= 1;
    const uint32_t nseg = (uint32_t)(in_dim / 256u);
    while (ksplit > 1u && ksplit > nseg) ksplit >>= 1;
    const uint32_t rpb = 8u / ksplit;
    const dim3 grid((unsigned)((out_dim + rpb - 1u) / rpb), 1);
    v41_pdl_register((const void *)v41_bf16_gemv_kernel<1u>);   /* the kernel v41_pdl_wait before reading x */
    #define V41_BFGEMV_LAUNCH(NT) v41_bf16_gemv_kernel<NT><<<grid, 256, 0, ds4_current_stream()>>>( \
        out, w, x, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)out_dim, ksplit, skip)
    switch (n_tok) {
        case 1: V41_BFGEMV_LAUNCH(1u); break;  case 2: V41_BFGEMV_LAUNCH(2u); break;
        case 3: V41_BFGEMV_LAUNCH(3u); break;  case 4: V41_BFGEMV_LAUNCH(4u); break;
        case 5: V41_BFGEMV_LAUNCH(5u); break;  case 6: V41_BFGEMV_LAUNCH(6u); break;
        case 7: V41_BFGEMV_LAUNCH(7u); break;  default: V41_BFGEMV_LAUNCH(8u); break;
    }
    #undef V41_BFGEMV_LAUNCH
    return cuda_ok(cudaGetLastError(), what);
}

/* ---- the two matmul entries (cuda_v41_1.inc.cu:275-322) ---- */
extern "C" int ds4_gpu_v41_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                             const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || n_tok == 0) return 0;
    const uint64_t wbytes = in_dim * out_dim * 4;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const float *W = (const float *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 f32 w");
    if (!W) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK && (in_dim % 128u) == 0u)
        return v41_f32_gemv(W, in_dim, out_dim, (const float *)x->ptr, (float *)out->ptr, n_tok, "v41 f32 gemv");
    fprintf(stderr, "ds4: [ds41] f32 prefill (n_tok %u > %u) is not ported yet\n", n_tok, V41_GEMV_MAX_TOK);
    return 0;
}
extern "C" int ds4_gpu_v41_matmul_bf16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                              uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                              const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !x || !g_cublas_ready || n_tok == 0) return 0;
    const uint64_t wbytes = in_dim * out_dim * 2;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const __nv_bfloat16 *W = (const __nv_bfloat16 *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 bf16 w");
    if (!W) return 0;
    if (n_tok <= V41_GEMV_MAX_TOK && (in_dim % 256u) == 0u)
        return v41_bf16_gemv(W, in_dim, out_dim, (const float *)x->ptr, (float *)out->ptr, n_tok, "v41 bf16 gemv", NULL);
    fprintf(stderr, "ds4: [ds41] bf16 prefill (n_tok %u > %u) is not ported yet\n", n_tok, V41_GEMV_MAX_TOK);
    return 0;
}

/* ---- embedding row fetch, fp4x32 arm (cuda_v41_1.inc.cu:365-391) ----
 * The artifact carries no fp4x32 tensor; ported for the dispatch's old-GGUF
 * arm (v41_embed).  One block per token row, 32 elements per 17-byte block. */
__global__ static void v41_embed_kernel(float *out, const int32_t *tok, const uint8_t *w, uint32_t n_vocab, uint32_t n_embd) {
    const uint32_t t = blockIdx.x, nb = n_embd / 32u;
    int32_t id = tok[t]; if (id < 0 || (uint32_t)id >= n_vocab) id = 0;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        const uint8_t *p = w + ((uint64_t)id * nb + b) * 17u;
        const float s = ds4_e8m0_to_f32(p[16]);
        float *o = out + (uint64_t)t * n_embd + b * 32u;
        for (int j = 0; j < 16; j++) {
            o[2 * j] = v41_bf16r(ds4_fp4_nibble_to_f32(p[j] & 0x0F) * s);
            o[2 * j + 1] = v41_bf16r(ds4_fp4_nibble_to_f32(p[j] >> 4) * s);
        }
    }
}
extern "C" int ds4_gpu_v41_embed_fp4x32_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map,
                                               uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab,
                                               uint32_t n_tok, uint32_t n_embd) {
    if (!out || !tokens || n_tok == 0 || (n_embd % 32u)) return 0;
    const uint64_t wbytes = (uint64_t)n_vocab * (n_embd / 32u) * 17u;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset, wbytes, "v41 embed");
    if (!w || out->bytes < (uint64_t)n_tok * n_embd * 4) return 0;
    v41_embed_kernel<<<n_tok, 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const int32_t *)tokens->ptr, w, n_vocab, n_embd);
    return cuda_ok(cudaGetLastError(), "v41 embed");
}

/* ---- rms_norm with weight -> bf16 (official RMSNorm: f32 compute, bf16
 * weight values, result .to(bf16)) — cuda_v41_1.inc.cu:394-411 ---- */
__global__ static void v41_rms_norm_kernel(float *out, const float *x, const float *w, uint32_t dim, float eps) {
    v41_pdl_wait();
    const uint32_t r = blockIdx.x; const float *xr = x + (uint64_t)r * dim; float *o = out + (uint64_t)r * dim;
    float s = 0.f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (uint32_t k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) o[i] = v41_bf16r(w[i] * (xr[i] * inv));
}
extern "C" int ds4_gpu_v41_rms_norm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map,
                                           uint64_t model_size, uint64_t weight_offset, uint32_t dim, uint32_t n_tok, float eps) {
    if (!out || !x) return 0;
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset, (uint64_t)dim * 4, "v41 norm w");
    if (!w) return 0;
    v41_rms_norm_kernel<<<n_tok, 256, 0, ds4_current_stream()>>>((float *)out->ptr, (const float *)x->ptr, w, dim, eps);
    return cuda_ok(cudaGetLastError(), "v41 rms norm");
}

/* ---- add + expand_hc (cuda_v41_1.inc.cu:413-433) ---- */
__global__ static void v41_add_kernel(float *a, const float *b, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
extern "C" int ds4_gpu_v41_add_tensor(ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t n) {
    if (!a || !b) return 0;
    v41_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ds4_current_stream()>>>((float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "v41 add");
}
__global__ static void v41_expand_hc_kernel(float *hc, const float *x, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t n = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const float v = x[(uint64_t)n * n_embd + d];
    for (uint32_t c = 0; c < n_hc; c++) hc[((uint64_t)n * n_hc + c) * n_embd + d] = v;
}
extern "C" int ds4_gpu_v41_expand_hc_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    if (!hc || !x) return 0;
    v41_expand_hc_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, ds4_current_stream()>>>((float *)hc->ptr, (const float *)x->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "v41 expand hc");
}
