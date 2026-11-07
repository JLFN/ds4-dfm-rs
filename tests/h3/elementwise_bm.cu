/*
 * Bounded reproducer for the DiT's two elementwise kernels, at the shape the
 * 768x768x107 denoise runs them: 18,799 rows, width 5,376 for AdaLN, and a
 * 18,799-row, 56-head, 128-dim QKV for the RoPE boundary.
 *
 *   tests/h3/elementwise_bm [rows] [width] [heads] [head_dim] [rope_half] [calls]
 *
 * `h3d_adaln_kernel` and `h3d_qkv_rope_kernel` are verbatim ports of
 * `/data/h3.c/h3_gpu_cuda.cu` (`:2656-2687`, `:3079-3136`), and both are
 * reduction-bound on one lane or on redundant broadcasts rather than on
 * bandwidth:
 *
 *   - AdaLN gives thread 0 a `width`-long chain of dependent FMAs fed by
 *     2-byte loads, with the other 255 threads parked at `__syncthreads`, so
 *     the block's critical path is one lane's memory latency.
 *   - QKV/RoPE has every one of a block's 128 threads walk the whole 128
 *     dimension twice for the Q and K sums, reading the same 512 bytes each
 *     time; the useful data per load is 4 bytes and the block issues 512 of
 *     them.
 *
 * Every arm below is timed at that shape and its output hashed, so the
 * variants can be compared bit for bit before any of them replaces the port.
 * `bm_*_ref` are the shipped kernels copied verbatim (the reproducer asserts
 * their output equals the in-tree entry point's), `*_nosum` removes the
 * reduction and leaves the elementwise pass (the control that says how much
 * of the call is reduction), and the candidate designs follow.
 *
 * Build: `make CUDA_ARCH=<arch> tests/h3/elementwise_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* The batch of one lane's loads: 16 elements is 32 bytes, i.e. two 16-byte
 * transactions in flight, and 16 registers of staging. */
#define BM_ADALN_BATCH 16
/* Shared staging for one AdaLN row, and the head dimension the QKV arms size
 * theirs for. */
#define BM_ADALN_WIDTH 8192
#define BM_QKV_MAX_DIM 256

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

static uint16_t bf16_of(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static float rand_unit(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return (float)((state >> 8) & 0xffffffu) / 8388608.0f - 1.0f;
}

static double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                         const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

static uint64_t fnv(const uint16_t *data, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < count; ++i) {
        hash ^= data[i] & 0xffu;
        hash *= 1099511628211ull;
        hash ^= data[i] >> 8;
        hash *= 1099511628211ull;
    }
    return hash;
}

/* --- AdaLN ---------------------------------------------------------------- */

/* The shipped kernel, verbatim. */
__global__ static void bm_adaln_ref(__nv_bfloat16 *output,
                                    const __nv_bfloat16 *input,
                                    const __nv_bfloat16 *weight,
                                    const __nv_bfloat16 *modulation,
                                    const uint32_t *row_map, uint32_t rows,
                                    uint32_t width, uint32_t slots,
                                    uint32_t shift_slot, uint32_t scale_slot,
                                    float epsilon, size_t input_offset) {
    __shared__ float inverse;
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *source = input + input_offset + (size_t)row * width;
    if (threadIdx.x == 0) {
        float square_sum = 0.0f;
        for (uint32_t k = 0; k < width; ++k) {
            float value = h3d_value(source[k]);
            square_sum = fmaf(value, value, square_sum);
        }
        inverse = rsqrtf(square_sum / (float)width + epsilon);
    }
    __syncthreads();
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized = h3d_value(source[column]) * inverse *
                           h3d_value(weight[column]);
        float shift =
            h3d_value(modulation[base + (size_t)shift_slot * width + column]);
        float scale =
            h3d_value(modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3d_store(normalized * (1.0f + scale) + shift);
    }
}

/* The elementwise pass alone: the sum replaced by a constant. */
__global__ static void bm_adaln_nosum(__nv_bfloat16 *output,
                                      const __nv_bfloat16 *input,
                                      const __nv_bfloat16 *weight,
                                      const __nv_bfloat16 *modulation,
                                      const uint32_t *row_map, uint32_t rows,
                                      uint32_t width, uint32_t slots,
                                      uint32_t shift_slot, uint32_t scale_slot,
                                      float epsilon, size_t input_offset) {
    __shared__ float inverse;
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *source = input + input_offset + (size_t)row * width;
    if (threadIdx.x == 0) {
        inverse = rsqrtf(1.0f / (float)width + epsilon);
    }
    __syncthreads();
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized = h3d_value(source[column]) * inverse *
                           h3d_value(weight[column]);
        float shift =
            h3d_value(modulation[base + (size_t)shift_slot * width + column]);
        float scale =
            h3d_value(modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3d_store(normalized * (1.0f + scale) + shift);
    }
}

/* The same chain, fed by 16-byte loads: the batch is read into registers
 * first and consumed in index order afterwards, so the accumulation and every
 * rounding is the reference's, and one lane has 16 loads in flight instead of
 * a load per dependent add. */
__global__ static void bm_adaln_batched(__nv_bfloat16 *output,
                                        const __nv_bfloat16 *input,
                                        const __nv_bfloat16 *weight,
                                        const __nv_bfloat16 *modulation,
                                        const uint32_t *row_map, uint32_t rows,
                                        uint32_t width, uint32_t slots,
                                        uint32_t shift_slot,
                                        uint32_t scale_slot, float epsilon,
                                        size_t input_offset) {
    __shared__ float inverse;
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *source = input + input_offset + (size_t)row * width;
    if (threadIdx.x == 0) {
        float square_sum = 0.0f;
        uint32_t k = 0;
        while (k < width && (((uintptr_t)(source + k)) & 15u) != 0) {
            float value = h3d_value(source[k]);
            square_sum = fmaf(value, value, square_sum);
            ++k;
        }
        for (; k + BM_ADALN_BATCH <= width; k += BM_ADALN_BATCH) {
            float batch[BM_ADALN_BATCH];
            const uint4 *words = (const uint4 *)(source + k);
            #pragma unroll
            for (uint32_t word = 0; word < BM_ADALN_BATCH / 8; ++word) {
                uint4 packed = words[word];
                uint32_t parts[4] = {packed.x, packed.y, packed.z, packed.w};
                #pragma unroll
                for (uint32_t part = 0; part < 4; ++part) {
                    batch[word * 8 + part * 2] =
                        h3d_value(__ushort_as_bfloat16((unsigned short)parts[part]));
                    batch[word * 8 + part * 2 + 1] = h3d_value(
                        __ushort_as_bfloat16((unsigned short)(parts[part] >> 16)));
                }
            }
            #pragma unroll
            for (uint32_t i = 0; i < BM_ADALN_BATCH; ++i) {
                square_sum = fmaf(batch[i], batch[i], square_sum);
            }
        }
        for (; k < width; ++k) {
            float value = h3d_value(source[k]);
            square_sum = fmaf(value, value, square_sum);
        }
        inverse = rsqrtf(square_sum / (float)width + epsilon);
    }
    __syncthreads();
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized = h3d_value(source[column]) * inverse *
                           h3d_value(weight[column]);
        float shift =
            h3d_value(modulation[base + (size_t)shift_slot * width + column]);
        float scale =
            h3d_value(modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3d_store(normalized * (1.0f + scale) + shift);
    }
}

/* The same chain with 64 elements (eight 16-byte loads) in flight per
 * iteration instead of 16. */
__global__ static void bm_adaln_batched64(__nv_bfloat16 *output,
                                          const __nv_bfloat16 *input,
                                          const __nv_bfloat16 *weight,
                                          const __nv_bfloat16 *modulation,
                                          const uint32_t *row_map, uint32_t rows,
                                          uint32_t width, uint32_t slots,
                                          uint32_t shift_slot,
                                          uint32_t scale_slot, float epsilon,
                                          size_t input_offset) {
    __shared__ float inverse;
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *source = input + input_offset + (size_t)row * width;
    if (threadIdx.x == 0) {
        float square_sum = 0.0f;
        uint32_t k = 0;
        for (; k + 64 <= width; k += 64) {
            uint4 words[8];
            const uint4 *at = (const uint4 *)(source + k);
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                words[i] = at[i];
            }
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                uint32_t parts[4] = {words[i].x, words[i].y, words[i].z,
                                     words[i].w};
                #pragma unroll
                for (uint32_t part = 0; part < 4; ++part) {
                    float low = h3d_value(
                        __ushort_as_bfloat16((unsigned short)parts[part]));
                    float high = h3d_value(__ushort_as_bfloat16(
                        (unsigned short)(parts[part] >> 16)));
                    square_sum = fmaf(low, low, square_sum);
                    square_sum = fmaf(high, high, square_sum);
                }
            }
        }
        for (; k < width; ++k) {
            float value = h3d_value(source[k]);
            square_sum = fmaf(value, value, square_sum);
        }
        inverse = rsqrtf(square_sum / (float)width + epsilon);
    }
    __syncthreads();
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized = h3d_value(source[column]) * inverse *
                           h3d_value(weight[column]);
        float shift =
            h3d_value(modulation[base + (size_t)shift_slot * width + column]);
        float scale =
            h3d_value(modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3d_store(normalized * (1.0f + scale) + shift);
    }
}

/* The row staged in shared memory once by the whole block, so the chain's
 * links come from shared memory and the elementwise pass reads it there too. */
__global__ static void bm_adaln_staged(__nv_bfloat16 *output,
                                       const __nv_bfloat16 *input,
                                       const __nv_bfloat16 *weight,
                                       const __nv_bfloat16 *modulation,
                                       const uint32_t *row_map, uint32_t rows,
                                       uint32_t width, uint32_t slots,
                                       uint32_t shift_slot, uint32_t scale_slot,
                                       float epsilon, size_t input_offset) {
    __shared__ float inverse;
    __shared__ __align__(16) __nv_bfloat16 staged[BM_ADALN_WIDTH];
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *source = input + input_offset + (size_t)row * width;

    for (uint32_t k = threadIdx.x * 8; k < width; k += blockDim.x * 8) {
        if (k + 8 <= width) {
            *(uint4 *)(staged + k) = *(const uint4 *)(source + k);
        } else {
            for (uint32_t i = 0; i < 8 && k + i < width; ++i) {
                staged[k + i] = source[k + i];
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float square_sum = 0.0f;
        for (uint32_t k = 0; k < width; ++k) {
            float value = h3d_value(staged[k]);
            square_sum = fmaf(value, value, square_sum);
        }
        inverse = rsqrtf(square_sum / (float)width + epsilon);
    }
    __syncthreads();
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized = h3d_value(staged[column]) * inverse *
                           h3d_value(weight[column]);
        float shift =
            h3d_value(modulation[base + (size_t)shift_slot * width + column]);
        float scale =
            h3d_value(modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3d_store(normalized * (1.0f + scale) + shift);
    }
}

/* The sum reduced across the block instead of walked by one lane: the price of
 * the reference's order, for AdaLN's row as for the head below. */
__global__ static void bm_adaln_reduce(__nv_bfloat16 *output,
                                       const __nv_bfloat16 *input,
                                       const __nv_bfloat16 *weight,
                                       const __nv_bfloat16 *modulation,
                                       const uint32_t *row_map, uint32_t rows,
                                       uint32_t width, uint32_t slots,
                                       uint32_t shift_slot, uint32_t scale_slot,
                                       float epsilon, size_t input_offset) {
    __shared__ float partial[32];
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *source = input + input_offset + (size_t)row * width;

    float square_sum = 0.0f;
    for (uint32_t k = threadIdx.x; k < width; k += blockDim.x) {
        float value = h3d_value(source[k]);
        square_sum = fmaf(value, value, square_sum);
    }
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);
    }
    if (lane == 0) {
        partial[warp] = square_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (uint32_t w = 0; w < (blockDim.x >> 5); ++w) {
            total += partial[w];
        }
        partial[0] = rsqrtf(total / (float)width + epsilon);
    }
    __syncthreads();
    float inverse = partial[0];
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float normalized = h3d_value(source[column]) * inverse *
                           h3d_value(weight[column]);
        float shift =
            h3d_value(modulation[base + (size_t)shift_slot * width + column]);
        float scale =
            h3d_value(modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3d_store(normalized * (1.0f + scale) + shift);
    }
}



/* --- QKV / RoPE ----------------------------------------------------------- */

/* The shipped kernel, verbatim. */
__global__ static void bm_qkv_ref(
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
        const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t dimension = blockIdx.z * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.x;
    if (row >= sequence || head >= heads || dimension >= head_dim) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim * 3;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float q = __bfloat162float(qkv[q_base + d]);
        float k = __bfloat162float(qkv[k_base + d]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = rsqrtf(q_sum / (float)head_dim + epsilon);
    float k_inverse = rsqrtf(k_sum / (float)head_dim + epsilon);

    float qw = __bfloat162float(q_weight[dimension]);
    float kw = __bfloat162float(k_weight[dimension]);
    float q0 = __bfloat162float(qkv[q_base + dimension]) * q_inverse * qw;
    float k0 = __bfloat162float(qkv[k_base + dimension]) * k_inverse * kw;
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = __bfloat162float(qkv[q_base + pair]) * q_inverse *
                   __bfloat162float(q_weight[pair]);
        float k1 = __bfloat162float(qkv[k_base + pair]) * k_inverse *
                   __bfloat162float(k_weight[pair]);
        float c = __bfloat162float(rope_cos[(size_t)row * rope_half + rope_index]);
        float s = __bfloat162float(rope_sin[(size_t)row * rope_half + rope_index]);
        if (dimension < rope_half) {
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else {
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
    }
    size_t output_index = ((size_t)head * sequence + row) * head_dim + dimension;
    query[output_index] = __float2bfloat16(q0);
    key[output_index] = __float2bfloat16(k0);
    value[output_index] = qkv[v_base + dimension];
}

/* The write pass alone: both inverses replaced by a constant, so the block
 * still reads its Q/K head, applies the rotation and writes Q, K and V. */
__global__ static void bm_qkv_nosum(
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
        const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t dimension = blockIdx.z * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.x;
    if (row >= sequence || head >= heads || dimension >= head_dim) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim * 3;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    float q_inverse = rsqrtf(1.0f / (float)head_dim + epsilon);
    float k_inverse = q_inverse;

    float qw = __bfloat162float(q_weight[dimension]);
    float kw = __bfloat162float(k_weight[dimension]);
    float q0 = __bfloat162float(qkv[q_base + dimension]) * q_inverse * qw;
    float k0 = __bfloat162float(qkv[k_base + dimension]) * k_inverse * kw;
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = __bfloat162float(qkv[q_base + pair]) * q_inverse *
                   __bfloat162float(q_weight[pair]);
        float k1 = __bfloat162float(qkv[k_base + pair]) * k_inverse *
                   __bfloat162float(k_weight[pair]);
        float c = __bfloat162float(rope_cos[(size_t)row * rope_half + rope_index]);
        float s = __bfloat162float(rope_sin[(size_t)row * rope_half + rope_index]);
        if (dimension < rope_half) {
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else {
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
    }
    size_t output_index = ((size_t)head * sequence + row) * head_dim + dimension;
    query[output_index] = __float2bfloat16(q0);
    key[output_index] = __float2bfloat16(k0);
    value[output_index] = qkv[v_base + dimension];
}

/* The reference's grid and order, with the head's Q and K staged in shared
 * memory once: one lane then walks the 128-element chain for both sums, so
 * the rounding is the reference's and no other thread repeats the walk. */
__global__ static void bm_qkv_staged(
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
        const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon) {
    __shared__ __nv_bfloat16 staged[2 * BM_QKV_MAX_DIM];
    __shared__ float inverse[2];
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.x;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim * 3;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        staged[d] = qkv[q_base + d];
        staged[head_dim + d] = qkv[k_base + d];
        value[((size_t)head * sequence + row) * head_dim + d] = qkv[v_base + d];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float q_sum = 0.0f, k_sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float q = h3d_value(staged[d]);
            float k = h3d_value(staged[head_dim + d]);
            q_sum = fmaf(q, q, q_sum);
            k_sum = fmaf(k, k, k_sum);
        }
        inverse[0] = rsqrtf(q_sum / (float)head_dim + epsilon);
        inverse[1] = rsqrtf(k_sum / (float)head_dim + epsilon);
    }
    __syncthreads();
    float q_inverse = inverse[0];
    float k_inverse = inverse[1];

    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float qw = h3d_value(q_weight[dimension]);
        float kw = h3d_value(k_weight[dimension]);
        float q0 = h3d_value(staged[dimension]) * q_inverse * qw;
        float k0 = h3d_value(staged[head_dim + dimension]) * k_inverse * kw;
        if (dimension < rope_half * 2) {
            uint32_t rope_index = dimension % rope_half;
            uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                     dimension - rope_half;
            float q1 = h3d_value(staged[pair]) * q_inverse *
                       h3d_value(q_weight[pair]);
            float k1 = h3d_value(staged[head_dim + pair]) * k_inverse *
                       h3d_value(k_weight[pair]);
            float c =
                h3d_value(rope_cos[(size_t)row * rope_half + rope_index]);
            float s =
                h3d_value(rope_sin[(size_t)row * rope_half + rope_index]);
            if (dimension < rope_half) {
                q0 = q0 * c - q1 * s;
                k0 = k0 * c - k1 * s;
            } else {
                q0 = q0 * c + q1 * s;
                k0 = k0 * c + k1 * s;
            }
        }
        size_t output_index =
            ((size_t)head * sequence + row) * head_dim + dimension;
        query[output_index] = h3d_store(q0);
        key[output_index] = h3d_store(k0);
    }
}

/* The same grid with the sums reduced across the block instead of walked by
 * one lane: every thread squares its own element and the block folds the
 * partials. This is the price of the reference's order - the sum's rounding
 * changes, so the outputs will not match bit for bit. */
__global__ static void bm_qkv_reduce(
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
        const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon) {
    __shared__ float partial[64];
    uint32_t dimension = blockIdx.z * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.x;
    /* No early return: the block still has barriers below, so an inactive
     * thread contributes zero and writes nothing. */
    int active = row < sequence && head < heads && dimension < head_dim;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim * 3;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    float q = active ? __bfloat162float(qkv[q_base + dimension]) : 0.0f;
    float k = active ? __bfloat162float(qkv[k_base + dimension]) : 0.0f;
    float q_square = q * q;
    float k_square = k * k;

    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_square += __shfl_down_sync(0xffffffffu, q_square, offset);
        k_square += __shfl_down_sync(0xffffffffu, k_square, offset);
    }
    if (lane == 0) {
        partial[warp] = q_square;
        partial[32 + warp] = k_square;
    }
    __syncthreads();
    uint32_t warps = (blockDim.x + 31u) >> 5u;
    if (threadIdx.x == 0) {
        float q_total = 0.0f, k_total = 0.0f;
        for (uint32_t w = 0; w < warps; ++w) {
            q_total += partial[w];
            k_total += partial[32 + w];
        }
        partial[0] = rsqrtf(q_total / (float)head_dim + epsilon);
        partial[1] = rsqrtf(k_total / (float)head_dim + epsilon);
    }
    __syncthreads();
    float q_inverse = partial[0];
    float k_inverse = partial[1];

    if (!active) return;
    float qw = __bfloat162float(q_weight[dimension]);
    float kw = __bfloat162float(k_weight[dimension]);
    float q0 = q * q_inverse * qw;
    float k0 = k * k_inverse * kw;
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = __bfloat162float(qkv[q_base + pair]) * q_inverse *
                   __bfloat162float(q_weight[pair]);
        float k1 = __bfloat162float(qkv[k_base + pair]) * k_inverse *
                   __bfloat162float(k_weight[pair]);
        float c = __bfloat162float(rope_cos[(size_t)row * rope_half + rope_index]);
        float s = __bfloat162float(rope_sin[(size_t)row * rope_half + rope_index]);
        if (dimension < rope_half) {
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else {
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
    }
    size_t output_index = ((size_t)head * sequence + row) * head_dim + dimension;
    query[output_index] = __float2bfloat16(q0);
    key[output_index] = __float2bfloat16(k0);
    value[output_index] = qkv[v_base + dimension];
}

/* --- host ----------------------------------------------------------------- */

struct shapes {
    uint32_t rows;
    uint32_t width;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t rope_half;
    uint32_t slots;
    int calls;
};

int main(int argc, char **argv) {
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    shapes shape;
    shape.rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 18799;
    shape.width = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 5376;
    shape.heads = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 56;
    shape.head_dim = argc > 4 ? (uint32_t)std::atoi(argv[4]) : 128;
    shape.rope_half = argc > 5 ? (uint32_t)std::atoi(argv[5]) : 48;
    shape.slots = 6;
    shape.calls = argc > 6 ? std::atoi(argv[6]) : 5;
    if (shape.head_dim > BM_QKV_MAX_DIM) fail("shape", "head_dim above staging");

    const size_t rows = shape.rows, width = shape.width;
    const size_t elements = rows * width;
    const size_t rope_elements = rows * shape.rope_half;
    const size_t qkv_elements = (size_t)shape.rows * shape.heads * shape.head_dim;

    printf("shape: %zu rows, width %zu, %u heads of %u (rope %u), %d calls\n",
           rows, width, shape.heads, shape.head_dim, shape.rope_half,
           shape.calls);

    uint32_t state = 20261029u;
    std::vector<uint16_t> host_input(elements), host_weight(width);
    std::vector<uint16_t> host_modulation((size_t)shape.slots * width);
    std::vector<uint32_t> host_map(rows);
    std::vector<uint16_t> host_qkv(qkv_elements * 3);
    std::vector<uint16_t> host_norm(shape.head_dim);
    std::vector<uint16_t> host_cos(rope_elements), host_sin(rope_elements);
    for (size_t i = 0; i < elements; ++i) {
        host_input[i] = bf16_of(rand_unit(state));
    }
    for (size_t i = 0; i < width; ++i) {
        host_weight[i] = bf16_of(1.0f + 0.25f * rand_unit(state));
    }
    for (size_t i = 0; i < host_modulation.size(); ++i) {
        host_modulation[i] = bf16_of(0.5f * rand_unit(state));
    }
    for (size_t i = 0; i < rows; ++i) {
        host_map[i] = (uint32_t)(rand_unit(state) * 0.5f + 0.5f) % shape.slots;
    }
    for (size_t i = 0; i < host_qkv.size(); ++i) {
        host_qkv[i] = bf16_of(rand_unit(state));
    }
    for (size_t i = 0; i < shape.head_dim; ++i) {
        host_norm[i] = bf16_of(1.0f + 0.25f * rand_unit(state));
    }
    for (size_t i = 0; i < rope_elements; ++i) {
        host_cos[i] = bf16_of(cosf(2.0f * rand_unit(state)));
        host_sin[i] = bf16_of(sinf(2.0f * rand_unit(state)));
    }

    auto *input = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *output = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *weight = h3d_alloc(gpu, H3D_BF16, width, error, sizeof(error));
    auto *modulation =
        h3d_alloc(gpu, H3D_BF16, host_modulation.size(), error, sizeof(error));
    auto *map = h3d_alloc(gpu, H3D_U32, rows, error, sizeof(error));
    auto *qkv = h3d_alloc(gpu, H3D_BF16, qkv_elements * 3, error, sizeof(error));
    auto *query = h3d_alloc(gpu, H3D_BF16, qkv_elements, error, sizeof(error));
    auto *key = h3d_alloc(gpu, H3D_BF16, qkv_elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_BF16, qkv_elements, error, sizeof(error));
    auto *q_weight = h3d_alloc(gpu, H3D_BF16, shape.head_dim, error, sizeof(error));
    auto *k_weight = h3d_alloc(gpu, H3D_BF16, shape.head_dim, error, sizeof(error));
    auto *ropec =
        h3d_alloc(gpu, H3D_BF16, rope_elements, error, sizeof(error));
    auto *ropes =
        h3d_alloc(gpu, H3D_BF16, rope_elements, error, sizeof(error));
    if (!input || !output || !weight || !modulation || !map || !qkv || !query ||
        !key || !value || !q_weight || !k_weight || !ropec || !ropes) {
        fail("alloc", error);
    }
    if (!h3d_upload(gpu, input, host_input.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, weight, host_weight.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, modulation, host_modulation.data(), error,
                    sizeof(error)) ||
        !h3d_upload(gpu, map, host_map.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, qkv, host_qkv.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, q_weight, host_norm.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, k_weight, host_norm.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, ropec, host_cos.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, ropes, host_sin.data(), error, sizeof(error))) {
        fail("upload", error);
    }

    auto time_calls = [&](const char *label, double moved_mb, const auto &run) {
        run();
        if (h3d_sync(gpu, error, sizeof(error)) == 0) fail(label, error);
        auto begin = std::chrono::steady_clock::now();
        for (int call = 0; call < shape.calls; ++call) {
            run();
        }
        if (h3d_sync(gpu, error, sizeof(error)) == 0) fail(label, error);
        const double each = elapsed_ms(begin, std::chrono::steady_clock::now()) /
                            shape.calls;
        printf("  %-34s %9.3f ms   %7.1f GB/s\n", label, each,
               moved_mb / each);
        return each;
    };

    /* AdaLN moves the input, the output and the two modulation rows it reads:
     * 3 x rows x width x 2 bytes. */
    const double adaln_mb = 3.0 * (double)elements * 2.0 / 1048576.0;
    printf("adaln: moves %.0f MB (input, output and both modulation rows)\n",
           adaln_mb);
    const dim3 adaln_grid((unsigned)rows);
    const auto adaln_ref_launch = [&] {
        bm_adaln_ref<<<adaln_grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)weight->data,
            (const __nv_bfloat16 *)modulation->data, (const uint32_t *)map->data,
            (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1, 1e-5f, 0);
    };
    const double adaln_shipped = time_calls("shipped (h3d_adaln)", adaln_mb, [&] {
        if (!h3d_adaln(gpu, output, input, 0, weight, modulation, map,
                       (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1,
                       1e-5f, error, sizeof(error))) {
            fail("adaln", error);
        }
    });
    const double adaln_copy = time_calls("reference copy", adaln_mb, adaln_ref_launch);
    time_calls("no-sum control", adaln_mb, [&] {
        bm_adaln_nosum<<<adaln_grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)weight->data,
            (const __nv_bfloat16 *)modulation->data, (const uint32_t *)map->data,
            (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1, 1e-5f, 0);
    });
    time_calls("batched 16-element loads", adaln_mb, [&] {
        bm_adaln_batched<<<adaln_grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)weight->data,
            (const __nv_bfloat16 *)modulation->data, (const uint32_t *)map->data,
            (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1, 1e-5f, 0);
    });
    time_calls("batched 64-element loads", adaln_mb, [&] {
        bm_adaln_batched64<<<adaln_grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)weight->data,
            (const __nv_bfloat16 *)modulation->data, (const uint32_t *)map->data,
            (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1, 1e-5f, 0);
    });
    if (width <= BM_ADALN_WIDTH) {
        time_calls("row staged in shared memory", adaln_mb, [&] {
            bm_adaln_staged<<<adaln_grid, 256, 0, gpu->stream>>>(
                (__nv_bfloat16 *)output->data,
                (const __nv_bfloat16 *)input->data,
                (const __nv_bfloat16 *)weight->data,
                (const __nv_bfloat16 *)modulation->data,
                (const uint32_t *)map->data, (uint32_t)rows, (uint32_t)width,
                shape.slots, 0, 1, 1e-5f, 0);
        });
    } else {
        printf("  (the staged arm is skipped above the reproducer's %d-wide "
               "staging)\n", BM_ADALN_WIDTH);
    }
    time_calls("block reduction (order change)", adaln_mb, [&] {
        bm_adaln_reduce<<<adaln_grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)weight->data,
            (const __nv_bfloat16 *)modulation->data, (const uint32_t *)map->data,
            (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1, 1e-5f, 0);
    });
    /* The in-tree entry again, last, so the same kernel is timed at both ends
     * of the run and the reader can see how much of a spread is arm order. */
    time_calls("shipped again (h3d_adaln)", adaln_mb, [&] {
        if (!h3d_adaln(gpu, output, input, 0, weight, modulation, map,
                       (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1,
                       1e-5f, error, sizeof(error))) {
            fail("adaln", error);
        }
    });
    if (adaln_shipped < adaln_copy * 0.98) {
        printf("  note: the in-tree entry is %.1f%% off the copy on the same "
               "binary\n", 100.0 * (adaln_copy / adaln_shipped - 1.0));
    }

    /* Bit comparison. The batch changes when a lane's loads are issued, not
     * the order the values enter the chain, so the candidate must emit the
     * reference's bytes exactly. The copy is checked too: it is the same
     * arithmetic as the in-tree entry, and if the two disagree the in-tree
     * entry has changed under this reproducer. */
    std::vector<uint16_t> staging(elements);
    auto hash_adaln_output = [&]() -> uint64_t {
        if (!h3d_read(gpu, staging.data(), output, error, sizeof(error))) {
            fail("read", error);
        }
        return fnv(staging.data(), elements);
    };
    if (!h3d_adaln(gpu, output, input, 0, weight, modulation, map, (uint32_t)rows,
                   (uint32_t)width, shape.slots, 0, 1, 1e-5f, error,
                   sizeof(error))) {
        fail("adaln", error);
    }
    const uint64_t adaln_shipped_hash = hash_adaln_output();
    adaln_ref_launch();
    const uint64_t adaln_copy_hash = hash_adaln_output();
    bm_adaln_batched<<<adaln_grid, 256, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)weight->data,
        (const __nv_bfloat16 *)modulation->data, (const uint32_t *)map->data,
        (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1, 1e-5f, 0);
    const uint64_t adaln_batched_hash = hash_adaln_output();
    if (width <= BM_ADALN_WIDTH) {
        bm_adaln_staged<<<adaln_grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
            (const __nv_bfloat16 *)weight->data,
            (const __nv_bfloat16 *)modulation->data, (const uint32_t *)map->data,
            (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1, 1e-5f, 0);
    }
    const uint64_t adaln_staged_hash = hash_adaln_output();
    printf("adaln output: shipped %016llx  copy %016llx\n",
           (unsigned long long)adaln_shipped_hash,
           (unsigned long long)adaln_copy_hash);
    printf("              batched %016llx  staged %016llx\n",
           (unsigned long long)adaln_batched_hash,
           (unsigned long long)adaln_staged_hash);
    if (adaln_copy_hash != adaln_shipped_hash) {
        fail("adaln", "the verbatim copy and the in-tree entry diverge");
    }
    if (adaln_batched_hash == adaln_shipped_hash) {
        printf("  the batched arm equals the reference bit for bit\n");
    } else {
        printf("  the batched arm DIFFERS from the reference\n");
    }
    if (adaln_staged_hash == adaln_shipped_hash) {
        printf("  the staged arm equals the reference bit for bit\n");
    } else {
        printf("  the staged arm DIFFERS from the reference\n");
    }
    adaln_ref_launch();
    std::vector<uint16_t> reference_output(elements);
    if (!h3d_read(gpu, reference_output.data(), output, error, sizeof(error))) {
        fail("read", error);
    }
    bm_adaln_reduce<<<adaln_grid, 256, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)weight->data,
        (const __nv_bfloat16 *)modulation->data, (const uint32_t *)map->data,
        (uint32_t)rows, (uint32_t)width, shape.slots, 0, 1, 1e-5f, 0);
    if (!h3d_read(gpu, staging.data(), output, error, sizeof(error))) {
        fail("read", error);
    }
    size_t adaln_differing = 0;
    int adaln_worst = 0;
    for (size_t i = 0; i < elements; ++i) {
        if (reference_output[i] == staging[i]) continue;
        ++adaln_differing;
        int delta = (int)reference_output[i] - (int)staging[i];
        if (delta < 0) delta = -delta;
        if (delta > adaln_worst) adaln_worst = delta;
    }
    printf("  the reduction arm differs in %zu of %zu elements (%.3f%%), worst "
           "%d bf16 steps\n", adaln_differing, elements,
           100.0 * (double)adaln_differing / (double)elements, adaln_worst);

    /* QKV/RoPE moves the 3 x rows x heads x head_dim triple in and the same
     * triple out. */
    const double qkv_mb = 6.0 * (double)qkv_elements * 2.0 / 1048576.0;
    printf("qkv_rope: moves %.0f MB (the triple read and written)\n", qkv_mb);
    const dim3 qkv_grid((unsigned)shape.rows, shape.heads,
                        (shape.head_dim + 127) / 128);
    const auto qkv_ref_launch = [&] {
        bm_qkv_ref<<<qkv_grid, 128, 0, gpu->stream>>>(
            (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
            (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
            (const __nv_bfloat16 *)q_weight->data,
            (const __nv_bfloat16 *)k_weight->data,
            (const __nv_bfloat16 *)ropec->data,
            (const __nv_bfloat16 *)ropes->data, shape.rows, shape.heads,
            shape.head_dim, shape.rope_half, 1e-5f);
    };
    const double qkv_shipped = time_calls("shipped (h3d_qkv_rope)", qkv_mb, [&] {
        if (!h3d_qkv_rope(gpu, query, key, value, qkv, q_weight, k_weight, ropec,
                          ropes, shape.rows, shape.heads, shape.head_dim,
                          shape.rope_half, 1e-5f, error, sizeof(error))) {
            fail("qkv_rope", error);
        }
    });
    const double qkv_copy = time_calls("reference copy", qkv_mb, qkv_ref_launch);
    time_calls("no-sum control", qkv_mb, [&] {
        bm_qkv_nosum<<<qkv_grid, 128, 0, gpu->stream>>>(
            (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
            (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
            (const __nv_bfloat16 *)q_weight->data,
            (const __nv_bfloat16 *)k_weight->data,
            (const __nv_bfloat16 *)ropec->data,
            (const __nv_bfloat16 *)ropes->data, shape.rows, shape.heads,
            shape.head_dim, shape.rope_half, 1e-5f);
    });
    time_calls("staged one-lane sum", qkv_mb, [&] {
        bm_qkv_staged<<<qkv_grid, 128, 0, gpu->stream>>>(
            (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
            (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
            (const __nv_bfloat16 *)q_weight->data,
            (const __nv_bfloat16 *)k_weight->data,
            (const __nv_bfloat16 *)ropec->data,
            (const __nv_bfloat16 *)ropes->data, shape.rows, shape.heads,
            shape.head_dim, shape.rope_half, 1e-5f);
    });
    if (shape.head_dim == 128) {
        time_calls("block reduction (order change)", qkv_mb, [&] {
            bm_qkv_reduce<<<qkv_grid, 128, 0, gpu->stream>>>(
                (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
                (__nv_bfloat16 *)value->data,
                (const __nv_bfloat16 *)qkv->data,
                (const __nv_bfloat16 *)q_weight->data,
                (const __nv_bfloat16 *)k_weight->data,
                (const __nv_bfloat16 *)ropec->data,
                (const __nv_bfloat16 *)ropes->data, shape.rows, shape.heads,
                shape.head_dim, shape.rope_half, 1e-5f);
        });
    }
    if (qkv_shipped < qkv_copy * 0.98) {
        printf("  note: the in-tree entry is %.1f%% off the copy on the same "
               "binary\n", 100.0 * (qkv_copy / qkv_shipped - 1.0));
    }

    std::vector<uint16_t> reference_query(qkv_elements), reference_key(qkv_elements);
    std::vector<uint16_t> arm_query(qkv_elements), arm_key(qkv_elements);
    auto hash_qkv_output = [&]() -> uint64_t {
        if (!h3d_read(gpu, arm_query.data(), query, error, sizeof(error)) ||
            !h3d_read(gpu, arm_key.data(), key, error, sizeof(error))) {
            fail("read", error);
        }
        return fnv(arm_query.data(), qkv_elements) ^
               fnv(arm_key.data(), qkv_elements);
    };
    const auto adopt_reference = [&] {
        qkv_ref_launch();
        if (!h3d_read(gpu, reference_query.data(), query, error, sizeof(error)) ||
            !h3d_read(gpu, reference_key.data(), key, error, sizeof(error))) {
            fail("read", error);
        }
    };
    adopt_reference();
    const uint64_t qkv_copy_hash = fnv(reference_query.data(), qkv_elements) ^
                                   fnv(reference_key.data(), qkv_elements);
    if (!h3d_qkv_rope(gpu, query, key, value, qkv, q_weight, k_weight, ropec,
                      ropes, shape.rows, shape.heads, shape.head_dim,
                      shape.rope_half, 1e-5f, error, sizeof(error))) {
        fail("qkv_rope", error);
    }
    const uint64_t qkv_shipped_hash = hash_qkv_output();
    bm_qkv_staged<<<qkv_grid, 128, 0, gpu->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
        (const __nv_bfloat16 *)q_weight->data,
        (const __nv_bfloat16 *)k_weight->data, (const __nv_bfloat16 *)ropec->data,
        (const __nv_bfloat16 *)ropes->data, shape.rows, shape.heads,
        shape.head_dim, shape.rope_half, 1e-5f);
    const uint64_t qkv_staged_hash = hash_qkv_output();
    if (shape.head_dim != 128) {
        printf("qkv_rope output (Q ^ K):\n  shipped %016llx  copy %016llx  "
               "staged %016llx\n",
               (unsigned long long)qkv_shipped_hash,
               (unsigned long long)qkv_copy_hash,
               (unsigned long long)qkv_staged_hash);
        if (qkv_copy_hash != qkv_shipped_hash) {
            fail("qkv_rope", "the verbatim copy and the in-tree entry diverge");
        }
        printf("  (the reduction arm holds one dimension per thread and is "
               "skipped above head_dim 128)\n");
        h3d_free(input); h3d_free(output); h3d_free(weight); h3d_free(modulation);
        h3d_free(map); h3d_free(qkv); h3d_free(query); h3d_free(key);
        h3d_free(value); h3d_free(q_weight); h3d_free(k_weight); h3d_free(ropec);
        h3d_free(ropes);
        h3d_destroy(gpu);
        return 0;
    }
    bm_qkv_reduce<<<qkv_grid, 128, 0, gpu->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
        (const __nv_bfloat16 *)q_weight->data,
        (const __nv_bfloat16 *)k_weight->data, (const __nv_bfloat16 *)ropec->data,
        (const __nv_bfloat16 *)ropes->data, shape.rows, shape.heads,
        shape.head_dim, shape.rope_half, 1e-5f);
    const uint64_t qkv_reduced_hash = hash_qkv_output();

    printf("qkv_rope output (Q ^ K, FNV over the bf16 halves):\n");
    printf("  shipped %016llx  copy %016llx  staged %016llx  reduction %016llx\n",
           (unsigned long long)qkv_shipped_hash,
           (unsigned long long)qkv_copy_hash,
           (unsigned long long)qkv_staged_hash,
           (unsigned long long)qkv_reduced_hash);
    if (qkv_copy_hash != qkv_shipped_hash) {
        fail("qkv_rope", "the verbatim copy and the in-tree entry diverge");
    }
    if (qkv_staged_hash == qkv_shipped_hash) {
        printf("  the staged arm equals the reference bit for bit\n");
    } else {
        printf("  the staged arm DIFFERS from the reference\n");
    }
    size_t differing = 0;
    int worst = 0;
    for (size_t i = 0; i < qkv_elements; ++i) {
        if (reference_query[i] == arm_query[i]) continue;
        ++differing;
        int delta = (int)reference_query[i] - (int)arm_query[i];
        if (delta < 0) delta = -delta;
        if (delta > worst) worst = delta;
    }
    printf("  the reduction arm differs in %zu of %zu Q elements (%.3f%%), "
           "worst %d bf16 steps\n", differing, qkv_elements,
           100.0 * (double)differing / (double)qkv_elements, worst);

    h3d_free(input); h3d_free(output); h3d_free(weight); h3d_free(modulation);
    h3d_free(map); h3d_free(qkv); h3d_free(query); h3d_free(key);
    h3d_free(value); h3d_free(q_weight); h3d_free(k_weight); h3d_free(ropec);
    h3d_free(ropes);
    h3d_destroy(gpu);
    return 0;
}
