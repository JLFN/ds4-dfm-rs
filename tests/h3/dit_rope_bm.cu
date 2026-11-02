/*
 * Bounded reproducer for `h3d_qkv_rope`, the DiT's BF16 grouped QKV + QK-norm +
 * RoPE kernel: it times the model's own entry point at a denoise shape, hashes
 * Q/K/V so a variant can be checked bit-equal without rendering, times the
 * reference spelling the kernel replaced, two staging shapes that keep its
 * arithmetic order, and the traffic floor. Build it with
 * `make tests/h3/dit_rope_bm` (the same H3_NVCCFLAGS and sibling objects as
 * `tests/h3/attn_bm`), then run:
 *
 *   tests/h3/dit_rope_bm 37722 56 5   the 1344x768, 124-frame denoise shape
 *   tests/h3/dit_rope_bm 18432 56 5   the 768x768, 107-frame denoise shape
 *
 * The shape is the one `attn_bm 37722 56` prices attention at: 56 heads of 128
 * dimensions over `video_latent_t(frames) * (latent/2)^2` rows
 * (`crates/ds4-core/src/minimax_h3/dit.rs:2802` passes `ROPE_HALF` = 3 * 16 =
 * 48, so 96 of the 128 dimensions rotate). The kernel runs once per block, 51
 * times a step. A change must leave the Q/K/V `fnv` unchanged; only then are
 * its timings comparable.
 *
 * The variants, all of which reproduce `h3d_qkv_rope_reference_kernel`'s serial
 * RMS sum exactly (`fmaf`, dimension after dimension, in that order):
 *
 *   shipped   `h3d_qkv_rope` itself - the staged arm the model runs
 *   reference the kernel that arm replaced, one block per (row, head, 128 dims),
 *             every thread re-reading the head row to reach its own sum
 *   rows4/8   the shipped staging with 4 or 8 rows per block, one lane per row
 *             running that row's serial sum (same operands, same order)
 *   floor     the same outputs with no sums and no rotation: what the phase
 *             would cost if its arithmetic were free, so a variant that reads
 *             the floor is at its traffic ceiling and nothing is left in the
 *             kernel. Its `fnv` differs by construction.
 *
 * Measured 2026-11-02, one binary, this host, 37,722 rows: shipped 17.676 ms
 * (183.6 GB/s), reference 40.627, rows4 18.238, rows8 18.406, floor 14.926
 * (217.4 GB/s) - so the shipped shape stays, the kernel sits at 84% of its own
 * traffic floor, and the 51 calls a step it makes are worth 0.14 s at most
 * (`docs/minimax-h3/denoise-rope-ceiling-2026-11-02.md`).
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

static uint64_t fnv1a(const void *data, size_t bytes) {
    constexpr uint64_t offset = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset;
    auto *bytes_data = static_cast<const unsigned char *>(data);
    for (size_t index = 0; index < bytes; index++) {
        hash = (hash ^ bytes_data[index]) * prime;
    }
    return hash;
}

static void fill(std::vector<__nv_bfloat16> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = __float2bfloat16((float)(state >> 8) / 8388608.0f - 1.0f);
    }
}

/* The shipped kernel's thread-0 sum, lifted so a lane can run it on its own
 * row: the same uint4 reads, the same low/high halves, the same fmaf order. */
__device__ static void h3d_rope_inverse(const __nv_bfloat16 *q_row,
                                        const __nv_bfloat16 *k_row,
                                        uint32_t head_dim, float epsilon,
                                        float *inverse) {
    float q_sum = 0.0f, k_sum = 0.0f;
    const uint32_t bulk = head_dim - (head_dim % H3D_ELEMENTWISE_VECTOR);
    for (uint32_t d = 0; d < bulk; d += H3D_ELEMENTWISE_VECTOR) {
        uint4 q_word = *(const uint4 *)(q_row + d);
        uint4 k_word = *(const uint4 *)(k_row + d);
        uint32_t q_parts[4] = {q_word.x, q_word.y, q_word.z, q_word.w};
        uint32_t k_parts[4] = {k_word.x, k_word.y, k_word.z, k_word.w};
        #pragma unroll
        for (uint32_t part = 0; part < 4; ++part) {
            float q_low = h3d_value(__ushort_as_bfloat16((unsigned short)q_parts[part]));
            float k_low = h3d_value(__ushort_as_bfloat16((unsigned short)k_parts[part]));
            float q_high = h3d_value(
                __ushort_as_bfloat16((unsigned short)(q_parts[part] >> 16)));
            float k_high = h3d_value(
                __ushort_as_bfloat16((unsigned short)(k_parts[part] >> 16)));
            q_sum = fmaf(q_low, q_low, q_sum);
            k_sum = fmaf(k_low, k_low, k_sum);
            q_sum = fmaf(q_high, q_high, q_sum);
            k_sum = fmaf(k_high, k_high, k_sum);
        }
    }
    for (uint32_t d = bulk; d < head_dim; d++) {
        float q = h3d_value(q_row[d]);
        float k = h3d_value(k_row[d]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    inverse[0] = rsqrtf(q_sum / (float)head_dim + epsilon);
    inverse[1] = rsqrtf(k_sum / (float)head_dim + epsilon);
}

/* The shipped epilogue for one (row, head) pair, from the staged copy. */
__device__ static void h3d_rope_epilogue(
        __nv_bfloat16 *query, __nv_bfloat16 *key, const __nv_bfloat16 *q_row,
        const __nv_bfloat16 *k_row, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t row, uint32_t head,
        uint32_t sequence, uint32_t head_dim, uint32_t rope_half, float q_inverse,
        float k_inverse, uint32_t dimension) {
    float qw = h3d_value(q_weight[dimension]);
    float kw = h3d_value(k_weight[dimension]);
    float q0 = h3d_value(q_row[dimension]) * q_inverse * qw;
    float k0 = h3d_value(k_row[dimension]) * k_inverse * kw;
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = h3d_value(q_row[pair]) * q_inverse * h3d_value(q_weight[pair]);
        float k1 = h3d_value(k_row[pair]) * k_inverse * h3d_value(k_weight[pair]);
        float c = h3d_value(rope_cos[(size_t)row * rope_half + rope_index]);
        float s = h3d_value(rope_sin[(size_t)row * rope_half + rope_index]);
        if (dimension < rope_half) {
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else {
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
    }
    size_t output_index = ((size_t)head * sequence + row) * head_dim + dimension;
    query[output_index] = h3d_store(q0);
    key[output_index] = h3d_store(k0);
}

/* rows-per-block: one head, ROWS rows, each row's serial sum run by its own
 * lane. Bit-identical to the shipped kernel by construction - same operands,
 * same order - and it amortizes the two barriers and the block over ROWS rows. */
template <uint32_t ROWS>
__global__ static void rope_rows_kernel(
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
        const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, float epsilon) {
    __shared__ __align__(16) __nv_bfloat16 staged[2 * ROWS * H3D_QKV_STAGED_DIM];
    __shared__ float inverse[2 * ROWS];
    uint32_t head = blockIdx.y;
    uint32_t first = blockIdx.x * ROWS;
    size_t inner = (size_t)heads * head_dim;

    for (uint32_t index = threadIdx.x; index < ROWS * head_dim;
         index += blockDim.x) {
        uint32_t local = index / head_dim;
        uint32_t dimension = index % head_dim;
        uint32_t row = first + local;
        if (row >= sequence) continue;
        size_t row_base = (size_t)row * inner * 3;
        size_t q_base = row_base + (size_t)head * head_dim * 3;
        staged[local * head_dim + dimension] = qkv[q_base + dimension];
        staged[ROWS * head_dim + local * head_dim + dimension] =
            qkv[q_base + head_dim + dimension];
        value[((size_t)head * sequence + row) * head_dim + dimension] =
            qkv[q_base + 2 * head_dim + dimension];
    }
    __syncthreads();

    if (threadIdx.x < ROWS && first + threadIdx.x < sequence) {
        uint32_t local = threadIdx.x;
        h3d_rope_inverse(staged + local * head_dim,
                         staged + ROWS * head_dim + local * head_dim, head_dim,
                         epsilon, inverse + 2 * local);
    }
    __syncthreads();

    for (uint32_t index = threadIdx.x; index < ROWS * head_dim;
         index += blockDim.x) {
        uint32_t local = index / head_dim;
        uint32_t dimension = index % head_dim;
        uint32_t row = first + local;
        if (row >= sequence) continue;
        h3d_rope_epilogue(query, key, staged + local * head_dim,
                          staged + ROWS * head_dim + local * head_dim, q_weight,
                          k_weight, rope_cos, rope_sin, row, head, sequence,
                          head_dim, rope_half, inverse[2 * local],
                          inverse[2 * local + 1], dimension);
    }
}

/* The traffic floor: the same three reads and three writes, no sums, no
 * rotation, so its time is what the phase's traffic alone costs. */
__global__ static void rope_floor_kernel(
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
        const __nv_bfloat16 *qkv, uint32_t sequence, uint32_t heads,
        uint32_t head_dim) {
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.x;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim * 3;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        query[((size_t)head * sequence + row) * head_dim + dimension] =
            qkv[q_base + dimension];
        key[((size_t)head * sequence + row) * head_dim + dimension] =
            qkv[q_base + head_dim + dimension];
        value[((size_t)head * sequence + row) * head_dim + dimension] =
            qkv[q_base + 2 * head_dim + dimension];
    }
}

struct Tensors {
    h3d_tensor *q;
    h3d_tensor *k;
    h3d_tensor *v;
};

int main(int argc, char **argv) {
    const uint32_t sequence = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 37722;
    const uint32_t heads = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 56;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 5;
    constexpr uint32_t head_dim = 128;
    constexpr uint32_t rope_half = 48;
    constexpr float epsilon = 1e-5f;
    char error[512] = {};

    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    const size_t elements = (size_t)sequence * heads * head_dim;
    h3d_tensor *qkv = h3d_alloc(gpu, H3D_BF16, elements * 3, error, sizeof(error));
    h3d_tensor *q_norm = h3d_alloc(gpu, H3D_BF16, head_dim, error, sizeof(error));
    h3d_tensor *k_norm = h3d_alloc(gpu, H3D_BF16, head_dim, error, sizeof(error));
    h3d_tensor *cosines =
            h3d_alloc(gpu, H3D_BF16, (size_t)sequence * rope_half, error, sizeof(error));
    h3d_tensor *sines =
            h3d_alloc(gpu, H3D_BF16, (size_t)sequence * rope_half, error, sizeof(error));
    Tensors out = {h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error)),
                   h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error)),
                   h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error))};
    if (!qkv || !q_norm || !k_norm || !cosines || !sines || !out.q || !out.k ||
        !out.v) {
        fail("alloc", error);
    }

    std::vector<__nv_bfloat16> host_qkv(elements * 3), host_norm(head_dim),
            host_cos((size_t)sequence * rope_half),
            host_sin((size_t)sequence * rope_half);
    fill(host_qkv, 1);
    fill(host_norm, 2);
    fill(host_cos, 3);
    fill(host_sin, 4);
    for (size_t index = 0; index < host_cos.size(); index++) {
        host_cos[index] = __float2bfloat16(
                __bfloat162float(host_cos[index]) * 0.5f + 0.5f);
        host_sin[index] =
                __float2bfloat16(__bfloat162float(host_sin[index]) * 0.5f);
    }
    if (!h3d_upload(gpu, qkv, host_qkv.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, q_norm, host_norm.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, k_norm, host_norm.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, cosines, host_cos.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, sines, host_sin.data(), error, sizeof(error))) {
        fail("upload", error);
    }

    auto launch = [&](int variant) {
        switch (variant) {
            case 0:
                if (!h3d_qkv_rope(gpu, out.q, out.k, out.v, qkv, q_norm, k_norm,
                                  cosines, sines, sequence, heads, head_dim,
                                  rope_half, epsilon, error, sizeof(error))) {
                    fail("shipped rope", error);
                }
                break;
            case 1:
                h3d_qkv_rope_reference_kernel<<<dim3(sequence, heads,
                                                     (head_dim + 127) / 128),
                                                128, 0, gpu->stream>>>(
                        (__nv_bfloat16 *)out.q->data, (__nv_bfloat16 *)out.k->data,
                        (__nv_bfloat16 *)out.v->data, (const __nv_bfloat16 *)qkv->data,
                        (const __nv_bfloat16 *)q_norm->data,
                        (const __nv_bfloat16 *)k_norm->data,
                        (const __nv_bfloat16 *)cosines->data,
                        (const __nv_bfloat16 *)sines->data, sequence, heads,
                        head_dim, rope_half, epsilon);
                break;
            case 2:
                rope_rows_kernel<4><<<dim3((sequence + 3) / 4, heads), 128, 0,
                                      gpu->stream>>>(
                        (__nv_bfloat16 *)out.q->data, (__nv_bfloat16 *)out.k->data,
                        (__nv_bfloat16 *)out.v->data, (const __nv_bfloat16 *)qkv->data,
                        (const __nv_bfloat16 *)q_norm->data,
                        (const __nv_bfloat16 *)k_norm->data,
                        (const __nv_bfloat16 *)cosines->data,
                        (const __nv_bfloat16 *)sines->data, sequence, heads,
                        head_dim, rope_half, epsilon);
                break;
            case 3:
                rope_rows_kernel<8><<<dim3((sequence + 7) / 8, heads), 128, 0,
                                      gpu->stream>>>(
                        (__nv_bfloat16 *)out.q->data, (__nv_bfloat16 *)out.k->data,
                        (__nv_bfloat16 *)out.v->data, (const __nv_bfloat16 *)qkv->data,
                        (const __nv_bfloat16 *)q_norm->data,
                        (const __nv_bfloat16 *)k_norm->data,
                        (const __nv_bfloat16 *)cosines->data,
                        (const __nv_bfloat16 *)sines->data, sequence, heads,
                        head_dim, rope_half, epsilon);
                break;
            case 4:
                rope_floor_kernel<<<dim3(sequence, heads), 128, 0, gpu->stream>>>(
                        (__nv_bfloat16 *)out.q->data, (__nv_bfloat16 *)out.k->data,
                        (__nv_bfloat16 *)out.v->data, (const __nv_bfloat16 *)qkv->data,
                        sequence, heads, head_dim);
                break;
        }
    };

    const char *names[] = {"shipped (staged)", "reference (re-read)", "rows4",
                          "rows8", "floor (no math)"};
    std::vector<__nv_bfloat16> host(elements);
    uint64_t base[3] = {0, 0, 0};
    for (int variant = 0; variant < 5; variant++) {
        launch(variant);
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
        uint64_t hashes[3] = {0, 0, 0};
        h3d_tensor *outputs[3] = {out.q, out.k, out.v};
        for (int which = 0; which < 3; which++) {
            if (!h3d_read(gpu, host.data(), outputs[which], error, sizeof(error))) {
                fail("read", error);
            }
            hashes[which] = fnv1a(host.data(), host.size() * sizeof(host[0]));
        }
        if (variant == 0) {
            base[0] = hashes[0];
            base[1] = hashes[1];
            base[2] = hashes[2];
        }
        const bool same = hashes[0] == base[0] && hashes[1] == base[1] &&
                          hashes[2] == base[2];

        launch(variant);
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
        auto start = std::chrono::steady_clock::now();
        for (int call = 0; call < calls; call++) launch(variant);
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
        auto finish = std::chrono::steady_clock::now();
        const double ms =
                std::chrono::duration<double, std::milli>(finish - start).count() / calls;
        const double bytes = (double)elements * 2.0 * 6.0; /* qkv in, q/k/v out */
        std::printf("%-22s ms_per_call=%8.3f gbs=%6.1f identical=%s fnv=%016llx\n",
                    names[variant], ms, bytes / (ms * 1e6), same ? "yes" : "NO",
                    (unsigned long long)hashes[0]);
    }

    h3d_free(out.v);
    h3d_free(out.k);
    h3d_free(out.q);
    h3d_free(sines);
    h3d_free(cosines);
    h3d_free(k_norm);
    h3d_free(q_norm);
    h3d_free(qkv);
    h3d_destroy(gpu);
    return 0;
}
