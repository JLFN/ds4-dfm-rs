/*
 * Bounded reproducer for `h3d_video_qkv_rope_f32`, the video VAE decoder's
 * QKV + RMS-normalize + RoPE kernel: it times the decoder's own entry point at
 * a decode shape, hashes Q/K/V so a variant can be checked bit-equal without
 * decoding a clip, and times four staging shapes that keep the reference's
 * arithmetic order plus the traffic floor. Build it with
 * `make tests/h3/vae_rope_bm` (the same H3_NVCCFLAGS and sibling objects as
 * `tests/h3/attn_f32_bm`), then run:
 *
 *   tests/h3/vae_rope_bm 2805 32 50   the 320-pixel tile, 20x20 latents over a
 *                                    seven-latent chunk plus the five suffix
 *                                    rows, the shape the product canvas runs
 *   tests/h3/vae_rope_bm 2273 32 50    the 288-pixel tile, 18x18, the shape the
 *                                    512x512 canvas runs
 *
 * The shape is `CHUNK_LATENT_TIME * tile_h * tile_w + SUFFIX`, where
 * `configured_tile_pixels`/`tile_axis` pick the tile
 * (`crates/ds4-core/src/minimax_h3/vae_video.rs:183-257`): 1344x768 takes
 * 320-pixel tiles, 20x20, fifteen per chunk over seven chunks, so 105
 * submissions of 2,805 rows and 3,780 block calls. A change to the kernel must
 * leave the Q/K/V `fnv` unchanged at both shapes; only then are its timings
 * comparable.
 *
 * The variants, all of which reproduce the reference's serial RMS sum exactly
 * (`fmaf`, one dimension after another, in that order):
 *
 *   shipped   `h3d_video_qkv_rope_f32` itself - the arm the decoder runs
 *   v1        the Q/K row staged into shared, every thread summing from it
 *   v2        the same staging, the two sums run once by thread 0
 *   v3/v4     v2 with two or four heads per block (128/256-thread CTAs)
 *   floor     the same outputs with no sums and no rotation: what the phase
 *             would cost if its arithmetic were free, so a variant that reads
 *             0.6 ms is at the traffic floor and there is nothing left to win
 *             in the kernel. Its `fnv` differs by construction.
 *
 * Measured 2026-11-02, one binary, this host, 2,805 rows: shipped 0.894 ms,
 * v1 0.629, v2 0.638, v3 0.629, v4 0.622, floor 0.621 - which is why the
 * decoder runs v1's shape and why the kernel's 3.48 s became 2.45 s without a
 * digit of its arithmetic moving.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static void fill(std::vector<float> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = (float)(state >> 8) / 8388608.0f - 1.0f;
    }
}

/* v1: cooperative staging into shared, then every thread runs the shipped
 * kernel's serial sum from shared (same fmaf order, same operands), and writes
 * its own dimension. Bit-identical by construction. */
__global__ static void rope_v1_kernel(float *query, float *key, float *value,
                                      const float *qkv, const float *bias,
                                      const float *rope_cos,
                                      const float *rope_sin, uint32_t sequence,
                                      uint32_t heads, uint32_t head_dim,
                                      uint32_t rope_half, float epsilon) {
    extern __shared__ float smem[];
    float *q_row = smem;
    float *k_row = smem + head_dim;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t column_base = (size_t)head * head_dim * 3;
    size_t q_base = row_base + column_base;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_row[d] = h3d_qkv_bias(qkv, bias, q_base + d, column_base + d);
        k_row[d] = h3d_qkv_bias(qkv, bias, k_base + d, column_base + head_dim + d);
    }
    __syncthreads();

    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float q = q_row[d];
        float k = k_row[d];
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = rsqrtf(q_sum / (float)head_dim + epsilon);
    float k_inverse = rsqrtf(k_sum / (float)head_dim + epsilon);

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float q0 = q_row[d] * q_inverse;
        float k0 = k_row[d] * k_inverse;
        if (d < rope_half * 2) {
            uint32_t rope_index = d % rope_half;
            uint32_t pair = d < rope_half ? d + rope_half : d - rope_half;
            float q1 = q_row[pair] * q_inverse;
            float k1 = k_row[pair] * k_inverse;
            float c = rope_cos[(size_t)row * rope_half + rope_index];
            float s = rope_sin[(size_t)row * rope_half + rope_index];
            if (d < rope_half) {
                q0 = q0 * c - q1 * s;
                k0 = k0 * c - k1 * s;
            } else {
                q0 = q0 * c + q1 * s;
                k0 = k0 * c + k1 * s;
            }
        }
        size_t output = ((size_t)head * sequence + row) * head_dim + d;
        query[output] = q0;
        key[output] = k0;
        value[output] = h3d_qkv_bias(qkv, bias, v_base + d,
                                     column_base + 2 * head_dim + d);
    }
}

/* v2: the same staging, but the row's two serial sums run once (thread 0) and
 * the inverses are broadcast. The sum is the same chain over the same values,
 * so the inverses are the same bits. */
__global__ static void rope_v2_kernel(float *query, float *key, float *value,
                                      const float *qkv, const float *bias,
                                      const float *rope_cos,
                                      const float *rope_sin, uint32_t sequence,
                                      uint32_t heads, uint32_t head_dim,
                                      uint32_t rope_half, float epsilon) {
    extern __shared__ float smem[];
    float *q_row = smem;
    float *k_row = smem + head_dim;
    float *inverse = smem + 2 * head_dim;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t column_base = (size_t)head * head_dim * 3;
    size_t q_base = row_base + column_base;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_row[d] = h3d_qkv_bias(qkv, bias, q_base + d, column_base + d);
        k_row[d] = h3d_qkv_bias(qkv, bias, k_base + d, column_base + head_dim + d);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float q_sum = 0.0f, k_sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float q = q_row[d];
            float k = k_row[d];
            q_sum = fmaf(q, q, q_sum);
            k_sum = fmaf(k, k, k_sum);
        }
        inverse[0] = rsqrtf(q_sum / (float)head_dim + epsilon);
        inverse[1] = rsqrtf(k_sum / (float)head_dim + epsilon);
    }
    __syncthreads();

    float q_inverse = inverse[0];
    float k_inverse = inverse[1];
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float q0 = q_row[d] * q_inverse;
        float k0 = k_row[d] * k_inverse;
        if (d < rope_half * 2) {
            uint32_t rope_index = d % rope_half;
            uint32_t pair = d < rope_half ? d + rope_half : d - rope_half;
            float q1 = q_row[pair] * q_inverse;
            float k1 = k_row[pair] * k_inverse;
            float c = rope_cos[(size_t)row * rope_half + rope_index];
            float s = rope_sin[(size_t)row * rope_half + rope_index];
            if (d < rope_half) {
                q0 = q0 * c - q1 * s;
                k0 = k0 * c - k1 * s;
            } else {
                q0 = q0 * c + q1 * s;
                k0 = k0 * c + k1 * s;
            }
        }
        size_t output = ((size_t)head * sequence + row) * head_dim + d;
        query[output] = q0;
        key[output] = k0;
        value[output] = h3d_qkv_bias(qkv, bias, v_base + d,
                                     column_base + 2 * head_dim + d);
    }
}

/* v3: `rope_v2_kernel` with several heads per block, so the CTA is 128 or 256
 * threads instead of 64. Group `g` is the `head_dim` threads at `g*head_dim`,
 * and every group runs the same barriers. */
template <int GROUPS>
__global__ static void rope_v3_kernel(float *query, float *key, float *value,
                                      const float *qkv, const float *bias,
                                      const float *rope_cos,
                                      const float *rope_sin, uint32_t sequence,
                                      uint32_t heads, uint32_t head_dim,
                                      uint32_t rope_half, float epsilon) {
    extern __shared__ float smem[];
    uint32_t group = threadIdx.x / head_dim;
    uint32_t lane = threadIdx.x % head_dim;
    float *q_row = smem + (size_t)group * 2 * head_dim;
    float *k_row = q_row + head_dim;
    float *inverse = smem + (size_t)GROUPS * 2 * head_dim + group * 2;
    uint32_t head = blockIdx.y * GROUPS + group;
    uint32_t row = blockIdx.z;
    if (head >= heads) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t column_base = (size_t)head * head_dim * 3;
    size_t q_base = row_base + column_base;
    size_t k_base = q_base + head_dim;
    size_t v_base = k_base + head_dim;

    for (uint32_t d = lane; d < head_dim; d += head_dim) {
        q_row[d] = h3d_qkv_bias(qkv, bias, q_base + d, column_base + d);
        k_row[d] = h3d_qkv_bias(qkv, bias, k_base + d, column_base + head_dim + d);
    }
    __syncthreads();

    if (lane == 0) {
        float q_sum = 0.0f, k_sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float q = q_row[d];
            float k = k_row[d];
            q_sum = fmaf(q, q, q_sum);
            k_sum = fmaf(k, k, k_sum);
        }
        inverse[0] = rsqrtf(q_sum / (float)head_dim + epsilon);
        inverse[1] = rsqrtf(k_sum / (float)head_dim + epsilon);
    }
    __syncthreads();

    float q_inverse = inverse[0];
    float k_inverse = inverse[1];
    for (uint32_t d = lane; d < head_dim; d += head_dim) {
        float q0 = q_row[d] * q_inverse;
        float k0 = k_row[d] * k_inverse;
        if (d < rope_half * 2) {
            uint32_t rope_index = d % rope_half;
            uint32_t pair = d < rope_half ? d + rope_half : d - rope_half;
            float q1 = q_row[pair] * q_inverse;
            float k1 = k_row[pair] * k_inverse;
            float c = rope_cos[(size_t)row * rope_half + rope_index];
            float s = rope_sin[(size_t)row * rope_half + rope_index];
            if (d < rope_half) {
                q0 = q0 * c - q1 * s;
                k0 = k0 * c - k1 * s;
            } else {
                q0 = q0 * c + q1 * s;
                k0 = k0 * c + k1 * s;
            }
        }
        size_t output = ((size_t)head * sequence + row) * head_dim + d;
        query[output] = q0;
        key[output] = k0;
        value[output] = h3d_qkv_bias(qkv, bias, v_base + d,
                                     column_base + 2 * head_dim + d);
    }
}

/* v4 is the traffic floor, not a candidate: the same outputs with no sums and
 * no rotation. It prices what the phase costs when the arithmetic is free. */
__global__ static void rope_floor_kernel(float *query, float *key, float *value,
                                         const float *qkv, const float *bias,
                                         uint32_t sequence, uint32_t heads,
                                         uint32_t head_dim) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (row >= sequence || head >= heads || d >= head_dim) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t column_base = (size_t)head * head_dim * 3;
    size_t q_base = row_base + column_base;
    size_t output = ((size_t)head * sequence + row) * head_dim + d;
    query[output] = h3d_qkv_bias(qkv, bias, q_base + d, column_base + d);
    key[output] = h3d_qkv_bias(qkv, bias, q_base + head_dim + d,
                               column_base + head_dim + d);
    value[output] = h3d_qkv_bias(qkv, bias, q_base + 2 * head_dim + d,
                                 column_base + 2 * head_dim + d);
}

struct Tensors {
    h3d_tensor *q;
    h3d_tensor *k;
    h3d_tensor *v;
};

static void run_shipped(h3d_context *gpu, const Tensors &out,
                        const h3d_tensor *qkv, const h3d_tensor *bias,
                        const h3d_tensor *cosines, const h3d_tensor *sines,
                        uint32_t sequence, uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    char error[512] = {};
    if (!h3d_video_qkv_rope_f32(gpu, out.q, out.k, out.v, qkv, bias, cosines,
                                sines, sequence, heads, head_dim, rope_half,
                                epsilon, error, sizeof(error))) {
        fail("shipped rope", error);
    }
}

int main(int argc, char **argv) {
    const uint32_t sequence = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 2805;
    const uint32_t heads = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 32;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 50;
    constexpr uint32_t head_dim = 64;
    constexpr uint32_t rope_half = 24;
    constexpr float epsilon = 1e-5f;
    char error[512] = {};

    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    const size_t elements = (size_t)sequence * heads * head_dim;
    h3d_tensor *qkv = h3d_alloc(gpu, H3D_F32, elements * 3, error, sizeof(error));
    h3d_tensor *bias = h3d_alloc(gpu, H3D_F32, (size_t)heads * head_dim * 3,
                                 error, sizeof(error));
    h3d_tensor *cosines = h3d_alloc(gpu, H3D_F32, (size_t)sequence * rope_half,
                                    error, sizeof(error));
    h3d_tensor *sines = h3d_alloc(gpu, H3D_F32, (size_t)sequence * rope_half,
                                  error, sizeof(error));
    Tensors out = {h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error)),
                   h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error)),
                   h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error))};
    if (!qkv || !bias || !cosines || !sines || !out.q || !out.k || !out.v) {
        fail("alloc", error);
    }

    std::vector<float> host_qkv(elements * 3), host_bias((size_t)heads * head_dim * 3),
            host_cos((size_t)sequence * rope_half), host_sin((size_t)sequence * rope_half);
    fill(host_qkv, 1);
    fill(host_bias, 2);
    fill(host_cos, 3);
    fill(host_sin, 4);
    for (size_t index = 0; index < host_cos.size(); index++) {
        host_cos[index] = host_cos[index] * 0.5f + 0.5f;
        host_sin[index] = host_sin[index] * 0.5f;
    }
    if (!h3d_upload(gpu, qkv, host_qkv.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, bias, host_bias.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, cosines, host_cos.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, sines, host_sin.data(), error, sizeof(error))) {
        fail("upload", error);
    }

    const unsigned blocks_y = heads;
    auto launch = [&](int variant) {
        switch (variant) {
            case 0:
                run_shipped(gpu, out, qkv, bias, cosines, sines, sequence, heads,
                            head_dim, rope_half, epsilon);
                break;
            case 1:
                rope_v1_kernel<<<dim3(1, blocks_y, sequence), head_dim,
                                 3 * head_dim * sizeof(float), gpu->stream>>>(
                        (float *)out.q->data, (float *)out.k->data,
                        (float *)out.v->data, (const float *)qkv->data,
                        (const float *)bias->data, (const float *)cosines->data,
                        (const float *)sines->data, sequence, heads, head_dim,
                        rope_half, epsilon);
                break;
            case 2:
                rope_v2_kernel<<<dim3(1, blocks_y, sequence), head_dim,
                                 (2 * head_dim + 2) * sizeof(float), gpu->stream>>>(
                        (float *)out.q->data, (float *)out.k->data,
                        (float *)out.v->data, (const float *)qkv->data,
                        (const float *)bias->data, (const float *)cosines->data,
                        (const float *)sines->data, sequence, heads, head_dim,
                        rope_half, epsilon);
                break;
            case 3:
                rope_v3_kernel<2><<<dim3(1, (blocks_y + 1) / 2, sequence),
                                    2 * head_dim,
                                    (4 * head_dim + 4) * sizeof(float), gpu->stream>>>(
                        (float *)out.q->data, (float *)out.k->data,
                        (float *)out.v->data, (const float *)qkv->data,
                        (const float *)bias->data, (const float *)cosines->data,
                        (const float *)sines->data, sequence, heads, head_dim,
                        rope_half, epsilon);
                break;
            case 4:
                rope_v3_kernel<4><<<dim3(1, (blocks_y + 3) / 4, sequence),
                                    4 * head_dim,
                                    (8 * head_dim + 8) * sizeof(float), gpu->stream>>>(
                        (float *)out.q->data, (float *)out.k->data,
                        (float *)out.v->data, (const float *)qkv->data,
                        (const float *)bias->data, (const float *)cosines->data,
                        (const float *)sines->data, sequence, heads, head_dim,
                        rope_half, epsilon);
                break;
            case 5:
                rope_floor_kernel<<<dim3((head_dim + 63) / 64, blocks_y, sequence),
                                    64, 0, gpu->stream>>>(
                        (float *)out.q->data, (float *)out.k->data,
                        (float *)out.v->data, (const float *)qkv->data,
                        (const float *)bias->data, sequence, heads, head_dim);
                break;
        }
    };

    const char *names[] = {"shipped", "v1 shared/per-thread-sum",
                           "v2 shared/one-sum", "v3 2-heads", "v4 4-heads",
                           "floor (no math)"};
    std::vector<float> host(elements);
    uint64_t base[3] = {0, 0, 0};
    for (int variant = 0; variant < 6; variant++) {
        launch(variant);
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
        uint64_t hashes[3] = {0, 0, 0};
        h3d_tensor *outputs[3] = {out.q, out.k, out.v};
        for (int which = 0; which < 3; which++) {
            if (!h3d_read(gpu, host.data(), outputs[which], error, sizeof(error))) {
                fail("read", error);
            }
            hashes[which] = fnv1a(host.data(), host.size() * 4);
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
        const double bytes = (double)elements * 4.0 * 6.0; /* qkv in, q/k/v out */
        std::printf("%-28s ms_per_call=%8.3f gbs=%6.1f identical=%s fnv=%016llx\n",
                    names[variant], ms, bytes / (ms * 1e6), same ? "yes" : "NO",
                    (unsigned long long)hashes[0]);
    }

    h3d_free(out.v);
    h3d_free(out.k);
    h3d_free(out.q);
    h3d_free(sines);
    h3d_free(cosines);
    h3d_free(bias);
    h3d_free(qkv);
    h3d_destroy(gpu);
    return 0;
}
