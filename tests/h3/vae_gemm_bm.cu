/*
 * Bounded reproducer for the video VAE decoder's four F32 matmuls, the pair
 * `h3d_vae_linear` serves and the `H3_VAE_GEMM` option selects the compute mode
 * for. One call here is one decoder block's matmul set at the shape the 768x768
 * job runs, so a per-call figure times 1,944 block calls prices the block.
 *
 *   tests/h3/vae_gemm_bm            # 2,532 rows, the 768 job's tile
 *   tests/h3/vae_gemm_bm 2273       # the 512 job's tile
 *   tests/h3/vae_gemm_bm 2532 5     # five calls for the mean
 *
 * It prints each mode's time and output hash, then the deviation between the
 * modes on the same inputs, and it says plainly whether the two FAST modes are
 * one computation or two: on this toolchain they are one, bit for bit, which
 * means the two arms cannot be A/B'd against each other and only the pair
 * (F32, FAST) is a real comparison.
 *
 * Build: `make tests/h3/vae_gemm_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* The decoder block's own widths (`crates/ds4-core/src/minimax_h3/vae_video.rs`). */
enum { HIDDEN = 2048, INNER = 2048, FFN = 8192 };

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

/* Deterministic high-entropy F32 values in [-1, 1): every mantissa bit is
 * used, so a reduced-precision mode has something to round. */
static void fill(std::vector<float> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = (float)(state >> 8) / 8388608.0f - 1.0f;
    }
}

struct Matmul {
    const char *name;
    uint32_t input_dim;
    uint32_t output_dim;
    h3d_tensor *input;
    h3d_tensor *weight;
    h3d_tensor *bias;
    h3d_tensor *output;
};

int main(int argc, char **argv) {
    const uint32_t rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 2532;
    const int calls = argc > 2 ? std::atoi(argv[2]) : 2;
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) {
        fail("create", error);
    }

    const uint32_t shapes[4][2] = {
        {HIDDEN, INNER * 3}, {INNER, HIDDEN}, {HIDDEN, FFN * 2}, {FFN, HIDDEN}};
    const char *names[4] = {"qkv", "out", "ff1", "w2"};
    Matmul matmuls[4] = {};
    for (int index = 0; index < 4; index++) {
        const uint32_t input_dim = shapes[index][0], output_dim = shapes[index][1];
        Matmul &matmul = matmuls[index];
        matmul.name = names[index];
        matmul.input_dim = input_dim;
        matmul.output_dim = output_dim;
        matmul.input = h3d_alloc(gpu, H3D_F32, (size_t)rows * input_dim, error, sizeof(error));
        matmul.weight = h3d_alloc(gpu, H3D_F32, (size_t)output_dim * input_dim, error, sizeof(error));
        matmul.bias = h3d_alloc(gpu, H3D_F32, output_dim, error, sizeof(error));
        matmul.output = h3d_alloc(gpu, H3D_F32, (size_t)rows * output_dim, error, sizeof(error));
        if (!matmul.input || !matmul.weight || !matmul.bias || !matmul.output) {
            fail("alloc", error);
        }
        std::vector<float> host;
        host.resize((size_t)rows * input_dim);
        fill(host, 10u + index * 3u);
        h3d_upload(gpu, matmul.input, host.data(), error, sizeof(error));
        host.resize((size_t)output_dim * input_dim);
        fill(host, 11u + index * 3u);
        h3d_upload(gpu, matmul.weight, host.data(), error, sizeof(error));
        host.resize(output_dim);
        fill(host, 12u + index * 3u);
        h3d_upload(gpu, matmul.bias, host.data(), error, sizeof(error));
    }

    /* Every mode over the same inputs; the first is the reference path. */
    const char *modes[3] = {"fp32", "tf32", "bf16"};
    std::vector<std::vector<float>> results[3];
    double milliseconds[3] = {};
    for (int mode = 0; mode < 3; mode++) {
        if (!h3d_set_vae_option(gpu, "gemm", modes[mode])) {
            fail("option", modes[mode]);
        }
        auto start = std::chrono::steady_clock::now();
        for (int call = 0; call < calls; call++) {
            for (Matmul &matmul : matmuls) {
                if (!h3d_vae_linear(gpu, matmul.output, matmul.input, matmul.weight,
                                    matmul.bias, rows, matmul.input_dim,
                                    matmul.output_dim, error, sizeof(error))) {
                    fail("vae linear", error);
                }
            }
        }
        if (!h3d_sync(gpu, error, sizeof(error))) {
            fail("sync", error);
        }
        auto finish = std::chrono::steady_clock::now();
        milliseconds[mode] =
                std::chrono::duration<double, std::milli>(finish - start).count() / calls;
        uint64_t hash = 14695981039346656037ULL;
        for (Matmul &matmul : matmuls) {
            std::vector<float> host((size_t)rows * matmul.output_dim);
            if (!h3d_read(gpu, host.data(), matmul.output, error, sizeof(error))) {
                fail("read", error);
            }
            /* Mix the block's four outputs into one figure, so a mode that
             * differs anywhere moves it. */
            const uint64_t piece = fnv1a(host.data(), host.size() * sizeof(float));
            hash = (hash ^ piece) * 1099511628211ULL;
            results[mode].push_back(std::move(host));
        }
        std::printf("rows=%u mode=%-4s ms_per_block=%.3f out_fnv=%016llx\n", rows,
                    modes[mode], milliseconds[mode], (unsigned long long)hash);
    }

    auto compare = [&](const char *label, int left, int right) {
        double worst = 0.0, square = 0.0, scale = 0.0;
        size_t differing = 0, elements = 0;
        for (size_t piece = 0; piece < results[left].size(); piece++) {
            const std::vector<float> &a = results[left][piece];
            const std::vector<float> &b = results[right][piece];
            for (size_t index = 0; index < a.size(); index++) {
                const double delta = (double)a[index] - (double)b[index];
                if (a[index] != b[index]) differing++;
                worst = std::fmax(worst, std::fabs(delta));
                scale = std::fmax(scale, std::fabs((double)b[index]));
                square += delta * delta;
                elements++;
            }
        }
        const double rms = std::sqrt(square / (double)elements);
        std::printf("%-14s max_abs=%.6g rms=%.6g rel_rms=%.3g differing=%.2f%%\n",
                    label, worst, rms, scale > 0.0 ? rms / scale : 0.0,
                    100.0 * (double)differing / (double)elements);
    };
    compare("tf32 vs fp32", 1, 0);
    compare("bf16 vs fp32", 2, 0);
    compare("bf16 vs tf32", 2, 1);

    for (Matmul &matmul : matmuls) {
        h3d_free(matmul.output);
        h3d_free(matmul.bias);
        h3d_free(matmul.weight);
        h3d_free(matmul.input);
    }
    h3d_destroy(gpu);
    return 0;
}
