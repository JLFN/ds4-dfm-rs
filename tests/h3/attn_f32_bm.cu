/*
 * Bounded reproducer for `h3d_sdpa_f32`, the video VAE's F32 attention kernel:
 * it times the kernel at a decode shape and hashes the output, so a variant
 * can be measured and checked bit-equal without decoding a clip. Build it with
 * `make tests/h3/attn_f32_bm` (the same H3_NVCCFLAGS and sibling objects as
 * `tests/h3/attn_bm`), then run:
 *
 *   tests/h3/attn_f32_bm 2532 32 5     the 768x768 decoder tile, the shape the
 *                                      job actually runs
 *   tests/h3/attn_f32_bm 2273 32 5     the 512x512 decoder tile
 *   tests/h3/attn_f32_bm 16133 32 2    the campaign's gate shape: one chunk's
 *                                      whole 48x48 plane, seven latent steps
 *                                      plus the five suffix rows. It is a BOUND
 *                                      on any tile the decoder ever passes, not
 *                                      a shape it runs, so its timings are not
 *                                      an attribution of the job.
 *
 * The shapes the decoder passes are `CHUNK_LATENT_TIME * tile_h * tile_w +
 * SUFFIX`, where `configured_tile_pixels`/`tile_axis` pick the tile
 * (`crates/ds4-core/src/minimax_h3/vae_video.rs:156-205`): 768x768 takes 304-px
 * tiles, 19x19 latents, three per axis over six chunks, so 54 tile submissions
 * of 2,532 rows and 1,944 block calls; 512x512 takes 288-px tiles, 18x18, four
 * per chunk over three chunks, so 12 submissions of 2,273 rows and 432 block
 * calls. A change to the kernel must leave the printed `out_fnv` unchanged at
 * the gate shape; only then are its timings comparable.
 *
 * A trailing `--compare` runs the byte-exact arm and the `H3_VAE_*` arm on the
 * same inputs and prints the deviation between them (max absolute, rms, the
 * fraction of values that differ), which is the quantity a numerical change
 * has to be judged on. Both arms run in one process and one binary.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cmath>
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

/* Deterministic values in [-1, 1): the scaled scores stay finite, so the
 * softmax the kernel runs is the one the model runs. */
static void fill(std::vector<float> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = (float)(state >> 8) / 8388608.0f - 1.0f;
    }
}

int main(int argc, char **argv) {
    const uint32_t sequence = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 16133;
    const uint32_t heads = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 32;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 2;
    bool compare = false;
    for (int index = 1; index < argc; index++) {
        if (std::strcmp(argv[index], "--compare") == 0) compare = true;
    }
    constexpr uint32_t head_dim = 64;
    char error[512] = {};

    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) {
        fail("create", error);
    }

    const size_t elements = (size_t)sequence * heads * head_dim;
    h3d_tensor *query = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    h3d_tensor *key = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    h3d_tensor *value = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    h3d_tensor *output = h3d_alloc(gpu, H3D_F32, elements, error, sizeof(error));
    if (!query || !key || !value || !output) {
        fail("alloc", error);
    }

    std::vector<float> host(elements);
    uint32_t seed = 1;
    for (auto *tensor : {query, key, value}) {
        fill(host, seed++);
        if (!h3d_upload(gpu, tensor, host.data(), error, sizeof(error))) {
            fail("upload", error);
        }
    }

    const float scale = 1.0f / std::sqrt((float)head_dim);
    if (!h3d_sdpa_f32(gpu, output, query, key, value, sequence, heads, head_dim,
                      scale, error, sizeof(error))) {
        fail("sdpa", error);
    }
    if (!h3d_sync(gpu, error, sizeof(error))) {
        fail("sync", error);
    }

    auto start = std::chrono::steady_clock::now();
    for (int call = 0; call < calls; call++) {
        if (!h3d_sdpa_f32(gpu, output, query, key, value, sequence, heads,
                          head_dim, scale, error, sizeof(error))) {
            fail("sdpa", error);
        }
    }
    if (!h3d_sync(gpu, error, sizeof(error))) {
        fail("sync", error);
    }
    auto finish = std::chrono::steady_clock::now();
    const double total_ms =
            std::chrono::duration<double, std::milli>(finish - start).count();

    std::vector<float> result(elements);
    if (!h3d_read(gpu, result.data(), output, error, sizeof(error))) {
        fail("read", error);
    }

    std::printf("sequence=%u heads=%u calls=%d ms_per_call=%.3f out_fnv=%016llx\n",
                sequence, heads, calls, total_ms / calls,
                (unsigned long long)fnv1a(result.data(), result.size() * 4));
    std::printf("first=%.6f last=%.6f\n", result.front(), result.back());

    if (compare) {
        /* `result` already holds the arm the environment selected. The base is
         * the byte-exact kernel in the reference's CTA order, set through the
         * host option rather than the environment so both arms are one binary. */
        if (!h3d_set_vae_option(gpu, "attention", "scalar") ||
            !h3d_set_vae_option(gpu, "sdpa_order", "heads")) {
            fail("set reference options", "refused");
        }
        if (!h3d_sdpa_f32(gpu, output, query, key, value, sequence, heads,
                          head_dim, scale, error, sizeof(error))) {
            fail("sdpa reference", error);
        }
        std::vector<float> reference(elements);
        if (!h3d_read(gpu, reference.data(), output, error, sizeof(error))) {
            fail("read reference", error);
        }
        const char *attention = std::getenv("H3_VAE_ATTENTION");
        const char *order = std::getenv("H3_VAE_SDPA_ORDER");
        double worst = 0.0, square = 0.0;
        size_t differing = 0;
        for (size_t index = 0; index < elements; index++) {
            const double delta = (double)result[index] - (double)reference[index];
            if (result[index] != reference[index]) differing++;
            if (std::fabs(delta) > worst) worst = std::fabs(delta);
            square += delta * delta;
        }
        std::printf("compare base=scalar+heads arm=%s+%s\n",
                    attention ? attention : "scalar(default)",
                    order ? order : "heads(default)");
        std::printf("max_abs=%.8g rms=%.8g differing=%.2f%%\n", worst,
                    std::sqrt(square / (double)elements),
                    100.0 * (double)differing / (double)elements);
    }

    h3d_free(output);
    h3d_free(value);
    h3d_free(key);
    h3d_free(query);
    h3d_destroy(gpu);
    return 0;
}
