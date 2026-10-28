/*
 * Bounded reproducer for Sol-Attn's sparse forward: the attention the routing
 * mask was built for, timed against the dense attention at the shape the H3
 * denoise runs, at the taus the reference's recipe sweeps.
 *
 *   tests/h3/sol_forward_bm                 # 18,432 tokens, 56 heads, 3 calls
 *   tests/h3/sol_forward_bm 4352 56 5 1.25  # the 512x512 denoise shape
 *
 * The realized sparsity is reported beside every time: a mask that routes
 * nothing exact is a correctness pass and a performance failure, and the two
 * figures belong together. `count` is checked against the mask's own
 * population before any of it is believed.
 *
 * Build: `make tests/h3/sol_forward_bm`.
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

static uint16_t bf16_of(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

/* Blocks carry a per-block signal so the route is a mix of exact and folded;
 * V is an order of magnitude above K, as in the model's own scales. */
static void fill(std::vector<uint16_t> &query, std::vector<uint16_t> &key,
                 std::vector<uint16_t> &value, uint32_t sequence, uint32_t heads,
                 uint32_t seed) {
    uint32_t state = seed;
    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t block = 0; block < blocks; ++block) {
            state = state * 1664525u + 1013904223u;
            const float signal = (float)((state >> 8) & 0xffffffu) / 16777216.0f;
            const uint32_t first = block * H3D_SOL_BLOCK;
            const uint32_t rows = std::min<uint32_t>(H3D_SOL_BLOCK, sequence - first);
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
                    const size_t at =
                        ((size_t)head * sequence + first + row) * H3D_SOL_HEAD_DIM + dim;
                    state = state * 1664525u + 1013904223u;
                    const float noise = (float)((state >> 8) & 0xffffffu) / 16777216.0f;
                    state = state * 1664525u + 1013904223u;
                    const float other = (float)((state >> 8) & 0xffffffu) / 16777216.0f;
                    key[at] = bf16_of(0.5f * signal + 0.02f * (noise - 0.5f));
                    value[at] = bf16_of(2.0f * other);
                    query[at] = bf16_of(0.5f + other);
                }
            }
        }
    }
}

static double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                         const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

int main(int argc, char **argv) {
    const uint32_t sequence = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 18432;
    const uint32_t heads = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 56;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 3;
    const float tau = argc > 4 ? (float)std::atof(argv[4]) : 1.25f;
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    const uint32_t words = h3d_sol_words(sequence);
    const size_t elements = (size_t)sequence * heads * H3D_SOL_HEAD_DIM;
    const size_t summary = (size_t)blocks * heads * H3D_SOL_HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    printf("shape: %u tokens, %u heads, %u blocks, %u words per query block, "
           "%.0f MB per tensor\n",
           sequence, heads, blocks, words, (double)elements * 2.0 / 1048576.0);

    std::vector<uint16_t> host_query(elements), host_key(elements), host_value(elements);
    auto setup_start = std::chrono::steady_clock::now();
    fill(host_query, host_key, host_value, sequence, heads, 20261028u);
    printf("host fill: %.2f s\n", elapsed_ms(setup_start,
           std::chrono::steady_clock::now()) / 1000.0);

    auto *query = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *key = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *value = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *output = h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
    auto *kc = h3d_alloc(gpu, H3D_BF16, summary, error, sizeof(error));
    auto *vc = h3d_alloc(gpu, H3D_BF16, summary, error, sizeof(error));
    auto *mask = h3d_alloc(gpu, H3D_U32, (size_t)blocks * heads * words, error, sizeof(error));
    auto *count = h3d_alloc(gpu, H3D_U32, (size_t)blocks * heads, error, sizeof(error));
    auto *threshold = h3d_alloc(gpu, H3D_F32, (size_t)blocks * heads, error, sizeof(error));
    auto *query_bar = h3d_alloc(gpu, H3D_F32, summary, error, sizeof(error));
    auto *kc_mean = h3d_alloc(gpu, H3D_F32, (size_t)heads * H3D_SOL_HEAD_DIM, error, sizeof(error));
    auto *kc_var = h3d_alloc(gpu, H3D_F32, (size_t)heads * H3D_SOL_HEAD_DIM, error, sizeof(error));
    if (!query || !key || !value || !output || !kc || !vc || !mask || !count ||
        !threshold || !query_bar || !kc_mean || !kc_var) {
        fail("alloc", error);
    }
    h3d_sol_plan plan = {kc, vc, mask, count, threshold, query_bar, kc_mean, kc_var};
    if (!h3d_upload(gpu, query, host_query.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, key, host_key.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, value, host_value.data(), error, sizeof(error))) {
        fail("upload", error);
    }

    /* Each arm is timed over `calls` back-to-back calls with one sync, so the
     * figure is the work and not a launch or sync latency. */
    auto time_calls = [&](const char *label, const auto &run) {
        run();
        if (h3d_sync(gpu, error, sizeof(error)) == 0) fail(label, error);
        auto begin = std::chrono::steady_clock::now();
        for (int call = 0; call < calls; ++call) {
            run();
        }
        if (h3d_sync(gpu, error, sizeof(error)) == 0) fail(label, error);
        const double each = elapsed_ms(begin, std::chrono::steady_clock::now()) / calls;
        printf("  %-28s %9.3f ms per call (mean of %d)\n", label, each, calls);
        return each;
    };
    auto search = [&](float at) {
        if (!h3d_sol_search(gpu, &plan, query, key, value, sequence, heads, scale,
                            at, 0, 0, error, sizeof(error))) {
            fail("search", error);
        }
    };
    auto forward = [&] {
        if (!h3d_sol_forward(gpu, output, &plan, query, key, value, sequence, heads,
                             scale, error, sizeof(error))) {
            fail("forward", error);
        }
    };
    auto dense = [&] {
        if (!h3d_sdpa(gpu, output, query, key, value, sequence, heads,
                      H3D_SOL_HEAD_DIM, scale, error, sizeof(error))) {
            fail("sdpa", error);
        }
    };

    const double dense_ms = time_calls("dense attention (h3d_sdpa)", dense);
    printf("arms at tau %.2f:\n", (double)tau);
    search(tau);
    const double search_ms = time_calls("search (routing)", [&] { search(tau); });
    const double forward_ms = time_calls("sparse forward", forward);

    std::vector<uint32_t> host_count((size_t)blocks * heads, 0);
    std::vector<uint32_t> host_mask((size_t)blocks * heads * words, 0);
    if (!h3d_read(gpu, host_count.data(), count, error, sizeof(error)) ||
        !h3d_read(gpu, host_mask.data(), mask, error, sizeof(error))) {
        fail("read", error);
    }
    uint64_t exact = 0, pairs = (uint64_t)blocks * blocks * heads;
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t q_block = 0; q_block < blocks; ++q_block) {
            uint32_t set = 0;
            for (uint32_t word = 0; word < words; ++word) {
                set += __builtin_popcount(
                    host_mask[((size_t)q_block * heads + head) * words + word]);
            }
            if (set != host_count[(size_t)q_block * heads + head]) {
                fail("count", "the mask and its count disagree");
            }
            exact += set;
        }
    }
    const double share = 100.0 * (double)exact / (double)pairs;
    printf("route: %llu of %llu query-block/KV-block pairs exact (%.1f%%), "
           "%.1f%% folded\n",
           (unsigned long long)exact, (unsigned long long)pairs, share,
           100.0 - share);
    printf("sparse attention per call: %.3f ms forward + %.3f ms search = %.3f ms, "
           "%.2fx the dense %.3f ms\n",
           forward_ms, search_ms, forward_ms + search_ms,
           dense_ms / (forward_ms + search_ms), dense_ms);
    printf("the forward alone is %.2fx the dense %.3f ms\n",
           dense_ms / forward_ms, dense_ms);

    h3d_free(query); h3d_free(key); h3d_free(value); h3d_free(output);
    h3d_free(kc); h3d_free(vc); h3d_free(mask); h3d_free(count);
    h3d_free(threshold); h3d_free(query_bar); h3d_free(kc_mean); h3d_free(kc_var);
    h3d_destroy(gpu);
    return 0;
}
