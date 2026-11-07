/*
 * Bounded reproducer for Sol-Attn's search step: the routing decision that
 * picks which KV blocks attend exactly. It times the four kernels at the shape
 * the H3 denoise runs, and prices the search step against the dense attention
 * at the same shape, so its share of an attention call is a measurement rather
 * than an estimate.
 *
 *   tests/h3/sol_route_bm              # 18,432 tokens, 56 heads, 3 calls
 *   tests/h3/sol_route_bm 4352 56 5    # the 512x512 denoise shape
 *
 * One attention call is 50 body layers x 3 sparse steps in a Sol-H3 request
 * (49 layers with tau and one dense), and ~100 calls in this engine's 2-step
 * yardstick, so a per-call figure times 100 prices the step for that job.
 *
 * Build: `make tests/h3/sol_route_bm`.
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

/* Blocks carry a per-block signal so the route is a mix of exact and skipped;
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

    dim3 pool_grid(blocks, heads);
    dim3 threshold_grid(blocks, heads);
    dim3 route_grid((blocks + H3D_SOL_QM - 1) / H3D_SOL_QM, heads);
    /* Each kernel is timed over `calls` back-to-back launches with one sync, so
     * the figure is the kernel's own time and not a launch or sync latency. */
    auto time_kernel = [&](const char *label, const auto &launch) {
        launch();
        if (cudaDeviceSynchronize() != cudaSuccess) fail(label, "warmup");
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, gpu->stream);
        for (int call = 0; call < calls; ++call) {
            launch();
        }
        cudaEventRecord(end, gpu->stream);
        const cudaError_t status = cudaEventSynchronize(end);
        float total = 0.0f;
        cudaEventElapsedTime(&total, begin, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        if (status != cudaSuccess) fail(label, cudaGetErrorString(status));
        printf("  [%s] %.3f ms over %d back-to-back calls\n", label, (double)total,
               calls);
        return (double)total / calls;
    };
    const double pool = time_kernel("pool", [&] {
        h3d_sol_pool_kernel<<<pool_grid, H3D_SOL_THREADS, 0, gpu->stream>>>(
            (__nv_bfloat16 *)kc->data, (__nv_bfloat16 *)vc->data,
            (const __nv_bfloat16 *)key->data, (const __nv_bfloat16 *)value->data,
            sequence, blocks);
    });
    const double stats = time_kernel("stats", [&] {
        h3d_sol_stats_kernel<<<heads, H3D_SOL_THREADS, 0, gpu->stream>>>(
            (float *)kc_mean->data, (float *)kc_var->data,
            (const __nv_bfloat16 *)kc->data, blocks);
    });
    const double limit = time_kernel("threshold", [&] {
        h3d_sol_threshold_kernel<<<threshold_grid, H3D_SOL_WARP, 0, gpu->stream>>>(
            (float *)threshold->data, (float *)query_bar->data,
            (const float *)kc_mean->data, (const float *)kc_var->data,
            (const __nv_bfloat16 *)query->data, sequence, heads, scale, 1.25f);
    });
    const double route = time_kernel("route", [&] {
        h3d_sol_route_kernel<<<route_grid, H3D_SOL_WARP, 0, gpu->stream>>>(
            (uint32_t *)mask->data, (uint32_t *)count->data,
            (const __nv_bfloat16 *)kc->data, (const float *)query_bar->data,
            (const float *)threshold->data, heads, blocks, words, scale, 0, 0);
    });

    auto start = std::chrono::steady_clock::now();
    for (int call = 0; call < calls; ++call) {
        if (!h3d_sol_search(gpu, &plan, query, key, value, sequence, heads, scale,
                            1.25f, 0, 0, error, sizeof(error))) {
            fail("search", error);
        }
    }
    if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
    auto stop = std::chrono::steady_clock::now();
    const double whole = elapsed_ms(start, stop) / calls;

    start = std::chrono::steady_clock::now();
    for (int call = 0; call < calls; ++call) {
        if (!h3d_sdpa(gpu, output, query, key, value, sequence, heads,
                      H3D_SOL_HEAD_DIM, scale, error, sizeof(error))) {
            fail("sdpa", error);
        }
    }
    if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
    stop = std::chrono::steady_clock::now();
    const double dense = elapsed_ms(start, stop) / calls;

    std::vector<uint32_t> host_count((size_t)blocks * heads, 0);
    std::vector<uint32_t> host_mask((size_t)blocks * heads * words, 0);
    if (!h3d_read(gpu, host_count.data(), count, error, sizeof(error)) ||
        !h3d_read(gpu, host_mask.data(), mask, error, sizeof(error))) {
        fail("read", error);
    }
    uint64_t exact = 0, pairs = 0;
    for (uint32_t head = 0; head < heads; ++head) {
        for (uint32_t q_block = 0; q_block < blocks; ++q_block) {
            uint32_t set = 0;
            for (uint32_t word = 0; word < words; ++word) {
                set += __builtin_popcount(
                    host_mask[((size_t)q_block * heads + head) * words + word]);
            }
            exact += host_count[(size_t)q_block * heads + head];
            pairs += blocks;
            if (set != host_count[(size_t)q_block * heads + head]) {
                fail("count", "the mask and its count disagree");
            }
        }
    }
    const double share = 100.0 * (double)exact / (double)pairs;
    printf("kernel times per call (mean of %d):\n", calls);
    printf("  pool (kc, vc)   %8.3f ms\n", pool);
    printf("  kc statistics   %8.3f ms\n", stats);
    printf("  threshold       %8.3f ms\n", limit);
    printf("  route           %8.3f ms\n", route);
    printf("  search total    %8.3f ms  (%.1f%% of the dense attention below)\n",
           whole, 100.0 * whole / dense);
    printf("  dense attention %8.3f ms at the same shape (h3d_sdpa, tensor arm)\n",
           dense);
    printf("route: %llu of %llu query-block/KV-block pairs exact (%.1f%%), "
           "%.1f%% skipped\n",
           (unsigned long long)exact, (unsigned long long)pairs, share, 100.0 - share);

    h3d_free(query); h3d_free(key); h3d_free(value); h3d_free(output);
    h3d_free(kc); h3d_free(vc); h3d_free(mask); h3d_free(count);
    h3d_free(threshold); h3d_free(query_bar); h3d_free(kc_mean); h3d_free(kc_var);
    h3d_destroy(gpu);
    return 0;
}
