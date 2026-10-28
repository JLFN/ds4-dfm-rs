/*
 * Bounded reproducer for VSA's selector: the reference's own H3-stage sparsity
 * rule, priced against the tau search step it replaces and against the dense
 * attention it routes, at the shape the denoise runs.
 *
 *   tests/h3/sol_vsa_bm                      # 18,432 tokens, 56 heads
 *   tests/h3/sol_vsa_bm 4352 56 3 0.9 8      # the 512x512 shape, prefix 8
 *
 * The realized sparsity is reported beside every time, and `count` is checked
 * against the mask's own population before any of it is believed. The last
 * figure is the honest deviation measure: how many mask bits the port's F32
 * centroid order moves against the reference's own row order, sampled over a
 * few heads because the reference order costs blocks x 128 multiplies per
 * query block on the CPU.
 *
 * Build: `make tests/h3/sol_vsa_bm`.
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

static float of_bf16(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

/* Blocks carry a per-block signal so the selection has a margin to work with,
 * and V sits an order of magnitude above K as it does in the model. */
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

/* `compute_topk` (`video_sparse_attn.py:161-163`), the reference's formula. */
static uint32_t vsa_keep(double sparsity, uint32_t tiles) {
    const double keep = std::ceil((1.0 - sparsity) * (double)tiles);
    if (keep < 1.0) return 1u;
    if (keep > (double)tiles) return tiles;
    return (uint32_t)keep;
}

/* The reference's own order for one query block and head: its tile mean is the
 * sequential sum of the block's rows over its true row count (`_pool_tiles`,
 * `video_sparse_attn_h3.py:371-383`), and its score is the row-wise dot with
 * the pooled key tile (`:629`). The rule is `_build_block_mask` (`:387-410`). */
static std::vector<uint32_t> reference_words(
        const uint16_t *query, const float *kc, uint32_t sequence, uint32_t heads,
        uint32_t head, uint32_t q_block, uint32_t blocks, uint32_t words,
        uint32_t prefix, uint32_t keep, float scale) {
    const uint32_t first = q_block * H3D_SOL_BLOCK;
    uint32_t rows = sequence - first;
    if (rows > H3D_SOL_BLOCK) rows = H3D_SOL_BLOCK;
    std::vector<float> pooled(H3D_SOL_HEAD_DIM, 0.0f);
    for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
        float total = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            total += of_bf16(query[((size_t)head * sequence + first + row) *
                                       H3D_SOL_HEAD_DIM + dim]);
        }
        pooled[dim] = total / (float)rows;
    }
    std::vector<float> score(blocks, 0.0f);
    for (uint32_t block = 0; block < blocks; ++block) {
        float dot = 0.0f;
        for (uint32_t dim = 0; dim < H3D_SOL_HEAD_DIM; ++dim) {
            dot += pooled[dim] *
                   kc[((size_t)head * blocks + block) * H3D_SOL_HEAD_DIM + dim];
        }
        score[block] = dot * scale;
    }
    std::vector<uint32_t> bits(words, 0u);
    for (uint32_t block = 0; block < blocks; ++block) {
        bool exact = q_block < prefix || block < prefix;
        if (!exact) {
            uint32_t rank = 0;
            for (uint32_t other = prefix; other < blocks; ++other) {
                if (score[other] > score[block] ||
                    (score[other] == score[block] && other < block)) {
                    rank += 1;
                }
            }
            exact = rank < keep;
        }
        if (exact) bits[block / H3D_SOL_GROUP] |= 1u << (block % H3D_SOL_GROUP);
    }
    return bits;
}

int main(int argc, char **argv) {
    const uint32_t sequence = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 18432;
    const uint32_t heads = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 56;
    const int calls = argc > 3 ? std::atoi(argv[3]) : 3;
    const double sparsity = argc > 4 ? std::atof(argv[4]) : 0.9;
    const uint32_t prefix = argc > 5 ? (uint32_t)std::atoi(argv[5]) : 8;
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    const uint32_t blocks = (sequence + H3D_SOL_BLOCK - 1) / H3D_SOL_BLOCK;
    const uint32_t words = h3d_sol_words(sequence);
    const size_t elements = (size_t)sequence * heads * H3D_SOL_HEAD_DIM;
    const size_t summary = (size_t)blocks * heads * H3D_SOL_HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)H3D_SOL_HEAD_DIM);
    const uint32_t keep = vsa_keep(sparsity, blocks - prefix);
    printf("shape: %u tokens, %u heads, %u blocks (%u prefix, %u video), "
           "sparsity %.2f keeps %u video blocks\n",
           sequence, heads, blocks, prefix, blocks - prefix, sparsity, keep);

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
    auto dense = [&] {
        if (!h3d_sdpa(gpu, output, query, key, value, sequence, heads,
                      H3D_SOL_HEAD_DIM, scale, error, sizeof(error))) {
            fail("sdpa", error);
        }
    };
    auto tau_search = [&] {
        if (!h3d_sol_search(gpu, &plan, query, key, value, sequence, heads, scale,
                            1.25f, 0, 0, error, sizeof(error))) {
            fail("tau search", error);
        }
    };
    auto vsa_search = [&] {
        if (!h3d_sol_vsa(gpu, &plan, query, key, value, sequence, heads, scale,
                         sparsity, prefix, error, sizeof(error))) {
            fail("VSA search", error);
        }
    };
    auto forward = [&] {
        if (!h3d_sol_forward(gpu, output, &plan, query, key, value, sequence, heads,
                             scale, error, sizeof(error))) {
            fail("forward", error);
        }
    };

    const double dense_ms = time_calls("dense attention (h3d_sdpa)", dense);
    const double tau_ms = time_calls("tau search (threshold)", tau_search);
    const double vsa_ms = time_calls("VSA search (top-k)", vsa_search);
    const double forward_ms = time_calls("sparse forward", forward);
    printf("the selector adds %.3f ms over the tau path's own search step\n",
           vsa_ms - tau_ms);

    std::vector<uint32_t> host_count((size_t)blocks * heads, 0);
    std::vector<uint32_t> host_mask((size_t)blocks * heads * words, 0);
    std::vector<uint16_t> host_kc(summary);
    if (!h3d_read(gpu, host_count.data(), count, error, sizeof(error)) ||
        !h3d_read(gpu, host_mask.data(), mask, error, sizeof(error)) ||
        !h3d_read(gpu, host_kc.data(), kc, error, sizeof(error))) {
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
    printf("route: %llu of %llu query-block/KV-block pairs exact (%.2f%%), "
           "%.2f%% folded\n",
           (unsigned long long)exact, (unsigned long long)pairs, share, 100.0 - share);
    printf("per layer: %.3f ms VSA search + %.3f ms forward = %.3f ms against "
           "%.3f ms dense (%.2fx)\n",
           vsa_ms, forward_ms, vsa_ms + forward_ms, dense_ms,
           dense_ms / (vsa_ms + forward_ms));

    /* What the F32 centroid order moves against the reference's raw-row order.
     * Sampled: each pair costs blocks x 128 multiplies on one CPU thread. */
    std::vector<float> kc_float(summary);
    for (size_t at = 0; at < summary; ++at) kc_float[at] = of_bf16(host_kc[at]);
    const uint32_t sampled = std::min<uint32_t>(heads, 4);
    uint64_t bits = 0, moved = 0;
    for (uint32_t head = 0; head < sampled; ++head) {
        for (uint32_t q_block = 0; q_block < blocks; ++q_block) {
            const std::vector<uint32_t> want = reference_words(
                host_query.data(), kc_float.data(), sequence, heads, head, q_block,
                blocks, words, prefix, keep, scale);
            for (uint32_t word = 0; word < words; ++word) {
                const uint32_t got =
                    host_mask[((size_t)q_block * heads + head) * words + word];
                const uint32_t differ = want[word] ^ got;
                moved += (uint64_t)__builtin_popcount(differ);
                bits += (uint64_t)__builtin_popcount(got);
            }
        }
    }
    printf("F32 order: %llu of %llu mask bits differ from the reference's own row "
           "order over %u of %u heads (%.4f%%)\n",
           (unsigned long long)moved, (unsigned long long)bits, sampled, heads,
           100.0 * (double)moved / (double)(bits ? bits : 1));

    h3d_free(kc_var);
    h3d_free(kc_mean);
    h3d_free(query_bar);
    h3d_free(threshold);
    h3d_free(count);
    h3d_free(mask);
    h3d_free(vc);
    h3d_free(kc);
    h3d_free(output);
    h3d_free(value);
    h3d_free(key);
    h3d_free(query);
    h3d_destroy(gpu);
    return 0;
}
