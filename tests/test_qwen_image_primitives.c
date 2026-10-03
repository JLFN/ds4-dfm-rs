/* Qwen-Image-2.1 DiT primitive parity on CUDA, against the CPU oracle's math.
 *
 * The three P2 kernels with no upstream ggml-cuda equivalent:
 *
 *   1. affine-free LayerNorm over the feature axis, eps 1e-6;
 *   2. modulate: split the [hidden, 2] parameter on its second axis, select
 *      the row by the token's side of the prefix, multiply, tanh-gate and add
 *      the residual;
 *   3. gated MLP: out = up * silu(gate), separate and fused [2n, rows] layouts.
 *
 * Each runs at the real DiT geometry (hidden 4096, intermediate 12288, 128
 * text + 4096 image = 4224 joint tokens, decode row 1) and is diffed against
 * a double host mirror of crates/ds4-core/src/qwen_image/dit.rs.  The kernels
 * accumulate in F32 and the silu/tanh tails ride fast-math expf, so the gate
 * is relative RMS <= 1e-5 with max abs <= 1e-4 on O(1) data; the observed
 * numbers printed below sit an order of magnitude inside it.
 *
 * Build: make test-qwen-image-primitives */

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Real DiT geometry (recipe section 3.1/4). */
enum {
    kHidden = 4096,
    kIntermediate = 12288,
    kTextTokens = 128,   /* the text prefix of the reference run */
    kImageTokens = 4096, /* 64x64 latent grid, patch 1x1 */
    kJointTokens = kTextTokens + kImageTokens,
};

static const double kRelRmsTol = 1.0e-5;
static const double kMaxAbsTol = 1.0e-4;

static int g_failures = 0;

/* splitmix64: deterministic and version-independent, unlike rand(). */
static uint64_t g_rng = 0x9e3779b97f4a7c15ull;

static uint64_t next_u64(void) {
    g_rng += 0x9e3779b97f4a7c15ull;
    uint64_t z = g_rng;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static void fill_signed(float *dst, size_t count, float scale) {
    for (size_t i = 0; i < count; i++) {
        const float unit = (float)(next_u64() >> 40) / (float)(1u << 24);
        dst[i] = (2.0f * unit - 1.0f) * scale;
    }
}

static float *host_alloc(size_t count) {
    float *p = (float *) malloc(count * sizeof(float));
    if (!p) {
        fprintf(stderr, "FAIL: host allocation of %zu floats\n", count);
        exit(1);
    }
    return p;
}

static ds4_gpu_tensor *device_alloc(size_t count) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t) count * sizeof(float));
    if (!t) {
        fprintf(stderr, "FAIL: device allocation of %zu floats\n", count);
        exit(1);
    }
    return t;
}

static void device_write(ds4_gpu_tensor *t, const float *src, size_t count) {
    if (!ds4_gpu_tensor_write(t, 0, src, (uint64_t) count * sizeof(float))) {
        fprintf(stderr, "FAIL: tensor write of %zu floats\n", count);
        exit(1);
    }
}

static void device_read(const ds4_gpu_tensor *t, float *dst, size_t count) {
    if (!ds4_gpu_tensor_read(t, 0, dst, (uint64_t) count * sizeof(float))) {
        fprintf(stderr, "FAIL: tensor read of %zu floats\n", count);
        exit(1);
    }
}

/* relative RMS = sqrt(sum (got-want)^2 / sum want^2), plus the max absolute
 * deviation.  The kernel drifts -> both move by orders of magnitude. */
static void compare(const char *label, const float *got, const float *want,
                    size_t count) {
    double se = 0.0, sr = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double d = (double) got[i] - (double) want[i];
        se += d * d;
        sr += (double) want[i] * (double) want[i];
        if (fabs(d) > max_abs) { max_abs = fabs(d); }
    }
    const double rel_rms = sr > 0.0 ? sqrt(se / sr)
                                    : (se > 0.0 ? INFINITY : 0.0);
    const int ok = rel_rms <= kRelRmsTol && max_abs <= kMaxAbsTol;
    if (!ok) { g_failures++; }
    printf("%-46s max_abs=%.3e rel_rms=%.3e  %s\n", label, max_abs, rel_rms,
           ok ? "ok" : "FAIL");
}

/* dit.rs::layer_norm_row: double accumulation, f32 statistics and output. */
static void ref_layernorm(float *out, const float *x, uint32_t dim,
                          uint32_t rows) {
    for (uint32_t t = 0; t < rows; t++) {
        const float *src = x + (size_t) t * dim;
        float *dst = out + (size_t) t * dim;

        double sum = 0.0;
        for (uint32_t i = 0; i < dim; i++) { sum += src[i]; }
        const float mean = (float)(sum / (double) dim);

        /* Centered second pass: no E[x^2] - mean^2 cancellation on a row with
         * a large offset, which is what the block activations carry. */
        double var = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            const double d = (double) src[i] - (double) mean;
            var += d * d;
        }
        const float var32 = (float)(var / (double) dim);
        const float scale = (float)(1.0 / sqrt((double) var32 + 1e-6));

        for (uint32_t i = 0; i < dim; i++) {
            dst[i] = (src[i] - mean) * scale;
        }
    }
}

/* dit.rs::modulate: row 0 (`p + 1` or `tanh(p)`) on the image tokens after
 * the prefix, row 1 on the prefix itself. */
static void ref_modulate(float *out, const float *x, const float *residual,
                         const float *param, uint32_t hidden, uint32_t tokens,
                         uint32_t prefix, int gated) {
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t row = (t < prefix) ? 1u : 0u;
        for (uint32_t i = 0; i < hidden; i++) {
            const size_t index = (size_t) t * hidden + i;
            const float p = param[(size_t) row * hidden + i];
            const float factor = gated ? (float) tanh((double) p) : p + 1.0f;
            out[index] = (residual ? residual[index] : 0.0f) + x[index] * factor;
        }
    }
}

/* dit.rs::silu: x / (1 + exp(-x)). */
static void ref_mlp(float *out, const float *gate, const float *up,
                    size_t count) {
    for (size_t i = 0; i < count; i++) {
        const double g = gate[i];
        out[i] = (float)((double) up[i] * (g / (1.0 + exp(-g))));
    }
}

/* The fused [2n, rows] form: gate in feature chunk 0, up in chunk 1. */
static void ref_mlp_fused(float *out, const float *fused, uint32_t n,
                          uint32_t rows) {
    for (uint32_t t = 0; t < rows; t++) {
        const float *row = fused + (size_t) t * 2 * n;
        for (uint32_t i = 0; i < n; i++) {
            const double g = row[i];
            out[(size_t) t * n + i] =
                    (float)((double) row[n + i] * (g / (1.0 + exp(-g))));
        }
    }
}

static void test_layernorm(uint32_t rows) {
    const size_t count = (size_t) kHidden * rows;
    float *x = host_alloc(count);
    float *want = host_alloc(count);
    float *got = host_alloc(count);

    fill_signed(x, count, 2.0f);
    for (uint32_t t = 0; t < rows; t++) {
        const float offset = 0.3f * (float)(t % 5u);
        for (uint32_t i = 0; i < kHidden; i++) {
            x[(size_t) t * kHidden + i] += offset;
        }
    }
    ref_layernorm(want, x, kHidden, rows);

    ds4_gpu_tensor *tx = device_alloc(count);
    device_write(tx, x, count);
    if (!ds4_gpu_qwen_image_layernorm_tensor(tx, kHidden, rows)) {
        fprintf(stderr, "FAIL: layernorm launch refused\n");
        exit(1);
    }
    device_read(tx, got, count);

    char label[64];
    snprintf(label, sizeof(label), "layernorm rows=%u dim=%u", rows, kHidden);
    compare(label, got, want, count);

    ds4_gpu_tensor_free(tx);
    free(x);
    free(want);
    free(got);
}

static void test_modulate(uint32_t tokens, uint32_t prefix, int gated,
                          int use_residual) {
    const size_t count = (size_t) kHidden * tokens;
    float *x = host_alloc(count);
    float *param = host_alloc((size_t) kHidden * 2);
    float *residual = use_residual ? host_alloc(count) : NULL;
    float *want = host_alloc(count);
    float *got = host_alloc(count);

    fill_signed(x, count, 1.5f);
    fill_signed(param, (size_t) kHidden * 2, 1.5f);
    if (residual) { fill_signed(residual, count, 1.5f); }
    ref_modulate(want, x, residual, param, kHidden, tokens, prefix, gated);

    ds4_gpu_tensor *tx = device_alloc(count);
    ds4_gpu_tensor *tparam = device_alloc((size_t) kHidden * 2);
    ds4_gpu_tensor *tres = use_residual ? device_alloc(count) : NULL;
    device_write(tx, x, count);
    device_write(tparam, param, (size_t) kHidden * 2);
    if (tres) { device_write(tres, residual, count); }
    if (!ds4_gpu_qwen_image_modulate_tensor(tx, tparam, tres, kHidden, tokens,
                                            prefix, (uint32_t) gated)) {
        fprintf(stderr, "FAIL: modulate launch refused\n");
        exit(1);
    }
    device_read(tx, got, count);

    char label[80];
    snprintf(label, sizeof(label), "modulate tokens=%u prefix=%u %s%s", tokens,
             prefix, gated ? "gated" : "plain",
             use_residual ? "+residual" : "");
    compare(label, got, want, count);

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tparam);
    if (tres) { ds4_gpu_tensor_free(tres); }
    free(x);
    free(param);
    free(residual);
    free(want);
    free(got);
}

static void test_mlp(uint32_t rows, int fused) {
    const size_t count = (size_t) kIntermediate * rows;
    const size_t source = fused ? 2 * count : count;
    float *gate = host_alloc(source);
    float *up = fused ? NULL : host_alloc(count);
    float *want = host_alloc(count);
    float *got = host_alloc(count);

    /* The gate pre-activation spans about [-6, 6] so silu crosses both tails;
     * up stays O(1) like the proj output. */
    fill_signed(gate, source, 6.0f);
    if (!fused) { fill_signed(up, count, 1.5f); }
    if (fused) {
        ref_mlp_fused(want, gate, kIntermediate, rows);
    } else {
        ref_mlp(want, gate, up, count);
    }

    ds4_gpu_tensor *tgate = device_alloc(source);
    ds4_gpu_tensor *tout = device_alloc(count);
    ds4_gpu_tensor *tup = NULL;
    device_write(tgate, gate, source);
    int launched;
    if (fused) {
        launched = ds4_gpu_qwen_image_mlp_gated_fused_tensor(tout, tgate,
                                                            kIntermediate, rows);
    } else {
        tup = device_alloc(count);
        device_write(tup, up, count);
        launched = ds4_gpu_qwen_image_mlp_gated_tensor(tout, tgate, tup,
                                                       kIntermediate, rows);
    }
    if (!launched) {
        fprintf(stderr, "FAIL: gated MLP launch refused\n");
        exit(1);
    }
    device_read(tout, got, count);

    char label[64];
    snprintf(label, sizeof(label), "mlp_gated%s rows=%u n=%u",
             fused ? "_fused" : "", rows, kIntermediate);
    compare(label, got, want, count);

    ds4_gpu_tensor_free(tgate);
    ds4_gpu_tensor_free(tout);
    if (tup) { ds4_gpu_tensor_free(tup); }
    free(gate);
    free(up);
    free(want);
    free(got);
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        return 1;
    }

    printf("== Qwen-Image-2.1 DiT primitives (CUDA vs the oracle's host math) ==\n");
    test_layernorm(1);
    test_layernorm(kJointTokens);

    test_modulate(1, 0, 0, 0);
    test_modulate(kTextTokens + 1, kTextTokens, 1, 1);
    test_modulate(kJointTokens, kTextTokens, 0, 0);
    test_modulate(kJointTokens, kTextTokens, 1, 1);

    test_mlp(1, 0);
    test_mlp(kJointTokens, 0);
    test_mlp(1, 1);
    test_mlp(kJointTokens, 1);

    ds4_gpu_cleanup();
    printf("%s\n", g_failures ? "primitive checks FAILED"
                              : "all Qwen-Image primitive checks passed");
    return g_failures ? 1 : 0;
}
