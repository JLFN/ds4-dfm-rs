/* Qwen-Image-2.1 DiT parity on CUDA, against the CPU oracle's math.
 *
 * The P2 kernels with no upstream ggml-cuda equivalent:
 *
 *   1. affine-free LayerNorm over the feature axis, eps 1e-6;
 *   2. modulate: split the [hidden, 2] parameter on its second axis, select
 *      the row by the token's side of the prefix, multiply, tanh-gate and add
 *      the residual;
 *   3. gated MLP: out = up * silu(gate), separate and fused [2n, rows] layouts;
 *   4. rope3d: the 3-axis adjacent-pair rope (widths 16/56/56, theta 10000)
 *      applied from a prebuilt [tokens, 64, 2, 2] table;
 *   5. segment attention: queries [start, end) attend to keys [0, end), the
 *      text prefix causal, the image segment unmasked and bidirectional;
 *   6. the timestep path: the host-built 256-wide sinusoidal table (theta
 *      10000) through the 256 -> 4096 MLP, whose two silus run on the device;
 *   7. patch/unpatch at a 1x1 patch: the latent's channel-slowest layout and
 *      the joint matrix's channel-fastest one, a pure permutation.
 *
 * Each runs at the real DiT geometry (hidden 4096, intermediate 12288, 128
 * text + 4096 image = 4224 joint tokens, 32 heads x head dim 128, 64 latent
 * channels) and is diffed against a double host mirror of
 * crates/ds4-core/src/qwen_image (dit.rs and oracle.rs).  The kernels
 * accumulate in F32 and the silu/tanh/exp tails ride fast-math, so the gate is
 * relative RMS <= 1e-5 with max abs <= 1e-4 on O(1) data; the observed numbers
 * printed below sit an order of magnitude inside it.
 *
 * The attention case is fed the oracle's own normalized and roped q/k, so it
 * isolates the segment kernel; the rope case feeds the normalized q/k and
 * diffs the rotation.  The two compare separately because the text segment's
 * causal mask and the image segment's absence of one are different failures.
 * The timestep case likewise builds the oracle's own table and diffs the MLP
 * on the device only, since the table's sin/cos never run there.
 *
 * Build: make test-qwen-image-primitives */

#include "ds4_gpu.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Real DiT geometry (recipe section 3.1/4). */
enum {
    kHidden = 4096,
    kIntermediate = 12288,
    kTextTokens = 128,   /* the text prefix of the reference run */
    kImageTokens = 4096, /* 64x64 latent grid, patch 1x1 */
    kJointTokens = kTextTokens + kImageTokens,
    kHeads = 32,         /* hidden / head dim */
    kHeadDim = 128,
    kRopePairs = kHeadDim / 2, /* 8 + 28 + 28 over the three axes */
    kImageSide = 64,
    kTimeEmbedDim = 256, /* the sinusoidal embedding width */
    kChannels = 64,      /* the latent's in and out channels */
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

/* dit.rs::silu, the same double form ref_mlp uses inside its gate. */
static void ref_silu(float *x, size_t count) {
    for (size_t i = 0; i < count; i++) {
        const double v = x[i];
        x[i] = (float)(v / (1.0 + exp(-v)));
    }
}

/* dit.rs::timestep_embedding in the oracle's own f32 arithmetic: per column,
 * feature j < dim/2 is cos(t * freq) and feature j >= dim/2 the same angle's
 * sin, with freq = exp(-ln(theta) * j / (dim/2)).  The f32 rounding of the
 * argument is the oracle's too, and it is why this table is host-built: the
 * angles reach 1000 rad at the top of the flow grid, where an f32 argument
 * already costs 6e-5, and the device's fast-math cosf adds 5.5e-4 on top. */
static void ref_timestep_table(float *table, const float *times, uint32_t cols,
                               uint32_t dim) {
    const uint32_t half = dim / 2u;
    const float log_period = logf(10000.0f);

    for (uint32_t c = 0; c < cols; c++) {
        for (uint32_t j = 0; j < half; j++) {
            const float freq = expf(-log_period * (float) j / (float) half);
            const float arg = times[c] * freq;
            table[j + dim * c] = cosf(arg);
            table[j + half + dim * c] = sinf(arg);
        }
    }
}

/* dit.rs::matmul_rows (dot8) in double: W is [m, k] row-major, x is
 * feature-fastest [k, n] and the result is feature-fastest [m, n]. */
static void ref_linear(float *out, const float *w, uint32_t k, uint32_t m,
                       const float *x, uint32_t n) {
    for (uint32_t t = 0; t < n; t++) {
        for (uint32_t o = 0; o < m; o++) {
            double sum = 0.0;
            for (uint32_t i = 0; i < k; i++) {
                sum += (double) w[(size_t) o * k + i] *
                       (double) x[i + (size_t) k * t];
            }
            out[o + (size_t) m * t] = (float) sum;
        }
    }
}

/* dit.rs's patchify/unpatchify at a 1x1 patch: src [b, a] feature-fastest
 * becomes dst [a, b] feature-fastest, dst[j + b*i] = src[i + a*j].  The
 * caller picks which axis is which: patchify passes a = pixels, b = channels
 * (and reads the latent), unpatchify the pair the other way round. */
static void ref_transpose(float *dst, const float *src, uint32_t a, uint32_t b) {
    for (uint32_t i = 0; i < a; i++) {
        for (uint32_t j = 0; j < b; j++) {
            dst[j + (size_t) b * i] = src[i + (size_t) a * j];
        }
    }
}

/* dit.rs::rms_norm_row: the sum of squares in double, scale in f32, then
 * x * scale * w.  norm_q/norm_k are per-head vectors shared by every head. */
static void ref_rms_norm_row(float *x, const float *weight, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; i++) { sum += (double) x[i] * (double) x[i]; }

    const float mean = (float)(sum / (double) dim);
    const float scale = 1.0f / sqrtf(mean + 1e-6f);
    for (size_t i = 0; i < dim; i++) { x[i] = x[i] * scale * weight[i]; }
}

/* oracle.rs::build_layout for the text-to-image case: text tokens carry a
 * monotonic scalar on all three axes, then the image grid takes a constant
 * temporal id and a centered spatial one. */
static void ref_positions(float *pos, size_t text, size_t side) {
    size_t at = 0;

    for (size_t i = 0; i < text; i++) {
        pos[at * 3 + 0] = pos[at * 3 + 1] = pos[at * 3 + 2] = (float) i;
        at++;
    }

    const float temporal = (float) text;
    const float center = (float)(side - side / 2);
    for (size_t h = 0; h < side; h++) {
        for (size_t w = 0; w < side; w++) {
            pos[at * 3 + 0] = temporal;
            pos[at * 3 + 1] = (float) h - center;
            pos[at * 3 + 2] = (float) w - center;
            at++;
        }
    }
}

/* oracle.rs::rope_omega/rope_table: one rope per axis at widths 16/56/56,
 * theta 10000, concatenated on the pair axis.  Each pair is stored
 * [[cos, -sin], [sin, cos]] (cos at +0, sin at +2). */
static void ref_rope_table(float *table, const float *pos, size_t tokens) {
    static const uint32_t axes[3] = {16, 56, 56};
    const float theta = 10000.0f;
    size_t pair_offset = 0;

    for (size_t axis = 0; axis < 3; axis++) {
        const uint32_t dim = axes[axis];
        const uint32_t half = dim / 2;
        float omega[28]; /* the widest axis has 28 pairs */
        const float end = ((float) dim - 2.0f) / (float) dim;
        const float step = half > 1 ? end / (float)(half - 1) : 0.0f;

        for (uint32_t j = 0; j < half; j++) {
            omega[j] = 1.0f / powf(theta, (float) j * step);
        }

        for (size_t t = 0; t < tokens; t++) {
            const float id = pos[t * 3 + axis];
            for (uint32_t j = 0; j < half; j++) {
                const float angle = id * omega[j];
                const float c = cosf(angle);
                const float s = sinf(angle);
                float *base = table + (t * kRopePairs + pair_offset + j) * 4;
                base[0] = c;
                base[1] = -s;
                base[2] = s;
                base[3] = c;
            }
        }
        pair_offset += half;
    }
}

/* oracle.rs::apply_rope with RopePairing::Interleaved: adjacent pairs of the
 * head-major [head][token][head_dim] tensor. */
static void ref_apply_rope(float *x, const float *table, size_t tokens,
                           size_t heads, size_t dim) {
    const size_t pairs = dim / 2;

    for (size_t h = 0; h < heads; h++) {
        for (size_t t = 0; t < tokens; t++) {
            float *row = x + (h * tokens + t) * dim;
            const float *pe = table + t * pairs * 4;
            for (size_t j = 0; j < pairs; j++) {
                const float cos = pe[j * 4 + 0];
                const float sin = pe[j * 4 + 2];
                const float x0 = row[2 * j];
                const float x1 = row[2 * j + 1];
                row[2 * j] = x0 * cos - x1 * sin;
                row[2 * j + 1] = x0 * sin + x1 * cos;
            }
        }
    }
}

/* dit.rs::dot8: eight accumulators, a fixed horizontal sum, then the tail. */
static float ref_dot8(const float *a, const float *b, size_t n) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const size_t full = n / 8;

    for (size_t chunk = 0; chunk < full; chunk++) {
        for (size_t lane = 0; lane < 8; lane++) {
            acc[lane] += a[chunk * 8 + lane] * b[chunk * 8 + lane];
        }
    }

    float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
              + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (size_t i = full * 8; i < n; i++) { sum += a[i] * b[i]; }
    return sum;
}

/* One head's share of dit.rs::segment_attention: q row (head h, query) against
 * keys [0, end), the causal mask on the text prefix, softmax, then the oracle's
 * eight-lane PV order.  Queries own disjoint output runs, so splitting over
 * heads can never change a sum. */
typedef struct {
    const float *q, *k, *v;
    float *out;
    size_t head, heads, head_dim, tokens, start, end;
    int causal;
} RefAttnJob;

static void *ref_attn_head(void *arg) {
    const RefAttnJob *job = (const RefAttnJob *) arg;
    const size_t hidden = job->heads * job->head_dim;
    const size_t hd = job->head * job->head_dim;
    const float scale = 1.0f / sqrtf((float) job->head_dim);
    float *scores = host_alloc(job->end);

    for (size_t query = job->start; query < job->end; query++) {
        const float *q_row = job->q + (job->head * job->tokens + query) * job->head_dim;

        for (size_t key = 0; key < job->end; key++) {
            const float *k_row = job->k + (job->head * job->tokens + key) * job->head_dim;
            float score = ref_dot8(q_row, k_row, job->head_dim) * scale;
            if (job->causal && key > query) { score = -INFINITY; }
            scores[key] = score;
        }

        float max = -INFINITY;
        for (size_t key = 0; key < job->end; key++) { max = fmaxf(max, scores[key]); }

        float sum = 0.0f;
        for (size_t key = 0; key < job->end; key++) {
            scores[key] = expf(scores[key] - max);
            sum += scores[key];
        }
        for (size_t key = 0; key < job->end; key++) { scores[key] /= sum; }

        for (size_t d = 0; d < job->head_dim; d++) {
            float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            size_t key = 0;

            for (; key + 8 <= job->end; key += 8) {
                for (size_t lane = 0; lane < 8; lane++) {
                    acc[lane] += scores[key + lane] * job->v[(key + lane) * hidden + hd + d];
                }
            }

            float total = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                        + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for (; key < job->end; key++) {
                total += scores[key] * job->v[key * hidden + hd + d];
            }
            job->out[query * hidden + hd + d] = total;
        }
    }

    free(scores);
    return NULL;
}

static void ref_segment_attention(const float *q, const float *k, const float *v,
                                  size_t heads, size_t head_dim, size_t tokens,
                                  int causal, size_t start, size_t end,
                                  float *out) {
    pthread_t threads[kHeads];
    RefAttnJob jobs[kHeads];

    for (size_t h = 0; h < heads; h++) {
        jobs[h] = (RefAttnJob) { q, k, v, out, h, heads, head_dim, tokens,
                                 start, end, causal };
        if (pthread_create(&threads[h], NULL, ref_attn_head, &jobs[h]) != 0) {
            fprintf(stderr, "FAIL: pthread_create\n");
            exit(1);
        }
    }
    for (size_t h = 0; h < heads; h++) { pthread_join(threads[h], NULL); }
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

/* The position and attention path at the real DiT shape: 32 heads x 128, 128
 * text + 4096 image tokens.  The fixture carries the oracle's own normalized
 * q/k; the rope case diffs the rotation against it, then the attention case
 * consumes the oracle's roped tensors, so each kernel is isolated. */
static void test_attn_path(void) {
    const size_t rows = (size_t) kHeads * kJointTokens;
    const size_t qk_count = rows * kHeadDim;
    const size_t out_count = (size_t) kHidden * kJointTokens;
    const size_t pe_count = (size_t) kJointTokens * kRopePairs * 4;

    float *q = host_alloc(qk_count);
    float *k = host_alloc(qk_count);
    float *q_roped = host_alloc(qk_count);
    float *k_roped = host_alloc(qk_count);
    float *v = host_alloc(out_count);
    float *want = host_alloc(out_count);
    float *got = host_alloc(out_count);
    float *norm_q = host_alloc(kHeadDim);
    float *norm_k = host_alloc(kHeadDim);
    float *pos = host_alloc((size_t) kJointTokens * 3);
    float *pe = host_alloc(pe_count);

    fill_signed(q, qk_count, 1.0f);
    fill_signed(k, qk_count, 1.0f);
    fill_signed(v, out_count, 1.0f);
    fill_signed(norm_q, kHeadDim, 1.0f);
    fill_signed(norm_k, kHeadDim, 1.0f);

    /* dit.rs::attention, host side: the per-head norm_q/norm_k rows, then the
     * interleaved rope over the [head][token][head_dim] layout. */
    for (size_t h = 0; h < kHeads; h++) {
        for (size_t t = 0; t < kJointTokens; t++) {
            ref_rms_norm_row(q + (h * kJointTokens + t) * kHeadDim, norm_q, kHeadDim);
            ref_rms_norm_row(k + (h * kJointTokens + t) * kHeadDim, norm_k, kHeadDim);
        }
    }

    ref_positions(pos, kTextTokens, kImageSide);
    ref_rope_table(pe, pos, kJointTokens);

    memcpy(q_roped, q, qk_count * sizeof(float));
    memcpy(k_roped, k, qk_count * sizeof(float));
    ref_apply_rope(q_roped, pe, kJointTokens, kHeads, kHeadDim);
    ref_apply_rope(k_roped, pe, kJointTokens, kHeads, kHeadDim);

    ds4_gpu_tensor *tq = device_alloc(qk_count);
    ds4_gpu_tensor *tk = device_alloc(qk_count);
    ds4_gpu_tensor *tv = device_alloc(out_count);
    ds4_gpu_tensor *tout = device_alloc(out_count);
    ds4_gpu_tensor *tpe = device_alloc(pe_count);

    device_write(tq, q, qk_count);
    device_write(tk, k, qk_count);
    device_write(tpe, pe, pe_count);
    if (!ds4_gpu_qwen_image_rope3d_tensor(tq, tpe, kJointTokens, kHeads, kHeadDim) ||
        !ds4_gpu_qwen_image_rope3d_tensor(tk, tpe, kJointTokens, kHeads, kHeadDim)) {
        fprintf(stderr, "FAIL: rope3d launch refused\n");
        exit(1);
    }
    device_read(tq, got, qk_count);
    compare("rope3d q heads=32 tokens=4224", got, q_roped, qk_count);
    device_read(tk, got, qk_count);
    compare("rope3d k heads=32 tokens=4224", got, k_roped, qk_count);

    /* The oracle's segmentation: the causal text prefix first, then the
     * unmasked image segment, whose queries see the whole joint sequence. */
    ref_segment_attention(q_roped, k_roped, v, kHeads, kHeadDim, kJointTokens,
                          1, 0, kTextTokens, want);
    ref_segment_attention(q_roped, k_roped, v, kHeads, kHeadDim, kJointTokens,
                          0, kTextTokens, kJointTokens, want);

    device_write(tq, q_roped, qk_count);
    device_write(tk, k_roped, qk_count);
    device_write(tv, v, out_count);
    if (!ds4_gpu_qwen_image_attn_segment_tensor(tout, tq, tk, tv, kJointTokens,
                                                kHeads, kHeadDim, 0, kTextTokens, 1) ||
        !ds4_gpu_qwen_image_attn_segment_tensor(tout, tq, tk, tv, kJointTokens,
                                                kHeads, kHeadDim, kTextTokens,
                                                kJointTokens, 0)) {
        fprintf(stderr, "FAIL: segment attention launch refused\n");
        exit(1);
    }
    device_read(tout, got, out_count);

    compare("attn text prefix (causal) 128", got, want,
            (size_t) kTextTokens * kHidden);
    compare("attn image (unmasked) 4096x4224",
            got + (size_t) kTextTokens * kHidden,
            want + (size_t) kTextTokens * kHidden,
            (size_t) kImageTokens * kHidden);

    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tk);
    ds4_gpu_tensor_free(tv);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tpe);
    free(q);
    free(k);
    free(q_roped);
    free(k_roped);
    free(v);
    free(want);
    free(got);
    free(norm_q);
    free(norm_k);
    free(pos);
    free(pe);
}

/* In-place silu at the timestep path's real shape; the data span both tails. */
static void test_silu(uint32_t dim, uint32_t rows) {
    const size_t count = (size_t) dim * rows;
    float *x = host_alloc(count);
    float *want = host_alloc(count);
    float *got = host_alloc(count);

    fill_signed(x, count, 6.0f);
    memcpy(want, x, count * sizeof(float));
    ref_silu(want, count);

    ds4_gpu_tensor *tx = device_alloc(count);
    device_write(tx, x, count);
    if (!ds4_gpu_qwen_image_silu_tensor(tx, dim, rows)) {
        fprintf(stderr, "FAIL: silu launch refused\n");
        exit(1);
    }
    device_read(tx, got, count);

    char label[64];
    snprintf(label, sizeof(label), "silu dim=%u rows=%u", dim, rows);
    compare(label, got, want, count);

    ds4_gpu_tensor_free(tx);
    free(x);
    free(want);
    free(got);
}

/* With the zero timestep the table collapses to cos(0) = 1 over the low half
 * of the feature axis and sin(0) = 0 over the high half, so its two columns
 * must be bit-identical.  The device comparison cannot catch a swapped half
 * or column -- both sides are fed this same table -- so the layout is pinned
 * here instead. */
static void check_zero_table(const float *table, uint32_t dim) {
    const uint32_t half = dim / 2u;
    int ok = 1;

    for (uint32_t j = 0; j < half; j++) {
        ok = ok && table[j] == 1.0f && table[j + half] == 0.0f;
    }
    for (uint32_t i = 0; i < dim; i++) {
        ok = ok && table[i] == table[dim + i];
    }
    if (!ok) { g_failures++; }
    printf("%-46s %s\n", "timestep table t=0: cos=1, sin=0, equal columns",
           ok ? "ok" : "FAIL");
}

/* One denoise step's timestep path: the oracle's own f32 table for [t, 0]
 * through linear_1 (256 -> 4096) -> silu -> linear_2 (4096 -> 4096) -> silu,
 * the two GEMMs and both silus on the device.  Both Linears have n_tok = 2,
 * so the f32 GEMM enters through the stable-rows entry: this box's default
 * f32 entry is TF32 (measured 3.1e-4 rel RMS against an exact dot), three
 * decades outside the gate. */
static void test_timestep(float timestep, const void *map, uint64_t map_bytes,
                          uint64_t w1_offset, uint64_t w2_offset) {
    const uint32_t cols = 2; /* dit.rs: concat(timestep, 0) */
    const size_t table_count = (size_t) kTimeEmbedDim * cols;
    const size_t count = (size_t) kHidden * cols;
    const float times[2] = { timestep, 0.0f };
    const float *w1 = (const float *) ((const char *) map + w1_offset);
    const float *w2 = (const float *) ((const char *) map + w2_offset);

    float *table = host_alloc(table_count);
    float *want = host_alloc(count);
    float *want2 = host_alloc(count);
    float *got = host_alloc(count);

    ref_timestep_table(table, times, cols, kTimeEmbedDim);
    if (timestep == 0.0f) { check_zero_table(table, kTimeEmbedDim); }

    ref_linear(want, w1, kTimeEmbedDim, kHidden, table, cols);
    ref_silu(want, count);
    ref_linear(want2, w2, kHidden, kHidden, want, cols);
    ref_silu(want2, count);

    ds4_gpu_tensor *ttable = device_alloc(table_count);
    ds4_gpu_tensor *th1 = device_alloc(count);
    ds4_gpu_tensor *th2 = device_alloc(count);
    device_write(ttable, table, table_count);

    if (!ds4_gpu_matmul_f32_stable_rows_tensor(th1, map, map_bytes, w1_offset,
                                               kTimeEmbedDim, kHidden, ttable,
                                               cols) ||
        !ds4_gpu_qwen_image_silu_tensor(th1, kHidden, cols)) {
        fprintf(stderr, "FAIL: timestep linear_1/silu refused\n");
        exit(1);
    }
    device_read(th1, got, count);

    char label[64];
    snprintf(label, sizeof(label), "timestep mlp hidden t=%g", (double) timestep);
    compare(label, got, want, count);

    if (!ds4_gpu_matmul_f32_stable_rows_tensor(th2, map, map_bytes, w2_offset,
                                               kHidden, kHidden, th1, cols) ||
        !ds4_gpu_qwen_image_silu_tensor(th2, kHidden, cols)) {
        fprintf(stderr, "FAIL: timestep linear_2/silu refused\n");
        exit(1);
    }
    device_read(th2, got, count);
    snprintf(label, sizeof(label), "timestep mlp out t=%g", (double) timestep);
    compare(label, got, want2, count);

    ds4_gpu_tensor_free(ttable);
    ds4_gpu_tensor_free(th1);
    ds4_gpu_tensor_free(th2);
    free(table);
    free(want);
    free(want2);
    free(got);
}

/* Both directions of the 1x1 patch at one latent grid: a random latent
 * through the patch entry and a random token matrix through the unpatch
 * entry, each against the mirror.  Separate comparisons so a direction-
 * specific index error cannot hide behind the other one. */
static void test_patch(uint32_t channels, uint32_t pixels) {
    const size_t count = (size_t) channels * pixels;
    float *source = host_alloc(count);
    float *want = host_alloc(count);
    float *got = host_alloc(count);

    ds4_gpu_tensor *tsrc = device_alloc(count);
    ds4_gpu_tensor *tdst = device_alloc(count);
    char label[80];

    /* patchify: the latent [pixels, channels] gathered into [channels, pixels]. */
    fill_signed(source, count, 1.0f);
    ref_transpose(want, source, pixels, channels);

    device_write(tsrc, source, count);
    if (!ds4_gpu_qwen_image_patch_1x1_tensor(tdst, tsrc, channels, pixels)) {
        fprintf(stderr, "FAIL: patch launch refused\n");
        exit(1);
    }
    device_read(tdst, got, count);
    snprintf(label, sizeof(label), "patch_1x1 channels=%u pixels=%u", channels,
             pixels);
    compare(label, got, want, count);

    /* unpatch_crop: proj_out's [channels, pixels] back into the latent. */
    fill_signed(source, count, 1.0f);
    ref_transpose(want, source, channels, pixels);

    device_write(tsrc, source, count);
    if (!ds4_gpu_qwen_image_unpatch_crop_tensor(tdst, tsrc, channels, pixels)) {
        fprintf(stderr, "FAIL: unpatch launch refused\n");
        exit(1);
    }
    device_read(tdst, got, count);
    snprintf(label, sizeof(label), "unpatch_crop channels=%u pixels=%u",
             channels, pixels);
    compare(label, got, want, count);

    ds4_gpu_tensor_free(tsrc);
    ds4_gpu_tensor_free(tdst);
    free(source);
    free(want);
    free(got);
}

/* The timestep MLP's two Linears as one mapped blob, laid out the way the
 * artifact holds them (row-major, ne0 = in fastest), then both timestep
 * cases.  The map outlives the run and is freed by main after cleanup. */
static void *test_timestep_path(void) {
    const uint64_t off1 = 4096; /* aligned past the header the map may hold */
    const uint64_t off2 = off1 + (uint64_t) kHidden * kTimeEmbedDim * sizeof(float);
    const uint64_t bytes = off2 + (uint64_t) kHidden * kHidden * sizeof(float);

    void *map = NULL;
    if (posix_memalign(&map, off1, bytes)) {
        fprintf(stderr, "FAIL: weight map allocation of %llu bytes\n",
                (unsigned long long) bytes);
        exit(1);
    }
    memset(map, 0, bytes);
    fill_signed((float *) ((char *) map + off1),
                (size_t) kHidden * kTimeEmbedDim, 0.35f);
    fill_signed((float *) ((char *) map + off2), (size_t) kHidden * kHidden,
                0.02f);

    if (!ds4_gpu_set_model_map(map, bytes)) {
        fprintf(stderr, "FAIL: weight map registration\n");
        exit(1);
    }

    test_timestep(999.7f, map, bytes, off1, off2);
    test_timestep(0.0f, map, bytes, off1, off2);
    return map;
}

/* ---------------------------------------------------------------------------
 * Qwen-Image-2.1 VAE decoder primitives (P2 unit 11), against the P1 oracle
 * crates/ds4-core/src/qwen_image/vae.rs.
 *
 * The convs are compositions: im2col3x3 (or the 1x1 patch permutation) + the
 * tree's F16 GEMM + the bias add + the inverse permutation.  Their operands are
 * F16 (the reference F16-casts the conv weights and writes its im2col patches
 * in the weight's type), so the mirrors round activations with the oracle's own
 * f16_round and use F16-rounded weights, accumulated in double as the truth.
 * ------------------------------------------------------------------------- */

/* vae.rs::f16_round: round-to-nearest-even to the binary16 grid, overflow to
 * infinity, subnormals by magnitude with their sign kept. */
static float as_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float vae_f16_round(float value) {
    const uint32_t SIGN = 0x80000000u, INF_BITS = 0x7f800000u;
    const uint32_t NAN_BITS = 0x7fc00000u, MANT_BITS = 0x007fffffu;
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & SIGN;
    const int exponent = (int)((bits >> 23) & 0xffu) - 127;
    const uint32_t mantissa = bits & MANT_BITS;

    if ((bits & 0x7fffffffu) >= INF_BITS) {
        return as_float(mantissa == 0 ? (sign | INF_BITS) : (sign | NAN_BITS));
    }
    if (exponent > 15) { return as_float(sign | INF_BITS); }
    if (exponent >= -14) {
        const uint32_t keep = mantissa >> 13;
        const uint32_t rest = mantissa & 0x1fffu;
        uint32_t half = keep;
        if (rest > 0x1000u || (rest == 0x1000u && (keep & 1u))) { half += 1; }
        if (half == 0x400u) {
            if (exponent == 15) { return as_float(sign | INF_BITS); }
            return as_float(sign | (((uint32_t)(exponent + 128)) << 23));
        }
        return as_float(sign | (((uint32_t)(exponent + 127)) << 23) | (half << 13));
    }
    if (exponent < -25) { return as_float(sign); }
    const uint32_t steps = (uint32_t)rint(fabsf(value) * 16777216.0f);
    if (steps == 0) { return as_float(sign); }
    uint32_t leading = 31u;
    while (!(steps & (1u << leading))) { leading--; }
    const uint32_t biased = leading + 127u - 24u;
    return as_float(sign | (biased << 23) | ((steps - (1u << leading)) << (23 - leading)));
}

/* The map stores F16 weights; pack_f16 mirrors CUDA's __float2half (cvt.rn.f16
 * .f32), which was verified bit-identical to vae_f16_round on the oracle's grid,
 * subnormal, overflow and tie vectors. */
static uint16_t pack_f16(float value) {
    uint32_t x;
    memcpy(&x, &value, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t u = x & 0x7fffffffu;
    uint32_t result;
    uint32_t remainder;
    if (u >= 0x7f800000u) {
        remainder = 0;
        result = (u == 0x7f800000u) ? (sign | 0x7c00u) : 0x7fffu;
    } else if (u > 0x477fefffu) {
        remainder = 0x80000000u;
        result = sign | 0x7bffu;
    } else if (u >= 0x38800000u) {
        remainder = u << 19;
        u -= 0x38000000u;
        result = sign | (u >> 13);
    } else if (u < 0x33000001u) {
        remainder = u;
        result = sign;
    } else {
        const uint32_t exponent = u >> 23;
        const uint32_t shift = 0x7e - exponent;
        const uint32_t mantissa = (u & 0x7fffffu) | 0x800000u;
        remainder = mantissa << (32u - shift);
        result = (sign | (mantissa >> shift)) & 0xffffu;
    }
    if (remainder > 0x80000000u || (remainder == 0x80000000u && (result & 1u))) {
        result++;
    }
    return (uint16_t) result;
}

static float unpack_f16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int e = 0;
            while (!(m & 0x400u)) { m <<= 1; e++; }
            bits = sign | ((uint32_t)(113 - e) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    return as_float(bits);
}

/* ref_layernorm's companion: the same relative-RMS and max-abs report with a
 * caller-supplied bound (the F16 conv operands ride a wider tolerance than the
 * DiT's F32 kernels). */
static void compare_tol(const char *label, const float *got, const float *want,
                        size_t count, double rel_tol, double abs_tol) {
    double se = 0.0, sr = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double d = (double) got[i] - (double) want[i];
        se += d * d;
        sr += (double) want[i] * (double) want[i];
        if (fabs(d) > max_abs) { max_abs = fabs(d); }
    }
    const double rel_rms = sr > 0.0 ? sqrt(se / sr) : (se > 0.0 ? INFINITY : 0.0);
    const int ok = rel_rms <= rel_tol && max_abs <= abs_tol;
    if (!ok) { g_failures++; }
    printf("%-46s max_abs=%.3e rel_rms=%.3e  %s\n", label, max_abs, rel_rms,
           ok ? "ok" : "FAIL");
}

/* The F16 conv gates: two identical F16 operand sets summed in different orders
 * differ by accumulation drift only, measured well inside these bounds. */
static const double kF16RelRmsTol = 2.0e-4;
static const double kF16MaxAbsTol = 2.0e-3;

/* vae.rs::conv3x3 over a channel-slowest plane, F16 operands, double truth.
 * The operand rounding is hoisted out of the (oc, tap, ic) walk: it depends on
 * (pixel, ic) only, so rounding once per plane element is the same values at a
 * fraction of the mirror's cost. */
static void ref_vae_conv3x3(float *out, const float *x, const uint16_t *w,
                            const float *bias, uint32_t ci, uint32_t co,
                            uint32_t width, uint32_t height) {
    const uint32_t pixels = width * height;
    float *rounded = host_alloc((size_t) pixels * ci);
    for (size_t i = 0; i < (size_t) pixels * ci; i++) { rounded[i] = vae_f16_round(x[i]); }

    for (uint32_t oc = 0; oc < co; oc++) {
        for (uint32_t h = 0; h < height; h++) {
            for (uint32_t p = 0; p < width; p++) {
                double acc = 0.0;
                for (uint32_t tap = 0; tap < 9u; tap++) {
                    const int ih = (int) h + (int)(tap / 3u) - 1;
                    const int iw = (int) p + (int)(tap % 3u) - 1;
                    if (ih < 0 || ih >= (int) height || iw < 0 || iw >= (int) width) { continue; }
                    for (uint32_t ic = 0; ic < ci; ic++) {
                        acc += (double) rounded[(uint32_t)(iw + width * ih) + pixels * ic] *
                               (double) unpack_f16(w[tap + 9u * (ic + ci * oc)]);
                    }
                }
                out[(h * width + p) + pixels * oc] = (float) acc + bias[oc];
            }
        }
    }
    free(rounded);
}

/* vae.rs::conv1x1: one unchanged position per pixel, F16 operands. */
static void ref_vae_conv1x1(float *out, const float *x, const uint16_t *w,
                            const float *bias, uint32_t ci, uint32_t co,
                            uint32_t pixels) {
    float *rounded = host_alloc((size_t) pixels * ci);
    for (size_t i = 0; i < (size_t) pixels * ci; i++) { rounded[i] = vae_f16_round(x[i]); }

    for (uint32_t oc = 0; oc < co; oc++) {
        for (uint32_t p = 0; p < pixels; p++) {
            double acc = 0.0;
            for (uint32_t ic = 0; ic < ci; ic++) {
                acc += (double) rounded[p + pixels * ic] *
                       (double) unpack_f16(w[ic + ci * oc]);
            }
            out[p + pixels * oc] = (float) acc + bias[oc];
        }
    }
    free(rounded);
}

static void ref_vae_im2col3x3(float *dst, const float *src, uint32_t ci,
                              uint32_t width, uint32_t height) {
    const uint32_t pixels = width * height;
    const uint32_t in_dim = 9u * ci;
    for (uint32_t p = 0; p < pixels; p++) {
        for (uint32_t j = 0; j < in_dim; j++) {
            const uint32_t ic = j / 9u, tap = j % 9u;
            const int ih = (int)(p / width) + (int)(tap / 3u) - 1;
            const int iw = (int)(p % width) + (int)(tap % 3u) - 1;
            float value = 0.0f;
            if (ih >= 0 && ih < (int) height && iw >= 0 && iw < (int) width) {
                value = src[(uint32_t)(iw + width * ih) + pixels * ic];
            }
            dst[j + in_dim * p] = value;
        }
    }
}

/* vae.rs::rms_norm: sum of x^2, scale 1/sqrt(mean + 1e-12), x*scale*gamma. */
static void ref_vae_rms_norm(float *out, const float *x, const float *gamma,
                             uint32_t channels, uint32_t pixels) {
    for (uint32_t p = 0; p < pixels; p++) {
        double sum = 0.0;
        for (uint32_t c = 0; c < channels; c++) {
            const double v = x[p + pixels * c];
            sum += v * v;
        }
        const float mean = (float)(sum / (double) channels);
        const float scale = 1.0f / sqrtf(mean + 1e-12f);
        for (uint32_t c = 0; c < channels; c++) {
            out[p + pixels * c] = x[p + pixels * c] * scale * gamma[c];
        }
    }
}

static void ref_vae_nearest_up2(float *dst, const float *src, uint32_t channels,
                                uint32_t width, uint32_t height) {
    const uint32_t out_width = 2u * width, out_pixels = 4u * width * height;
    for (uint32_t c = 0; c < channels; c++) {
        for (uint32_t oy = 0; oy < 2u * height; oy++) {
            for (uint32_t ox = 0; ox < out_width; ox++) {
                const uint32_t at = (ox / 2u + width * (oy / 2u)) + width * height * c;
                dst[ox + out_width * oy + out_pixels * c] = src[at];
            }
        }
    }
}

/* vae.rs::dup_up3d's closed form, which the oracle tests against the literal
 * ggml concat/reshape/permute chain. */
static void ref_vae_dup_up3d(float *dst, const float *src, uint32_t cin,
                             uint32_t cout, uint32_t width, uint32_t height,
                             uint32_t factor_t) {
    const uint32_t fs = 2u, factor = fs * fs * factor_t;
    const uint32_t repeats = cout * factor / cin;
    const uint32_t out_width = fs * width, out_pixels = fs * fs * width * height;
    for (uint32_t oc = 0; oc < cout; oc++) {
        for (uint32_t y = 0; y < fs * height; y++) {
            for (uint32_t x = 0; x < out_width; x++) {
                const uint32_t m = fs * fs * (factor_t - 1u + factor_t * oc)
                                 + fs * (y % fs) + (x % fs);
                const uint32_t c_in = m / repeats;
                dst[x + out_width * y + out_pixels * oc] =
                    src[(x / fs + width * (y / fs)) + width * height * c_in];
            }
        }
    }
}

/* vae.rs::attention over one head: q/k/v are the channel thirds of the
 * feature-fastest [3*channels, tokens] fixture; softmax over all tokens.  The
 * accumulations are F32 in the oracle's own order (one sequential dot per key,
 * one sequential PV sum per feature), which the kernel mirrors, so the only
 * residual difference is the block max/denominator reduction order. */
static void ref_vae_attn(float *out, const float *qkv, uint32_t channels,
                         uint32_t tokens) {
    float *scores = host_alloc(tokens);
    const float scale = 1.0f / sqrtf((float) channels);
    for (uint32_t qt = 0; qt < tokens; qt++) {
        const float *q = qkv + (size_t) 3u * channels * qt;
        for (uint32_t kt = 0; kt < tokens; kt++) {
            const float *k = qkv + (size_t) 3u * channels * kt + channels;
            float dot = 0.0f;
            for (uint32_t f = 0; f < channels; f++) {
                dot += q[f] * k[f];
            }
            scores[kt] = dot * scale;
        }
        float max = -INFINITY;
        for (uint32_t kt = 0; kt < tokens; kt++) { max = fmaxf(max, scores[kt]); }
        float sum = 0.0f;
        for (uint32_t kt = 0; kt < tokens; kt++) {
            scores[kt] = expf(scores[kt] - max);
            sum += scores[kt];
        }
        for (uint32_t kt = 0; kt < tokens; kt++) { scores[kt] /= sum; }
        for (uint32_t f = 0; f < channels; f++) {
            float acc = 0.0f;
            for (uint32_t kt = 0; kt < tokens; kt++) {
                acc += scores[kt] *
                       qkv[(size_t)(2u * channels + f) + (size_t) 3u * channels * kt];
            }
            out[(size_t) channels * qt + f] = acc;
        }
    }
    free(scores);
}

/* Packs `w` as F16 into a map at a 16-byte offset; the map's own words are the
 * mirror's weights, so map and GEMM see exactly the same values. */
static uint64_t map_put_f16(void *map, uint64_t *cursor, const float *w,
                            size_t count) {
    const uint64_t off = (*cursor + 15u) & ~15ull;
    uint16_t *dst = (uint16_t *)((char *) map + off);
    for (size_t i = 0; i < count; i++) { dst[i] = pack_f16(w[i]); }
    *cursor = off + count * 2u;
    return off;
}

static void *vae_map_alloc(uint64_t bytes) {
    void *map = NULL;
    if (posix_memalign(&map, 4096, bytes)) {
        fprintf(stderr, "FAIL: VAE weight map allocation\n");
        exit(1);
    }
    memset(map, 0, bytes);
    return map;
}

/* One registered map serves every weight-based case: re-registering (and
 * freeing) a fresh host map per case trips CUDA's host-registration owner, and
 * the GEMM only ever reads a slice of it. */
#define VAE_MAP_BYTES (64ull << 20)
static void *g_vae_map = NULL;
static const uint64_t g_vae_map_bytes = VAE_MAP_BYTES;

static void vae_map_init(void) {
    g_vae_map = vae_map_alloc(g_vae_map_bytes);
    if (!ds4_gpu_set_model_map(g_vae_map, g_vae_map_bytes)) {
        fprintf(stderr, "FAIL: VAE weight map registration\n");
        exit(1);
    }
}

/* conv3x3 / conv1x1 as the caller composes them: patch permutation, the tree's
 * F16 GEMM, the per-channel bias add, then the inverse permutation. */
static void test_vae_conv(uint32_t cin, uint32_t cout, uint32_t width,
                          uint32_t height, int taps3) {
    const uint32_t pixels = width * height;
    if (pixels <= 8u) {
        fprintf(stderr, "FAIL: conv case below the tree's native F16 token path\n");
        exit(1);
    }
    const uint32_t in_dim = (taps3 ? 9u : 1u) * cin;
    const size_t weight_count = (size_t) in_dim * cout;
    const size_t out_count = (size_t) pixels * cout;

    float *x = host_alloc((size_t) pixels * cin);
    float *w = host_alloc(weight_count);
    float *bias = host_alloc(cout);
    float *want = host_alloc(out_count);
    float *got = host_alloc(out_count);
    fill_signed(x, (size_t) pixels * cin, 1.0f);
    fill_signed(w, weight_count, 0.3f);
    fill_signed(bias, cout, 0.5f);

    uint64_t cursor = 64;
    const uint64_t woff = map_put_f16(g_vae_map, &cursor, w, weight_count);
    const uint16_t *w16 = (const uint16_t *)((const char *) g_vae_map + woff);

    if (taps3) {
        ref_vae_conv3x3(want, x, w16, bias, cin, cout, width, height);
    } else {
        ref_vae_conv1x1(want, x, w16, bias, cin, cout, pixels);
    }

    ds4_gpu_tensor *tx = device_alloc((size_t) pixels * cin);
    ds4_gpu_tensor *tpatch = device_alloc((size_t) pixels * in_dim);
    ds4_gpu_tensor *tgemm = device_alloc(out_count);
    ds4_gpu_tensor *tbias = device_alloc(cout);
    ds4_gpu_tensor *tout = device_alloc(out_count);
    device_write(tx, x, (size_t) pixels * cin);
    device_write(tbias, bias, cout);

    const int patched = taps3
        ? ds4_gpu_qwen_image_im2col3x3_tensor(tpatch, tx, cin, width, height)
        : ds4_gpu_qwen_image_patch_1x1_tensor(tpatch, tx, cin, pixels);
    if (!patched ||
        !ds4_gpu_matmul_f16_tensor(tgemm, g_vae_map, g_vae_map_bytes, woff,
                                   in_dim, cout, tpatch, pixels) ||
        !ds4_gpu_qwen_image_bias_add_tensor(tgemm, tbias, cout, pixels) ||
        !ds4_gpu_qwen_image_unpatch_crop_tensor(tout, tgemm, cout, pixels)) {
        fprintf(stderr, "FAIL: VAE conv%s launch refused\n", taps3 ? "3x3" : "1x1");
        exit(1);
    }
    device_read(tout, got, out_count);

    char label[80];
    snprintf(label, sizeof(label), "vae conv%s %ux%u cin=%u cout=%u",
             taps3 ? "3x3" : "1x1", width, height, cin, cout);
    compare_tol(label, got, want, out_count, kF16RelRmsTol, kF16MaxAbsTol);

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tpatch);
    ds4_gpu_tensor_free(tgemm);
    ds4_gpu_tensor_free(tbias);
    ds4_gpu_tensor_free(tout);
    free(x);
    free(w);
    free(bias);
    free(want);
    free(got);
}

/* The im2col on its own, so a border-zero error cannot hide behind the GEMM. */
static void test_vae_im2col(uint32_t ci, uint32_t width, uint32_t height) {
    const uint32_t pixels = width * height, in_dim = 9u * ci;
    const size_t count = (size_t) in_dim * pixels;
    float *x = host_alloc((size_t) pixels * ci);
    float *want = host_alloc(count);
    float *got = host_alloc(count);
    fill_signed(x, (size_t) pixels * ci, 1.0f);
    ref_vae_im2col3x3(want, x, ci, width, height);

    ds4_gpu_tensor *tx = device_alloc((size_t) pixels * ci);
    ds4_gpu_tensor *tpatches = device_alloc(count);
    device_write(tx, x, (size_t) pixels * ci);
    if (!ds4_gpu_qwen_image_im2col3x3_tensor(tpatches, tx, ci, width, height)) {
        fprintf(stderr, "FAIL: VAE im2col launch refused\n");
        exit(1);
    }
    device_read(tpatches, got, count);

    char label[80];
    snprintf(label, sizeof(label), "vae im2col3x3 %ux%u ci=%u", width, height, ci);
    compare(label, got, want, count);

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tpatches);
    free(x);
    free(want);
    free(got);
}

static void test_vae_rms_norm(uint32_t channels, uint32_t pixels) {
    const size_t count = (size_t) channels * pixels;
    float *x = host_alloc(count);
    float *gamma = host_alloc(channels);
    float *want = host_alloc(count);
    float *got = host_alloc(count);
    fill_signed(x, count, 1.0f);
    fill_signed(gamma, channels, 1.0f);

    /* Pixel 0 is tiny: its mean square is near eps, so a wrong eps moves its
     * scale (1e-12 vs 1e-6 changes it by sqrt(2) here).  Pixel 0, channel c is
     * at index pixels*c in the channel-slowest plane. */
    for (uint32_t c = 0; c < channels; c++) {
        x[(size_t) pixels * c] = (c & 1u) ? 1.0e-6f : -1.0e-6f;
    }
    ref_vae_rms_norm(want, x, gamma, channels, pixels);

    ds4_gpu_tensor *tx = device_alloc(count);
    ds4_gpu_tensor *tgamma = device_alloc(channels);
    device_write(tx, x, count);
    device_write(tgamma, gamma, channels);
    if (!ds4_gpu_qwen_image_vae_rms_norm_tensor(tx, tgamma, channels, pixels)) {
        fprintf(stderr, "FAIL: VAE RMSNorm launch refused\n");
        exit(1);
    }
    device_read(tx, got, count);

    char label[80];
    snprintf(label, sizeof(label), "vae rms_norm ch=%u pixels=%u", channels, pixels);
    compare_tol(label, got, want, count, 1.0e-5, 1.0e-4);

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgamma);
    free(x);
    free(gamma);
    free(want);
    free(got);
}

static void test_vae_nearest_up2(uint32_t channels, uint32_t width,
                                 uint32_t height) {
    const size_t count = (size_t) width * height * channels;
    const size_t out_count = 4u * count;
    float *x = host_alloc(count);
    float *want = host_alloc(out_count);
    float *got = host_alloc(out_count);
    fill_signed(x, count, 1.0f);
    ref_vae_nearest_up2(want, x, channels, width, height);

    ds4_gpu_tensor *tx = device_alloc(count);
    ds4_gpu_tensor *tout = device_alloc(out_count);
    device_write(tx, x, count);
    if (!ds4_gpu_qwen_image_nearest_up2_tensor(tout, tx, channels, width, height)) {
        fprintf(stderr, "FAIL: VAE nearest up2 launch refused\n");
        exit(1);
    }
    device_read(tout, got, out_count);

    char label[80];
    snprintf(label, sizeof(label), "vae nearest_up2 %ux%u ch=%u", width, height, channels);
    compare(label, got, want, out_count);

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    free(x);
    free(want);
    free(got);
}

static void test_vae_dup_up3d(uint32_t cin, uint32_t cout, uint32_t width,
                              uint32_t height, uint32_t factor_t) {
    const size_t count = (size_t) width * height * cin;
    const size_t out_count = 4u * (size_t) width * height * cout;
    float *x = host_alloc(count);
    float *want = host_alloc(out_count);
    float *got = host_alloc(out_count);
    fill_signed(x, count, 1.0f);
    ref_vae_dup_up3d(want, x, cin, cout, width, height, factor_t);

    ds4_gpu_tensor *tx = device_alloc(count);
    ds4_gpu_tensor *tout = device_alloc(out_count);
    device_write(tx, x, count);
    if (!ds4_gpu_qwen_image_dup_up3d_tensor(tout, tx, cin, cout, width, height,
                                            factor_t)) {
        fprintf(stderr, "FAIL: VAE DupUp3D launch refused\n");
        exit(1);
    }
    device_read(tout, got, out_count);

    char label[96];
    snprintf(label, sizeof(label), "vae dup_up3d %ux%u in=%u out=%u ft=%u",
             width, height, cin, cout, factor_t);
    compare(label, got, want, out_count);

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    free(x);
    free(want);
    free(got);
}

static void test_vae_attn(uint32_t channels, uint32_t tokens) {
    const size_t count = (size_t) 3u * channels * tokens;
    float *qkv = host_alloc(count);
    float *want = host_alloc((size_t) channels * tokens);
    float *got = host_alloc((size_t) channels * tokens);
    fill_signed(qkv, count, 1.0f);
    ref_vae_attn(want, qkv, channels, tokens);

    ds4_gpu_tensor *tqkv = device_alloc(count);
    ds4_gpu_tensor *tout = device_alloc((size_t) channels * tokens);
    device_write(tqkv, qkv, count);
    if (!ds4_gpu_qwen_image_vae_attn_tensor(tout, tqkv, channels, tokens)) {
        fprintf(stderr, "FAIL: VAE attention launch refused\n");
        exit(1);
    }
    device_read(tout, got, (size_t) channels * tokens);

    char label[80];
    snprintf(label, sizeof(label), "vae attn ch=%u tokens=%u", channels, tokens);
    compare_tol(label, got, want, (size_t) channels * tokens, 1.0e-5, 1.0e-4);

    ds4_gpu_tensor_free(tqkv);
    ds4_gpu_tensor_free(tout);
    free(qkv);
    free(want);
    free(got);
}

/* The whole AttentionBlock: RMSNorm, to_qkv (1x1), the attention, proj (1x1)
 * and the residual add, at the decoder's real width.  The last add runs on the
 * host because the block's own residual is a plain F32 sum. */
static void test_vae_attention_block(uint32_t channels, uint32_t width,
                                     uint32_t height) {
    const uint32_t pixels = width * height;
    const size_t plane = (size_t) pixels * channels;
    const size_t qkv_count = 3u * plane;

    float *x = host_alloc(plane);
    float *gamma = host_alloc(channels);
    float *wqkv = host_alloc((size_t) channels * 3u * channels);
    float *bqkv = host_alloc(3u * channels);
    float *wproj = host_alloc((size_t) channels * channels);
    float *bproj = host_alloc(channels);
    float *normed = host_alloc(plane);
    float *qkv_plane = host_alloc(qkv_count);
    float *qkv_ff = host_alloc(qkv_count);
    float *attn_ff = host_alloc(plane);
    float *attn_plane = host_alloc(plane);
    float *want = host_alloc(plane);
    float *got = host_alloc(plane);
    fill_signed(x, plane, 1.0f);
    fill_signed(gamma, channels, 1.0f);
    fill_signed(wqkv, (size_t) channels * 3u * channels, 0.2f);
    fill_signed(bqkv, 3u * channels, 0.3f);
    fill_signed(wproj, (size_t) channels * channels, 0.2f);
    fill_signed(bproj, channels, 0.3f);

    uint64_t cursor = 64;
    const uint64_t qkv_off = map_put_f16(g_vae_map, &cursor, wqkv, (size_t) channels * 3u * channels);
    const uint64_t proj_off = map_put_f16(g_vae_map, &cursor, wproj, (size_t) channels * channels);
    const uint16_t *wqkv16 = (const uint16_t *)((const char *) g_vae_map + qkv_off);
    const uint16_t *wproj16 = (const uint16_t *)((const char *) g_vae_map + proj_off);

    /* The oracle's own channel order: q, k, v (vae.rs:636-638). */
    ref_vae_rms_norm(normed, x, gamma, channels, pixels);
    ref_vae_conv1x1(qkv_plane, normed, wqkv16, bqkv, channels, 3u * channels, pixels);
    for (uint32_t t = 0; t < pixels; t++) {
        for (uint32_t f = 0; f < 3u * channels; f++) {
            qkv_ff[f + (size_t) 3u * channels * t] = qkv_plane[t + (size_t) pixels * f];
        }
    }
    ref_vae_attn(attn_ff, qkv_ff, channels, pixels);
    for (uint32_t t = 0; t < pixels; t++) {
        for (uint32_t f = 0; f < channels; f++) {
            attn_plane[t + (size_t) pixels * f] = attn_ff[f + (size_t) channels * t];
        }
    }
    ref_vae_conv1x1(want, attn_plane, wproj16, bproj, channels, channels, pixels);
    for (size_t i = 0; i < plane; i++) { want[i] += x[i]; }

    ds4_gpu_tensor *tx = device_alloc(plane);
    ds4_gpu_tensor *tgamma = device_alloc(channels);
    ds4_gpu_tensor *tpatch = device_alloc(plane);
    ds4_gpu_tensor *tqkv = device_alloc(qkv_count);
    ds4_gpu_tensor *tbqkv = device_alloc(3u * channels);
    ds4_gpu_tensor *tattn = device_alloc(plane);
    ds4_gpu_tensor *tproj = device_alloc(plane);
    ds4_gpu_tensor *tbproj = device_alloc(channels);
    ds4_gpu_tensor *tout = device_alloc(plane);
    device_write(tx, x, plane);
    device_write(tgamma, gamma, channels);
    device_write(tbqkv, bqkv, 3u * channels);
    device_write(tbproj, bproj, channels);

    const int launched =
        ds4_gpu_qwen_image_vae_rms_norm_tensor(tx, tgamma, channels, pixels) &&
        ds4_gpu_qwen_image_patch_1x1_tensor(tpatch, tx, channels, pixels) &&
        ds4_gpu_matmul_f16_tensor(tqkv, g_vae_map, g_vae_map_bytes, qkv_off,
                                  channels, 3u * channels, tpatch, pixels) &&
        ds4_gpu_qwen_image_bias_add_tensor(tqkv, tbqkv, 3u * channels, pixels) &&
        ds4_gpu_qwen_image_vae_attn_tensor(tattn, tqkv, channels, pixels) &&
        ds4_gpu_matmul_f16_tensor(tproj, g_vae_map, g_vae_map_bytes, proj_off,
                                  channels, channels, tattn, pixels) &&
        ds4_gpu_qwen_image_bias_add_tensor(tproj, tbproj, channels, pixels) &&
        ds4_gpu_qwen_image_unpatch_crop_tensor(tout, tproj, channels, pixels);
    if (!launched) {
        fprintf(stderr, "FAIL: VAE attention block launch refused\n");
        exit(1);
    }
    device_read(tout, got, plane);
    for (size_t i = 0; i < plane; i++) { got[i] += x[i]; }

    char label[80];
    snprintf(label, sizeof(label), "vae attn block %ux%u ch=%u", width, height, channels);
    /* The proj GEMM rounds its input (the attention output) to F16.  The
     * kernel's fast-math expf and its parallel max/denominator differ from the
     * mirror by ~1e-7, which occasionally crosses an F16 rounding boundary: a
     * single flipped operand of the 1152-term proj dot moves the result by one
     * F16 ULP times a weight.  rel_rms stays inside the operand contract; the
     * max_abs bound carries that F16 boundary effect. */
    compare_tol(label, got, want, plane, 2.0e-4, 2.0e-2);

    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tgamma);
    ds4_gpu_tensor_free(tpatch);
    ds4_gpu_tensor_free(tqkv);
    ds4_gpu_tensor_free(tbqkv);
    ds4_gpu_tensor_free(tattn);
    ds4_gpu_tensor_free(tproj);
    ds4_gpu_tensor_free(tbproj);
    ds4_gpu_tensor_free(tout);
    free(x);
    free(gamma);
    free(wqkv);
    free(bqkv);
    free(wproj);
    free(bproj);
    free(normed);
    free(qkv_plane);
    free(qkv_ff);
    free(attn_ff);
    free(attn_plane);
    free(want);
    free(got);
}

/* The two by-name refusals the entries must make. */
static void test_vae_refusals(void) {
    const uint32_t channels = 4, tokens = 9000;
    /* Sized for 16384 so neither call can be refused by an undersized tensor. */
    ds4_gpu_tensor *qkv = device_alloc((size_t) 3u * channels * 16384u);
    ds4_gpu_tensor *out = device_alloc((size_t) channels * 16384u);
    ds4_gpu_tensor *small = device_alloc(4096);
    int bad = 0;

    /* DupUp3D asserts cout * factor % cin == 0; 2 * 4 % 3 != 0. */
    if (ds4_gpu_qwen_image_dup_up3d_tensor(small, small, 3, 2, 2, 2, 1) != 0) { bad++; }
    /* 9000 is over kMaxAttentionPixels (8192) but inside the 48 KiB shared
     * limit, so only the named pixel refusal can stop the launch. */
    if (ds4_gpu_qwen_image_vae_attn_tensor(out, qkv, channels, tokens) != 0) { bad++; }
    /* 16384 is a 128x128 latent, the first real size over the limit. */
    if (ds4_gpu_qwen_image_vae_attn_tensor(out, qkv, channels, 16384) != 0) { bad++; }
    if (bad) { g_failures++; }
    printf("%-46s %s\n", "vae refusals (DupUp3D ratio, 16384-token attn)",
           bad ? "FAIL" : "ok");
    ds4_gpu_tensor_free(qkv);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(small);
}

/* The map's F16 encoder must round-trip to vae_f16_round; otherwise every
 * weight-based comparison would be measuring the encoder, not the kernel. */
static void check_f16_round_trip(void) {
    const float values[] = {0.0f, 1.0f, -2.0f, 0.5f, 2048.0f, -65504.0f,
        6.1035156e-5f, 65504.0f, 65520.0f, -70000.0f, 0.1f, 1.0f + 1.0f/2048.0f,
        1.0f + 3.0f/2048.0f, ldexpf(1.0f, -24), ldexpf(1.0f, -25),
        0.75f * ldexpf(1.0f, -24), ldexpf(1.0f, -30), -ldexpf(1.0f, -24),
        -0.75f * ldexpf(1.0f, -24), -1e-5f, -6e-5f};
    int ok = 1;
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        const float rounded = vae_f16_round(values[i]);
        const float back = unpack_f16(pack_f16(values[i]));
        ok = ok && memcmp(&rounded, &back, sizeof(float)) == 0;
    }
    if (!ok) { g_failures++; }
    printf("%-46s %s\n", "vae f16 encoder round-trip", ok ? "ok" : "FAIL");
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

    printf("== Qwen-Image-2.1 DiT attention path (CUDA vs the oracle's host math) ==\n");
    test_attn_path();

    printf("== Qwen-Image-2.1 DiT glue (CUDA vs the oracle's host math) ==\n");
    test_silu(kHidden, 2);
    void *weights = test_timestep_path();
    test_patch(kChannels, 4096);  /* 64x64 latent, the reference run */
    test_patch(kChannels, 16384); /* 128x128 latent */
    test_patch(kChannels, 27556); /* 166x166 latent: a partial 32x32 tile */

    printf("== Qwen-Image-2.1 VAE decoder primitives (CUDA vs the oracle's host math) ==\n");
    vae_map_init();
    check_f16_round_trip();
    test_vae_refusals();

    test_vae_im2col(3, 5, 3);        /* every border tap, tiny channel count */
    test_vae_im2col(kChannels, 16, 16);
    test_vae_rms_norm(1152, 256);    /* middle width, the P1 16x16 latent grid */
    test_vae_nearest_up2(1152, 16, 16);

    /* The up ladder's DupUp3D ratios (qwen_image.rs::vae_decoder_dims:
     * 1152,1152,1152,576,288,144, and VAE_TEMPORAL_UPSAMPLE), plus an
     * asymmetric ratio the oracle also covers. */
    test_vae_dup_up3d(1152, 1152, 16, 16, 2);
    test_vae_dup_up3d(1152, 576, 16, 16, 2);
    test_vae_dup_up3d(576, 288, 16, 16, 2);
    test_vae_dup_up3d(288, 144, 16, 16, 1);
    test_vae_dup_up3d(6, 3, 3, 2, 1);

    /* The convs as caller-side compositions.  conv1 64->1152, the middle
     * 1152->1152 residuals, a down-ladder 1152->576 and the head's 144->4. */
    test_vae_conv(64, 1152, 16, 16, 1);
    test_vae_conv(1152, 1152, 16, 16, 1);
    test_vae_conv(1152, 576, 16, 16, 1);
    test_vae_conv(144, 4, 16, 16, 1);
    test_vae_conv(64, 1152, 5, 3, 1);      /* border padding */
    test_vae_conv(64, 64, 16, 16, 0);
    test_vae_conv(1152, 3456, 16, 16, 0);  /* to_qkv */
    test_vae_conv(1152, 1152, 16, 16, 0);  /* proj */
    test_vae_conv(1152, 1152, 5, 3, 0);

    test_vae_attn(1152, 256);   /* the middle block's real geometry */
    test_vae_attn(1152, 1024);  /* the shape the task names */
    test_vae_attention_block(1152, 16, 16);

    ds4_gpu_cleanup();
    free(weights);
    free(g_vae_map);
    printf("%s\n", g_failures ? "Qwen-Image checks FAILED"
                              : "all Qwen-Image checks passed");
    return g_failures ? 1 : 0;
}
