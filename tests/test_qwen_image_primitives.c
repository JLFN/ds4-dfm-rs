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
 *      text prefix causal, the image segment unmasked and bidirectional.
 *
 * Each runs at the real DiT geometry (hidden 4096, intermediate 12288, 128
 * text + 4096 image = 4224 joint tokens, 32 heads x head dim 128) and is
 * diffed against a double host mirror of crates/ds4-core/src/qwen_image
 * (dit.rs and oracle.rs).  The kernels accumulate in F32 and the silu/tanh/
 * exp tails ride fast-math, so the gate is relative RMS <= 1e-5 with max abs
 * <= 1e-4 on O(1) data; the observed numbers printed below sit an order of
 * magnitude inside it.
 *
 * The attention case is fed the oracle's own normalized and roped q/k, so it
 * isolates the segment kernel; the rope case feeds the normalized q/k and
 * diffs the rotation.  The two compare separately because the text segment's
 * causal mask and the image segment's absence of one are different failures.
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

    ds4_gpu_cleanup();
    printf("%s\n", g_failures ? "Qwen-Image checks FAILED"
                              : "all Qwen-Image checks passed");
    return g_failures ? 1 : 0;
}
