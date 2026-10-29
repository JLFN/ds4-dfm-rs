/*
 * The Gemma4 text tower's primitives against host transcriptions of the
 * reference.
 *
 * Reference: HuggingFace `gemma4` at transformers 5.14.1
 * (`/data/refs/transformers-5.14.1/transformers/models/gemma4/
 * modeling_gemma4.py`, the pin in `requirements/stage2-observed.txt:7`) plus
 * the two ATen expressions the tower's numerics rest on
 * (`/data/refs/torch-2.13.0-aten`). There is no Python in this campaign, so
 * every check is a host transcription of the same expression, cited beside it;
 * the external check on the same math is the operator's golden dump of the
 * tower's own hidden states.
 *
 *   tests/h3/gemma_tower_bm [rows]
 *
 * Checks: the embedding gather and scale; the RMS norm with and without a
 * weight; the tanh GELU; the elementwise product and the layer-scalar scale;
 * both RoPE conventions (a 256-wide sliding table and a 512-wide proportional
 * one) plus the rotation; the additive mask (causal, sliding, padded); and
 * attention on both layer shapes.
 *
 * The host oracles reproduce the kernels' own reduction trees, so a check with
 * no transcendental in it is bit-exact and the gate says so; where the
 * arithmetic does call one (`rsqrtf`, `tanhf`, `expf`, `powf`, `cosf`, `sinf`)
 * the host's libm and the device's are not the same implementation, and the
 * check reports how many BF16 values moved instead of pretending otherwise.
 *
 * Build: `make CUDA_ARCH=<arch> tests/h3/gemma_tower_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool passed, const char *name) {
    std::printf("%s %s\n", passed ? "PASS" : "FAIL", name);
    failures += !passed;
}

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

static uint32_t lcg = 987654321u;

static float next_unit(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)((lcg >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f;
}

static float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

static uint16_t bits_of(__nv_bfloat16 value) {
    uint16_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

/* How far a device result is from the host's: the number of elements whose
 * BF16 bits differ, the relative RMS over the whole tensor, and the worst
 * relative deviation. */
struct Diff {
    size_t differ;
    double rel_rms;
    double worst;
};

static Diff compare(const std::vector<__nv_bfloat16> &got,
                    const std::vector<__nv_bfloat16> &want) {
    Diff result{0, 0.0, 0.0};
    double square = 0.0, reference = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (bits_of(got[i]) != bits_of(want[i])) ++result.differ;
        const double g = __bfloat162float(got[i]);
        const double w = __bfloat162float(want[i]);
        square += (g - w) * (g - w);
        reference += w * w;
        const double scale = std::fabs(w) > 1e-12 ? std::fabs(w) : 1e-12;
        result.worst = std::max(result.worst, std::fabs(g - w) / scale);
    }
    result.rel_rms = reference > 0.0 ? std::sqrt(square / reference) : 0.0;
    return result;
}

/* The kernel's warp reduction (`__shfl_down_sync` over 32 lanes): every step
 * reads the pre-step values, so the host snapshots them. Only lane 0 is read
 * afterwards, so lanes whose source is out of range do not matter. */
static float host_warp_sum(float lane[32]) {
    float previous[32];
    for (uint32_t offset = 16; offset; offset >>= 1) {
        std::memcpy(previous, lane, sizeof(previous));
        for (uint32_t l = 0; l < offset; ++l) lane[l] += previous[l + offset];
    }
    return lane[0];
}

/* The kernel's block reduction over 256 partials: the halving tree, each step
 * writing only the lower half, so no snapshot is needed. */
static float host_tree(float partial[256]) {
    for (uint32_t stride = 128; stride; stride >>= 1) {
        for (uint32_t t = 0; t < stride; ++t) partial[t] += partial[t + stride];
    }
    return partial[0];
}

static float host_max_tree(float partial[256]) {
    for (uint32_t stride = 128; stride; stride >>= 1) {
        for (uint32_t t = 0; t < stride; ++t) {
            partial[t] = std::fmax(partial[t], partial[t + stride]);
        }
    }
    return partial[0];
}

/* `Gemma4RMSNorm.forward` (`modeling_gemma4.py:210-214`) with the kernel's own
 * tree: `x * pow(mean(x^2) + eps, -0.5)` in F32, the optional `* weight`, then
 * the cast back. `pow(x, -0.5)` on CUDA is `rsqrtf`
 * (`PowKernel.cu:170-179` -> `UnaryOpsKernel.cu:96` ->
 * `CUDAMathCompat.h:122`). */
static void host_rms(const __nv_bfloat16 *input, uint32_t rows, uint32_t width,
                     const std::vector<__nv_bfloat16> &weight, float epsilon,
                     std::vector<__nv_bfloat16> &output) {
    for (uint32_t row = 0; row < rows; ++row) {
        const __nv_bfloat16 *source = input + (size_t)row * width;
        float partial[256];
        for (uint32_t t = 0; t < 256; ++t) {
            float sum = 0.0f;
            for (uint32_t column = t; column < width; column += 256) {
                const float value = __bfloat162float(source[column]);
                sum = fmaf(value, value, sum);
            }
            partial[t] = sum;
        }
        const float inverse =
            rsqrtf(host_tree(partial) / (float)width + epsilon);
        __nv_bfloat16 *destination = output.data() + (size_t)row * width;
        for (uint32_t column = 0; column < width; ++column) {
            float value = __bfloat162float(source[column]) * inverse;
            if (!weight.empty()) value *= __bfloat162float(weight[column]);
            destination[column] = __float2bfloat16(value);
        }
    }
}

/* `Gemma4TextMLP.forward` (`:1088-1090`) with ATen's tanh gelu
 * (`ActivationGeluKernel.cu:21-31`). */
static float host_gelu(float x) {
    constexpr float beta = 1.41421356237309504880f * 1.12837916709551257390f *
                           0.5f;
    constexpr float kappa = 0.044715f;
    const float cube = x * x * x;
    return 0.5f * x * (1.0f + tanhf(beta * (x + kappa * cube)));
}

/* `apply_rotary_pos_emb` (`:793-811`) with `rotate_half` (`:786-791`): two
 * products and the sum, each rounded to BF16 as torch's elementwise ops do.
 * The row is snapshotted first: `rotate_half` reads the two halves against each
 * other, and torch materialises it before the products, so an in-place oracle
 * would read a half-written row (which is the bug this check caught in the
 * kernel). */
static void host_rope(__nv_bfloat16 *x, const __nv_bfloat16 *cos,
                      const __nv_bfloat16 *sin, uint32_t rows, uint32_t heads,
                      uint32_t head_dim) {
    const uint32_t half = head_dim / 2;
    std::vector<float> staged(head_dim);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            const size_t base = ((size_t)row * heads + head) * head_dim;
            const size_t table = (size_t)row * head_dim;
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                staged[dimension] = __bfloat162float(x[base + dimension]);
            }
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                const uint32_t partner =
                    dimension < half ? dimension + half : dimension - half;
                const float rotated =
                    dimension < half ? -staged[partner] : staged[partner];
                const float turned = bf16_round(
                    staged[dimension] *
                    __bfloat162float(cos[table + dimension]));
                const float spin = bf16_round(
                    rotated * __bfloat162float(sin[table + dimension]));
                x[base + dimension] = __float2bfloat16(turned + spin);
            }
        }
    }
}

/* `eager_mask` (`masking_utils.py:608-610`): keep is `kv <= q` and, with a
 * window, `kv > q - window`, intersected with the padding mask on the key
 * axis; the fill is `finfo(bfloat16).min`. */
static void host_mask(std::vector<__nv_bfloat16> &mask, uint32_t rows,
                      uint32_t window, const std::vector<uint32_t> &padding) {
    for (uint32_t q = 0; q < rows; ++q) {
        for (uint32_t kv = 0; kv < rows; ++kv) {
            bool keep = kv <= q;
            if (keep && window) {
                keep = (int64_t)kv > (int64_t)q - (int64_t)window;
            }
            if (keep && !padding.empty()) keep = padding[kv] != 0;
            mask[(size_t)q * rows + kv] =
                __float2bfloat16(keep ? 0.0f : H3D_GEMMA_MASK_FILL);
        }
    }
}

/* `eager_attention_forward` (`:824-858`) at `scaling = 1.0`, with the kernel's
 * own trees: the warp's strided dot and shuffle sum, the block's max and sum
 * reductions, and one BF16 rounding per product. */
static void host_attention(const std::vector<__nv_bfloat16> &query,
                           const std::vector<__nv_bfloat16> &key,
                           const std::vector<__nv_bfloat16> &value,
                           const std::vector<__nv_bfloat16> &mask,
                           uint32_t rows, uint32_t heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale,
                           std::vector<__nv_bfloat16> &output) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            const uint32_t source = head / (heads / kv_heads);
            const __nv_bfloat16 *q_row =
                query.data() + ((size_t)row * heads + head) * head_dim;
            float qv[H3D_GEMMA_HEAD_DIM_MAX];
            for (uint32_t i = 0; i < head_dim; ++i) {
                qv[i] = __bfloat162float(q_row[i]);
            }

            std::vector<float> score(rows);
            for (uint32_t k = 0; k < rows; ++k) {
                const __nv_bfloat16 *k_row =
                    key.data() + ((size_t)k * kv_heads + source) * head_dim;
                float lane[32] = {};
                for (uint32_t l = 0; l < 32; ++l) {
                    for (uint32_t i = l; i < head_dim; i += 32) {
                        lane[l] = fmaf(qv[i], __bfloat162float(k_row[i]),
                                       lane[l]);
                    }
                }
                float weighted = bf16_round(host_warp_sum(lane) * scale);
                if (!mask.empty()) {
                    weighted = bf16_round(
                        weighted +
                        __bfloat162float(mask[(size_t)row * rows + k]));
                }
                score[k] = weighted;
            }

            float partial[256];
            for (uint32_t t = 0; t < 256; ++t) {
                float maximum = -INFINITY;
                for (uint32_t k = t; k < rows; k += 256) {
                    maximum = std::fmax(maximum, score[k]);
                }
                partial[t] = maximum;
            }
            const float maximum = host_max_tree(partial);
            for (uint32_t t = 0; t < 256; ++t) {
                float total = 0.0f;
                for (uint32_t k = t; k < rows; k += 256) {
                    const float weight = expf(score[k] - maximum);
                    score[k] = weight;
                    total += weight;
                }
                partial[t] = total;
            }
            const float inverse = 1.0f / host_tree(partial);

            const uint32_t per_lane = head_dim / H3D_GEMMA_THREADS;
            float accumulator[2] = {0.0f, 0.0f};
            __nv_bfloat16 *out_row =
                output.data() + ((size_t)row * heads + head) * head_dim;
            for (uint32_t warp = 0; warp < H3D_GEMMA_WARPS; ++warp) {
                for (uint32_t l = 0; l < 32; ++l) {
                    const uint32_t begin =
                        warp * (head_dim / H3D_GEMMA_WARPS) + l * per_lane;
                    accumulator[0] = 0.0f;
                    accumulator[1] = 0.0f;
                    for (uint32_t k = 0; k < rows; ++k) {
                        const float weight = bf16_round(score[k] * inverse);
                        const __nv_bfloat16 *v_row =
                            value.data() +
                            ((size_t)k * kv_heads + source) * head_dim;
                        for (uint32_t e = 0; e < per_lane; ++e) {
                            accumulator[e] = fmaf(
                                weight, __bfloat162float(v_row[begin + e]),
                                accumulator[e]);
                        }
                    }
                    for (uint32_t e = 0; e < per_lane; ++e) {
                        out_row[begin + e] =
                            __float2bfloat16(accumulator[e]);
                    }
                }
            }
        }
    }
}

static void report(const char *name, const Diff &diff, double tolerance) {
    check(diff.rel_rms <= tolerance, name);
    std::printf("  %zu differ, rel RMS %.3g, worst %.3g\n", diff.differ,
                diff.rel_rms, diff.worst);
}

/* For the checks whose arithmetic has no transcendental in it: nothing may
 * move, and a difference is a real difference rather than two libms. */
static void report_exact(const char *name, const Diff &diff,
                         size_t elements) {
    check(diff.differ == 0, name);
    std::printf("  %zu of %zu differ, rel RMS %.3g\n", diff.differ, elements,
                diff.rel_rms);
}

/* Where a head-major attention difference sits, for the first few of them. */
static void dump_attention(const std::vector<__nv_bfloat16> &got,
                           const std::vector<__nv_bfloat16> &want,
                           uint32_t heads, uint32_t dim) {
    uint32_t shown = 0;
    for (size_t i = 0; i < want.size() && shown < 6; ++i) {
        if (bits_of(got[i]) == bits_of(want[i])) continue;
        const size_t inner = (size_t)heads * dim;
        std::printf("  [%zu] row %zu head %zu dim %zu: got %.6g want %.6g\n", i,
                    i / inner, (i % inner) / dim, (i % inner) % dim,
                    __bfloat162float(got[i]), __bfloat162float(want[i]));
        ++shown;
    }
}

int main(int argc, char **argv) {
    char error[512] = {};
    setvbuf(stdout, NULL, _IONBF, 0);
    const uint32_t rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 128;
    const uint32_t heads = 16;

    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);
    std::printf("rows %u, heads %u\n", rows, heads);

    /* ---- the embedding: a gather and one BF16 product ---- */
    {
        const uint32_t vocabulary = 1024, width = 3840;
        std::vector<__nv_bfloat16> table((size_t)vocabulary * width);
        for (__nv_bfloat16 &value : table) {
            value = __float2bfloat16(next_unit());
        }
        std::vector<uint32_t> ids(rows);
        for (uint32_t &id : ids) id = (uint32_t)((next_unit() + 1.0f) * 511.5f);
        /* `embed_scale` is `hidden_size ** 0.5` cast to BF16 (`:1477`). */
        const float scale = bf16_round(std::sqrt(3840.0f));

        std::vector<__nv_bfloat16> want((size_t)rows * width);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t column = 0; column < width; ++column) {
                want[(size_t)row * width + column] = __float2bfloat16(
                    __bfloat162float(table[(size_t)ids[row] * width + column]) *
                    scale);
            }
        }

        h3d_tensor *device_table =
            h3d_alloc(gpu, H3D_BF16, table.size(), error, sizeof(error));
        h3d_tensor *device_ids =
            h3d_alloc(gpu, H3D_U32, rows, error, sizeof(error));
        h3d_tensor *device_out =
            h3d_alloc(gpu, H3D_BF16, want.size(), error, sizeof(error));
        if (!device_table || !device_ids || !device_out ||
            !h3d_upload(gpu, device_table, table.data(), error,
                        sizeof(error)) ||
            !h3d_upload(gpu, device_ids, ids.data(), error, sizeof(error)) ||
            !h3d_embedding(gpu, device_out, device_table, device_ids, rows,
                           width, scale, error, sizeof(error))) {
            fail("embedding", error);
        }
        std::vector<__nv_bfloat16> got(want.size());
        if (!h3d_read(gpu, got.data(), device_out, error, sizeof(error))) {
            fail("read embedding", error);
        }
        report_exact("embedding matches the gather and BF16 scale",
                     compare(got, want), want.size());
        h3d_free(device_out);
        h3d_free(device_ids);
        h3d_free(device_table);
    }

    /* ---- the RMS norm, with and without the weight ---- */
    {
        const uint32_t width = 3840;
        std::vector<__nv_bfloat16> input((size_t)rows * width);
        for (__nv_bfloat16 &value : input) {
            value = __float2bfloat16(next_unit() * 2.0f);
        }
        std::vector<__nv_bfloat16> weight(width);
        for (__nv_bfloat16 &value : weight) {
            value = __float2bfloat16(next_unit() + 1.5f);
        }
        h3d_tensor *device_input =
            h3d_alloc(gpu, H3D_BF16, input.size(), error, sizeof(error));
        h3d_tensor *device_weight =
            h3d_alloc(gpu, H3D_BF16, width, error, sizeof(error));
        h3d_tensor *device_out =
            h3d_alloc(gpu, H3D_BF16, input.size(), error, sizeof(error));
        if (!device_input || !device_weight || !device_out ||
            !h3d_upload(gpu, device_input, input.data(), error,
                        sizeof(error)) ||
            !h3d_upload(gpu, device_weight, weight.data(), error,
                        sizeof(error))) {
            fail("rms upload", error);
        }

        std::vector<__nv_bfloat16> want(input.size()), got(input.size());
        host_rms(input.data(), rows, width, weight, 1e-6f, want);
        if (!h3d_rms_norm(gpu, device_out, device_input, device_weight, rows,
                          width, 1e-6f, error, sizeof(error)) ||
            !h3d_read(gpu, got.data(), device_out, error, sizeof(error))) {
            fail("rms weight", error);
        }
        report("RMS norm with a weight matches the F32 reference",
               compare(got, want), 1e-3);

        host_rms(input.data(), rows, width, std::vector<__nv_bfloat16>(), 1e-6f,
                 want);
        if (!h3d_rms_norm(gpu, device_out, device_input, NULL, rows, width,
                          1e-6f, error, sizeof(error)) ||
            !h3d_read(gpu, got.data(), device_out, error, sizeof(error))) {
            fail("rms weightless", error);
        }
        report("RMS norm without a weight matches (Gemma's v_norm)",
               compare(got, want), 1e-3);
        h3d_free(device_out);
        h3d_free(device_weight);
        h3d_free(device_input);
    }

    /* ---- the tanh GELU, the product and the layer scalar ---- */
    {
        const size_t elements = (size_t)rows * 15360;
        std::vector<__nv_bfloat16> left(elements), right(elements);
        for (size_t i = 0; i < elements; ++i) {
            left[i] = __float2bfloat16(next_unit() * 4.0f);
            right[i] = __float2bfloat16(next_unit() * 2.0f);
        }
        const __nv_bfloat16 scalar = __float2bfloat16(1.0f);
        h3d_tensor *device_left =
            h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
        h3d_tensor *device_right =
            h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
        h3d_tensor *device_scalar =
            h3d_alloc(gpu, H3D_BF16, 1, error, sizeof(error));
        h3d_tensor *device_out =
            h3d_alloc(gpu, H3D_BF16, elements, error, sizeof(error));
        if (!device_left || !device_right || !device_scalar || !device_out ||
            !h3d_upload(gpu, device_left, left.data(), error, sizeof(error)) ||
            !h3d_upload(gpu, device_right, right.data(), error,
                        sizeof(error)) ||
            !h3d_upload(gpu, device_scalar, &scalar, error, sizeof(error))) {
            fail("elementwise upload", error);
        }

        std::vector<__nv_bfloat16> want(elements), got(elements);
        for (size_t i = 0; i < elements; ++i) {
            want[i] = __float2bfloat16(host_gelu(__bfloat162float(left[i])));
        }
        if (!h3d_gelu_tanh(gpu, device_out, device_left, elements, error,
                           sizeof(error)) ||
            !h3d_read(gpu, got.data(), device_out, error, sizeof(error))) {
            fail("gelu", error);
        }
        report("tanh GELU matches the ATen expression", compare(got, want),
               1e-3);

        for (size_t i = 0; i < elements; ++i) {
            want[i] = __float2bfloat16(__bfloat162float(left[i]) *
                                       __bfloat162float(right[i]));
        }
        if (!h3d_mul(gpu, device_out, device_left, device_right, elements,
                     error, sizeof(error)) ||
            !h3d_read(gpu, got.data(), device_out, error, sizeof(error))) {
            fail("mul", error);
        }
        report_exact("the MLP product matches", compare(got, want),
                     want.size());

        for (size_t i = 0; i < elements; ++i) {
            want[i] = __float2bfloat16(__bfloat162float(left[i]) *
                                       __bfloat162float(scalar));
        }
        if (!h3d_scale_bf16(gpu, device_out, device_left, device_scalar,
                            elements, error, sizeof(error)) ||
            !h3d_read(gpu, got.data(), device_out, error, sizeof(error))) {
            fail("scale", error);
        }
        report_exact("the layer scalar matches", compare(got, want),
                     want.size());
        h3d_free(device_out);
        h3d_free(device_scalar);
        h3d_free(device_right);
        h3d_free(device_left);
    }

    /* ---- both RoPE conventions, table and rotation ---- */
    struct RopeShape {
        const char *name;
        uint32_t dim, angles, kv_heads;
        float base;
    };
    const RopeShape rope_shapes[2] = {
        {"sliding (dim 256, default, theta 1e4)", 256, 128, 8, 1e4f},
        {"full (dim 512, proportional, theta 1e6)", 512, 64, 1, 1e6f},
    };
    for (const RopeShape &shape : rope_shapes) {
        const uint32_t dim = shape.dim, kv_heads = shape.kv_heads;
        std::vector<__nv_bfloat16> cos((size_t)rows * dim);
        std::vector<__nv_bfloat16> sin((size_t)rows * dim);
        std::vector<__nv_bfloat16> want_cos((size_t)rows * dim),
            want_sin((size_t)rows * dim);
        /* `inv_freq` per `:1157-1161` / `modeling_rope_utils.py:236-250`. */
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t d = 0; d < dim; ++d) {
                const uint32_t angle = d < dim / 2 ? d : d - dim / 2;
                const float frequency =
                    angle < shape.angles
                        ? 1.0f / powf(shape.base,
                                      2.0f * (float)angle / (float)dim)
                        : 0.0f;
                const float value = frequency * (float)row;
                want_cos[(size_t)row * dim + d] = __float2bfloat16(cosf(value));
                want_sin[(size_t)row * dim + d] = __float2bfloat16(sinf(value));
            }
        }

        h3d_tensor *device_cos =
            h3d_alloc(gpu, H3D_BF16, cos.size(), error, sizeof(error));
        h3d_tensor *device_sin =
            h3d_alloc(gpu, H3D_BF16, sin.size(), error, sizeof(error));
        if (!device_cos || !device_sin ||
            !h3d_gemma_rope_table(gpu, device_cos, device_sin, rows, dim,
                                  shape.angles, shape.base, error,
                                  sizeof(error)) ||
            !h3d_read(gpu, cos.data(), device_cos, error, sizeof(error)) ||
            !h3d_read(gpu, sin.data(), device_sin, error, sizeof(error))) {
            fail("rope table", error);
        }
        char name[128];
        std::snprintf(name, sizeof(name), "%s: cos table", shape.name);
        report(name, compare(cos, want_cos), 1e-3);
        std::snprintf(name, sizeof(name), "%s: sin table", shape.name);
        report(name, compare(sin, want_sin), 1e-3);

        std::vector<__nv_bfloat16> query((size_t)rows * heads * dim),
            key((size_t)rows * kv_heads * dim);
        for (__nv_bfloat16 &value : query) {
            value = __float2bfloat16(next_unit());
        }
        for (__nv_bfloat16 &value : key) {
            value = __float2bfloat16(next_unit());
        }
        std::vector<__nv_bfloat16> want_query = query, want_key = key;
        host_rope(want_query.data(), cos.data(), sin.data(), rows, heads, dim);
        host_rope(want_key.data(), cos.data(), sin.data(), rows, kv_heads, dim);

        h3d_tensor *device_query =
            h3d_alloc(gpu, H3D_BF16, query.size(), error, sizeof(error));
        h3d_tensor *device_key =
            h3d_alloc(gpu, H3D_BF16, key.size(), error, sizeof(error));
        if (!device_query || !device_key ||
            !h3d_upload(gpu, device_query, query.data(), error,
                        sizeof(error)) ||
            !h3d_upload(gpu, device_key, key.data(), error, sizeof(error)) ||
            !h3d_gemma_rope(gpu, device_query, device_key, device_cos,
                            device_sin, rows, heads, kv_heads, dim, error,
                            sizeof(error)) ||
            !h3d_read(gpu, query.data(), device_query, error, sizeof(error)) ||
            !h3d_read(gpu, key.data(), device_key, error, sizeof(error))) {
            fail("rope", error);
        }
        std::snprintf(name, sizeof(name), "%s: rotated query", shape.name);
        report(name, compare(query, want_query), 1e-3);
        std::snprintf(name, sizeof(name), "%s: rotated key", shape.name);
        report(name, compare(key, want_key), 1e-3);

        /* ---- the mask and attention on this layer's shape ---- */
        for (uint32_t window : {0u, 32u}) {
            std::vector<uint32_t> padding(rows, 1);
            for (uint32_t i = 0; i < rows / 8; ++i) padding[i] = 0;
            std::vector<__nv_bfloat16> want_mask((size_t)rows * rows);
            host_mask(want_mask, rows, window, padding);

            h3d_tensor *device_padding =
                h3d_alloc(gpu, H3D_U32, rows, error, sizeof(error));
            h3d_tensor *device_mask =
                h3d_alloc(gpu, H3D_BF16, want_mask.size(), error,
                          sizeof(error));
            std::vector<__nv_bfloat16> got_mask(want_mask.size());
            if (!device_padding || !device_mask ||
                !h3d_upload(gpu, device_padding, padding.data(), error,
                            sizeof(error)) ||
                !h3d_gemma_mask(gpu, device_mask, rows, window, device_padding,
                                error, sizeof(error)) ||
                !h3d_read(gpu, got_mask.data(), device_mask, error,
                          sizeof(error))) {
                fail("mask", error);
            }
            std::snprintf(name, sizeof(name), "%s: mask (window %u, padded)",
                          shape.name, window);
            report_exact(name, compare(got_mask, want_mask), want_mask.size());

            /* `v` is its own tensor: a `k_eq_v` layer takes V from the raw
             * `k_proj` output through the weightless norm, not from the
             * normalised and rotated key, so the check must not conflate them.
             */
            std::vector<__nv_bfloat16> value((size_t)rows * kv_heads * dim);
            for (__nv_bfloat16 &entry : value) {
                entry = __float2bfloat16(next_unit());
            }
            std::vector<__nv_bfloat16> attention((size_t)rows * heads * dim);
            std::vector<__nv_bfloat16> want_attention(attention.size());
            host_attention(query, key, value, got_mask, rows, heads, kv_heads,
                           dim, 1.0f, want_attention);
            h3d_tensor *device_value =
                h3d_alloc(gpu, H3D_BF16, value.size(), error, sizeof(error));
            h3d_tensor *device_attention =
                h3d_alloc(gpu, H3D_BF16, attention.size(), error,
                          sizeof(error));
            if (!device_value || !device_attention ||
                !h3d_upload(gpu, device_value, value.data(), error,
                            sizeof(error)) ||
                !h3d_gemma_attention(gpu, device_attention, device_query,
                                     device_key, device_value, device_mask,
                                     rows, heads, kv_heads, dim, 1.0f, error,
                                     sizeof(error)) ||
                !h3d_read(gpu, attention.data(), device_attention, error,
                          sizeof(error))) {
                fail("attention", error);
            }
            std::snprintf(name, sizeof(name), "%s: attention (window %u)",
                          shape.name, window);
            const Diff attention_diff = compare(attention, want_attention);
            report(name, attention_diff, 1e-3);
            if (attention_diff.differ) {
                dump_attention(attention, want_attention, heads, dim);
            }
            h3d_free(device_attention);
            h3d_free(device_value);
            h3d_free(device_mask);
            h3d_free(device_padding);
        }

        h3d_free(device_key);
        h3d_free(device_query);
        h3d_free(device_sin);
        h3d_free(device_cos);
    }

    h3d_destroy(gpu);
    std::printf("%s\n", failures ? "FAILURES" : "all checks passed");
    return failures ? 1 : 0;
}
