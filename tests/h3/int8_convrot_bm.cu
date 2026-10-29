/*
 * The Gemma4 encoder's INT8 convrot route, step by step against a host
 * transcription of the reference.
 *
 * Reference: `comfy-kitchen` 0.2.26 at `/data/refs/comfy-kitchen-v0.2.26`,
 * called by the release through `runtime/cache_ops/int8_policy.py:58-107`.
 * There is no Python in this campaign, so the "eager reference" of the port
 * plan is a host implementation of the same five steps, cited line by line
 * below; the external check on the same math is the portable gate
 * (`tests/h3-int8_convrot_gate`), which reproduces the release's own INT8
 * weights from the BF16 sibling they were quantized from.
 *
 *   tests/h3/int8_convrot_bm [rows] [k] [n] [group]
 *
 * Defaults are the cached prompt's shape and Gemma-4-12B's `q_proj`
 * (`model.layers.0.self_attn.q_proj`, [4096, 3840], group 256).
 *
 * Checks: the Hadamard sign matrix against a host Kronecker build; the
 * rotation; the row scale and the INT8 values; the int32 accumulator against a
 * host 64-bit dot product; the epilogue's two F32 multiplies and its BF16
 * rounding; the chain end to end; and the deviation from the BF16 linear the
 * route replaces. Every step but the rotation is exact, because every step is
 * integer or a single rounding with the reference's own association.
 *
 * Build: `make CUDA_ARCH=<arch> tests/h3/int8_convrot_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
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

static uint32_t lcg = 12345u;

static float next_unit(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)((lcg >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f;
}

static float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

static void for_each_row(uint32_t rows,
                         const std::function<void(uint32_t, uint32_t)> &body) {
    unsigned workers = std::thread::hardware_concurrency();
    if (workers < 1) workers = 1;
    if (workers > 16) workers = 16;
    if ((uint32_t)workers > rows) workers = rows;
    std::vector<std::thread> pool;
    for (unsigned worker = 0; worker < workers; ++worker) {
        pool.emplace_back([&, worker] {
            for (uint32_t row = worker; row < rows; row += workers) {
                body(row, worker);
            }
        });
    }
    for (std::thread &thread : pool) thread.join();
}

/* `_build_hadamard` (`tensor/int8_utils.py:10-38`): `h4`, Kronecker-multiplied
 * until the size, each step appending the new factor on the right. Returned as
 * its signs, which is all the port stores. */
static void host_hadamard(uint32_t size, std::vector<int8_t> &signs) {
    static const int8_t h4[16] = {1, 1,  1, -1, 1, 1, -1, 1,
                                  1, -1, 1, 1,  -1, 1, 1, 1};
    std::vector<int8_t> matrix(h4, h4 + 16);
    uint32_t current = 4;
    while (current < size) {
        std::vector<int8_t> next((size_t)current * 4 * current * 4);
        for (uint32_t i = 0; i < current * 4; ++i) {
            for (uint32_t j = 0; j < current * 4; ++j) {
                next[(size_t)i * current * 4 + j] =
                    (int8_t)(matrix[(size_t)(i / 4) * current + (j / 4)] *
                             h4[(i % 4) * 4 + (j % 4)]);
            }
        }
        matrix.swap(next);
        current *= 4;
    }
    signs.swap(matrix);
}

/* `_rotate_activation` (`:55-70`) in the accumulator's own precision and in
 * the order the kernel walks: `out[j] = recip * sum_i signs[i][j] * in[i]`.
 * `T` is float for the port's arithmetic and double for the exactly rounded
 * answer, whose difference bounds what the reference's own tile order can
 * cost. */
template <typename T>
static void host_rotate(const std::vector<float> &in, uint32_t rows,
                        uint32_t columns, uint32_t group,
                        const std::vector<int8_t> &signs,
                        std::vector<float> &out) {
    const T recip = (T)(1.0f / sqrtf((float)group));
    const uint32_t groups = columns / group;
    for_each_row(rows, [&](uint32_t row, uint32_t) {
        for (uint32_t index = 0; index < groups; ++index) {
            size_t base = (size_t)row * columns + (size_t)index * group;
            for (uint32_t j = 0; j < group; ++j) {
                T sum = 0;
                for (uint32_t i = 0; i < group; ++i) {
                    const int8_t sign = signs[(size_t)i * group + j];
                    sum += sign > 0 ? (T)in[base + i] : -(T)in[base + i];
                }
                out[base + j] = bf16_round((float)(sum * recip));
            }
        }
    });
}

/* `quantize_int8_rowwise` (`:857-871`) with `_int8_scale_for_math` (`:815-817`)
 * and `_round_int8` (`:820-826`): `abs_max` over the BF16 row, `scale =
 * (abs_max.float() / 127).clamp(min=1e-30)`, then the BF16 divisor, the BF16
 * quotient, `round_` (half to even) and the clamp. */
static void host_quantize_row(const float *row, size_t columns, float *scale,
                              int8_t *quantized) {
    float amax = 0.0f;
    for (size_t index = 0; index < columns; ++index) {
        amax = fmaxf(amax, fabsf(row[index]));
    }
    *scale = fmaxf(amax / H3D_INT8_MAX, H3D_INT8_MIN_SCALE);
    float divisor = bf16_round(*scale);
    if (divisor == 0.0f) divisor = bf16_round(H3D_INT8_TINY_BF16);
    for (size_t index = 0; index < columns; ++index) {
        float value = bf16_round(row[index] / divisor);
        value = rintf(value);
        quantized[index] = (int8_t)fminf(fmaxf(value, -128.0f), H3D_INT8_MAX);
    }
}

/* `chunk.float() * (x_scale * weight_scale)` then `.to(bf16)` then
 * `+ bias` in BF16 (`:1035-1047`). */
static float host_epilogue(int32_t accumulator, float row_scale,
                           float weight_scale, const float *bias, size_t at) {
    float value = bf16_round((float)accumulator * (row_scale * weight_scale));
    if (bias) value = bf16_round(value + bias[at]);
    return value;
}

struct Mismatch {
    size_t count = 0;
    double worst = 0.0;
};

static void report(const char *what, const Mismatch &mismatch, size_t total) {
    std::printf("%-28s %zu of %zu differ (worst %.6g)\n", what, mismatch.count,
                total, mismatch.worst);
}

int main(int argc, char **argv) {
    char error[512] = {};
    setvbuf(stdout, NULL, _IONBF, 0);
    const uint32_t rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 1024;
    const uint32_t k = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 3840;
    const uint32_t n = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 4096;
    const uint32_t group = argc > 4 ? (uint32_t)std::atoi(argv[4]) : 256;

    std::printf("rows %u, k %u, n %u, group %u\n", rows, k, n, group);

    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    /* ---- the Hadamard sign matrix ---- */
    for (uint32_t size : {4u, 16u, 64u, 256u}) {
        std::vector<int8_t> signs;
        host_hadamard(size, signs);
        h3d_tensor *words = h3d_alloc(gpu, H3D_U32, (size_t)size * ((size + 31) / 32),
                                      error, sizeof(error));
        if (!words || !h3d_int8_hadamard(gpu, words, size, error, sizeof(error))) {
            fail("hadamard", error);
        }
        std::vector<uint32_t> got((size_t)size * ((size + 31) / 32));
        if (!h3d_read(gpu, got.data(), words, error, sizeof(error))) {
            fail("read hadamard", error);
        }
        size_t wrong = 0;
        const uint32_t per_row = (size + 31) / 32;
        for (uint32_t i = 0; i < size; ++i) {
            for (uint32_t j = 0; j < size; ++j) {
                const uint32_t bit = (got[(size_t)i * per_row + (j >> 5)] >> (j & 31u)) & 1u;
                if ((bit != 0) != (signs[(size_t)i * size + j] > 0)) ++wrong;
            }
        }
        char name[64];
        std::snprintf(name, sizeof(name), "hadamard %u matches kron(h4)", size);
        check(wrong == 0, name);
        if (wrong) std::printf("  %zu sign bits wrong\n", wrong);
        h3d_free(words);
    }

    const uint32_t padded = h3d_int8_pad_rows(rows);

    /* The sign matrix, built once by the reference's own Kronecker rule. */
    std::vector<int8_t> signs;
    host_hadamard(group, signs);

    /* ---- operands ----
     * The activation is a BF16 row block. The weight is what the release's
     * converter would have written: the source weight rotated in groups by the
     * same matrix (`_rotate_weight`, `tensor/int8_utils.py:42-52`, which
     * multiplies by `h.T`; the matrix is symmetric) and then quantized
     * row-wise, exactly as a `*.comfy_quant` tensor carries it. The route's
     * own claim is `(H A) . (H W) = A . W`, so the BF16 linear it replaces is
     * the dequantized rotated weight against the unrotated activation. */
    std::vector<float> host_a((size_t)rows * k);
    std::vector<uint16_t> bytes_a((size_t)rows * k);
    for (size_t index = 0; index < host_a.size(); ++index) {
        /* Occasional large values, as an activation row's outliers are: the
         * row's one absmax decides the whole row's scale. */
        float magnitude = (index % 4096 == 0) ? 6.0f : 1.0f;
        host_a[index] = bf16_round(next_unit() * magnitude);
        __nv_bfloat16 encoded = __float2bfloat16(host_a[index]);
        std::memcpy(&bytes_a[index], &encoded, 2);
    }
    std::vector<float> host_w_src((size_t)n * k), host_w_rot((size_t)n * k);
    for (size_t index = 0; index < host_w_src.size(); ++index) {
        float magnitude = (index % 4096 == 0) ? 4.0f : 1.0f;
        host_w_src[index] = bf16_round(next_unit() * magnitude);
    }
    host_rotate<float>(host_w_src, n, k, group, signs, host_w_rot);
    std::vector<int8_t> host_w((size_t)n * k);
    std::vector<float> host_ws(n);
    std::vector<uint16_t> bytes_bias(n);
    std::vector<float> host_bias(n);
    for (uint32_t j = 0; j < n; ++j) {
        host_quantize_row(&host_w_rot[(size_t)j * k], k, &host_ws[j],
                          &host_w[(size_t)j * k]);
        host_bias[j] = bf16_round(next_unit() * 0.25f);
        __nv_bfloat16 encoded = __float2bfloat16(host_bias[j]);
        std::memcpy(&bytes_bias[j], &encoded, 2);
    }

    h3d_tensor *input = h3d_alloc(gpu, H3D_BF16, host_a.size(), error, sizeof(error));
    h3d_tensor *weight = h3d_alloc(gpu, H3D_I8, host_w.size(), error, sizeof(error));
    h3d_tensor *weight_scale = h3d_alloc(gpu, H3D_F32, n, error, sizeof(error));
    h3d_tensor *bias = h3d_alloc(gpu, H3D_BF16, n, error, sizeof(error));
    h3d_tensor *output = h3d_alloc(gpu, H3D_BF16, (size_t)rows * n, error, sizeof(error));
    h3d_tensor *output_bf16 = h3d_alloc(gpu, H3D_BF16, (size_t)rows * n, error, sizeof(error));
    struct h3d_int8_plan plan = {};
    plan.words = h3d_alloc(gpu, H3D_U32, (size_t)group * ((group + 31) / 32), error, sizeof(error));
    plan.rotated = h3d_alloc(gpu, H3D_BF16, (size_t)rows * k, error, sizeof(error));
    plan.quantized = h3d_alloc(gpu, H3D_I8, (size_t)padded * k, error, sizeof(error));
    plan.row_scales = h3d_alloc(gpu, H3D_F32, padded, error, sizeof(error));
    plan.accumulator = h3d_alloc(gpu, H3D_I32, (size_t)padded * n, error, sizeof(error));
    if (!input || !weight || !weight_scale || !bias || !output || !output_bf16 ||
        !plan.words || !plan.rotated || !plan.quantized || !plan.row_scales ||
        !plan.accumulator) {
        fail("alloc", error);
    }
    if (!h3d_upload(gpu, input, bytes_a.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, weight, host_w.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, weight_scale, host_ws.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, bias, bytes_bias.data(), error, sizeof(error)) ||
        !h3d_int8_hadamard(gpu, plan.words, group, error, sizeof(error))) {
        fail("upload", error);
    }

    /* ---- the chain ---- */
    if (!h3d_linear_int8_convrot(gpu, output, input, weight, weight_scale, bias,
                                 rows, k, n, group, &plan, error, sizeof(error))) {
        fail("int8 convrot", error);
    }

    /* ---- 1. the rotation ---- */
    std::vector<uint16_t> got_rotated((size_t)rows * k);
    if (!h3d_read(gpu, got_rotated.data(), plan.rotated, error, sizeof(error))) {
        fail("read rotated", error);
    }
    std::vector<float> want_f32((size_t)rows * k), want_f64((size_t)rows * k);
    host_rotate<float>(host_a, rows, k, group, signs, want_f32);
    host_rotate<double>(host_a, rows, k, group, signs, want_f64);
    Mismatch rotation, rotation_exact;
    for (size_t index = 0; index < want_f32.size(); ++index) {
        float got = __bfloat162float(*(__nv_bfloat16 *)&got_rotated[index]);
        if (got != want_f32[index]) {
            ++rotation.count;
            rotation.worst = fmax(rotation.worst, fabs((double)got - want_f32[index]));
        }
        if (got != want_f64[index]) {
            ++rotation_exact.count;
            rotation_exact.worst =
                fmax(rotation_exact.worst, fabs((double)got - want_f64[index]));
        }
    }
    report("rotation vs host F32 order", rotation, want_f32.size());
    report("rotation vs exact (double)", rotation_exact, want_f64.size());

    /* ---- 2. the row scales ---- */
    std::vector<float> got_scales(padded);
    std::vector<int8_t> got_quantized((size_t)padded * k);
    if (!h3d_read(gpu, got_scales.data(), plan.row_scales, error, sizeof(error)) ||
        !h3d_read(gpu, got_quantized.data(), plan.quantized, error, sizeof(error))) {
        fail("read quantized", error);
    }
    Mismatch scales, values;
    std::vector<float> want_scales(padded);
    std::vector<int8_t> want_quantized((size_t)padded * k);
    for_each_row(rows, [&](uint32_t row, uint32_t) {
        std::vector<float> line(k);
        for (uint32_t i = 0; i < k; ++i) {
            line[i] = __bfloat162float(*(__nv_bfloat16 *)&got_rotated[(size_t)row * k + i]);
        }
        host_quantize_row(line.data(), k, &want_scales[row],
                          &want_quantized[(size_t)row * k]);
    });
    for (uint32_t row = 0; row < rows; ++row) {
        if (got_scales[row] != want_scales[row]) {
            ++scales.count;
            scales.worst = fmax(scales.worst, fabs((double)got_scales[row] -
                                                   (double)want_scales[row]) /
                                                  (double)want_scales[row]);
        }
        for (uint32_t i = 0; i < k; ++i) {
            size_t at = (size_t)row * k + i;
            if (got_quantized[at] != want_quantized[at]) {
                ++values.count;
                values.worst = fmax(values.worst,
                                    fabs((double)got_quantized[at] - want_quantized[at]));
            }
        }
    }
    report("row scales", scales, rows);
    report("int8 values", values, (size_t)rows * k);

    /* ---- 3. the int32 accumulator ---- */
    std::vector<int32_t> got_accumulator((size_t)padded * n);
    if (!h3d_read(gpu, got_accumulator.data(), plan.accumulator, error,
                  sizeof(error))) {
        fail("read accumulator", error);
    }
    /* The host dot is 64-bit and the device's is 32-bit, so the comparison is
     * the accumulator's own contract: |dot| < 2^31 for these operands, which
     * the count of saturated rows would expose. */
    std::vector<double> dots((size_t)rows * n, 0.0);
    for_each_row(rows, [&](uint32_t row, uint32_t) {
        for (uint32_t j = 0; j < n; ++j) {
            int64_t sum = 0;
            for (uint32_t i = 0; i < k; ++i) {
                sum += (int64_t)got_quantized[(size_t)row * k + i] *
                       (int64_t)host_w[(size_t)j * k + i];
            }
            dots[(size_t)row * n + j] = (double)sum;
        }
    });
    Mismatch accumulate;
    size_t out_of_range = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t j = 0; j < n; ++j) {
            size_t at = (size_t)row * n + j;
            if (fabs(dots[at]) >= 2147483647.0) ++out_of_range;
            if ((double)got_accumulator[at] != dots[at]) {
                ++accumulate.count;
                accumulate.worst =
                    fmax(accumulate.worst, fabs((double)got_accumulator[at] - dots[at]));
            }
        }
    }
    if (out_of_range) {
        std::printf("  %zu accumulator elements exceed the int32 range\n", out_of_range);
    }
    report("int32 accumulate", accumulate, (size_t)rows * n);

    /* ---- 4. the epilogue ---- */
    std::vector<uint16_t> got_output((size_t)rows * n);
    if (!h3d_read(gpu, got_output.data(), output, error, sizeof(error))) {
        fail("read output", error);
    }
    Mismatch epilogue;
    for_each_row(rows, [&](uint32_t row, uint32_t) {
        for (uint32_t j = 0; j < n; ++j) {
            size_t at = (size_t)row * n + j;
            float want = host_epilogue(got_accumulator[at], got_scales[row],
                                       host_ws[j], host_bias.data(), j);
            float got = __bfloat162float(*(__nv_bfloat16 *)&got_output[at]);
            if (got != want) {
                ++epilogue.count;
                epilogue.worst = fmax(epilogue.worst, fabs((double)got - want));
            }
        }
    });
    report("epilogue vs host", epilogue, (size_t)rows * n);

    /* ---- 5. the BF16 linear the route replaces ---- */
    /* The grouped Hadamard is orthogonal, so `(H A) . (H W) = A . W`: the
     * linear the route approximates is the source weight against the
     * unrotated activation, and the deviation below is the route's total
     * error, activation and weight quantization together. */
    std::vector<uint16_t> bytes_wref((size_t)n * k);
    for (size_t at = 0; at < host_w_src.size(); ++at) {
        __nv_bfloat16 encoded = __float2bfloat16(host_w_src[at]);
        std::memcpy(&bytes_wref[at], &encoded, 2);
    }
    h3d_tensor *weight_reference =
        h3d_alloc(gpu, H3D_BF16, host_w_src.size(), error, sizeof(error));
    if (!weight_reference ||
        !h3d_upload(gpu, weight_reference, bytes_wref.data(), error, sizeof(error)) ||
        !h3d_linear(gpu, output_bf16, input, weight_reference, bias, rows, k, n,
                    H3D_BF16, error, sizeof(error))) {
        fail("bf16 linear", error);
    }
    std::vector<uint16_t> got_bf16((size_t)rows * n);
    if (!h3d_read(gpu, got_bf16.data(), output_bf16, error, sizeof(error))) {
        fail("read bf16", error);
    }
    double sum = 0.0, sum_diff = 0.0, worst = 0.0;
    for (size_t at = 0; at < got_bf16.size(); ++at) {
        float reference = __bfloat162float(*(__nv_bfloat16 *)&got_bf16[at]);
        float got = __bfloat162float(*(__nv_bfloat16 *)&got_output[at]);
        double delta = fabs((double)got - (double)reference);
        sum += (double)reference * reference;
        sum_diff += delta * delta;
        worst = fmax(worst, delta);
    }
    std::printf("INT8 convrot vs the source BF16 linear over %zu elements: "
                "RMS %.6g against %.6g (rel %.4g), worst abs %.6g\n",
                got_bf16.size(), sqrt(sum_diff / got_bf16.size()),
                sqrt(sum / got_bf16.size()), sqrt(sum_diff / sum), worst);

    /* ---- 6. the pieces' cost ---- */
    auto median = [](std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    bool timed_ok = true;
    auto timed = [&](auto &&body) {
        std::vector<double> samples;
        for (int repeat = 0; repeat < 4; ++repeat) {
            auto start = std::chrono::steady_clock::now();
            if (!body()) {
                timed_ok = false;
                return 0.0;
            }
            cudaStreamSynchronize(gpu->stream);
            auto end = std::chrono::steady_clock::now();
            if (repeat) {
                samples.push_back(
                    std::chrono::duration<double, std::milli>(end - start).count());
            }
        }
        return median(samples);
    };
    const double rotate_ms = timed([&] {
        return h3d_int8_rotate(gpu, plan.rotated, plan.words, input, rows, k,
                               group, error, sizeof(error));
    });
    const double scale_ms = timed([&] {
        return h3d_int8_scale_rows(gpu, plan.row_scales, plan.rotated, rows, k,
                                   error, sizeof(error)) &&
               h3d_int8_quantize_rows(gpu, plan.quantized, plan.row_scales,
                                      plan.rotated, rows, k, error, sizeof(error));
    });
    const double matmul_ms = timed([&] {
        return h3d_linear_int8_raw(gpu, plan.accumulator, plan.quantized, weight,
                                   padded, k, n, error, sizeof(error));
    });
    const double epilogue_ms = timed([&] {
        return h3d_int8_scale_epilogue(gpu, output, plan.accumulator,
                                       plan.row_scales, weight_scale, rows, n,
                                       error, sizeof(error));
    });
    const double chain_ms = timed([&] {
        return h3d_linear_int8_convrot(gpu, output, input, weight, weight_scale,
                                       bias, rows, k, n, group, &plan, error,
                                       sizeof(error));
    });
    if (!timed_ok) fail("timing", error);
    std::printf("per call %u x %u x %u (group %u): rotate %.3f ms, "
                "scale+quantize %.3f ms, int8 matmul %.3f ms, epilogue %.3f ms, "
                "chain %.3f ms\n",
                rows, k, n, group, rotate_ms, scale_ms, matmul_ms, epilogue_ms,
                chain_ms);

    std::printf("%s\n", failures == 0 ? "INT8-CONVROT-OK" : "INT8-CONVROT-OFF");
    h3d_destroy(gpu);
    return failures == 0 ? 0 : 1;
}
