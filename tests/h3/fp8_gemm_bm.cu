/*
 * The release's FP8 projection route on its own: `h3d_fp8_scale`,
 * `h3d_fp8_quantize` and `h3d_linear_e4m3` on synthetic operands at shapes
 * the DiT runs, checked against a host implementation of FastVideo's own two
 * steps and against the BF16 path the route replaces.
 *
 *   tests/h3/fp8_gemm_bm [rows] [k] [n] [mode]
 *
 * `mode` 0 runs the whole route, 1 stops after the quantization (the bisect
 * between the two kernels and the matmul). The default is 18,799 x 5,376 x
 * 21,504, the DiT's QKV projection at job A's row count.
 *
 * Build: `make CUDA_ARCH=<arch> tests/h3/fp8_gemm_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* The reference's scale floor is the native side's own constant
 * (`H3D_FP8_MIN_SCALE`), so the host check cannot drift from the kernel. */

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

/* A deterministic operand with a wide range: the reference's tensorwise scale
 * is one number for the whole tensor, so a few large values decide it. */
static uint32_t lcg = 12345u;

static float next_unit(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)((lcg >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f;
}

static float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

/* E4M3: sign, 4-bit exponent biased by 7, 3-bit mantissa; exponent 0 is
 * subnormal and exponent 15 with mantissa 7 is NaN, which the clamp excludes. */
static double e4m3_decode(uint8_t byte) {
    unsigned exponent = (byte >> 3) & 15u, mantissa = byte & 7u;
    double value = exponent ? (1.0 + mantissa / 8.0) * ldexp(1.0, (int)exponent - 7)
                            : (mantissa / 8.0) * ldexp(1.0, -6);
    return (byte & 0x80u) ? -value : value;
}

static double e4m3_ulp(double value) {
    double magnitude = fabs(value);
    if (magnitude < ldexp(1.0, -6)) return ldexp(1.0, -9);
    int exponent = 0;
    frexp(magnitude, &exponent);
    return ldexp(1.0, exponent - 4);
}

/* `(x / x_scale.to(x.dtype)).clamp(-448, 448)`: the divisor and the quotient
 * are both rounded to the operand's own dtype, as the reference's BF16
 * arithmetic rounds them. */
static float expected_quantized(float value, float scale) {
    float quotient = bf16_round(value / bf16_round(scale));
    if (quotient > H3D_FP8_MAX) return H3D_FP8_MAX;
    if (quotient < -H3D_FP8_MAX) return -H3D_FP8_MAX;
    return quotient;
}

/* One tensor's quantization against the host arithmetic. Returns the number of
 * bytes that disagree with the host quotient by more than half an E4M3 ulp. */
static size_t check_quantized(const char *name, const std::vector<float> &values,
                             const std::vector<uint8_t> &stored, float scale) {
    if (stored.size() < values.size()) fail(name, "short device copy");
    double amax = 0.0;
    for (float value : values) amax = fmax(amax, fabs((double)value));
    float want_scale = (float)fmax(amax / H3D_FP8_MAX, (double)H3D_FP8_MIN_SCALE);
    double scale_error = fabs((double)scale - (double)want_scale) / want_scale;
    size_t wrong = 0;
    double worst = 0.0;
    for (size_t index = 0; index < values.size(); ++index) {
        double got = e4m3_decode(stored[index]);
        double want = expected_quantized(values[index], want_scale);
        double over = fabs(got - want) - 0.5 * e4m3_ulp(want);
        if (over > 1e-9) {
            ++wrong;
            worst = fmax(worst, over);
        }
    }
    std::printf("%-6s %zu elements, scale %.6g (host %.6g, rel %.2g), "
                "bytes off by more than half an ulp: %zu (worst %.3g)\n",
                name, values.size(), (double)scale, (double)want_scale, scale_error,
                wrong, worst);
    return wrong;
}

/* A local copy of the matmul so one run can price the ways of handing cuBLASLt
 * the two scales, without touching the production entry point for each try:
 *
 *   variant 0  alpha 1 and no scales at all: `linear_fp8_bm`'s configuration
 *   variant 1  the two scales as `CUBLASLT_MATMUL_DESC_A/B_SCALE_POINTER`
 *   variant 2  the scales folded into a host alpha
 *   variant 3  the scales folded into a device alpha, device pointer mode
 *   variant 4  the two scales as outer vectors: `A/B_SCALE_MODE` set to
 *              `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`, so A carries one
 *              F32 per row and B one per output channel, and element (i, j)
 *              of the product takes both - the reference's own
 *              `granularity="channel"` (`fp8_config.py`) - with no workspace
 *   variant 5  the same, with a 64 MiB device workspace, the bisect between
 *              the route and the workspace it may need
 *   variant 6  the outer-vector mode on BF16 operands, the control that says
 *              whether the mode itself is missing or only its E4M3 combination
 */
static int local_gemm(h3d_context *gpu, h3d_tensor *out, h3d_tensor *a,
                      h3d_tensor *a_scale, h3d_tensor *w, h3d_tensor *w_scale,
                      uint32_t rows, uint32_t k, uint32_t n, int variant,
                      char *error, size_t error_size) {
    cublasLtMatmulDesc_t op = NULL;
    cublasLtMatrixLayout_t la = NULL, lb = NULL, lc = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    /* The layouts follow the operands, so the BF16 control (variant 6) runs
     * the same route as the E4M3 arms instead of a second hand-built call. */
    const cudaDataType_t operand_type = (a->dtype == H3D_BF16) ? CUDA_R_16BF
                                                               : CUDA_R_8F_E4M3;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    cublasPointerMode_t device_mode = CUBLAS_POINTER_MODE_DEVICE;
    float host_alpha = 1.0f, host_beta = 0.0f;
    const void *a_scale_ptr = a_scale->data, *b_scale_ptr = w_scale->data;
    float device_alpha = 1.0f;
    void *alpha_bytes = NULL;
    void *workspace = NULL;
    size_t workspace_bytes = 0;
    cublasStatus_t status = cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F,
                                                     CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &transpose, sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS && variant == 1) {
        status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                               &a_scale_ptr, sizeof(a_scale_ptr));
    }
    if (status == CUBLAS_STATUS_SUCCESS && variant == 1) {
        status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                               &b_scale_ptr, sizeof(b_scale_ptr));
    }
    if (variant == 2) {
        float scale_a = 0.0f, scale_b = 0.0f;
        if (!h3d_read(gpu, &scale_a, a_scale, error, error_size) ||
            !h3d_read(gpu, &scale_b, w_scale, error, error_size)) {
            return 0;
        }
        host_alpha = scale_a * scale_b;
    }
    if (variant == 3) {
        /* Two device floats: the product and beta. */
        if (cudaMalloc(&alpha_bytes, 2 * sizeof(float)) != cudaSuccess) {
            h3d_set_error(error, error_size, "no device scalars");
            return 0;
        }
        float zeros[2] = {0.0f, 0.0f};
        cudaMemcpy(alpha_bytes, zeros, sizeof(zeros), cudaMemcpyHostToDevice);
        /* One thread computes the product where it is wanted. */
        float host_a = 0.0f, host_b = 0.0f;
        if (!h3d_read(gpu, &host_a, a_scale, error, error_size) ||
            !h3d_read(gpu, &host_b, w_scale, error, error_size)) {
            return 0;
        }
        device_alpha = host_a * host_b;
        cudaMemcpy(alpha_bytes, &device_alpha, sizeof(float), cudaMemcpyHostToDevice);
        status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &device_mode, sizeof(device_mode));
    }
    if (variant == 4 || variant == 5 || variant == 6) {
        int scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &a_scale_ptr, sizeof(a_scale_ptr));
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &b_scale_ptr, sizeof(b_scale_ptr));
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                    &scale_mode, sizeof(scale_mode));
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                    &scale_mode, sizeof(scale_mode));
        }
    }
    if (variant == 5 && status == CUBLAS_STATUS_SUCCESS) {
        workspace_bytes = 64u << 20;
        if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
            h3d_set_error(error, error_size, "no workspace");
            status = CUBLAS_STATUS_ALLOC_FAILED;
        }
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&la, operand_type, rows, k, k);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&lb, operand_type, n, k, k);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, rows, n, n);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                  &row_major, sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                  &row_major, sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                  &row_major, sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        const void *alpha = (variant == 3) ? alpha_bytes : (const void *)&host_alpha;
        const void *beta = (variant == 3) ? (void *)((char *)alpha_bytes + sizeof(float))
                                          : (const void *)&host_beta;
        status = cublasLtMatmul(gpu->blas, op, alpha, a->data, la, w->data, lb, beta,
                                out->data, lc, out->data, lc, NULL, workspace,
                                workspace_bytes, gpu->stream);
    }
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (op) cublasLtMatmulDescDestroy(op);
    if (alpha_bytes) cudaFree(alpha_bytes);
    if (workspace) cudaFree(workspace);
    if (status != CUBLAS_STATUS_SUCCESS) {
        h3d_set_error(error, error_size, "local gemm status %d", (int)status);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    char error[512] = {};
    /* Unbuffered, so a variant that dies leaves its own last line behind. */
    setvbuf(stdout, NULL, _IONBF, 0);
    const uint32_t rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 18799;
    const uint32_t k = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 5376;
    const uint32_t n = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 21504;
    const int mode = argc > 4 ? std::atoi(argv[4]) : 0;

    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);

    std::vector<uint8_t> host_a((size_t)rows * k * 2), host_w((size_t)n * k * 2);
    std::vector<float> values_a((size_t)rows * k), values_w((size_t)n * k);
    for (size_t index = 0; index < values_a.size(); ++index) {
        /* Occasional large values, as a weight row's outliers are. */
        float magnitude = (index % 4096 == 0) ? 6.0f : 1.0f;
        values_a[index] = bf16_round(next_unit() * magnitude);
        __nv_bfloat16 encoded = __float2bfloat16(values_a[index]);
        std::memcpy(&host_a[index * 2], &encoded, 2);
    }
    for (size_t index = 0; index < values_w.size(); ++index) {
        float magnitude = (index % 4096 == 0) ? 6.0f : 1.0f;
        values_w[index] = bf16_round(next_unit() * magnitude);
        __nv_bfloat16 encoded = __float2bfloat16(values_w[index]);
        std::memcpy(&host_w[index * 2], &encoded, 2);
    }

    h3d_tensor *a_bf16 = h3d_alloc(gpu, H3D_BF16, values_a.size(), error, sizeof(error));
    h3d_tensor *w_bf16 = h3d_alloc(gpu, H3D_BF16, values_w.size(), error, sizeof(error));
    h3d_tensor *a_fp8 = h3d_alloc(gpu, H3D_FP8, values_a.size(), error, sizeof(error));
    h3d_tensor *w_fp8 = h3d_alloc(gpu, H3D_FP8, values_w.size(), error, sizeof(error));
    h3d_tensor *a_scale = h3d_alloc(gpu, H3D_F32, 1, error, sizeof(error));
    h3d_tensor *w_scale = h3d_alloc(gpu, H3D_F32, 1, error, sizeof(error));
    h3d_tensor *out_bf16 = h3d_alloc(gpu, H3D_BF16, (size_t)rows * n, error, sizeof(error));
    h3d_tensor *out_fp8 = h3d_alloc(gpu, H3D_BF16, (size_t)rows * n, error, sizeof(error));
    if (!a_bf16 || !w_bf16 || !a_fp8 || !w_fp8 || !a_scale || !w_scale ||
        !out_bf16 || !out_fp8) {
        fail("alloc", error);
    }
    /* The per-row probe needs scale vectors, not the one-element tensorwise
     * scales: `h3d_fp8_scale` and `h3d_fp8_quantize` each require
     * `h3d_is(scale, H3D_F32, 1)`, so the vectors get their own tensors. */
    h3d_tensor *row_scale = NULL, *col_scale = NULL, *out_control = NULL;
    if (mode >= 6) {
        row_scale = h3d_alloc(gpu, H3D_F32, rows, error, sizeof(error));
        col_scale = h3d_alloc(gpu, H3D_F32, n, error, sizeof(error));
        out_control = h3d_alloc(gpu, H3D_BF16, (size_t)rows * n, error, sizeof(error));
        if (!row_scale || !col_scale || !out_control) {
            fail("alloc outer", error);
        }
    }
    if (!h3d_upload(gpu, a_bf16, host_a.data(), error, sizeof(error)) ||
        !h3d_upload(gpu, w_bf16, host_w.data(), error, sizeof(error))) {
        fail("upload", error);
    }
    std::printf("rows %u, k %u, n %u, mode %d\n", rows, k, n, mode);

    if (!h3d_fp8_scale(gpu, a_scale, a_bf16, error, sizeof(error)) ||
        !h3d_fp8_scale(gpu, w_scale, w_bf16, error, sizeof(error)) ||
        !h3d_fp8_quantize(gpu, a_fp8, a_scale, a_bf16, values_a.size(), error, sizeof(error)) ||
        !h3d_fp8_quantize(gpu, w_fp8, w_scale, w_bf16, values_w.size(), error, sizeof(error))) {
        fail("quantize", error);
    }
    {
        std::vector<uint8_t> got_a(values_a.size()), got_w(values_w.size());
        float scale_a = 0.0f, scale_w = 0.0f;
        if (!h3d_read(gpu, got_a.data(), a_fp8, error, sizeof(error)) ||
            !h3d_read(gpu, got_w.data(), w_fp8, error, sizeof(error)) ||
            !h3d_read(gpu, &scale_a, a_scale, error, sizeof(error)) ||
            !h3d_read(gpu, &scale_w, w_scale, error, sizeof(error))) {
            fail("read quantized", error);
        }
        check_quantized("A", values_a, got_a, scale_a);
        check_quantized("W", values_w, got_w, scale_w);
        if (mode == 1) {
            std::printf("mode 1: quantization only\n");
            return 0;
        }
    }

    /* The BF16 path is the reference the route replaces. */
    if (!h3d_linear(gpu, out_bf16, a_bf16, w_bf16, NULL, rows, k, n, H3D_BF16,
                    error, sizeof(error))) {
        fail("bf16 linear", error);
    }
    auto median = [](std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    /* Modes 2..5 run the local variants, one per invocation, so a variant that
     * crashes or hangs takes only its own process with it. Modes 6..8 are the
     * outer-vector probe: 6 all-ones scales, 7 a power-of-two pattern, 8 the
     * same pattern with a device workspace; 9 times the no-scale GEMM alone,
     * and 10 runs the same outer-vector mode on BF16 operands instead of E4M3.
     */
    if (mode == 9) {
        /* The no-scale E4M3 GEMM on its own: the floor a per-row arm that
         * applies both scale vectors in an epilogue of its own would pay. */
        std::vector<double> control_ms;
        for (int repeat = 0; repeat < 4; ++repeat) {
            auto t0 = std::chrono::steady_clock::now();
            if (!local_gemm(gpu, out_control, a_fp8, a_scale, w_fp8, w_scale, rows, k, n,
                            0, error, sizeof(error))) {
                fail("control gemm", error);
            }
            cudaStreamSynchronize(gpu->stream);
            auto t1 = std::chrono::steady_clock::now();
            if (repeat) {
                control_ms.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }
        std::printf("no-scale fp8 gemm per call: %.3f ms (median of %zu)\n",
                    median(control_ms), control_ms.size());
        h3d_destroy(gpu);
        return 0;
    } else if (mode == 10) {
        /* Whether the mode itself is missing, separated from its E4M3
         * combination: the same attributes and the same vectors, applied to
         * the BF16 operands the route replaces. */
        std::vector<float> row_host(rows, 1.0f), col_host(n, 1.0f);
        row_host[0] = 4.0f;
        col_host[0] = 8.0f;
        if (!h3d_upload(gpu, row_scale, row_host.data(), error, sizeof(error)) ||
            !h3d_upload(gpu, col_scale, col_host.data(), error, sizeof(error))) {
            fail("upload scales", error);
        }
        if (!local_gemm(gpu, out_control, a_bf16, row_scale, w_bf16, col_scale, rows, k, n,
                        6, error, sizeof(error))) {
            fail("bf16 outer gemm", error);
        }
        std::vector<uint16_t> base((size_t)rows * n), outer((size_t)rows * n);
        if (!h3d_read(gpu, base.data(), out_bf16, error, sizeof(error)) ||
            !h3d_read(gpu, outer.data(), out_control, error, sizeof(error))) {
            fail("read outer", error);
        }
        double sum = 0.0, sum_diff = 0.0, gap = 0.0;
        for (size_t index = 0; index < outer.size(); ++index) {
            float reference = __bfloat162float(*(__nv_bfloat16 *)&base[index]);
            float got = __bfloat162float(*(__nv_bfloat16 *)&outer[index]);
            double want = (double)reference * (double)row_host[index / n] *
                          (double)col_host[index % n];
            double delta = fabs((double)got - want);
            sum += want * want;
            sum_diff += delta * delta;
            gap = fmax(gap, delta);
        }
        std::printf("bf16 outer vs scaled bf16 linear: RMS %.6g against %.6g "
                    "(rel %.3g), worst abs %.6g\n", sqrt(sum_diff / outer.size()),
                    sqrt(sum / outer.size()), sqrt(sum_diff / sum), gap);
        std::printf("%s\n", (sqrt(sum_diff / sum) < 0.05) ? "BF16-OUTER-OK"
                                                          : "BF16-OUTER-OFF");
        h3d_destroy(gpu);
        return 0;
    } else if (mode >= 2 && mode <= 5) {
        std::printf("local gemm variant %d\n", mode - 2);
        if (!local_gemm(gpu, out_fp8, a_fp8, a_scale, w_fp8, w_scale, rows, k, n,
                        mode - 2, error, sizeof(error))) {
            fail("local gemm", error);
        }
    } else if (mode >= 6) {
        /* Whether cuBLASLt will express the reference's `granularity="channel"`
         * at all, and what it costs. A's scale is one F32 per row and B's one
         * per output channel (`cublasLt.h`: in the outer-vector mode the
         * vectors "are expected to have M and N elements respectively"), and
         * the control is the same operands with no scales at all.
         *
         * The pattern is a power of two, which makes the check exact: rounding
         * to BF16 commutes with a power-of-two multiply, so element (i, j) of
         * the outer run must equal the control's own element times
         * row[i] * col[j]. */
        const bool patterned = (mode >= 7);
        const int variant = (mode >= 8) ? 5 : 4;
        std::vector<float> row_host(rows, 1.0f), col_host(n, 1.0f);
        if (patterned) {
            row_host[0] = 4.0f;
            col_host[0] = 8.0f;
        }
        if (!h3d_upload(gpu, row_scale, row_host.data(), error, sizeof(error)) ||
            !h3d_upload(gpu, col_scale, col_host.data(), error, sizeof(error))) {
            fail("upload scales", error);
        }
        std::printf("outer-vec variant %d, %s scales\n", variant,
                    patterned ? "power-of-two" : "all-one");
        std::vector<double> control_ms, outer_ms;
        for (int repeat = 0; repeat < 4; ++repeat) {
            auto t0 = std::chrono::steady_clock::now();
            if (!local_gemm(gpu, out_control, a_fp8, a_scale, w_fp8, w_scale, rows, k, n,
                            0, error, sizeof(error))) {
                fail("control gemm", error);
            }
            cudaStreamSynchronize(gpu->stream);
            auto t1 = std::chrono::steady_clock::now();
            if (!local_gemm(gpu, out_fp8, a_fp8, row_scale, w_fp8, col_scale, rows, k, n,
                            variant, error, sizeof(error))) {
                fail("outer gemm", error);
            }
            cudaStreamSynchronize(gpu->stream);
            auto t2 = std::chrono::steady_clock::now();
            if (repeat == 0) {
                continue; /* the first pair pays the kernels' own setup */
            }
            control_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            outer_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        std::vector<uint16_t> control((size_t)rows * n), outer((size_t)rows * n);
        if (!h3d_read(gpu, control.data(), out_control, error, sizeof(error)) ||
            !h3d_read(gpu, outer.data(), out_fp8, error, sizeof(error))) {
            fail("read outer", error);
        }
        double sum = 0.0, sum_diff = 0.0, gap = 0.0;
        for (size_t index = 0; index < outer.size(); ++index) {
            float reference = __bfloat162float(*(__nv_bfloat16 *)&control[index]);
            float got = __bfloat162float(*(__nv_bfloat16 *)&outer[index]);
            double want = (double)reference * (double)row_host[index / n] *
                          (double)col_host[index % n];
            double delta = fabs((double)got - want);
            sum += want * want;
            sum_diff += delta * delta;
            gap = fmax(gap, delta);
        }
        std::printf("outer vs scaled control: RMS %.6g against %.6g (rel %.3g), "
                    "worst abs %.6g\n", sqrt(sum_diff / outer.size()),
                    sqrt(sum / outer.size()), sqrt(sum_diff / sum), gap);
        std::printf("per call: control %.3f ms, outer %.3f ms (median of %zu)\n",
                    median(control_ms), median(outer_ms), control_ms.size());
        std::printf("%s\n", (sqrt(sum_diff / sum) < 0.05) ? "FP8-OUTER-OK" : "FP8-OUTER-OFF");
        h3d_destroy(gpu);
        return 0;
    } else if (!h3d_linear_e4m3(gpu, out_fp8, a_fp8, a_scale, w_fp8, w_scale,
                                rows, k, n, error, sizeof(error))) {
        fail("fp8 linear", error);
    }
    std::vector<uint16_t> got_bf16((size_t)rows * n), got_fp8((size_t)rows * n);
    if (!h3d_read(gpu, got_bf16.data(), out_bf16, error, sizeof(error)) ||
        !h3d_read(gpu, got_fp8.data(), out_fp8, error, sizeof(error))) {
        fail("read output", error);
    }
    double sum = 0.0, sum_diff = 0.0, worst = 0.0;
    for (size_t index = 0; index < got_bf16.size(); ++index) {
        float reference = __bfloat162float(*(__nv_bfloat16 *)&got_bf16[index]);
        float quantized = __bfloat162float(*(__nv_bfloat16 *)&got_fp8[index]);
        double delta = fabs((double)quantized - (double)reference);
        sum += (double)reference * reference;
        sum_diff += delta * delta;
        worst = fmax(worst, delta);
    }
    std::printf("FP8 vs BF16 over %zu elements: RMS %.6g against %.6g "
                "(rel %.3g), worst abs %.6g\n",
                got_bf16.size(), sqrt(sum_diff / got_bf16.size()),
                sqrt(sum / got_bf16.size()),
                sqrt(sum_diff / sum), worst);
    std::printf("%s\n", (sqrt(sum_diff / sum) < 0.05) ? "FP8-GEMM-OK" : "FP8-GEMM-OFF");
    h3d_destroy(gpu);
    return 0;
}
