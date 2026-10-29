/*
 * What FP8 weights would buy the DiT's four projections: cuBLASLt at the same
 * shapes and the same row-major TRANSB configuration `h3d_linear` uses, in
 * BF16 (the shipped path) and in E4M3 x E4M3 with a BF16 output.
 *
 *   tests/h3/linear_fp8_bm [rows] [calls]
 *
 * job A's shapes: 18,799 rows and (K, N) of 5376/21504 (qkv), 7168/5376
 * (out), 5376/14336 (fc1) and 14336/5376 (fc2), each once per layer per
 * evaluation, 50 layers.
 *
 * This prices the arithmetic, not a quantization scheme: the operands are
 * synthetic and unscaled (alpha = 1), so the number below is the GEMM's own
 * ceiling. It does not include quantizing the activations, per-tensor or
 * per-row scales, or the accuracy a W8A8 scheme would lose.
 *
 * Build: `make CUDA_ARCH=<arch> tests/h3/linear_fp8_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct shape {
    const char *name;
    uint32_t input_dim;   /* K */
    uint32_t output_dim;  /* N */
};

static const shape SHAPES[] = {
    {"qkv", 5376, 21504},
    {"out", 7168, 5376},
    {"fc1", 5376, 14336},
    {"fc2", 14336, 5376},
};

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

/* One call of the same configuration `h3d_linear_lt_compute` builds:
 * C(rows x N) = A(rows x K) x B(N x K)^T in row-major, alpha 1, beta 0. */
static int gemm(h3d_context *gpu, void *output, const void *input,
                const void *weight, uint32_t rows, uint32_t input_dim,
                uint32_t output_dim, cudaDataType_t type,
                cudaDataType_t output_type, char *error, size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t input_layout = NULL, weight_layout = NULL,
                           output_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
            sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&input_layout, type, rows,
                                            input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&weight_layout, type, output_dim,
                                            input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&output_layout, output_type, rows,
                                            output_dim, output_dim);
    }
    cublasLtOrder_t order = row_major;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
            output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(gpu->blas, operation, &alpha, input,
                                input_layout, weight, weight_layout, &beta,
                                output, output_layout, output, output_layout,
                                NULL, NULL, 0, gpu->stream);
    }
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (status != CUBLAS_STATUS_SUCCESS) {
        h3d_set_error(error, error_size, "cuBLASLt gemm failed");
        return 0;
    }
    return 1;
}

static double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                         const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

int main(int argc, char **argv) {
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) fail("create", error);
    const uint32_t rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 18799;
    const int calls = argc > 2 ? std::atoi(argv[2]) : 3;
    /* Elements to shift the operands by, to price what a misaligned slice
     * costs: cuBLASLt's tensor-core kernels assume 16-byte (align8 in the
     * kernel names) operands, and the model's tensors are views into larger
     * allocations. */
    const uint32_t drop = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 0;

    /* One allocation per tensor, sized for the largest shape (fc2's K). */
    size_t input_bytes = (size_t)rows * SHAPES[0].input_dim * 2;
    size_t weight_bytes = 0, output_bytes = 0;
    for (const shape &s : SHAPES) {
        size_t k = (size_t)s.input_dim, n = (size_t)s.output_dim;
        if (rows * k * 2 > input_bytes) input_bytes = rows * k * 2;
        if (n * k * 2 > weight_bytes) weight_bytes = n * k * 2;
        if (rows * n * 2 > output_bytes) output_bytes = rows * n * 2;
    }
    std::vector<uint8_t> host_a(input_bytes, 0x38), host_b(weight_bytes, 0x38);
    void *a = NULL, *b = NULL, *c = NULL;
    if (cudaMalloc((void **)&a, input_bytes + 32) != cudaSuccess ||
        cudaMalloc((void **)&b, weight_bytes + 32) != cudaSuccess ||
        cudaMalloc((void **)&c, output_bytes + 32) != cudaSuccess) {
        fail("cudaMalloc", "out of device memory for the bench tensors");
    }
    a = (char *)a + 2 * drop;
    b = (char *)b + 2 * drop;
    cudaMemcpy(a, host_a.data(), host_a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b.data(), host_b.size(), cudaMemcpyHostToDevice);

    printf("rows %u, %d calls per arm, operands shifted %u element(s) "
           "(%u bytes); FP32 accumulate, BF16 output, alpha 1 (unscaled)\n",
           rows, calls, drop, 2 * drop);
    printf("%-4s %8s %8s %10s %10s %12s %10s\n", "gemm", "K", "N", "TFLOP",
           "bf16 ms", "bf16 TFLOPS", "fp8 ms");

    double bf16_total = 0.0, fp8_total = 0.0;
    for (const shape &s : SHAPES) {
        const double tflop = 2.0 * rows * s.input_dim * s.output_dim / 1e12;
        if (!gemm(gpu, c, a, b, rows, s.input_dim, s.output_dim, CUDA_R_16BF,
                  CUDA_R_16BF, error, sizeof(error))) {
            fail("bf16", error);
        }
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
        auto begin = std::chrono::steady_clock::now();
        for (int call = 0; call < calls; ++call) {
            if (!gemm(gpu, c, a, b, rows, s.input_dim, s.output_dim,
                      CUDA_R_16BF, CUDA_R_16BF, error, sizeof(error))) {
                fail("bf16", error);
            }
        }
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
        const double bf16_ms =
            elapsed_ms(begin, std::chrono::steady_clock::now()) / calls;

        if (!gemm(gpu, c, a, b, rows, s.input_dim, s.output_dim,
                  CUDA_R_8F_E4M3, CUDA_R_16BF, error, sizeof(error))) {
            fail("fp8", error);
        }
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
        begin = std::chrono::steady_clock::now();
        for (int call = 0; call < calls; ++call) {
            if (!gemm(gpu, c, a, b, rows, s.input_dim, s.output_dim,
                      CUDA_R_8F_E4M3, CUDA_R_16BF, error, sizeof(error))) {
                fail("fp8", error);
            }
        }
        if (!h3d_sync(gpu, error, sizeof(error))) fail("sync", error);
        const double fp8_ms =
            elapsed_ms(begin, std::chrono::steady_clock::now()) / calls;

        printf("%-4s %8u %8u %10.3f %10.3f %12.1f %10.3f\n", s.name,
               s.input_dim, s.output_dim, tflop, bf16_ms,
               tflop / bf16_ms * 1000.0, fp8_ms);
        bf16_total += bf16_ms;
        fp8_total += fp8_ms;
    }
    printf("50 layers: bf16 %.0f ms, fp8 %.0f ms per evaluation "
           "(bf16 at its measured rate; fp8 unscaled)\n",
           bf16_total * 50.0, fp8_total * 50.0);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    h3d_destroy(gpu);
    return 0;
}
