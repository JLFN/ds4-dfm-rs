/*
 * Bounded reproducer for the video VAE decoder's four F32 matmuls, the pair
 * `h3d_vae_linear` serves and the `H3_VAE_GEMM` option selects the compute mode
 * for. One call here is one decoder block's matmul set at the shape the 768x768
 * job runs, so a per-call figure times 1,944 block calls prices the block.
 *
 *   tests/h3/vae_gemm_bm            # 2,532 rows, the 768 job's tile
 *   tests/h3/vae_gemm_bm 2273       # the 512 job's tile
 *   tests/h3/vae_gemm_bm 2532 5     # five calls for the mean
 *
 * It prints each mode's time and output hash, then the deviation between the
 * modes on the same inputs, and it says plainly whether the two FAST modes are
 * one computation or two: on this toolchain they are one, bit for bit, which
 * means the two arms cannot be A/B'd against each other and only the pair
 * (F32, FAST) is a real comparison.
 *
 * It then prices the one candidate the attribution leaves: the same four
 * matmuls with BF16 operands and an F32 output, which is what a BF16 kernel
 * would run against the TF32-class kernel cuBLASLt picks for `FAST_16BF`. That
 * figure is the GEMM's own ceiling, without the per-call activation rounding
 * the arm also pays.
 *
 * Finally it runs the arm the port now has (`H3_VAE_GEMM=bf16i`,
 * `h3d_vae_linear_bf16` behind the two casts `vae_video::project` uses), which
 * is the ceiling plus the rounding it leaves out, and reports the deviation
 * between the widened result and the F32 path's.
 *
 * Build: `make tests/h3/vae_gemm_bm`.
 */
#include "../../native/h3/h3_dit_gpu.cu"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* The decoder block's own widths (`crates/ds4-core/src/minimax_h3/vae_video.rs`). */
enum { HIDDEN = 2048, INNER = 2048, FFN = 8192 };

static void fail(const char *what, const char *detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(2);
}

static uint64_t fnv1a(const void *data, size_t bytes) {
    constexpr uint64_t offset = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset;
    auto *bytes_data = static_cast<const unsigned char *>(data);
    for (size_t index = 0; index < bytes; index++) {
        hash = (hash ^ bytes_data[index]) * prime;
    }
    return hash;
}

/* Deterministic high-entropy F32 values in [-1, 1): every mantissa bit is
 * used, so a reduced-precision mode has something to round. */
static void fill(std::vector<float> &host, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 1u;
    for (auto &item : host) {
        state = state * 1664525u + 1013904223u;
        item = (float)(state >> 8) / 8388608.0f - 1.0f;
    }
}

struct Matmul {
    const char *name;
    uint32_t input_dim;
    uint32_t output_dim;
    h3d_tensor *input;
    h3d_tensor *weight;
    h3d_tensor *bias;
    h3d_tensor *output;
};

/* Candidate probe, NOT a port path: the port's matmul with BF16 operands, so
 * the four block shapes can be priced under a BF16 kernel instead of the
 * TF32-class one cuBLASLt picks for `CUBLAS_COMPUTE_32F_FAST_16BF` (the class
 * the 2026-10-29 attribution found: `cutlass_..._s1688gemm_*` at 36 TFLOPS).
 * It mirrors `h3d_linear_lt_compute`'s layouts, TRANSB, row-major order, alpha
 * 1 and beta 0, and changes the operand types and optionally the output type.
 * `CUDA_R_32F` output is refused on the workstation's sm_89 with
 * CUBLAS_STATUS_INTERNAL_ERROR; `CUDA_R_16BF` output is the DiT's own path. */
static int bf16_matmul(h3d_context *context, void *output,
                       cudaDataType_t output_type, const void *input,
                       const void *weight, uint32_t rows, uint32_t input_dim,
                       uint32_t output_dim, char *error, size_t error_size) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t input_layout = NULL;
    cublasLtMatrixLayout_t weight_layout = NULL;
    cublasLtMatrixLayout_t output_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status =
            cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(
                operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
                sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&input_layout, CUDA_R_16BF, rows,
                                            input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_16BF,
                                            output_dim, input_dim, input_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&output_layout, output_type, rows,
                                            output_dim, output_dim);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
                input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
                sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
                weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
                sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutSetAttribute(
                output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
                sizeof(row_major));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmul(context->blas, operation, &alpha, input,
                                input_layout, weight, weight_layout, &beta,
                                output, output_layout, output, output_layout,
                                NULL, NULL, 0, context->stream);
    }
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    return h3d_blas_ok(status, "cuBLASLt BF16 linear", error, error_size);
}

int main(int argc, char **argv) {
    const uint32_t rows = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 2532;
    const int calls = argc > 2 ? std::atoi(argv[2]) : 2;
    bool f32_output = false;
    for (int index = 1; index < argc; index++) {
        if (std::strcmp(argv[index], "--f32out") == 0) {
            f32_output = true;
        }
    }
    char error[512] = {};
    h3d_context *gpu = h3d_create(error, sizeof(error));
    if (!gpu) {
        fail("create", error);
    }

    const uint32_t shapes[4][2] = {
        {HIDDEN, INNER * 3}, {INNER, HIDDEN}, {HIDDEN, FFN * 2}, {FFN, HIDDEN}};
    const char *names[4] = {"qkv", "out", "ff1", "w2"};
    Matmul matmuls[4] = {};
    for (int index = 0; index < 4; index++) {
        const uint32_t input_dim = shapes[index][0], output_dim = shapes[index][1];
        Matmul &matmul = matmuls[index];
        matmul.name = names[index];
        matmul.input_dim = input_dim;
        matmul.output_dim = output_dim;
        matmul.input = h3d_alloc(gpu, H3D_F32, (size_t)rows * input_dim, error, sizeof(error));
        matmul.weight = h3d_alloc(gpu, H3D_F32, (size_t)output_dim * input_dim, error, sizeof(error));
        matmul.bias = h3d_alloc(gpu, H3D_F32, output_dim, error, sizeof(error));
        matmul.output = h3d_alloc(gpu, H3D_F32, (size_t)rows * output_dim, error, sizeof(error));
        if (!matmul.input || !matmul.weight || !matmul.bias || !matmul.output) {
            fail("alloc", error);
        }
        std::vector<float> host;
        host.resize((size_t)rows * input_dim);
        fill(host, 10u + index * 3u);
        h3d_upload(gpu, matmul.input, host.data(), error, sizeof(error));
        host.resize((size_t)output_dim * input_dim);
        fill(host, 11u + index * 3u);
        h3d_upload(gpu, matmul.weight, host.data(), error, sizeof(error));
        host.resize(output_dim);
        fill(host, 12u + index * 3u);
        h3d_upload(gpu, matmul.bias, host.data(), error, sizeof(error));
    }

    /* Every mode over the same inputs; the first is the reference path. */
    const char *modes[3] = {"fp32", "tf32", "bf16"};
    std::vector<std::vector<float>> results[3];
    double milliseconds[3] = {};
    for (int mode = 0; mode < 3; mode++) {
        if (!h3d_set_vae_option(gpu, "gemm", modes[mode])) {
            fail("option", modes[mode]);
        }
        auto start = std::chrono::steady_clock::now();
        for (int call = 0; call < calls; call++) {
            for (Matmul &matmul : matmuls) {
                if (!h3d_vae_linear(gpu, matmul.output, matmul.input, matmul.weight,
                                    matmul.bias, rows, matmul.input_dim,
                                    matmul.output_dim, error, sizeof(error))) {
                    fail("vae linear", error);
                }
            }
        }
        if (!h3d_sync(gpu, error, sizeof(error))) {
            fail("sync", error);
        }
        auto finish = std::chrono::steady_clock::now();
        milliseconds[mode] =
                std::chrono::duration<double, std::milli>(finish - start).count() / calls;
        uint64_t hash = 14695981039346656037ULL;
        for (Matmul &matmul : matmuls) {
            std::vector<float> host((size_t)rows * matmul.output_dim);
            if (!h3d_read(gpu, host.data(), matmul.output, error, sizeof(error))) {
                fail("read", error);
            }
            /* Mix the block's four outputs into one figure, so a mode that
             * differs anywhere moves it. */
            const uint64_t piece = fnv1a(host.data(), host.size() * sizeof(float));
            hash = (hash ^ piece) * 1099511628211ULL;
            results[mode].push_back(std::move(host));
        }
        std::printf("rows=%u mode=%-4s ms_per_block=%.3f out_fnv=%016llx\n", rows,
                    modes[mode], milliseconds[mode], (unsigned long long)hash);
    }

    auto compare = [&](const char *label, int left, int right) {
        double worst = 0.0, square = 0.0, scale = 0.0;
        size_t differing = 0, elements = 0;
        for (size_t piece = 0; piece < results[left].size(); piece++) {
            const std::vector<float> &a = results[left][piece];
            const std::vector<float> &b = results[right][piece];
            for (size_t index = 0; index < a.size(); index++) {
                const double delta = (double)a[index] - (double)b[index];
                if (a[index] != b[index]) differing++;
                worst = std::fmax(worst, std::fabs(delta));
                scale = std::fmax(scale, std::fabs((double)b[index]));
                square += delta * delta;
                elements++;
            }
        }
        const double rms = std::sqrt(square / (double)elements);
        std::printf("%-14s max_abs=%.6g rms=%.6g rel_rms=%.3g differing=%.2f%%\n",
                    label, worst, rms, scale > 0.0 ? rms / scale : 0.0,
                    100.0 * (double)differing / (double)elements);
    };
    compare("tf32 vs fp32", 1, 0);
    compare("bf16 vs fp32", 2, 0);
    compare("bf16 vs tf32", 2, 1);

    /* The candidate probe: the same four shapes with BF16 operands. The inputs
     * are the same values rounded once on the host, so the figure is the GEMM's
     * own ceiling; rounding an activation per call would add a 20.7 MB read and
     * a 10.4 MB write, 0.13 ms at this host's 250 GB/s, and is not in it. */
    std::vector<h3d_tensor *> rounded_input(4), rounded_weight(4), narrowed(4);
    for (int index = 0; index < 4; index++) {
        const uint32_t input_dim = shapes[index][0], output_dim = shapes[index][1];
        std::vector<float> host;
        host.resize((size_t)rows * input_dim);
        fill(host, 10u + index * 3u);
        std::vector<__nv_bfloat16> narrow(host.size());
        for (size_t item = 0; item < host.size(); item++) {
            narrow[item] = __float2bfloat16(host[item]);
        }
        rounded_input[index] = h3d_alloc(gpu, H3D_BF16, host.size(), error, sizeof(error));
        if (!rounded_input[index] ||
            !h3d_upload(gpu, rounded_input[index], narrow.data(), error, sizeof(error))) {
            fail("bf16 input", error);
        }
        host.resize((size_t)output_dim * input_dim);
        fill(host, 11u + index * 3u);
        narrow.resize(host.size());
        for (size_t item = 0; item < host.size(); item++) {
            narrow[item] = __float2bfloat16(host[item]);
        }
        rounded_weight[index] = h3d_alloc(gpu, H3D_BF16, host.size(), error, sizeof(error));
        if (!rounded_weight[index] ||
            !h3d_upload(gpu, rounded_weight[index], narrow.data(), error, sizeof(error))) {
            fail("bf16 weight", error);
        }
        narrowed[index] = h3d_alloc(gpu, H3D_BF16, (size_t)rows * output_dim, error,
                                    sizeof(error));
        if (!narrowed[index]) {
            fail("bf16 output", error);
        }
    }
    if (!h3d_sync(gpu, error, sizeof(error))) {
        fail("bf16 sync", error);
    }

    /* The F32-output variant is opt-in: sm_89 refuses it with
     * CUBLAS_STATUS_INTERNAL_ERROR and leaves the context unusable, so a normal
     * run must not touch it. `--f32out` attempts it once and reports. */
    if (f32_output) {
        error[0] = 0;
        const bool ok = bf16_matmul(gpu, matmuls[0].output, CUDA_R_32F,
                                    rounded_input[0], rounded_weight[0], rows,
                                    matmuls[0].input_dim, matmuls[0].output_dim,
                                    error, sizeof(error));
        std::printf("bf16 operands, F32 output: %s (%s)\n", ok ? "accepted" : "refused",
                    ok ? "no error" : error);
        if (!ok || !h3d_sync(gpu, error, sizeof(error))) {
            std::printf("bf16 operands, F32 output: the context is unusable after it\n");
            return 3;
        }
    }

    /* cuBLASLt pays a one-time cost on the first BF16 call (algorithm selection
     * and, on sm_89, a JIT); it lands in the mean if the first call is kept, so
     * it is discarded and each call is printed. */
    double rounded_ms = 0.0;
    for (int call = 0; call <= calls; call++) {
        auto start = std::chrono::steady_clock::now();
        for (int index = 0; index < 4; index++) {
            if (!bf16_matmul(gpu, narrowed[index]->data, CUDA_R_16BF,
                             rounded_input[index]->data, rounded_weight[index]->data,
                             rows, matmuls[index].input_dim,
                             matmuls[index].output_dim, error, sizeof(error))) {
                std::fprintf(stderr, "bf16 matmul %s: %s\n", matmuls[index].name, error);
                std::exit(2);
            }
        }
        if (!h3d_sync(gpu, error, sizeof(error))) {
            fail("sync", error);
        }
        auto finish = std::chrono::steady_clock::now();
        const double this_call =
                std::chrono::duration<double, std::milli>(finish - start).count();
        std::printf("  bf16i call %d: %.3f ms%s\n", call, this_call,
                    call == 0 ? " (discarded)" : "");
        if (call > 0) {
            rounded_ms += this_call;
        }
    }
    rounded_ms /= (double)calls;
    std::printf("rows=%u mode=%-4s ms_per_block=%.3f (BF16 operands, GEMM only)\n", rows,
                "bf16i", rounded_ms);
    std::vector<float> probe;
    for (int index = 0; index < 4; index++) {
        std::vector<__nv_bfloat16> narrow((size_t)rows * matmuls[index].output_dim);
        if (!h3d_read(gpu, narrow.data(), narrowed[index], error, sizeof(error))) {
            fail("read", error);
        }
        for (__nv_bfloat16 value : narrow) {
            probe.push_back(__bfloat162float(value));
        }
    }

    double worst = 0.0, square = 0.0, scale = 0.0;
    size_t differing = 0, elements = 0;
    size_t at = 0;
    for (size_t piece = 0; piece < results[0].size(); piece++) {
        for (float reference : results[0][piece]) {
            const double delta = (double)probe[at++] - (double)reference;
            if (probe[at - 1] != reference) differing++;
            worst = std::fmax(worst, std::fabs(delta));
            scale = std::fmax(scale, std::fabs((double)reference));
            square += delta * delta;
            elements++;
        }
    }
    const double rms = std::sqrt(square / (double)elements);
    std::printf("%-14s max_abs=%.6g rms=%.6g rel_rms=%.3g differing=%.2f%%\n",
                "bf16i vs fp32", worst, rms, scale > 0.0 ? rms / scale : 0.0,
                100.0 * (double)differing / (double)elements);

    /* The arm as the port runs it (`vae_video::project`): narrow the
     * activation, GEMM on BF16, widen the result. The activation scratch is the
     * port's own size (`sequence * FFN`, its largest activation is `w2`'s), and
     * each `narrowed[]` result is exactly the shape its projection writes, as
     * the port's single result scratch is for the largest of the four. */
    h3d_tensor *activations =
            h3d_alloc(gpu, H3D_BF16, (size_t)rows * FFN, error, sizeof(error));
    std::vector<h3d_tensor *> widened(4);
    for (int index = 0; index < 4; index++) {
        widened[index] = h3d_alloc(gpu, H3D_F32, (size_t)rows * matmuls[index].output_dim,
                                   error, sizeof(error));
        if (!widened[index]) {
            fail("arm output", error);
        }
    }
    if (!activations) {
        fail("arm scratch", error);
    }

    double arm_ms = 0.0;
    for (int call = 0; call <= calls; call++) {
        auto start = std::chrono::steady_clock::now();
        for (int index = 0; index < 4; index++) {
            const Matmul &matmul = matmuls[index];
            const size_t in = (size_t)rows * matmul.input_dim;
            const size_t out = (size_t)rows * matmul.output_dim;
            if (!h3d_cast_f32_to_bf16(gpu, activations, matmul.input, in, error, sizeof(error)) ||
                !h3d_vae_linear_bf16(gpu, narrowed[index], activations,
                                     rounded_weight[index], rows, matmul.input_dim,
                                     matmul.output_dim, error, sizeof(error)) ||
                !h3d_cast_bf16_to_f32(gpu, widened[index], narrowed[index], out, error,
                                      sizeof(error))) {
                fail("arm", error);
            }
        }
        if (!h3d_sync(gpu, error, sizeof(error))) {
            fail("arm sync", error);
        }
        auto finish = std::chrono::steady_clock::now();
        const double this_call =
                std::chrono::duration<double, std::milli>(finish - start).count();
        std::printf("  bf16i arm call %d: %.3f ms%s\n", call, this_call,
                    call == 0 ? " (discarded)" : "");
        if (call > 0) {
            arm_ms += this_call;
        }
    }
    arm_ms /= (double)calls;
    std::printf("rows=%u mode=%-4s ms_per_block=%.3f (the arm: cast, GEMM, cast)\n", rows,
                "bf16i", arm_ms);

    worst = square = scale = 0.0;
    differing = elements = 0;
    for (size_t piece = 0; piece < results[0].size(); piece++) {
        std::vector<float> host((size_t)rows * matmuls[piece].output_dim);
        if (!h3d_read(gpu, host.data(), widened[piece], error, sizeof(error))) {
            fail("arm read", error);
        }
        for (size_t index = 0; index < host.size(); index++) {
            const double delta = (double)host[index] - (double)results[0][piece][index];
            if (host[index] != results[0][piece][index]) differing++;
            worst = std::fmax(worst, std::fabs(delta));
            scale = std::fmax(scale, std::fabs((double)results[0][piece][index]));
            square += delta * delta;
            elements++;
        }
    }
    const double arm_rms = std::sqrt(square / (double)elements);
    std::printf("%-14s max_abs=%.6g rms=%.6g rel_rms=%.3g differing=%.2f%%\n",
                "arm vs fp32", worst, arm_rms, scale > 0.0 ? arm_rms / scale : 0.0,
                100.0 * (double)differing / (double)elements);

    for (int index = 0; index < 4; index++) {
        h3d_free(widened[index]);
        h3d_free(narrowed[index]);
        h3d_free(rounded_weight[index]);
        h3d_free(rounded_input[index]);
    }
    h3d_free(activations);
    for (Matmul &matmul : matmuls) {
        h3d_free(matmul.output);
        h3d_free(matmul.bias);
        h3d_free(matmul.weight);
        h3d_free(matmul.input);
    }
    h3d_destroy(gpu);
    return 0;
}
