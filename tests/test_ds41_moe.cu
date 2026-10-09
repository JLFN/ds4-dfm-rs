/* test_ds41_moe.cu — P4-2 gate: the tensor-level V4.1 routed-MoE entry
 * (ds4_gpu_v41_routed_moe_tensor) against the Rust emulation
 * (crates/ds4-core/examples/ds41_moe_ref.rs).
 *
 *   test_ds41_moe <moe.blob> <moe.cases.txt> <moe.ref.f32>
 *
 * The fixture (tests/fixtures/ds41/vq/gen_moe.py) carries two case kinds:
 *   onehot — K=1, w=[1.0], a one-hot activation, and a probe expert whose
 *            down matrix decodes to a single nonzero column, so every dot in
 *            the chain is a single-term sum: the entry must be BIT-EXACT
 *            against the emulation.
 *   random — dense x, K=6 across experts: the f32 accumulation order (warp
 *            shuffles, FMA contraction) differs from the emulation's, so the
 *            compare is within TOL_REL, reported per case.
 *
 * The blob header is read host-side by the entry itself (ver/nc), and the
 * device pointer comes from the native range resolver — this test drives the
 * same path the forward will.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

#define IN_DIM 2048u
#define MID_DIM 2048u
#define OUT_DIM 512u
#define NEXP 2u
#define TOL_REL 1e-3f   /* measured far below; see the P4-2 evidence */

typedef struct {
    char kind[16];
    uint32_t k;
    int32_t *sel;
    float *w;
    float *x;
} moe_case;

static void *xread(const char *path, uint64_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "test_ds41_moe: cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "test_ds41_moe: read %s\n", path); exit(2); }
    fclose(f);
    *size = (uint64_t)n;
    return buf;
}

static uint32_t hex32(const char *s) { return (uint32_t)strtoul(s, NULL, 16); }

static moe_case *parse_cases(const char *path, uint32_t *n_cases) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "test_ds41_moe: cannot open %s\n", path); exit(2); }
    moe_case *cases = NULL;
    uint32_t n = 0, cap = 0;
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (strncmp(line, "case ", 5) == 0) {
            if (n == cap) { cap = cap ? cap * 2u : 8u; cases = (moe_case *)realloc(cases, cap * sizeof *cases); }
            moe_case *c = &cases[n++];
            memset(c, 0, sizeof *c);
            unsigned idx = 0, k = 0;
            if (sscanf(line, "case %u %15s K=%u", &idx, c->kind, &k) != 3) { fprintf(stderr, "test_ds41_moe: bad case line\n"); exit(2); }
            c->k = k;
            c->sel = (int32_t *)malloc(k * 4);
            c->w = (float *)malloc(k * 4);
            c->x = (float *)malloc(IN_DIM * 4);
            continue;
        }
        moe_case *c = &cases[n - 1];
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        if (strcmp(tok, "sel") == 0) {
            for (uint32_t i = 0; i < c->k && (tok = strtok(NULL, " \t\n")); i++) c->sel[i] = (int32_t)strtol(tok, NULL, 10);
        } else if (strcmp(tok, "w") == 0) {
            for (uint32_t i = 0; i < c->k && (tok = strtok(NULL, " \t\n")); i++) { uint32_t b = hex32(tok); memcpy(&c->w[i], &b, 4); }
        } else if (strcmp(tok, "x") == 0) {
            for (uint32_t i = 0; i < IN_DIM && (tok = strtok(NULL, " \t\n")); i++) { uint32_t b = hex32(tok); memcpy(&c->x[i], &b, 4); }
        }
    }
    fclose(f);
    *n_cases = n;
    return cases;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <moe.blob> <moe.cases.txt> <moe.ref.f32>\n", argv[0]);
        return 2;
    }
    uint64_t blob_bytes = 0, ref_bytes = 0;
    void *blob = xread(argv[1], &blob_bytes);
    float *ref = (float *)xread(argv[3], &ref_bytes);
    uint32_t n_cases = 0;
    moe_case *cases = parse_cases(argv[2], &n_cases);
    if (ref_bytes != (uint64_t)n_cases * OUT_DIM * 4) {
        fprintf(stderr, "test_ds41_moe: ref has %llu bytes, expected %u\n",
                (unsigned long long)ref_bytes, n_cases * OUT_DIM * 4u);
        return 2;
    }

    ds4_gpu_tensor *x_t = ds4_gpu_tensor_alloc(IN_DIM * 4);
    ds4_gpu_tensor *w_t = ds4_gpu_tensor_alloc(64 * 4);      /* K <= 64 in this fixture */
    ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc(64 * 4);
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(OUT_DIM * 4);
    if (!x_t || !w_t || !sel_t || !out_t) { fprintf(stderr, "test_ds41_moe: tensor alloc\n"); return 2; }

    int n_fail = 0;
    float worst_rel = 0.0f;
    for (uint32_t ci = 0; ci < n_cases; ci++) {
        moe_case *c = &cases[ci];
        if (c->k > 64u) { fprintf(stderr, "test_ds41_moe: K %u over the fixture bound\n", c->k); return 2; }
        ds4_gpu_tensor_write(x_t, 0, c->x, IN_DIM * 4);
        ds4_gpu_tensor_write(w_t, 0, c->w, c->k * 4);
        ds4_gpu_tensor_write(sel_t, 0, c->sel, c->k * 4);
        const int rc = ds4_gpu_v41_routed_moe_tensor(out_t, blob, blob_bytes, 0, blob_bytes,
                                                     IN_DIM, MID_DIM, OUT_DIM, sel_t, w_t, NEXP, c->k,
                                                     10.0f, x_t, 0, 1);
        if (!rc) { fprintf(stderr, "test_ds41_moe: case %u: entry refused\n", ci); n_fail++; continue; }
        float got[OUT_DIM];
        ds4_gpu_tensor_read(out_t, 0, got, OUT_DIM * 4);
        const float *want = ref + (uint64_t)ci * OUT_DIM;

        int bit_exact = 1, n_diff = 0;
        float max_rel = 0.0f, max_abs = 0.0f;
        for (uint32_t o = 0; o < OUT_DIM; o++) {
            uint32_t a, b;
            memcpy(&a, &got[o], 4);
            memcpy(&b, &want[o], 4);
            if (a == b) continue;
            bit_exact = 0;
            n_diff++;
            const float absd = fabsf(got[o] - want[o]);
            if (absd > max_abs) max_abs = absd;
            /* Relative only where the reference is significant; a near-zero
             * denominator would report noise as a huge ratio. */
            if (fabsf(want[o]) >= 1e-3f) {
                const float rel = absd / fabsf(want[o]);
                if (rel > max_rel) max_rel = rel;
            }
        }
        if (max_rel > worst_rel) worst_rel = max_rel;
        const int onehot = strcmp(c->kind, "onehot") == 0;
        const int pass = onehot ? bit_exact : (max_rel <= TOL_REL);
        if (!pass && getenv("DS41_MOE_DEBUG")) {
            for (uint32_t o = 0; o < 8; o++)
                printf("  dbg[%u] got %.6g want %.6g\n", o, (double)got[o], (double)want[o]);
        }
        if (!pass) n_fail++;
        printf("case %u %-6s K=%u: %s%s (max rel %.3e max abs %.3e diffs %d)\n",
               ci, c->kind, c->k,
               pass ? "PASS" : "FAIL",
               onehot ? " bit-exact" : "",
               max_rel, max_abs, n_diff);
    }
    printf("DS41 MoE entry: %s (%u cases, worst rel %.3e, tol %.0e)\n",
           n_fail ? "FAIL" : "PASS", n_cases, worst_rel, (double)TOL_REL);
    return n_fail ? 1 : 0;
}
