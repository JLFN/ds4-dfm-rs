/* tests/test_ds41_generate.cu — the greedy-generate gate harness (unit E, E0).
 *
 * Runs the port's generate entry (ds4_v41_generate_argmax, ds4_ds41_forward.inc)
 * over a golden prompt's exact ids and compares the produced token ids against
 * the engine's own --emit-trace capture:
 *
 *   test_ds41_generate <gguf> <ids_file> <engine_log> <n_predict>
 *                      [--engram-dir <shard dir>] [--no-engram]
 *
 * The engine log is the stderr of
 *   ds4 --cuda -m <gguf> --engram-dir <dir> --gen-ids <ids_file> -n N
 *       --temp 0 --no-dspark --emit-trace
 * whose `[emit] <absolute position> <token id>` lines are the golden sequence
 * (core_v41_api.c:246/387; the ids are the exact instrument — the printed text
 * loses the token boundaries).  The port must reproduce the sequence token for
 * token, positions included (a position mismatch = the state advanced wrong).
 *
 * The engine's generate path has NO no-engram switch (only the score path
 * takes --v41-no-engram), so the gate runs with the engram layers live: the
 * harness hashes each step's rows itself — the C mirror of the Rust host's
 * engram.rs (itself verified against the golden erows), fed through the
 * ds41_engram_feed the port's forward consumes.  --no-engram is a debug-only
 * port-side mode (no engine golden exists for it). */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>

extern "C" {
#include "../ds4.h"
#include "../ds4_gpu.h"
#include "../ds41_forward.h"
}

/* ---- shared helpers (the score harness's, kept local) ---- */

static bool parse_ids(const char *path, std::vector<int> *ids) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "harness: cannot read %s\n", path); return false; }
    long v;
    while (fscanf(f, "%ld", &v) == 1) ids->push_back((int)v);
    fclose(f);
    return !ids->empty();
}

static bool parse_engine_emit(const char *path, std::vector<int> *pos, std::vector<int> *ids) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "harness: cannot read %s\n", path); return false; }
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        /* the CLI's token text goes to stdout and [emit] to stderr, both into
         * one file: "[emit] 8 455\nThe[emit] 9 8397" — find the marker anywhere */
        const char *p = strstr(line, "[emit]");
        unsigned pp; int t;
        if (p && sscanf(p, "[emit] %u %d", &pp, &t) == 2) {
            pos->push_back((int)pp);
            ids->push_back(t);
        }
    }
    fclose(f);
    return !ids->empty();
}

static bool pread_exact(int fd, void *dst, size_t len, uint64_t off) {
    uint8_t *p = (uint8_t *)dst; size_t left = len; uint64_t o = off;
    while (left) {
        const ssize_t r = pread(fd, p, left, (off_t)o);
        if (r <= 0) return false;
        p += r; left -= (size_t)r; o += (uint64_t)r;
    }
    return true;
}

/* ---- the live engram feed (unit E) ----
 * The engine hashes rows per step inside its forward; the port consumes a
 * caller-built feed, so the harness owns the hash here — a line-for-line C
 * mirror of crates/ds4-core/src/engram.rs rows(), which mirrors the engine's
 * v41_engram_hash (core_v41_engram.c:94-116) and was verified against the
 * golden erows in P2/P4-3.  The table layouts are the engine's (layer-major),
 * NOT the row-major order the GGUF dims suggest: reading them by dims
 * silently produces different rows, not an error. */

struct engram_layer {
    uint32_t il, cols, hd, stride;
    uint64_t nrows, woff, soff;
    int fd;
    void *raw;      /* pinned [np][cols][stride]; row 0 is refreshed per step */
};

struct gen_feed {
    ds41_engram_hash_ref ref;
    std::vector<int32_t> token_map;
    std::vector<int64_t> mult, prim, offs;
    std::vector<int32_t> hist;      /* absolute position -> token id */
    std::vector<engram_layer> layers;
    uint32_t np;
    uint32_t emitted;
    int32_t eos;
    std::vector<int> *ids;          /* emitted ids for the comparison */
};

/* The model map is the host mmap (registered for device access); the same
 * cudaMemcpyDefault read the engine's loader uses handles either side. */
static bool read_map(const void *map, uint64_t off, uint64_t bytes, void *dst) {
    return cudaMemcpy(dst, (const char *)map + off, (size_t)bytes, cudaMemcpyDefault) == cudaSuccess;
}

static bool hash_tables(gen_feed *g) {
    const ds41_engram_hash_ref *r = &g->ref;
    g->token_map.resize(r->token_map_bytes / 4);
    g->mult.resize(r->mult_bytes / 8);
    g->prim.resize(r->prim_bytes / 8);
    g->offs.resize(r->off_bytes / 8);
    if (!read_map(r->map, r->token_map_off, r->token_map_bytes, g->token_map.data()) ||
        !read_map(r->map, r->mult_off, r->mult_bytes, g->mult.data()) ||
        !read_map(r->map, r->prim_off, r->prim_bytes, g->prim.data()) ||
        !read_map(r->map, r->off_off, r->off_bytes, g->offs.data())) return false;
    return true;
}

/* The rows for position `p` of engram layer index `ei` (engram.rs rows()). */
static void ehash_rows(const gen_feed *g, int64_t p, uint32_t ei, std::vector<int64_t> *rows) {
    const ds41_engram_hash_ref *r = &g->ref;
    const int ng = (int)r->max_ngram, nh = (int)r->heads;
    int64_t prod[16];
    for (int k = 0; k < ng; k++) {
        const int64_t back = p - k;
        int64_t cid;
        if (back < 0) {
            cid = r->pad;
        } else {
            const int32_t tok = g->hist[(size_t)back];
            cid = (tok >= 0 && (uint32_t)tok < r->n_vocab) ? (int64_t)g->token_map[(size_t)tok] : r->pad;
        }
        const int64_t m = g->mult[(size_t)ei * ng + k];
        prod[k] = (int64_t)((uint64_t)cid * (uint64_t)m);   /* the engine's wrap-around multiply */
    }
    rows->clear();
    int64_t rolling = prod[0];
    for (int i = 1; i < ng; i++) {
        rolling ^= prod[i];
        for (int head = 0; head < nh; head++) {
            const int64_t pr = g->prim[((size_t)ei * (ng - 1) + (i - 1)) * nh + head];
            int64_t rr = rolling % pr;
            if (rr < 0) rr += pr;
            rows->push_back(rr + g->offs[(size_t)ei * (ng - 1) * nh + (size_t)(i - 1) * nh + head]);
        }
    }
}

static bool feed_fill(gen_feed *g, uint32_t slot, int64_t pos) {
    std::vector<int64_t> rows;
    for (uint32_t k = 0; k < g->layers.size(); k++) {
        engram_layer &L = g->layers[k];
        ehash_rows(g, pos, k, &rows);
        if (rows.size() != L.cols) { fprintf(stderr, "harness: hash cols %zu != meta %u\n", rows.size(), L.cols); return false; }
        uint8_t *dst = (uint8_t *)L.raw + (size_t)slot * L.cols * L.stride;
        for (uint32_t c = 0; c < L.cols; c++) {
            const int64_t r = rows[c];
            if (r < 0 || (uint64_t)r >= L.nrows) { fprintf(stderr, "harness: engram row %lld out of range\n", (long long)r); return false; }
            uint8_t *d = dst + (size_t)c * L.stride;
            if (!pread_exact(L.fd, d, L.hd, L.woff + (uint64_t)r * L.hd) ||
                !pread_exact(L.fd, d + L.hd, L.hd / 32u, L.soff + (uint64_t)r * (L.hd / 32u))) return false;
        }
    }
    return true;
}

/* FNV-1a 64 over a byte range (the digest both sides print for the payload
 * check: DS41_FEED_DIGEST). */
static uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

static FILE *feed_digest_file(void) {
    static FILE *f = NULL;
    static int tried = 0;
    if (!tried) { tried = 1; const char *p = getenv("DS41_FEED_DIGEST"); if (p) f = fopen(p, "a"); }
    return f;
}

/* The emit hook: record the token, then refresh every engram layer's row 0
 * with the rows for this token's position — the forward's next step (n=1)
 * reads exactly that slot. */
static int emit_feed(int token, void *ud) {
    gen_feed *g = (gen_feed *)ud;
    const int64_t pos = (int64_t)g->np + g->emitted;
    g->hist[(size_t)pos] = token;
    if (!g->layers.empty() && !feed_fill(g, 0, pos)) { fprintf(stderr, "harness: feed refresh failed at pos %lld\n", (long long)pos); return 1; }
    FILE *df = feed_digest_file();
    if (df) {
        for (uint32_t k = 0; k < g->layers.size(); k++) {
            const engram_layer &L = g->layers[k];
            fprintf(df, "host pos %lld k %u fnv %016llx\n", (long long)pos, k, (unsigned long long)fnv1a(L.raw, (size_t)L.cols * L.stride));
        }
        fflush(df);
    }
    g->ids->push_back(token);
    g->emitted++;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <gguf> <ids_file> <engine_log> <n_predict> [--engram-dir <dir>] [--no-engram]\n", argv[0]);
        return 2;
    }
    const char *gguf = argv[1], *ids_path = argv[2], *englog = argv[3];
    const int n_predict = atoi(argv[4]);
    const char *engram_dir = NULL;
    int no_engram = 0;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--engram-dir") == 0 && i + 1 < argc) { engram_dir = argv[++i]; continue; }
        if (strcmp(argv[i], "--no-engram") == 0) { no_engram = 1; continue; }
        fprintf(stderr, "harness: unknown argument %s\n", argv[i]);
        return 2;
    }

    std::vector<int> prompt;
    if (!parse_ids(ids_path, &prompt)) return 2;
    std::vector<int> gpos, gids;
    if (!parse_engine_emit(englog, &gpos, &gids)) {
        fprintf(stderr, "harness: no [emit] lines in %s (was the engine run with --emit-trace?)\n", englog);
        return 2;
    }
    printf("engine golden: %zu emitted tokens (positions %d..%d)\n", gids.size(),
           gpos.empty() ? -1 : gpos.front(), gpos.empty() ? -1 : gpos.back());

    ds4_engine_options opt;
    memset(&opt, 0, sizeof opt);
    opt.model_path = gguf;
    opt.backend = DS4_BACKEND_CUDA;
    opt.defer_boot_prewarm = true;   /* the V4.1 boot prewarm has no session graph in the port yet */
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) { fprintf(stderr, "harness: engine open failed\n"); return 1; }

    /* The engram feed: the tables from the model map, the rows hashed per
     * position, the raw bytes pread from the shards (the score harness's
     * --engram-dir rule: keep the metadata basename, prefix the directory). */
    gen_feed g;
    memset(&g.ref, 0, sizeof g.ref);
    g.np = (uint32_t)prompt.size();
    /* THE E0 gate bug: the emit hook's row position is np + emitted, and
     * emitted was never initialized — uninitialized stack garbage that is
     * deterministic per binary (the 485e9c9 harness diverged at the first
     * step, the b36a865 one passed; identical addresses, identical payload
     * digests, so the garbage was invisible to every instrument).  Zero it,
     * and keep DS41_EMIT_OFFSET as the deliberate reproduction: any nonzero
     * offset shifts every live-hashed engram row the same way the garbage did. */
    g.emitted = 0;
    g.eos = 0;
    g.ids = NULL;
    if (const char *eo = getenv("DS41_EMIT_OFFSET")) {
        g.emitted = (uint32_t)atoi(eo);
        printf("emit offset: %u (deliberate)\n", g.emitted);
    }
    const int n_eng = ds4_v41_engram_count(e);
    if (!no_engram && n_eng > 0) {
        if (!engram_dir) { fprintf(stderr, "harness: the model has engram layers; pass --engram-dir <dir> (or --no-engram for the port-only debug mode)\n"); return 2; }
        if (!ds4_v41_engram_hash_ref(e, &g.ref) || !hash_tables(&g)) { fprintf(stderr, "harness: engram hash tables unavailable\n"); return 1; }
        g.hist.assign((size_t)g.np + (size_t)n_predict + 8u, 0);
        for (uint32_t i = 0; i < g.np; i++) g.hist[i] = prompt[i];
        for (int k = 0; k < n_eng; k++) {
            engram_layer L;
            char tpath[4096];
            uint64_t nrows = 0, woff = 0, soff = 0;
            uint32_t hd = 0, cols = 0;
            if (!ds4_v41_engram_meta(e, k, &L.il, tpath, sizeof tpath, &nrows, &woff, &soff, &hd, &cols)) { fprintf(stderr, "harness: engram meta %d failed\n", k); return 1; }
            std::string shard = tpath;
            const size_t slash = shard.find_last_of('/');
            shard = std::string(engram_dir) + "/" + (slash == std::string::npos ? shard : shard.substr(slash + 1));
            L.cols = cols; L.hd = hd; L.stride = hd + hd / 32u; L.nrows = nrows; L.woff = woff; L.soff = soff;
            L.fd = open(shard.c_str(), O_RDONLY);
            if (L.fd < 0) { fprintf(stderr, "harness: cannot open engram shard %s\n", shard.c_str()); return 1; }
            L.raw = ds4_gpu_host_alloc((uint64_t)g.np * cols * L.stride);
            if (!L.raw) { fprintf(stderr, "harness: pinned feed allocation failed\n"); return 1; }
            if (getenv("DS41_ADDR_DUMP")) {   /* the A/B layout discriminator: identical addresses rule the layout class out */
                void *dva = NULL;
                (void)cudaHostGetDevicePointer(&dva, L.raw, 0);
                printf("addr feed k %u host %p dev %p\n", k, L.raw, dva);
            }
            g.layers.push_back(L);
        }
        for (uint32_t i = 0; i < g.np; i++) if (!feed_fill(&g, i, (int64_t)i)) { fprintf(stderr, "harness: prompt feed fill failed at %u\n", i); return 1; }
        printf("engram feed: %zu layers, cols %u, head_dim %u, %u prompt rows hashed\n",
               g.layers.size(), g.layers.empty() ? 0 : g.layers[0].cols, g.layers.empty() ? 0 : g.layers[0].hd, g.np);
        {   /* host-side digest of the prompt region the prefill will upload (dev pos0 0 n np) */
            FILE *df = feed_digest_file();
            if (df) {
                for (uint32_t k = 0; k < g.layers.size(); k++) {
                    const engram_layer &L = g.layers[k];
                    fprintf(df, "host prefill k %u n %u fnv %016llx\n", k, g.np,
                            (unsigned long long)fnv1a(L.raw, (size_t)g.np * L.cols * L.stride));
                }
                fflush(df);
            }
        }
        /* Self-check: the engine's own captured rows for this prompt (the
         * golden erows next to the ids file, written by the score capture)
         * must equal what this harness's hash produces — the P2 instrument,
         * re-run here so a hash bug can never masquerade as a port bug.
         * DS41_SKIP_SELFCHECK=1 disables it (the A/B discrimination test:
         * binary A differed from B only by this block). */
        const int skip_sc = getenv("DS41_SKIP_SELFCHECK") != NULL;
        for (uint32_t k = 0; !skip_sc && k < g.layers.size(); k++) {
            std::string ids_path_s = ids_path;
            const size_t dot = ids_path_s.rfind(".ids");
            const std::string base = (dot == std::string::npos) ? ids_path_s : ids_path_s.substr(0, dot);
            char ep[4600];
            snprintf(ep, sizeof ep, "%s.logits.bin.erows_L%02u.txt", base.c_str(), g.layers[k].il);
            FILE *ef = fopen(ep, "r");
            if (!ef) { printf("hash self-check L%02u: %s absent, skipped\n", g.layers[k].il, ep); continue; }
            std::vector<int64_t> erows;
            long long v;
            while (fscanf(ef, "%lld", &v) == 1) erows.push_back(v);
            fclose(ef);
            size_t bad = 0;
            const size_t want = (size_t)g.np * g.layers[k].cols;
            if (erows.size() != want) { printf("hash self-check L%02u: golden %zu rows != %zu expected\n", g.layers[k].il, erows.size(), want); continue; }
            std::vector<int64_t> rows;
            for (uint32_t i = 0; i < g.np; i++) {
                ehash_rows(&g, (int64_t)i, k, &rows);
                for (uint32_t c = 0; c < g.layers[k].cols; c++) if (rows[c] != erows[(size_t)i * g.layers[k].cols + c]) bad++;
            }
            printf("hash self-check L%02u: %zu/%zu rows differ vs the golden erows\n", g.layers[k].il, bad, want);
        }
    }

    std::vector<int> oids;
    g.ids = &oids;
    ds41_engram_feed feed;
    memset(&feed, 0, sizeof feed);
    for (uint32_t k = 0; k < g.layers.size(); k++) feed.raw[k] = g.layers[k].raw;   /* k == the engram index (ds41_forward.h) */
    /* DS41_GEN_DELAY_MS=<n>: a bare delay before the generate call (the
     * timing-vs-content discriminator; no reads, no allocations). */
    if (const char *dm = getenv("DS41_GEN_DELAY_MS")) {
        const int ms = atoi(dm);
        if (ms > 0) {
            struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
            nanosleep(&ts, NULL);
            printf("pre-generate delay: %d ms\n", ms);
        }
    }
    const int rc = ds4_v41_generate_argmax(e, prompt.data(), (int)prompt.size(), n_predict,
                                           no_engram || n_eng == 0 ? 1 : 0,
                                           g.layers.empty() ? NULL : &feed, emit_feed, &g);
    if (rc != 0) { fprintf(stderr, "harness: generate entry failed\n"); return 1; }

    size_t diffs = 0, first = SIZE_MAX;
    const size_t n = gids.size() < oids.size() ? gids.size() : oids.size();
    for (size_t i = 0; i < n; i++) {
        if (gids[i] != oids[i]) { diffs++; if (first == SIZE_MAX) first = i; }
    }
    int pos_bad = 0;   /* the engine's own accounting: the first emitted token sits at position np */
    for (size_t i = 0; i < gids.size(); i++) if (gpos[i] != (int)prompt.size() + (int)i) pos_bad++;
    printf("port: %zu emitted tokens; ids match %zu/%zu, first divergence %s; engine positions contiguous from %zu: %s\n",
           oids.size(), n - diffs, n, first == SIZE_MAX ? "none" : std::to_string((long)first).c_str(),
           prompt.size(), pos_bad ? "NO" : "yes");
    if (first != SIZE_MAX) {
        const size_t lo = first >= 4 ? first - 4 : 0;
        for (size_t i = lo; i < n && i < first + 4; i++)
            printf("  %5zu  engine %d  port %d%s\n", i, gids[i], oids[i], i == first ? "  <-- first" : "");
    }
    printf("engine ids:");
    for (size_t i = 0; i < gids.size(); i++) printf(" %d", gids[i]);
    printf("\nport   ids:");
    for (size_t i = 0; i < oids.size(); i++) printf(" %d", oids[i]);
    printf("\n");
    const bool pass = (diffs == 0) && (gids.size() == oids.size()) && !pos_bad;
    printf("DS41 generate gate: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
