/* ds41_forward.h — the V4.1 (ds41) native forward's public surface (P4-4).
 *
 * The score entry is the C-side gate driver: it runs the teacher-forced
 * forward over `ids` and writes the logits file (header {n, vocab} then n rows
 * of f32) plus, for n <= 64 single-block runs, the per-layer dumps
 * <out>.x_Lnn.bin / <out>.y_Lnn.bin (MoE in/out), <out>.hce_Lnn.bin (hc after
 * the engram) and <out>.erows_Lnn.txt (the engram row ids) — the same golden
 * capture protocol the engine's --score-ids uses.
 *
 * Engram rows are host work by design: the port keeps the hash + pread on the
 * Rust host and the native forward consumes a pinned feed (the engine's
 * background pool / io_uring path stays behind; the port's first unit is the
 * eager single-state forward).  raw buffers MUST come from
 * ds4_gpu_host_alloc — the upload is a pinned zero-copy write.
 */
#ifndef DS41_FORWARD_H
#define DS41_FORWARD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS41_MAX_ENGRAM 4

typedef struct {
    /* Per engram layer: pinned host bytes [n][cols][head_dim + head_dim/32]
     * (the raw table row: the e4m3 plane followed by its ue8m0 tail) and the
     * row ids the caller hashed, [n][cols].  rows may be NULL (it only feeds
     * the erows dump). */
    const void *raw[DS41_MAX_ENGRAM];
    const int64_t *rows[DS41_MAX_ENGRAM];
} ds41_engram_feed;

/* Teacher-forced scoring of `ids` through an opened V4.1 engine.  Writes the
 * logits file to out_path and prints the segment PPL; returns 0 on success.
 * no_engram skips the engram layers (the compare-only mode); when the model
 * has engram layers and no feed is supplied, the forward refuses by name
 * rather than silently scoring without them. */
int ds4_v41_score_ids(void *engine, const int *ids, int n_ids, const char *out_path,
                      int no_engram, int chunk, const ds41_engram_feed *feed);

/* Engram table metadata for the feed builder: the engine's C loader filled it
 * from the GGUF, so a caller that hashes + preads rows itself (the Rust host,
 * or this port's C harness) does not re-parse the metadata.  count = how many
 * engram layers the model carries.  k indexes them (k is also the engram
 * index used in ds41_engram_feed).  The path is the GGUF's converter path;
 * a caller with the shards elsewhere keeps the basename and prefixes its own
 * directory (the engine's --engram-dir rule). */
int ds4_v41_engram_count(void *engine);
int ds4_v41_engram_meta(void *engine, int k, uint32_t *il, char *path, int path_cap,
                        uint64_t *rows, uint64_t *weight_off, uint64_t *scale_off,
                        uint32_t *head_dim, uint32_t *cols);

/* Zchain gate support (unit D): the model shape the sidecar headers are
 * validated against (the engine's DS4_N_EXPERT / DS4_N_EMBD) and the
 * router-bias reference for layer il (the engine's model_find_tensor +
 * m->map/m->size, core_v41_amp.c:104).  The C harness builds its loader from
 * these; production loads the same files on the Rust host (sidecar.rs) and
 * calls the two GPU stores. */
int ds4_v41_shape(void *engine, uint32_t *n_layer, uint32_t *n_expert, uint32_t *n_embd);
int ds4_v41_router_bias_ref(void *engine, uint32_t il, const void **map, uint64_t *size, uint64_t *offset);

/* Diagnostic (P4-4 head investigation): the forward's head block dumps the
 * bytes a weight resolve returns (device range or mapped pointer) and the raw
 * mapping bytes for the same span when DS41_DUMP_HEAD=<prefix> is set, so a
 * wrong resolve shows up as a byte diff.  Both write `path` and return 1 on
 * success. */
int ds4_gpu_v41_debug_read_weight(const void *model_map, uint64_t model_size,
                                  uint64_t offset, uint64_t bytes, const char *path);
int ds4_gpu_v41_debug_dump_raw(const void *model_map, uint64_t model_size,
                               uint64_t offset, uint64_t bytes, const char *path);

#ifdef __cplusplus
}
#endif
#endif
