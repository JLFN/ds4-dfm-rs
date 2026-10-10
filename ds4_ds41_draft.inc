/* ds4_ds41_draft.inc — DeepSeek V4.1 (ds41) DSpark draft towers, host side.
 * Ported from the C engine at /data/YoungAi (commit 3946dbc), files
 * src/core/core_v41_draft.c (every function cites its engine line) and the
 * window/attention halves in src/core/core_v41_attn.c:225-286.  Included by
 * ds4.c right after ds4_ds41_forward.inc (same TU: it calls the forward's
 * statics).
 *
 * One draft round: draft a block of `block` positions in one forward (three
 * towers, 128 experts top-3, window attention only), one main-model verify of
 * 1+k positions, accept the longest matching prefix, roll the rest back — the
 * engine's rule is that temperature-0 greedy output is byte-identical with
 * drafting on and off (pure time saving, not approximation).
 *
 *   main_x      = main_norm(main_proj(the main model's target-layer attention
 *                 inputs, captured by v41_layer into the main-hidden ring))
 *   block input = [last real token, noise x (B-1)] embeddings (noise id from
 *                 the metadata)
 *   towers      = the same v41_layer body with the tower weights (draft=1)
 *   per position= head(out_norm(h)) row i + markov_head(token_i) bias ->
 *                 argmax -> token_{i+1}; confidence = proj([h_i; markov_embed_i])
 *
 * Failure modes (the engine's, kept): 1. capturing main_hidden at the layer
 * OUTPUT instead of the attention input does not error, acceptance drops to
 * about 1; 2. letting the draft block's KV into the history window does not
 * error, acceptance rots over the session; 3. a GGUF without the draft
 * parameters leaves ready=0 and the caller decodes one token at a time.
 *
 * Not ported here (named): the per-round CUDA graph (core_v41_draft.c:326-360
 * — the port's eager path IS the engine's direct dispatch, E3 adds the graph),
 * the amp/distillation sidecars (core_v41_draft_amp.c), the confidence
 * scheduler's cost accounting (core_draft_sched.c, E2), --dspark-block, and
 * the sampling draft arm (dev_sample).  The dense per-expert MoE exists for
 * the per-expert artifact form; this artifact's towers carry VQ blobs. */
#ifndef DS4_DS41_DRAFT_INC
#define DS4_DS41_DRAFT_INC

#define DS4_V41_MKCACHE_SLOTS 256u   /* core_v41.h:168 (64 slots 50% hit / 256 slots 59%) */

typedef struct {
    ds4_v41_state st;          /* the towers' own buffers (draft=1, layer id = tower index) */
    ds4_gpu_tensor *mainx_raw; /* main_proj output (pre-norm) */
    ds4_gpu_tensor *mk_embed;  /* [cap][rank] per-position markov embed (confidence reads it) */
    ds4_gpu_tensor *mk_cur;    /* [rank] this position's row into markov_head */
    ds4_gpu_tensor *mk_bias;   /* [V] this position's logit bias */
    ds4_gpu_tensor *mk_cache, *mk_cache_ids, *mk_cache_next, *mk_hit;   /* [slots][V] f32 / [slots] id (-1 empty) / [1] rotation / [1] hit */
    ds4_gpu_tensor *conf_in, *conf;   /* [cap][E+rank] / [cap] */
    ds4_gpu_tensor *ids, *ids_next;   /* i32 [cap]: [0] = the real token, [1..block] = drafts; argmax landing slot */
    ds4_gpu_tensor *h;         /* [cap][E] the head's hc_pre output (confidence reads it too) */
    uint64_t *exp_off[DS4_MTP_MAX_TOWERS];   /* per-expert form: [3][n_expert] offsets; NULL for blobs */
    ds4_gpu_tensor *mainh_lin; /* [SWA][n_target][E] contiguous rows out of the main ring (main_proj's batch) */
    int64_t win_end;           /* newest absolute position in the tower windows (-1 = empty); the fill tops up (win_end, pos_main] */
    uint32_t block, ready;
    int32_t host_ids[DS4_MTP_MAX_BLOCK + 1];
    float host_conf[DS4_MTP_MAX_BLOCK];
    int32_t *p_tok, *p_bpos, *p_wpos, *p_first, *p_ids;   /* pinned slots the puts upload from */
    float *p_onehot, *p_conf;
    ds4_gpu_tensor *firstd;    /* device int: the ring-rows start (graph path; the eager path passes the value directly) */
    int cap_mode;              /* capture mode: 1 changes the puts to zero-copy kernels (always 0 until E3) */
    uint32_t rounds;           /* draft rounds produced */
} ds4_v41_draft;

/* core_v41_draft.c:25-33 — f32 vs bf16 BY THE REGISTERED GGUF TYPE, never
 * assumed.  The 2026-09-15 crash: the converter's plan_small stores
 * gate/markov/confidence as f32 while the main path's same-named matrices are
 * bf16; reading f32 bytes as bf16 does not error, it feeds garbage (acceptance
 * 0.14/5). */
static bool v41_small_matmul(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
                             uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n) {
    if (w->type == DS4_TENSOR_BF16)
        return ds4_gpu_v41_matmul_bf16_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n) != 0;
    if (w->type == DS4_TENSOR_F32)
        return ds4_gpu_v41_matmul_f32_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n) != 0;
    fprintf(stderr, "ds4: [v41] drafter: %.*s type %u is neither f32 nor bf16\n", (int)w->name.len, w->name.ptr, w->type);
    return false;
}

/* core_v41_draft.c:36-57 — the per-tower expert offset table [tower][3 x
 * n_expert], built once at bind time.  Why not in the kernel: a name lookup +
 * 128 range parses per layer is pure host waste. */
static bool v41_draft_exp_off(ds4_engine *e, ds4_v41_draft *dr) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    for (uint32_t T = 0; T < v->mtp_towers; T++) {
        /* Blob towers (the 100 GB recipe): no table; the state carries the blob
         * and the MoE takes the main fused kernel (v41_moe). */
        if (e->weights.v41_towers.exps_vq[T]) {
            dr->st.tower_exps_vq[T] = e->weights.v41_towers.exps_vq[T];
            dr->exp_off[T] = NULL; dr->st.tower_exp_off[T] = NULL;
            continue;
        }
        uint64_t *off = xmalloc((size_t)3u * v->mtp_experts * 8u);
        for (uint32_t i = 0; i < v->mtp_experts; i++) {
            const ds4_tensor *g = e->weights.v41_towers.exp_gate[T][i];
            const ds4_tensor *u = e->weights.v41_towers.exp_up[T][i];
            const ds4_tensor *d = e->weights.v41_towers.exp_down[T][i];
            if (!g || !u || !d) { free(off); fprintf(stderr, "ds4: [v41] draft tower %u expert %u tensors missing\n", T, i); return false; }
            off[i] = g->abs_offset; off[v->mtp_experts + i] = u->abs_offset; off[2u * v->mtp_experts + i] = d->abs_offset;
        }
        dr->exp_off[T] = off;
        dr->st.tower_exp_off[T] = off;
    }
    return true;
}

/* core_v41_draft.c:103-150 — the draft state's buffer list, minus amp /
 * distillation / graph tensors (named above). */
static bool v41_draft_state_alloc(ds4_v41_draft *dr, uint32_t cap, uint32_t towers) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint32_t FF = DS4_N_FF_EXP, R = v->mtp_markov_rank;
    const uint32_t pcap = cap > SWA ? cap : SWA;
    const uint64_t low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, mix = 2u * HC + HC * HC;
    ds4_v41_state *st = &dr->st;
    bool ok = true;
    st->draft = 1; st->cap_tok = cap; st->ctx = cap; st->idx_owner = -1; st->cand_owner = -1;
    st->tok = v41_alloc(cap * 4, &ok);          st->pos = v41_alloc((uint64_t)pcap * 4, &ok);
    st->hc = v41_alloc((uint64_t)cap * HC * E * 4, &ok);   st->hc2 = v41_alloc((uint64_t)cap * HC * E * 4, &ok);
    st->mix = v41_alloc((uint64_t)cap * mix * 4, &ok);      st->pre = v41_alloc((uint64_t)cap * HC * 4, &ok);
    st->post = v41_alloc((uint64_t)cap * HC * 4, &ok);      st->comb = v41_alloc((uint64_t)cap * HC * HC * 4, &ok);
    st->pre_mix = v41_alloc((uint64_t)cap * HC * 4, &ok);
    st->x = v41_alloc((uint64_t)cap * E * 4, &ok);          st->xn = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->qr = v41_alloc((uint64_t)cap * Q * 4, &ok);         st->qrn = v41_alloc((uint64_t)cap * Q * 4, &ok);
    st->q = v41_alloc((uint64_t)cap * NH * HD * 4, &ok);
    st->kv = v41_alloc((uint64_t)pcap * HD * 4, &ok);       st->kvn = v41_alloc((uint64_t)pcap * HD * 4, &ok);
    st->wintmp = v41_alloc((uint64_t)SWA * HD * 4, &ok);
    st->o = v41_alloc((uint64_t)cap * NH * HD * 4, &ok);    st->low = v41_alloc((uint64_t)cap * low * 4, &ok);
    st->attn_out = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->glog = v41_alloc((uint64_t)cap * v->mtp_experts * 4, &ok);
    st->sel = v41_alloc((uint64_t)cap * v->mtp_used * 4, &ok);
    st->rw = v41_alloc((uint64_t)cap * v->mtp_used * 4, &ok);
    st->routed = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->sg = v41_alloc((uint64_t)cap * FF * 4, &ok);        st->su = v41_alloc((uint64_t)cap * FF * 4, &ok);
    st->sh = v41_alloc((uint64_t)cap * FF * 4, &ok);        st->so = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->y = v41_alloc((uint64_t)cap * E * 4, &ok);
    st->logits = v41_alloc((uint64_t)cap * DS4_N_VOCAB * 4, &ok);
    st->main_x = v41_alloc((uint64_t)pcap * E * 4, &ok);
    /* The tower windows must hold one full window top-up (up to SWA rows land
     * at once after a prefill); sizing them block+1 was the 2026-09-18 crash
     * that silently disabled speculation (the fill failed quietly and "spec ==
     * pure decode" stayed green because no round ever drafted). */
    for (uint32_t T = 0; T < towers; T++) st->win[T] = v41_alloc((uint64_t)(SWA + pcap) * HD * 4, &ok);
    dr->mainx_raw = v41_alloc((uint64_t)pcap * E * 4, &ok);
    dr->mk_embed = v41_alloc((uint64_t)cap * R * 4, &ok);
    dr->mk_cur = v41_alloc((uint64_t)R * 4, &ok);
    dr->mk_bias = v41_alloc((uint64_t)DS4_N_VOCAB * 4, &ok);
    dr->mk_cache = v41_alloc((uint64_t)DS4_V41_MKCACHE_SLOTS * DS4_N_VOCAB * 4, &ok);
    dr->mk_cache_ids = v41_alloc((uint64_t)DS4_V41_MKCACHE_SLOTS * 4, &ok);
    dr->mk_cache_next = v41_alloc(16, &ok);
    dr->mk_hit = v41_alloc(16, &ok);
    if (ok) {
        int32_t neg[DS4_V41_MKCACHE_SLOTS];
        for (uint32_t i = 0; i < DS4_V41_MKCACHE_SLOTS; i++) neg[i] = -1;
        const uint32_t zero4[4] = { 0u, 0u, 0u, 0u };
        if (!ds4_gpu_tensor_write(dr->mk_cache_ids, 0, neg, sizeof neg) || !ds4_gpu_tensor_write(dr->mk_cache_next, 0, zero4, 16)) ok = false;
    }
    dr->conf_in = v41_alloc((uint64_t)cap * (E + R) * 4, &ok);
    dr->conf = v41_alloc((uint64_t)cap * 4, &ok);
    dr->ids = v41_alloc((uint64_t)cap * 4, &ok);
    dr->ids_next = v41_alloc(16, &ok);
    dr->h = v41_alloc((uint64_t)cap * E * 4, &ok);
    dr->mainh_lin = v41_alloc((uint64_t)SWA * v->n_mtp_target * E * 4, &ok);
    dr->win_end = -1;
    dr->p_tok = ds4_gpu_host_alloc((uint64_t)cap * 4);
    dr->p_bpos = ds4_gpu_host_alloc((uint64_t)cap * 4);
    /* p_wpos sizes by the window, not the graph's rows bucket: the first
     * post-prefill round tops up up to SWA rows (core_v41_draft.c:148-151). */
    dr->p_wpos = ds4_gpu_host_alloc((uint64_t)SWA * 4);
    dr->p_first = ds4_gpu_host_alloc(4);
    dr->p_ids = ds4_gpu_host_alloc((uint64_t)cap * 4);
    dr->p_conf = ds4_gpu_host_alloc((uint64_t)cap * 4);
    dr->p_onehot = ds4_gpu_host_alloc((uint64_t)cap * DS4_N_HC * 4);
    dr->firstd = v41_alloc(16, &ok);
    if (!dr->p_tok || !dr->p_bpos || !dr->p_wpos || !dr->p_first || !dr->p_ids || !dr->p_conf || !dr->p_onehot) ok = false;
    if (ok) for (uint32_t i = 0; i < cap * DS4_N_HC; i++) dr->p_onehot[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    return ok;
}

/* core_v41_draft.c:94-175 — arm the drafter; ready=0 means the caller decodes
 * one token at a time (no draft parameters in the GGUF, no towers). */
static void v41_draft_free(ds4_v41_draft *dr);
static bool v41_draft_alloc(ds4_engine *e, ds4_v41_draft *dr) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    memset(dr, 0, sizeof *dr);
    if (!v->mtp_towers || !v->mtp_block || !v->n_mtp_target) return false;
    if (!e->weights.v41_towers.main_proj || !e->weights.v41_towers.markov_embd || !e->weights.v41_towers.confidence) return false;
    const uint32_t B = v->mtp_block, cap = B + 1u;   /* +1: the verify batch runs up to 1+block rows; main_x sizes by it too */
    if (!v41_draft_state_alloc(dr, cap, v->mtp_towers)) { v41_draft_free(dr); return false; }
    if (!v41_draft_exp_off(e, dr)) { v41_draft_free(dr); return false; }
    dr->block = B;   /* the engine's --dspark-block clamp is not ported (named) */
    dr->ready = 1;
    fprintf(stderr, "ds4: [v41] DSpark drafter armed: %u towers x %u experts top-%u (%s), block %u, target layers",
            v->mtp_towers, v->mtp_experts, v->mtp_used, e->weights.v41_towers.exps_vq[0] ? "VQ blob" : "per-expert fp4x32", B);
    for (uint32_t i = 0; i < v->n_mtp_target; i++) fprintf(stderr, " L%02d", (int)v->mtp_target[i]);
    fprintf(stderr, "\n");
    return true;
}

/* core_v41_draft.c:179-200, minus the graph teardown. */
static void v41_draft_free(ds4_v41_draft *dr) {
    ds4_v41_state *st = &dr->st;
    ds4_gpu_host_free(dr->p_tok); ds4_gpu_host_free(dr->p_bpos); ds4_gpu_host_free(dr->p_wpos); ds4_gpu_host_free(dr->p_first);
    ds4_gpu_host_free(dr->p_ids); ds4_gpu_host_free(dr->p_conf); ds4_gpu_host_free(dr->p_onehot);
    dr->p_tok = dr->p_bpos = dr->p_wpos = dr->p_first = dr->p_ids = NULL; dr->p_conf = dr->p_onehot = NULL;
    ds4_gpu_tensor **all[] = { &st->tok, &st->pos, &st->hc, &st->hc2, &st->mix, &st->pre, &st->post, &st->comb, &st->pre_mix,
        &st->x, &st->xn, &st->qr, &st->qrn, &st->q, &st->kv, &st->kvn, &st->wintmp, &st->o, &st->low, &st->attn_out,
        &st->glog, &st->sel, &st->rw, &st->routed, &st->sg, &st->su, &st->sh, &st->so, &st->y, &st->logits, &st->main_x,
        &dr->mainx_raw, &dr->mk_embed, &dr->mk_cur, &dr->mk_bias, &dr->conf_in, &dr->conf, &dr->ids, &dr->ids_next, &dr->h,
        &dr->mainh_lin, &dr->firstd, &dr->mk_cache, &dr->mk_cache_ids, &dr->mk_cache_next, &dr->mk_hit };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) { if (*all[i]) ds4_gpu_tensor_free(*all[i]); *all[i] = NULL; }
    for (uint32_t T = 0; T < DS4_MTP_MAX_TOWERS; T++) {
        if (st->win[T]) { ds4_gpu_tensor_free(st->win[T]); st->win[T] = NULL; }
        free(dr->exp_off[T]); dr->exp_off[T] = NULL; st->tower_exp_off[T] = NULL; st->tower_exps_vq[T] = NULL;
    }
    dr->ready = 0;
}

/* core_v41_draft.c:202-211 — main_x = main_norm(main_proj(mainh_lin rows)),
 * the official DSparkBlock.forward_embed.  main_proj dispatches on the
 * registered type (FP8 / q4_K / fp4x32) via v41_tproj. */
static bool v41_draft_main_x(ds4_engine *e, ds4_v41_draft *dr, uint32_t rows) {
    const ds4_model *m = &e->model;
    const uint64_t in = (uint64_t)DS4_N_EMBD * g_ds4_v41.n_mtp_target;
    if (!v41_tproj(m, dr->mainx_raw, e->weights.v41_towers.main_proj, in, DS4_N_EMBD, dr->mainh_lin, rows, 1)) return false;
    return ds4_gpu_v41_rms_norm_tensor(dr->st.main_x, dr->mainx_raw, m->map, m->size,
                                       e->weights.v41_towers.main_norm->abs_offset, DS4_N_EMBD, rows, DS4_RMS_EPS) != 0;
}

/* core_v41_draft.c:213-219 — host -> device; capture mode would switch to the
 * zero-copy kernel (E3), the eager path writes synchronously. */
static bool v41_draft_put(const ds4_v41_draft *dr, ds4_gpu_tensor *t, uint64_t off, const void *pinned, uint64_t bytes) {
    return dr->cap_mode ? ds4_gpu_tensor_write_zerocopy(t, off, pinned, bytes) != 0 : ds4_gpu_tensor_write(t, off, pinned, bytes) != 0;
}

/* core_v41_draft.c:221-230 — the round's host slots: the window top-up
 * positions / ring start row + the block input [real token, noise x (B-1)]
 * (official draft_input_ids) and block positions pos0..pos0+B-1.  Every path
 * (direct, capture, replay) must call it: the graph bakes only the slot
 * ADDRESSES, values are read at replay time (the 2026-09-24 conviction:
 * replay read capture-time positions, the window rotted, acceptance fell
 * 0.47 -> 0.03 with no error). */
static void v41_draft_slots(const ds4_v41_state *main_st, ds4_v41_draft *dr, int64_t first, uint32_t rows, uint32_t pos0) {
    for (uint32_t i = 0; i < rows; i++) dr->p_wpos[i] = (int32_t)(first + (int64_t)i);   /* absolute positions (rope) */
    if (rows) dr->p_first[0] = (int32_t)(first % (int64_t)main_st->mainh_cap);
    for (uint32_t i = 0; i < dr->block; i++) {
        dr->p_tok[i] = i ? (int32_t)g_ds4_v41.mtp_noise_id : dr->host_ids[0];
        dr->p_bpos[i] = (int32_t)(pos0 + i);
    }
    __sync_synchronize();
}

/* core_v41_attn.c:239-261 — the draft tower's attention (official
 * DSparkAttention).  Three differences from the main route: no compressed
 * KV/indexer (compress_ratio == 0), the window holds the main model's
 * per-position main_x projections (pushed by v41_draft_push_main), and all n
 * positions of the block see each other (full_block).
 * The block's KV must NOT enter the history window: the draft is a trial, and
 * only main-confirmed positions belong in the window (else the next round
 * drafts from a history containing rejected drafts — no error, acceptance
 * rots). */
static bool v41_draft_attention(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_model *m = &e->model;
    const ds4_layer_weights *l = &e->weights.v41_towers.tower[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, SWA = DS4_N_SWA;
    const uint64_t rowb = (uint64_t)HD * 4;
    if (!v41_tproj(m, st->qr, l->attn_q_a, E, Q, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->qrn, st->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS)) return false;
    if (!v41_tproj(m, st->q, l->attn_q_b, Q, (uint64_t)NH * HD, st->qrn, n, 1)) return false;
    if (!v41_rope(st->q, st->pos, n, NH, HD, 0, false)) return false;
    if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->xn, n, 1)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS)) return false;
    if (!v41_rope(st->kvn, st->pos, n, 1, HD, 0, false)) return false;
    if (!ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, n, HD, 32)) return false;
    if (!ds4_gpu_tensor_copy(st->win[il], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)n * rowb)) return false;
    /* ring=0: the tower window is a linear segment maintained by
     * v41_draft_push_main, not the main route's ring. */
    if (!ds4_gpu_v41_sparse_attn_tensor(st->o, st->q, st->win[il], NULL, NULL, m->map, m->size,
                                        l->attn_sinks->abs_offset, n, st->pos0, SWA, 0, 0, 0, NH, HD,
                                        (float)(1.0 / sqrt((double)HD)), 1, 0, 0u, NULL, 0u)) return false;   /* full window, no dirty slots */
    if (!v41_rope(st->o, st->pos, n, NH, HD, 0, true)) return false;
    const uint32_t grp = NH / DS4_N_OUT_GROUP;
    if (!v41_tproj_grouped(m, st->low, l->attn_output_a, DS4_N_OUT_GROUP, (uint64_t)grp * HD, DS4_N_LORA_O, st->o, n, 1)) return false;
    return v41_tproj(m, st->attn_out, l->attn_output_b, (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O, E, st->low, n, 1);
}

/* core_v41_attn.c:263-280 — project `rows` CONFIRMED positions' main_x into
 * every tower's window (official self.window_kv_cache[start_pos % win] =
 * main_kv).  The window is a linear buffer, not a ring: new rows go to the
 * block region, then the whole thing shifts left by rows — same effect, one
 * modulo less.  st->pos holds the rows' absolute positions (rope). */
static bool v41_draft_push_main(ds4_engine *e, ds4_v41_state *st, uint32_t rows) {
    const ds4_model *m = &e->model;
    const uint32_t E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, SWA = DS4_N_SWA;
    const uint64_t rowb = (uint64_t)HD * 4;
    if (!rows) return true;
    for (uint32_t T = 0; T < g_ds4_v41.mtp_towers; T++) {
        const ds4_layer_weights *l = &e->weights.v41_towers.tower[T];
        if (!v41_tproj(m, st->kv, l->attn_kv, E, HD, st->main_x, rows, 1)) return false;
        if (!ds4_gpu_v41_rms_norm_tensor(st->kvn, st->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, rows, DS4_RMS_EPS)) return false;
        if (!v41_rope(st->kvn, st->pos, rows, 1, HD, 0, false)) return false;
        if (!ds4_gpu_v41_act_quant_fp8_tensor(st->kvn, rows, HD, 32)) return false;
        if (!ds4_gpu_tensor_copy(st->win[T], (uint64_t)SWA * rowb, st->kvn, 0, (uint64_t)rows * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->wintmp, 0, st->win[T], (uint64_t)rows * rowb, (uint64_t)SWA * rowb)) return false;
        if (!ds4_gpu_tensor_copy(st->win[T], 0, st->wintmp, 0, (uint64_t)SWA * rowb)) return false;
    }
    return true;
}

/* core_v41_draft.c:232-299 — one draft block forward: towers -> head ->
 * per-position markov greedy -> confidence.  Slots are filled by
 * v41_draft_slots (ids[0] = the last real token). */
static bool v41_draft_block(ds4_engine *e, ds4_v41_draft *dr, uint32_t pos0) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const ds4_model *m = &e->model;
    ds4_v41_state *st = &dr->st;
    const uint32_t B = dr->block, E = DS4_N_EMBD, R = v->mtp_markov_rank;
    st->n = B; st->pos0 = pos0;
    if (!v41_draft_put(dr, st->tok, 0, dr->p_tok, (uint64_t)B * 4) || !v41_draft_put(dr, st->pos, 0, dr->p_bpos, (uint64_t)B * 4)) return false;
    if (!v41_draft_put(dr, st->pre_mix, 0, dr->p_onehot, (uint64_t)B * DS4_N_HC * 4)) return false;
    if (!v41_embed(m, st->x, st->tok, e->weights.token_embd, DS4_N_VOCAB, B, E)) return false;
    if (!ds4_gpu_v41_expand_hc_tensor(st->hc, st->x, E, DS4_N_HC, B)) return false;
    for (uint32_t T = 0; T < v->mtp_towers; T++) if (!v41_layer(e, st, T)) return false;
    /* Exit: the official mtp[-1] borrows the main model's head; the norm is
     * mtp.2.norm (out_norm).  The amp correction (dr->ampK) is not ported:
     * without the sidecar it is the identity anyway (core_v41_draft.c:252). */
    if (!ds4_gpu_v41_hc_pre_tensor(dr->h, st->hc, st->pre_mix, E, DS4_N_HC, B)) return false;
    if (!ds4_gpu_v41_rms_norm_tensor(st->xn, dr->h, m->map, m->size, e->weights.v41_towers.out_norm->abs_offset, E, B, DS4_RMS_EPS)) return false;
    if (!v41_tproj(m, st->logits, e->weights.output, E, DS4_N_VOCAB, st->xn, B, 0)) return false;
    /* Per position, all on device (one D2H per position would cost the whole
     * speculation gain): logits[i] += markov_head(token_i) -> argmax ->
     * token_{i+1}. */
    const uint64_t nrow = e->weights.v41_towers.markov_embd->ndim > 1 ? e->weights.v41_towers.markov_embd->dim[1] : DS4_N_VOCAB;
    const uint32_t eb = e->weights.v41_towers.markov_embd->type == DS4_TENSOR_BF16 ? 2u : 4u;
    /* Bias cache: only the original bf16 markov_head path caches (a hit
     * returns the whole GEMV grid; the bias comes from the slot). */
    const bool mkc = dr->mk_cache && e->weights.v41_towers.markov_head->type == DS4_TENSOR_BF16;
    for (uint32_t i = 0; i < B; i++) {
        if (mkc && !ds4_gpu_v41_mkcache_lookup_tensor(dr->mk_hit, dr->mk_cache_ids, dr->mk_cache_next, dr->ids, i, DS4_V41_MKCACHE_SLOTS)) return false;
        if (!ds4_gpu_v41_row_gather_tensor(dr->mk_embed, m->map, m->size, e->weights.v41_towers.markov_embd->abs_offset,
                                           nrow, R, eb, dr->ids, i, i)) return false;   /* the confidence head reads the original row */
        if (!ds4_gpu_v41_row_gather_tensor(dr->mk_cur, m->map, m->size, e->weights.v41_towers.markov_embd->abs_offset,
                                           nrow, R, eb, dr->ids, i, 0)) return false;
        if (mkc) {
            if (!ds4_gpu_v41_matmul_bf16_skip_tensor(dr->mk_bias, m->map, m->size, e->weights.v41_towers.markov_head->abs_offset,
                                                     R, DS4_N_VOCAB, dr->mk_cur, 1, dr->mk_hit)) return false;
        } else if (!v41_small_matmul(m, dr->mk_bias, e->weights.v41_towers.markov_head, R, DS4_N_VOCAB, dr->mk_cur, 1)) return false;
        if (mkc ? !ds4_gpu_v41_mkcache_add_tensor(st->logits, i, dr->mk_bias, dr->mk_cache, dr->mk_hit, DS4_N_VOCAB)
                : !ds4_gpu_v41_row_add_tensor(st->logits, i, dr->mk_bias, DS4_N_VOCAB)) return false;
        /* argmax writes only the first int of its own row slot: land it in
         * ids_next, then copy to ids[i+1] (official output_ids[:, i+1]). */
        if (!ds4_gpu_v41_argmax_tensor(dr->ids_next, st->logits, i, DS4_N_VOCAB)) return false;
        if (!ds4_gpu_tensor_copy(dr->ids, (uint64_t)(i + 1u) * 4, dr->ids_next, 0, 4)) return false;
    }
    /* confidence = proj([h_i; markov_embed_i]) */
    for (uint32_t i = 0; i < B; i++) {
        if (!ds4_gpu_tensor_copy(dr->conf_in, (uint64_t)i * (E + R) * 4, dr->h, (uint64_t)i * E * 4, (uint64_t)E * 4)) return false;
        if (!ds4_gpu_tensor_copy(dr->conf_in, ((uint64_t)i * (E + R) + E) * 4, dr->mk_embed, (uint64_t)i * R * 4, (uint64_t)R * 4)) return false;
    }
    return v41_small_matmul(m, dr->conf, e->weights.v41_towers.confidence, (uint64_t)E + R, 1, dr->conf_in, B);
}

/* core_v41_draft.c:301-315 — top the window up: rows [first, first+rows) out
 * of the main ring -> main_proj/main_norm -> push into the tower windows.
 * These three steps MUST speak up on failure (the 2026-09-18 conviction: a
 * silent return false made the caller read "no draft this round", an
 * undersized buffer disabled the whole speculation path and the "spec ==
 * pure decode" gate stayed green). */
static bool v41_draft_fill(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, uint32_t rows) {
    if (!v41_draft_put(dr, dr->st.pos, 0, dr->p_wpos, (uint64_t)rows * 4)) { fprintf(stderr, "ds4: [v41] drafter: writing %u window positions failed\n", rows); return false; }
    if (dr->cap_mode && !ds4_gpu_tensor_write_zerocopy(dr->firstd, 0, dr->p_first, 4)) { fprintf(stderr, "ds4: [v41] drafter: writing the start row failed\n"); return false; }
    if (!ds4_gpu_v41_ring_rows_tensor(dr->mainh_lin, main_st->mainh, g_ds4_v41.n_mtp_target * DS4_N_EMBD, main_st->mainh_cap,
                                      (uint32_t)dr->p_first[0], rows, dr->cap_mode ? dr->firstd : NULL)) {
        fprintf(stderr, "ds4: [v41] drafter: taking %u rows from the main_hidden ring failed\n", rows); return false;
    }
    if (!v41_draft_main_x(e, dr, rows)) { fprintf(stderr, "ds4: [v41] drafter: main_proj/main_norm over %u rows failed\n", rows); return false; }
    if (!v41_draft_push_main(e, &dr->st, rows)) { fprintf(stderr, "ds4: [v41] drafter: pushing %u rows into the tower windows failed\n", rows); return false; }
    /* the block attention's main_x is only the last row (the official
     * forward_spec's main_hidden is the current position) */
    if (rows > 1 && !ds4_gpu_tensor_copy(dr->st.main_x, 0, dr->st.main_x, (uint64_t)(rows - 1u) * DS4_N_EMBD * 4,
                                         (uint64_t)DS4_N_EMBD * 4)) return false;
    return true;
}

/* core_v41_draft.c:317-324 — one round's GPU part (fill + ids[0] + block),
 * shared by the direct and (future) capture paths; the only difference is
 * whether the host writes go through sync copies or zero-copy kernels. */
static bool v41_draft_gpu_round(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int64_t first, uint32_t rows, uint32_t pos0) {
    v41_draft_slots(main_st, dr, first, rows, pos0);
    if (rows && !v41_draft_fill(e, main_st, dr, rows)) return false;
    if (!v41_draft_put(dr, dr->ids, 0, dr->p_tok, 4)) return false;   /* p_tok[0] = the real token (caller-filled) */
    return v41_draft_block(e, dr, pos0);
}

/* core_v41_draft.c:373-416, the direct path only (the graph round and the
 * scheduler's warm/cost accounting are E2/E3).  Returns 0 = cannot draft
 * (missing inputs), 2 = the direct dispatch ran synchronously (results in
 * host_ids/host_conf).  The engine's return 1 (graph in flight) never
 * happens here. */
static int v41_draft_launch(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main) {
    if (!dr->ready || !main_st->mainh) return 0;
    if (main_st->mainh_end != (int64_t)pos_main) {   /* the main forward did not write this position into the ring (CED skip / early stop) = inputs missing */
        fprintf(stderr, "ds4: [v41] drafter: main_hidden ring ends at %lld != pos_main %u, no draft this round\n", (long long)main_st->mainh_end, pos_main);
        return 0;
    }
    /* Top up (win_end, pos_main] — rounds that sat out and graph-warmed steps
     * are all caught here; a gap wider than the window rebuilds it. */
    const uint32_t SWA = DS4_N_SWA;
    int64_t first = dr->win_end + 1;
    if ((int64_t)pos_main - first + 1 > (int64_t)SWA) first = (int64_t)pos_main - (int64_t)SWA + 1;
    if (first < 0) first = 0;
    const uint32_t rows = (uint32_t)((int64_t)pos_main - first + 1);
    if (rows) {
        const int64_t avail_first = main_st->mainh_end - (int64_t)main_st->mainh_n + 1;
        if (first < avail_first) {
            fprintf(stderr, "ds4: [v41] drafter: window needs %lld..%u, the ring's contiguous run starts at %lld (%u rows), no draft this round\n",
                    (long long)first, pos_main, (long long)avail_first, main_st->mainh_n);
            return 0;
        }
    }
    dr->host_ids[0] = tok; dr->p_tok[0] = tok;
    const uint32_t pos0 = pos_main + 1u;
    if (!v41_draft_gpu_round(e, main_st, dr, first, rows, pos0)) return 0;
    if (rows) dr->win_end = pos_main;
    if (!ds4_gpu_synchronize()) return 0;
    if (!ds4_gpu_tensor_read(dr->ids, 0, dr->host_ids, (uint64_t)(dr->block + 1u) * 4)) return 0;
    if (!ds4_gpu_tensor_read(dr->conf, 0, dr->host_conf, (uint64_t)dr->block * 4)) return 0;
    dr->rounds++;
    return 2;
}
static bool v41_draft_step(ds4_engine *e, ds4_v41_state *main_st, ds4_v41_draft *dr, int32_t tok, uint32_t pos_main) {
    return v41_draft_launch(e, main_st, dr, tok, pos_main) == 2;   /* no graph path: launch only returns 0 or 2 */
}

/* ---- the confidence scheduler (core_draft_sched.c:31-97) ----
 * How many positions to verify each round: the drafter's own confidence says
 * how likely position j survives, and the engine measures the costs per
 * request (walked-graph step / one draft round / a verify of n rows) instead
 * of baking constants — a kernel that gets faster leaves stale constants
 * behind and the scheduler then blocks speculation entirely (the 2026-09-17
 * conviction: zero draft rounds in 40 tokens after a new kernel landed).
 *   round yield = 1 + sum_{j<=k} prod_{i<=j} rho*sigmoid(conf_i)
 *   round cost  = draft + verify(1+k rows)
 * pick k = argmax yield/cost; > 1 (vs one pure-decode step) means worth it.
 * k only affects speed, never the output: temperature-0 spec == pure decode
 * byte for byte (the d1 gate re-runs on every change).  rho (the acceptance
 * calibration for sampling) is history-only, so it is reproducible at temp 0;
 * the measured costs are not, which is why the trace gate pins k
 * (--dspark-verify / verify_k). */
typedef struct {
    double pred1, real1;
    double ms[DS4_MTP_MAX_BLOCK + 2u]; uint32_t cnt[DS4_MTP_MAX_BLOCK + 2u];
    double draft_ms; uint32_t draft_n;
} v41_sched;

static float v41_sched_sigmoid(float c) { return isfinite(c) ? 1.0f / (1.0f + expf(-c)) : 0.0f; }   /* non-finite conf = no confidence */

/* Sampling calibration (core_draft_sched.c:33-36): the conf head learned
 * "does greedy pick the same token", which is systematically optimistic under
 * sampling.  rho = first-position hits / sum of first sigmoids over this
 * request's drafted rounds, multiplied into every position's sigma (capped
 * at 1).  Observed every round, k=0 included, or a first-round rejection
 * deadlocks rho at 0. */
static void v41_sched_observe(v41_sched *s, const float *conf, int hit) {
    if (!s) return;
    s->pred1 += (double)v41_sched_sigmoid(conf[0]);
    s->real1 += hit ? 1.0 : 0.0;
}
/* Cost samples: n_rows = 1 is a single-token step, >= 2 a verify of 1+k rows. */
static void v41_sched_cost(v41_sched *s, uint32_t n_rows, double ms) {
    if (!s || n_rows == 0u || n_rows > DS4_MTP_MAX_BLOCK + 1u || !(ms > 0.0)) return;
    s->ms[n_rows] += ms; s->cnt[n_rows]++;
}
static void v41_sched_draft_cost(v41_sched *s, double ms) { if (s && ms > 0.0) { s->draft_ms += ms; s->draft_n++; } }

/* Weighted least squares ms(n) = v1 + tok*(n-1) over the per-n means (counts
 * as weights); one distinct x degenerates to a constant.  false = no samples. */
static bool v41_sched_fit(const v41_sched *s, double *v1, double *tok) {
    double W = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    uint32_t nx = 0; double x0 = 0;
    for (uint32_t n = 1; n <= DS4_MTP_MAX_BLOCK + 1u; n++) {
        if (!s->cnt[n]) continue;
        const double w = (double)s->cnt[n], x = (double)(n - 1u), y = s->ms[n] / w;
        if (nx == 0) x0 = x;
        if (nx == 0 || x != x0) nx++;
        W += w; Sx += w * x; Sy += w * y; Sxx += w * x * x; Sxy += w * x * y;
    }
    if (W <= 0.0) return false;
    const double den = W * Sxx - Sx * Sx;
    if (nx < 2u || den <= 0.0) { *v1 = Sy / W; *tok = 0.0; return true; }
    *tok = (W * Sxy - Sx * Sy) / den;
    *v1 = (Sy - *tok * Sx) / W;
    return true;
}

/* conf[0..block-1] = per-position confidence logits.  Returns how many
 * positions to verify (0 = not even the draft is worth its cost);
 * *value_out = the predicted yield/cost ratio (1.0 = break-even with one
 * pure-decode step). */
static uint32_t v41_draft_pick_k(const float *conf, uint32_t block, float *value_out, const v41_sched *s) {
    if (value_out) *value_out = 1.0f;
    if (!s || block == 0u) return block;
    /* Startup: no verify samples => measure the longest block once; one n>=2
     * point => add an n=2 (or the longest) point so the fit has a slope. */
    uint32_t nmeas = 0, first = 0;
    for (uint32_t n = 2; n <= block + 1u; n++) if (s->cnt[n]) { if (!nmeas) first = n; nmeas++; }
    if (nmeas == 0u) return block;
    if (nmeas == 1u && !s->cnt[1]) return first == 2u ? block : 1u;
    double v1, tok;
    if (!v41_sched_fit(s, &v1, &tok)) return block;
    /* One pure-decode step's ms: the real single-token samples when present,
     * else the fit at n=1 (verify 1 row ~ one step). */
    const double step = s->cnt[1] ? s->ms[1] / (double)s->cnt[1] : v1;
    const double draft = s->draft_n ? s->draft_ms / (double)s->draft_n : 0.0;
    if (!(step > 0.0)) return block;
    const float rho = s->pred1 > 0.0 ? (float)(s->real1 / s->pred1) : 1.0f;
    float surv = 1.0f, sum = 0.0f;
    double best = 1.0 / (draft + v1);   /* k=0: the draft is wasted, one single-token step runs */
    uint32_t bestk = 0;
    for (uint32_t j = 0; j < block; j++) {
        const float pj = rho * v41_sched_sigmoid(conf[j]);
        surv *= pj < 1.0f ? pj : 1.0f;
        sum += surv;
        const double cost = draft + v1 + tok * (double)(j + 1u);   /* verify 1+(j+1) rows */
        const double val = (1.0 + (double)sum) / (cost > 0.0 ? cost : step);
        if (val > best) { best = val; bestk = j + 1u; }
    }
    if (value_out) *value_out = (float)(best * step);   /* 1.0 = break-even with one pure-decode step */
    return bestk;
}

/* The DS41_GEN_LOGITS dump (see the generate entry): one full logits row per
 * emitted position, in emission order. */
static void v41_gen_dump_row(ds4_v41_state *st, FILE *fd) {
    if (!fd) return;
    float *row = xmalloc((size_t)DS4_N_VOCAB * 4);
    if (row && ds4_gpu_tensor_read(st->logits, (uint64_t)st->last_logit_row * DS4_N_VOCAB * 4, row, (uint64_t)DS4_N_VOCAB * 4)) {
        fwrite(row, 4, DS4_N_VOCAB, fd);
    }
    free(row);
}

/* ---- the greedy generate entry (core_v41_api.c:426-470 + the round drive
 * :175-425, the eager subset) ----
 * Prompt prefill in chunks (the shared v41_prefill_chunk), then decode steps:
 * device argmax on the last logits row -> emit -> next token.  The engine's
 * generate path sets head_last_only so a prefill block computes ONE logits
 * row (the head GEMV on xlast, core_v41_forward.c:334-342).
 * Spec decode runs by default when the GGUF carries the draft towers and
 * parameters (core_v41_api.c:219-221): each round drafts a block, verifies
 * 1+k positions in one batch, accepts the longest matching prefix and rolls
 * the rest back.  Temperature-0 output is byte-identical with drafting on and
 * off — pure time saving.  verify_k pins the round's k (the engine's
 * --dspark-verify; the deterministic trace gate uses it), 0 = the confidence
 * scheduler.  Named limits: sampling and the penalties, the decode graph, the
 * draft graph and the emit-trace diagnostics are not ported (their units);
 * no_engram mirrors the engine's --v41-no-engram. */
int ds4_v41_generate_argmax(void *engine, const int *prompt, int n_prompt, int n_predict, int no_engram, int verify_k,
                            const ds41_engram_feed *feed, int (*emit)(int token, void *ud), void *ud) {
    ds4_engine *e = (ds4_engine *)engine;
    if (!e || !prompt || n_prompt < 1 || DS4_MODEL_VARIANT != DS4_VARIANT_V41) return 1;
    const uint32_t np = (uint32_t)n_prompt;
    uint32_t ctx = g_ds4_v41.ctx;
    if (ctx == 0u) { fprintf(stderr, "ds4: [v41] model metadata has no deepseek4.context_length\n"); return 1; }
    if (np + 1u > ctx) { fprintf(stderr, "ds4: [v41] prompt %u tokens over context %u\n", np, ctx); return 1; }
    /* allocate only what this run can reach (core_v41_api.c:120-135) */
    const uint64_t need = (uint64_t)np + (uint64_t)(n_predict > 0 ? n_predict : 0) + DS4_MTP_MAX_BLOCK + 2u;
    if ((uint64_t)ctx > need) ctx = (uint32_t)need;
    const int room = ctx > np ? (int)(ctx - np) : 0;
    if (n_predict > room) n_predict = room;
    const uint32_t cap = DS4_V41_CHUNK < np ? DS4_V41_CHUNK : np;
    ds4_v41_state st;
    if (!v41_state_alloc(&st, cap, ctx, DS4_MTP_MAX_BLOCK + 2u)) return 1;
    st.head_last_only = 1;
    st.no_engram = no_engram;   /* the gate's compare-only mode (the engine's --v41-no-engram) */
    st.efeed = feed;            /* rows: the feed's prepare callback runs per block (spec), or the caller pre-fills and sets pos0 */
    /* Diagnostic (kept): DS41_GEN_LOGITS=<path> appends the full logits row of
     * every emitted position (f32, DS4_N_VOCAB per row) so the step path can
     * be diffed against the score path on the same sequence. */
    const char *gen_lg = getenv("DS41_GEN_LOGITS");
    FILE *gen_fd = gen_lg ? fopen(gen_lg, "wb") : NULL;
    ds4_gpu_tensor *am = ds4_gpu_tensor_alloc((uint64_t)(DS4_MTP_MAX_BLOCK + 2u) * 16u);   /* per-row 16 B slots (sampling layout; argmax uses the first int) */
    /* Spec decode arms when the GGUF carries the towers + draft parameters
     * (core_v41_api.c:219-221); ready=0 keeps the single-token loop.  The
     * engine's --no-dspark equivalent is the env switch (the CLI flag lands
     * with the serving unit). */
    ds4_v41_draft dr;
    const bool spec = !getenv("DS41_NO_DSPARK") && v41_draft_alloc(e, &dr);
    v41_sched cal; memset(&cal, 0, sizeof cal);
    uint32_t spec_rounds = 0, spec_acc = 0, spec_skipped = 0, spec_hist[DS4_MTP_MAX_BLOCK + 1];
    for (uint32_t i = 0; i <= DS4_MTP_MAX_BLOCK; i++) spec_hist[i] = 0;
    double ms_draft = 0, ms_snap = 0, ms_verify = 0, ms_argmax = 0;
    int rc = 1;
    int32_t tok = 0;
    int produced = 0;
    do {
        if (!am) break;
        const double t0 = now_sec();
        bool ok = true;
        for (uint32_t c0 = 0; ok && c0 < np;) ok = v41_prefill_chunk(e, &st, prompt, np, &c0, cap);
        if (!ok) break;
        fprintf(stderr, "[v41] prefill %u token %.1fs (%.1f t/s)\n", np, now_sec() - t0, (double)np / (now_sec() - t0 + 1e-9));
        if (getenv("DS41_ADDR_DUMP")) {   /* the A/B layout discriminator (see the harness) */
            fprintf(stderr, "addr am %p logits %p x %p xn %p tok %p pos %p\n", ds4_gpu_tensor_ptr(am), ds4_gpu_tensor_ptr(st.logits),
                    ds4_gpu_tensor_ptr(st.x), ds4_gpu_tensor_ptr(st.xn), ds4_gpu_tensor_ptr(st.tok), ds4_gpu_tensor_ptr(st.pos));
            for (uint32_t k = 0; k < g_ds4_v41.n_engram; k++)
                if (st.eraw[k]) fprintf(stderr, "addr eraw[%u] %p\n", k, ds4_gpu_tensor_ptr(st.eraw[k]));
            if (st.erows) fprintf(stderr, "addr erows %p ekv %p\n", ds4_gpu_tensor_ptr(st.erows), st.ekv ? ds4_gpu_tensor_ptr(st.ekv) : NULL);
        }
        /* the last prefill position's logits -> the first token (last_logit_row
         * says which row the head wrote; row 0 under head_last_only) */
        if (!ds4_gpu_v41_argmax_tensor(am, st.logits, st.last_logit_row, DS4_N_VOCAB) ||
            ds4_gpu_synchronize() == 0 ||
            ds4_gpu_tensor_read(am, 0, &tok, 4) == 0) break;
        v41_gen_dump_row(&st, gen_fd);
        {   /* the engine's one-shot argmax self-check (core_v41_api.c:193-200):
             * a wrong kernel would silently emit wrong tokens, so compare the
             * device index against a host scan of the same row once. */
            float *row = xmalloc((size_t)DS4_N_VOCAB * 4);
            if (row && ds4_gpu_tensor_read(st.logits, (uint64_t)st.last_logit_row * DS4_N_VOCAB * 4, row, (uint64_t)DS4_N_VOCAB * 4)) {
                uint32_t best = 0;
                for (uint32_t v = 1; v < DS4_N_VOCAB; v++) if (row[v] > row[best]) best = v;
                if ((int32_t)best != tok) fprintf(stderr, "ds4: *[v41] argmax kernel %d != host %u*\n", tok, best);
            }
            free(row);
        }
        rc = 0;
        bool stop = false;
        while (!stop && produced < n_predict) {
            produced++;
            if (emit && emit((int)tok, ud) != 0) break;   /* the CLI's EOS convention: nonzero = stop */
            if ((int)tok == ds4_token_eos(e)) break;      /* the engine's own check when emit does not stop */
            if (st.n_past + 1u > st.ctx) { fprintf(stderr, "[v41] context full %u\n", st.ctx); break; }
            if (!spec) {   /* no drafter armed: the plain single-token step */
                if (!v41_forward(e, &st, &tok, 1)) { rc = 1; break; }
                if (!ds4_gpu_v41_argmax_tensor(am, st.logits, st.last_logit_row, DS4_N_VOCAB) ||
                    ds4_gpu_synchronize() == 0 ||
                    ds4_gpu_tensor_read(am, 0, &tok, 4) == 0) { rc = 1; break; }
                v41_gen_dump_row(&st, gen_fd);
                continue;
            }
            /* ---- one spec round (core_v41_api.c:284-390, the eager subset) ---- */
            const double tr0 = now_sec();
            uint32_t k = 0;
            int32_t batch[DS4_MTP_MAX_BLOCK + 1];
            batch[0] = tok;
            float sched_val = 0.f;
            bool drafted = false;
            const uint32_t pos_round = st.n_past;   /* the round's first position (diagnostic) */
            const int dr_launch = v41_draft_launch(e, &st, &dr, (int32_t)tok, st.n_past - 1u);
            if (dr_launch != 0) {
                drafted = true;
                const uint32_t use = verify_k ? (uint32_t)verify_k
                                              : v41_draft_pick_k(dr.host_conf, dr.block, &sched_val, &cal);
                if (!verify_k && sched_val < 1.0f) spec_skipped++;   /* predicted not to repay even the draft */
                k = dr.block < use ? dr.block : use;
                for (uint32_t i = 0; i < k; i++) batch[i + 1u] = dr.host_ids[i + 1u];
            }
            const uint32_t nb = 1u + k;
            int32_t want[DS4_MTP_MAX_BLOCK + 1];
            const double tr1 = now_sec();
            if (k && !v41_spec_snapshot(&st, nb)) { rc = 1; break; }
            const double tr2 = now_sec();
            if (!v41_forward(e, &st, batch, nb)) { rc = 1; break; }
            const double tr3 = now_sec();
            if (!v41_device_next(&st, am, 0u, nb, want)) { rc = 1; break; }
            if (drafted) {
                /* Cost accounting (core_v41_api.c:335-341): the engine counts a
                 * direct dispatch as steady whenever that n cannot walk a graph;
                 * the port has no graph, so every dispatch is steady. */
                v41_sched_draft_cost(&cal, (tr1 - tr0) * 1e3);
                v41_sched_cost(&cal, nb, (tr3 - tr2) * 1e3);
                if (k) { ms_draft += (tr1 - tr0) * 1e3; ms_snap += (tr2 - tr1) * 1e3;
                         ms_verify += (tr3 - tr2) * 1e3; ms_argmax += (now_sec() - tr3) * 1e3; }
            }
            uint32_t a = 0;
            while (a < k && want[a] == batch[a + 1u]) a++;   /* accept the longest matching prefix */
            /* Observe the first position every drafted round (k=0 included), or
             * the calibration deadlocks at rho=0 (core_draft_sched.c:33). */
            if (drafted && !verify_k) v41_sched_observe(&cal, dr.host_conf, k ? (a >= 1u) : (want[0] == dr.host_ids[1]));
            /* The engine's per-round account (core_v41_api.c:346-357), the
             * spec trace gate's instrument. */
            if (drafted && g_ds4_v41_prof) {
                fprintf(stderr, "[dspark] pos %u k %u val %.3f acc %u conf", pos_round, k, (double)sched_val, a);
                for (uint32_t i = 0; i < dr.block; i++) {
                    const float c = dr.host_conf[i];
                    fprintf(stderr, " %.3f", isfinite(c) ? 1.0 / (1.0 + exp(-(double)c)) : 0.0);
                }
                fprintf(stderr, " | draft");
                for (uint32_t i = 0; i < k; i++) fprintf(stderr, " %d", batch[i + 1u]);
                fprintf(stderr, " | main");
                for (uint32_t i = 0; i < nb; i++) fprintf(stderr, " %d", want[i]);
                fprintf(stderr, "\n");
            }
            if (k) {
                spec_rounds++; spec_acc += a; spec_hist[a]++;
                if (!v41_spec_rollback(&st, 1u + a)) { rc = 1; break; }
            }
            const int tok_next = (int)want[a];
            /* The earned tokens: the draft matched the main model, emit them
             * straight (core_v41_api.c:384-390; the n_predict cap holds). */
            for (uint32_t i = 0; i < a; i++) {
                if (produced >= n_predict) break;
                produced++;
                const int t = (int)batch[i + 1u];
                if ((emit && emit(t, ud) != 0) || t == ds4_token_eos(e)) { stop = true; break; }
            }
            tok = tok_next;
        }
        /* The engine's summary (core_v41_api.c:393-406; its graph host-gap
         * line is not ported). */
        if (spec_rounds) {
            fprintf(stderr, "\n[v41] DSpark: %u rounds (scheduler skipped %u), average accept %.2f/%u; histogram",
                    spec_rounds, spec_skipped, (double)spec_acc / (double)spec_rounds, dr.block);
            for (uint32_t i = 0; i <= dr.block; i++) fprintf(stderr, " %u:%u", i, spec_hist[i]);
            const double R = (double)spec_rounds, ea = 1.0 + (double)spec_acc / R;
            const double per = (ms_draft + ms_snap + ms_verify + ms_argmax) / R;
            fprintf(stderr, "\n[v41] one round %.1f ms = draft %.1f + snapshot/rollback %.1f + verify %.1f + argmax %.1f"
                            "; yield %.2f token => %.1f ms/token\n",
                    per, ms_draft / R, ms_snap / R, ms_verify / R, ms_argmax / R, ea, per / ea);
        }
    } while (0);
    if (gen_fd) fclose(gen_fd);
    if (spec) v41_draft_free(&dr);
    if (am) ds4_gpu_tensor_free(am);
    v41_state_free(&st);
    return rc;
}

#endif /* DS4_DS41_DRAFT_INC */
