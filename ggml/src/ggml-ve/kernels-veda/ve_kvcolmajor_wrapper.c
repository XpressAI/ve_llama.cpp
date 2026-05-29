/*
 * VEDA + OMP wrappers for the columnar KV cache path (Stage 3).
 *
 * Three exports the backend dispatches:
 *
 *   ve_kvcache_mirror_to_colmajor_hbm
 *     Mirror one src row (F32, just-computed Q/K/V projection) into the
 *     col-major BF16 shadow at row index `row_idx`. Called from ops/
 *     set_rows.cpp right after the regular SET_ROWS, so the shadow stays
 *     in sync with the original cache as decode advances. The shadow's
 *     seq dim has unit stride, so this is a strided F32-write-into-BF16
 *     (NCC vectorises across head_dim per kv_head).
 *
 *   ve_flash_attn_ext_f32q_bf16kv_colmajor_hbm
 *     VEDA-launchable FA. Takes the col-major shadow KV ptrs, F32 query,
 *     optional F16 mask. Internally OMP-parallel over query heads; each
 *     head runs compute_KQ_colmajor + softmax + compute_SV_colmajor.
 *
 * The LLVM-VE-RV intrinsic kernels compute_KQ_colmajor and
 * compute_SV_colmajor live in flash_attn_bf16_colmajor.c. Softmax lives
 * here because NCC vectorises expf and the LLVM-VE-RV clang does not.
 */

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <veda/device.h>

typedef uint16_t bf16;

/* From flash_attn_bf16_colmajor.c (LLVM-VE-RV). */
extern void compute_KQ_colmajor(float * S,
                                const float * qh,
                                const bf16 * k_h,
                                int head_dim,
                                int seq_len,
                                int seq_max);
extern void compute_SV_colmajor(float * out_h,
                                const float * S,
                                const bf16 * v_h,
                                int head_dim,
                                int seq_len,
                                int seq_max);

/* ---------------------------------------------------------------------- */
/* Vectorised softmax: scale + mask, max-reduce, expf, normalise. NCC's
 * auto-vec recognises the expf loop and emits __vec_expf. */
/* ---------------------------------------------------------------------- */
static void softmax_nccvec(float * S, int seq_len,
                           const float * mask, float scale, float slope) {
    float maxv = -INFINITY;
    if (mask) {
#pragma _NEC ivdep
        for (int s = 0; s < seq_len; s++) {
            float v = S[s] * scale + mask[s] * slope;
            S[s] = v;
            if (v > maxv) maxv = v;
        }
    } else {
#pragma _NEC ivdep
        for (int s = 0; s < seq_len; s++) {
            float v = S[s] * scale;
            S[s] = v;
            if (v > maxv) maxv = v;
        }
    }

    float sumv = 0.0f;
#pragma _NEC ivdep
    for (int s = 0; s < seq_len; s++) {
        float e = expf(S[s] - maxv);
        S[s] = e;
        sumv += e;
    }
    float inv = (sumv > 0.0f) ? (1.0f / sumv) : 0.0f;
#pragma _NEC ivdep
    for (int s = 0; s < seq_len; s++) {
        S[s] *= inv;
    }
}

/* ---------------------------------------------------------------------- */
/* Per-head orchestration: K·Q → softmax → S·V. */
/* ---------------------------------------------------------------------- */
static void fa_colmajor_single_head(float * out_h,
                                    const float * qh,
                                    const bf16 * k_h,
                                    const bf16 * v_h,
                                    const float * mask,
                                    int head_dim,
                                    int seq_len,
                                    int seq_max,
                                    float scale,
                                    float slope) {
    /* 8192 covers max ctx ≥ 8192 — comfortably fits VE stack. */
    float S[8192] __attribute__((aligned(64)));
    compute_KQ_colmajor(S, qh, k_h, head_dim, seq_len, seq_max);
    softmax_nccvec(S, seq_len, mask, scale, slope);
    compute_SV_colmajor(out_h, S, v_h, head_dim, seq_len, seq_max);
}

/* ---------------------------------------------------------------------- */
/* F16 -> F32 conversion. Mask values are essentially {0, -inf, finite}, so
 * we don't bother with subnormals — treat them as zero. NCC vectorises
 * this loop (no branches, no calls). */
/* ---------------------------------------------------------------------- */
static void mask_f16_to_f32(const uint16_t * h, float * f, int n) {
#pragma _NEC ivdep
    for (int s = 0; s < n; s++) {
        uint32_t h32   = (uint32_t) h[s];
        uint32_t sign  = (h32 & 0x8000u) << 16;
        uint32_t exp_r = (h32 & 0x7c00u);                  /* exp field in place 10..14 */
        uint32_t mant  = (h32 & 0x03ffu);
        uint32_t inf_bits = sign | 0x7F800000u | (mant << 13);
        uint32_t nrm_bits = sign | ((exp_r + (112u << 10u)) << 13u) | (mant << 13);
        uint32_t is_inf   = -(uint32_t)(exp_r == 0x7C00u);
        uint32_t is_zero  = -(uint32_t)(exp_r == 0u);
        uint32_t bits     = (inf_bits & is_inf) | (nrm_bits & ~is_inf);
        bits = bits & ~is_zero;                            /* subnormal/zero -> 0 */
        bits |= (sign & is_zero);                          /* preserve signed zero */
        memcpy(&f[s], &bits, 4);
    }
}

/* ---------------------------------------------------------------------- */
/* VEDA-launchable FA entry point.
 *
 * Mask layout matches the existing FA dispatch: F16, staged into HMEM by
 * the backend. For single-token decode it is a flat [seq_len] vector
 * (op->src[3]->ne[1] == 1). We convert F16 -> F32 once here, share the
 * F32 copy across all heads. */
/* ---------------------------------------------------------------------- */
uint64_t ve_flash_attn_ext_f32q_bf16kv_colmajor_hbm(
    VEDAdeviceptr out_vptr,
    VEDAdeviceptr q_vptr,
    VEDAdeviceptr k_col_vptr,
    VEDAdeviceptr v_col_vptr,
    VEDAdeviceptr mask_hbm,           /* F16 mask in HBM, may be 0 (no mask) */
    uint64_t head_dim,
    uint64_t n_q_heads,
    uint64_t n_kv_heads,
    uint64_t seq_len,
    uint64_t seq_max,
    uint64_t q_head_stride_bytes,    /* q->nb[2] — stride between heads in bytes */
    uint64_t out_head_stride_bytes,  /* dst->nb[2] */
    uint64_t scale_bits,             /* float bit-pattern */
    uint64_t slope_bits) {
    float * out;
    float * q;
    bf16 *  k_col;
    bf16 *  v_col;
    const uint16_t * mask_f16 = NULL;

    if (vedaMemPtr((void **)&out,   out_vptr)   != 0) return 1;
    if (vedaMemPtr((void **)&q,     q_vptr)     != 0) return 2;
    if (vedaMemPtr((void **)&k_col, k_col_vptr) != 0) return 3;
    if (vedaMemPtr((void **)&v_col, v_col_vptr) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void **)(void*)&mask_f16, mask_hbm) != 0) return 5;

    float scale, slope;
    uint32_t sb = (uint32_t) scale_bits, lb = (uint32_t) slope_bits;
    memcpy(&scale, &sb, 4);
    memcpy(&slope, &lb, 4);

    int hd = (int) head_dim;
    int nq = (int) n_q_heads;
    int nk = (int) n_kv_heads;
    int sl = (int) seq_len;
    int sm = (int) seq_max;
    int q_hstride   = (int)(q_head_stride_bytes   / sizeof(float));
    int out_hstride = (int)(out_head_stride_bytes / sizeof(float));

    /* F16 -> F32 mask once, shared across heads. 8192 covers ctx >= 8192. */
    float mask_f32[8192] __attribute__((aligned(64)));
    float * mask = NULL;
    if (mask_f16 != NULL) {
        mask_f16_to_f32(mask_f16, mask_f32, sl);
        mask = mask_f32;
    }

#pragma omp parallel
    {
        int h;
#pragma omp for private(h)
        for (h = 0; h < nq; h++) {
            int kv_h = h * nk / nq;
            const float * qh   = q     + (size_t) h    * q_hstride;
            float *       oh   = out   + (size_t) h    * out_hstride;
            const bf16 *  k_h  = k_col + (size_t) kv_h * hd * sm;
            const bf16 *  v_h  = v_col + (size_t) kv_h * hd * sm;
            fa_colmajor_single_head(oh, qh, k_h, v_h, mask,
                                    hd, sl, sm, scale, 1.0f);
        }
    }
    return 0;
}

/* Direct inner variant for compiled graph kernels. Caller must already be
 * inside an OpenMP parallel region; this function shares heads with the caller's
 * team and does not launch VEDA or convert pointers. The compiled graph uses
 * causal seq_len instead of an explicit mask, matching its row-major FA inner. */
void attention_f32q_bf16kv_colmajor_inner_strided(
    float * out,
    const float * q,
    const bf16 * k_col,
    const bf16 * v_col,
    int head_dim,
    int n_q_heads,
    int n_kv_heads,
    int seq_len,
    int seq_max,
    float scale,
    size_t q_head_stride_bytes,
    size_t out_head_stride_bytes) {

    int hd = head_dim;
    int nq = n_q_heads;
    int nk = n_kv_heads;
    int sl = seq_len;
    int sm = seq_max;
    int q_hstride   = (int)(q_head_stride_bytes   / sizeof(float));
    int out_hstride = (int)(out_head_stride_bytes / sizeof(float));

    int h;
#pragma omp for private(h)
    for (h = 0; h < nq; h++) {
        int kv_h = h * nk / nq;
        const float * qh   = q     + (size_t) h    * q_hstride;
        float *       oh   = out   + (size_t) h    * out_hstride;
        const bf16 *  k_h  = k_col + (size_t) kv_h * hd * sm;
        const bf16 *  v_h  = v_col + (size_t) kv_h * hd * sm;
        fa_colmajor_single_head(oh, qh, k_h, v_h, NULL,
                                hd, sl, sm, scale, 1.0f);
    }
}

/* ---------------------------------------------------------------------- */
/* Mirror a single just-written K (or V) row into the col-major shadow.
 *
 * src_hbm: F32 [head_dim * n_kv_heads], the freshly computed row in
 *          row-major (laid out as kv_head varies slower than head_dim,
 *          matching how SET_ROWS' src looks).
 * shadow_hbm: BF16 [n_kv_heads][head_dim][seq_max] — the persistent
 *             col-major shadow allocated by the backend.
 * row_idx: which seq position this row occupies in the cache.
 *
 * Per token we mirror two of these (one for K, one for V). The work is
 * head_dim * n_kv_heads F32→BF16 conversions plus a strided store; we
 * vectorise across head_dim per kv_head. The store has stride seq_max
 * (not unit) — unavoidable because the shadow's unit-stride axis is seq.
 * Still cheap: head_dim * n_kv_heads = 1024 ops per token per layer.
 * ---------------------------------------------------------------------- */
/* Mirror nrows rows from src into shadow at [row_start, row_start+nrows).
 *
 *   src is F32 [nrows][channels] row-major (the same shape SET_ROWS' src
 *   has). shadow is BF16 [channels][seq_max] col-major.
 *
 * For nr==1 (decode) this is the original single-row scatter: one stream
 * of F32->BF16 conversions writing one column slot per channel. For
 * nr>1 (prompt prefill) we iterate the row dim around it.
 *
 * NCC cannot vectorise either layout — the source is row-major (stride-1
 * in channels) and the destination is col-major (stride-seq_max in
 * channels), so one axis is always a strided store/load. We pick the
 * outer/inner ordering that minimises absolute work: outer over rows so
 * each row scans channels sequentially (better hardware prefetch for the
 * F32 source) and accepts the scalar stride-sm BF16 store.
 *
 * Cost: nrows × channels scalar stores. For Llama-3.2-3B prompt-eval
 * (channels=1024, nr≤32, 28 layers × 2 caches) this is ≤ 1.8M stores per
 * prompt ≈ 1-2 ms. For decode nr=1 it's ≤ 1024 stores per call ≈ 1 μs.
 *
 * On the day we want to make this fast we'd write a tiled BF16 transpose
 * in LLVM-VE-RV intrinsics — but the absolute time isn't the bottleneck
 * relative to the FA speedup the shadow enables.
 */
uint64_t ve_kvcache_mirror_to_colmajor_hbm(
    VEDAdeviceptr src_vptr,
    VEDAdeviceptr shadow_vptr,
    uint64_t row_start,
    uint64_t nrows,
    uint64_t channels,        /* head_dim * n_kv_heads */
    uint64_t seq_max) {
    float * src;
    bf16 *  shadow;
    if (vedaMemPtr((void **)&src,    src_vptr)    != 0) return 1;
    if (vedaMemPtr((void **)&shadow, shadow_vptr) != 0) return 2;

    int nc = (int) channels;
    int sm = (int) seq_max;
    int rs = (int) row_start;
    int nr = (int) nrows;

    for (int r = 0; r < nr; r++) {
        const float * src_row = src + (size_t) r * nc;
        int           ri      = rs + r;
#pragma _NEC ivdep
        for (int c = 0; c < nc; c++) {
            uint32_t u;
            memcpy(&u, &src_row[c], 4);
            shadow[(size_t) c * sm + ri] = (bf16) (u >> 16);
        }
    }
    return 0;
}
