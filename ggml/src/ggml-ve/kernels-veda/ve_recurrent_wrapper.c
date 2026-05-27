/*
 * VEDA wrappers for the recurrent-layer ops used by Qwen3.5 / Qwen3.6
 * (gated delta net + 1D causal conv). Both ops are pure F32 and run
 * entirely in HBM; nothing transits HMEM at compute time.
 *
 * Exports:
 *   ve_ssm_conv_f32_hbm
 *     1D depthwise causal conv that feeds the gated delta net. Per
 *     output element: dot product of length d_conv (typically 4) along
 *     the sliding-window input with the per-channel weight row. NCC
 *     auto-vectorises the channel loop, the d_conv loop unrolls.
 *
 *   ve_gated_delta_net_f32_hbm
 *     The fused linear-attention + state-update kernel (Yang et al.,
 *     "Gated Delta Networks"). Per token per head: scale state by
 *     exp(g), compute delta from k, rank-1 state update, attention
 *     output from q. State is S_v x S_v floats per head (typically
 *     128x128 = 64 KB) -- comfortably resident in cache during the
 *     per-head iteration. OMP-parallel over (head, sequence) pairs.
 *
 * Both kernels match the GGML CPU reference (ggml-cpu/ops.cpp) exactly;
 * see ggml_compute_forward_ssm_conv_f32 and
 * ggml_compute_forward_gated_delta_net_one_chunk.
 */

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <veda/device.h>

/* ---------------------------------------------------------------------- */
/* SSM_CONV: 1D causal convolution with sliding window.
 *
 * src0 (sliding window) : F32 [ncs, d_inner, n_seqs]   (ncs = d_conv-1 + n_t)
 * src1 (conv weight)    : F32 [d_conv, d_inner]
 * dst                   : F32 [d_inner, n_t, n_seqs]
 *
 * Strides (in floats) passed explicitly because the backend already has
 * them and the kernel shouldn't have to assume row-contiguous layouts:
 *   src0:  nb1_floats  along d_inner   (typically ncs)
 *          nb2_floats  along n_seqs    (typically ncs * d_inner)
 *   dst:   nb1_floats  along n_t       (typically d_inner)
 *          nb2_floats  along n_seqs    (typically d_inner * n_t)
 *
 * Per output[s, t, ch]:
 *   sum_{k=0..d_conv-1} src0[s, ch, t+k] * src1[ch, k]
 *
 * Parallelism: OMP over channels. For Qwen3.5 decode n_t=n_seqs=1, so
 * the work per call is d_inner * d_conv FMAs -- ~16k operations. Launch
 * overhead dominates; keep the kernel lean.
 */
uint64_t ve_ssm_conv_f32_hbm(
    VEDAdeviceptr dst_vptr,
    VEDAdeviceptr src0_vptr,
    VEDAdeviceptr src1_vptr,
    uint64_t d_conv,
    uint64_t d_inner,
    uint64_t n_t,
    uint64_t n_seqs,
    uint64_t src0_nb1_floats,
    uint64_t src0_nb2_floats,
    uint64_t dst_nb1_floats,
    uint64_t dst_nb2_floats) {

    float * dst;
    const float * src0;
    const float * src1;
    if (vedaMemPtr((void **)&dst,  dst_vptr)  != 0) return 1;
    if (vedaMemPtr((void **)(void*)&src0, src0_vptr) != 0) return 2;
    if (vedaMemPtr((void **)(void*)&src1, src1_vptr) != 0) return 3;

    const int nc    = (int) d_conv;
    const int di    = (int) d_inner;
    const int nt    = (int) n_t;
    const int ns    = (int) n_seqs;
    const int sn1   = (int) src0_nb1_floats;
    const int sn2   = (int) src0_nb2_floats;
    const int dn1   = (int) dst_nb1_floats;
    const int dn2   = (int) dst_nb2_floats;

#pragma omp parallel for collapse(2)
    for (int s = 0; s < ns; ++s) {
        for (int t = 0; t < nt; ++t) {
            float       * dout = dst  + s * dn2 + t * dn1;
            const float * sin0 = src0 + s * sn2;

            /* Channel loop. NCC vectorises across `ch`; the inner k
             * loop fully unrolls (d_conv == 4 in Qwen3.5). The two
             * source strides (sn1 channel-step on src0, nc on src1)
             * are stride-1-along-k inside per-channel rows. */
#pragma _NEC ivdep
            for (int ch = 0; ch < di; ++ch) {
                const float * sk = sin0 + ch * sn1 + t;       /* [k]   */
                const float * wk = src1 + ch * nc;            /* [k]   */
                float sum = 0.0f;
                for (int k = 0; k < nc; ++k) {
                    sum += sk[k] * wk[k];
                }
                dout[ch] = sum;
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* Generic strided F32->F32 copy with same logical shape on both sides.
 * Used for the conv_state slide ("conv_state_last := conv_state_update")
 * in Qwen3.5 recurrent layers, where both source and destination are
 * non-contiguous views of the same underlying conv-state buffer.
 *
 * Strides are passed in *floats*, not bytes (the host already has the
 * byte strides and divides by 4 before the call). Up to 4D.
 *
 * Inner-most dim is contiguous on both sides (ne0 stride == 1) -- that
 * lets NCC vectorise the innermost loop. The outer three dims iterate
 * scalar-strided, which is fine for the conv_state slide where each
 * outer iteration carries ~d_inner contiguous floats.
 */
uint64_t ve_copy_strided_f32_hbm(
    VEDAdeviceptr dst_vptr,
    VEDAdeviceptr src_vptr,
    uint64_t ne0, uint64_t ne1, uint64_t ne2, uint64_t ne3,
    uint64_t src_nb1_f, uint64_t src_nb2_f, uint64_t src_nb3_f,
    uint64_t dst_nb1_f, uint64_t dst_nb2_f, uint64_t dst_nb3_f) {

    float * dst;
    const float * src;
    if (vedaMemPtr((void **)&dst, dst_vptr) != 0) return 1;
    if (vedaMemPtr((void **)(void*)&src, src_vptr) != 0) return 2;

    const int n0 = (int) ne0;
    const int n1 = (int) ne1;
    const int n2 = (int) ne2;
    const int n3 = (int) ne3;
    const int sn1 = (int) src_nb1_f, sn2 = (int) src_nb2_f, sn3 = (int) src_nb3_f;
    const int dn1 = (int) dst_nb1_f, dn2 = (int) dst_nb2_f, dn3 = (int) dst_nb3_f;

#pragma omp parallel for collapse(3) if (n1*n2*n3 >= 8)
    for (int i3 = 0; i3 < n3; ++i3) {
        for (int i2 = 0; i2 < n2; ++i2) {
            for (int i1 = 0; i1 < n1; ++i1) {
                const float * s = src + i3 * sn3 + i2 * sn2 + i1 * sn1;
                float       * d = dst + i3 * dn3 + i2 * dn2 + i1 * dn1;
#pragma _NEC ivdep
                for (int i0 = 0; i0 < n0; ++i0) {
                    d[i0] = s[i0];
                }
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* GATED_DELTA_NET (Qwen3.5 / Qwen3.6 recurrent layer).
 *
 * Mirrors ggml_compute_forward_gated_delta_net_one_chunk in
 * ggml-cpu/ops.cpp. Inputs:
 *
 *   q, k : F32 [S_v, neq1, n_tokens, neq3]   (GQA: nev1 = rq1*neq1)
 *   v    : F32 [S_v, H,    n_tokens, n_seqs]
 *   g    : F32 [neg0, neg1, n_tokens, n_seqs]  with neg0 in {1, S_v}
 *   beta : F32 [1,    neb1, n_tokens, n_seqs]
 *   state: F32 [S_v*S_v*H, K, n_seqs] -- K snapshot slots
 *
 * Output dst is two regions concatenated in memory:
 *   attn  : F32 [S_v, H, n_tokens, n_seqs]
 *   state : F32 [S_v*S_v*H, n_seqs, K]
 *
 * Per (head, seq):
 *   1. copy input state slot 0 into a S_v*S_v workspace
 *   2. for each token t in n_tokens:
 *        a. scale state by exp(g)        (per-row if kda, else global)
 *        b. delta[j] = (v[j] - dot(row j, k)) * beta
 *        c. row j += delta[j] * k        (rank-1 update)
 *        d. attn[j] = dot(row j, q) * scale
 *        e. if K>1, store snapshot at slot t-shift when in range
 *
 * For decode (n_tokens=1, K=1) the work per (head, seq) is:
 *   - 1 state scale          : S_v*S_v FMAs
 *   - S_v dot products of S_v: S_v*S_v FMAs (delta)
 *   - rank-1 update          : S_v*S_v FMAs
 *   - S_v dot products       : S_v*S_v FMAs (attn)
 * = 4*S_v^2 FMAs per (head, seq). With S_v=128, H=32: ~2M FMAs per token.
 *
 * OMP-parallel over (head, seq); each thread runs the head-loop body
 * with all inner state ops vectorising trivially under NCC (length-S_v
 * F32 reductions and updates).
 */

/* Per-head body. State is held in a thread-local S_v*S_v scratch passed
 * by the caller (so we can hoist allocation out of the omp parallel and
 * have one well-aligned chunk per thread). */
static void gated_delta_net_one_head(
    float *       attn_out_h,      /* dst attn for this (head, seq) */
    float *       state_out_h,     /* dst state for this (head, seq) */
    const float * q_base,
    const float * k_base,
    const float * v_base,
    const float * g_base,
    const float * beta_base,
    const float * state_in_h,      /* slot-0 state ptr for this (head, seq) */
    float *       work,            /* scratch S_v*S_v floats, aligned */
    float *       delta,           /* scratch S_v floats */
    int           S_v,
    int           kda,
    int           n_tokens,
    int           K,
    int           shift,
    int           state_size_per_snap, /* S_v*S_v*H*n_seqs floats */
    int           h_idx_in_seq,        /* iv1 (head index inside seq) */
    int           seq_in_dst,          /* iv3 */
    int           H,
    int           q_nb1_f, int q_nb2_f,
    int           k_nb1_f, int k_nb2_f,
    int           v_nb1_f, int v_nb2_f,
    int           g_nb1_f, int g_nb2_f,
    int           b_nb1_f, int b_nb2_f,
    int           attn_step_floats,    /* S_v * H -- stride to next token */
    float         scale) {

    /* Copy input state into work buffer. */
    for (int idx = 0; idx < S_v * S_v; ++idx) work[idx] = state_in_h[idx];

    float * attn = attn_out_h;

    for (int t = 0; t < n_tokens; ++t) {
        const float * q_d = q_base    + t * q_nb2_f;
        const float * k_d = k_base    + t * k_nb2_f;
        const float * v_d = v_base    + t * v_nb2_f;
        const float   bv  = beta_base[t * b_nb2_f];
        const float * g_d = g_base    + t * g_nb2_f;

        /* (a) scale state by exp(g). */
        if (kda) {
            /* per-row: row j of work has S_v contiguous floats, scale by exp(g[i]). */
#pragma _NEC ivdep
            for (int i = 0; i < S_v; ++i) {
                delta[i] = expf(g_d[i]);   /* reuse delta as exp(g) scratch */
            }
            for (int j = 0; j < S_v; ++j) {
                float * row = work + j * S_v;
#pragma _NEC ivdep
                for (int i = 0; i < S_v; ++i) row[i] *= delta[i];
            }
        } else {
            const float eg = expf(g_d[0]);
            const int tot = S_v * S_v;
#pragma _NEC ivdep
            for (int idx = 0; idx < tot; ++idx) work[idx] *= eg;
        }

        /* (b) delta[j] = (v[j] - dot(row j, k)) * beta. */
        for (int j = 0; j < S_v; ++j) {
            const float * row = work + j * S_v;
            float sum = 0.0f;
#pragma _NEC ivdep
            for (int i = 0; i < S_v; ++i) sum += row[i] * k_d[i];
            delta[j] = (v_d[j] - sum) * bv;
        }

        /* (c) rank-1 update: row j += delta[j] * k. */
        for (int j = 0; j < S_v; ++j) {
            float * row = work + j * S_v;
            const float dj = delta[j];
#pragma _NEC ivdep
            for (int i = 0; i < S_v; ++i) row[i] += dj * k_d[i];
        }

        /* (d) attn[j] = dot(row j, q) * scale. */
        for (int j = 0; j < S_v; ++j) {
            const float * row = work + j * S_v;
            float sum = 0.0f;
#pragma _NEC ivdep
            for (int i = 0; i < S_v; ++i) sum += row[i] * q_d[i];
            attn[j] = sum * scale;
        }
        attn += attn_step_floats;

        /* (e) per-token snapshot when K>1 and slot in range. */
        if (K > 1) {
            const int target_slot = t - shift;
            if (target_slot >= 0 && target_slot < K) {
                float * snap = state_out_h + target_slot * state_size_per_snap;
                /* state_out_h already includes the (iv3*H+iv1)*S_v*S_v offset,
                 * but per-snapshot indexing needs the base of THIS snapshot's
                 * head block. The caller passes state_out_h as
                 * state_out_base + (iv3*H+iv1)*S_v*S_v so we add the slot
                 * step directly. */
                for (int idx = 0; idx < S_v * S_v; ++idx) snap[idx] = work[idx];
            }
        }
    }

    /* K==1 path: final state lives where state_out_h points. */
    if (K == 1) {
        for (int idx = 0; idx < S_v * S_v; ++idx) state_out_h[idx] = work[idx];
    }
}

/* Up to S_v=256 head_dim. State scratch is 256*256*4 = 256 KB per thread.
 * 8 cores * 256 KB = 2 MB total -- comfortable. */
#define GDN_MAX_SV 256

uint64_t ve_gated_delta_net_f32_hbm(
    VEDAdeviceptr dst_vptr,
    VEDAdeviceptr q_vptr,
    VEDAdeviceptr k_vptr,
    VEDAdeviceptr v_vptr,
    VEDAdeviceptr g_vptr,
    VEDAdeviceptr beta_vptr,
    VEDAdeviceptr state_vptr,
    uint64_t S_v,
    uint64_t H,
    uint64_t n_tokens,
    uint64_t n_seqs,
    uint64_t K,
    uint64_t kda_flag,
    /* Q strides (floats) */
    uint64_t neq1, uint64_t neq3,
    uint64_t qnb1_f, uint64_t qnb2_f, uint64_t qnb3_f,
    /* K strides */
    uint64_t nek1, uint64_t nek3,
    uint64_t knb1_f, uint64_t knb2_f, uint64_t knb3_f,
    /* V strides */
    uint64_t vnb1_f, uint64_t vnb2_f, uint64_t vnb3_f,
    /* g strides + leading dim */
    uint64_t neg0,
    uint64_t gnb1_f, uint64_t gnb2_f, uint64_t gnb3_f,
    /* beta strides */
    uint64_t bnb1_f, uint64_t bnb2_f, uint64_t bnb3_f,
    /* state per-seq stride */
    uint64_t state_seq_stride_f) {

    float * dst;
    const float * q;  const float * k;  const float * v;
    const float * g;  const float * beta;
    const float * state_in;
    if (vedaMemPtr((void **)&dst,                   dst_vptr)   != 0) return 1;
    if (vedaMemPtr((void **)(void*)&q,     q_vptr)     != 0) return 2;
    if (vedaMemPtr((void **)(void*)&k,     k_vptr)     != 0) return 3;
    if (vedaMemPtr((void **)(void*)&v,     v_vptr)     != 0) return 4;
    if (vedaMemPtr((void **)(void*)&g,     g_vptr)     != 0) return 5;
    if (vedaMemPtr((void **)(void*)&beta,  beta_vptr)  != 0) return 6;
    if (vedaMemPtr((void **)(void*)&state_in, state_vptr) != 0) return 7;

    const int Sv = (int) S_v;
    const int Hi = (int) H;
    const int nt = (int) n_tokens;
    const int ns = (int) n_seqs;
    const int Ki = (int) K;
    const int kda = kda_flag ? 1 : 0;
    if (Sv <= 0 || Sv > GDN_MAX_SV) return 10;

    /* Output layout matches CPU:
     *   attn_scores : S_v * H * n_tokens * n_seqs   floats
     *   new_states  : S_v * S_v * H * n_seqs * K    floats
     */
    const int attn_score_elems    = Sv * Hi * nt * ns;
    const int state_size_per_snap = Sv * Sv * Hi * ns;
    float * attn_out_base  = dst;
    float * state_out_base = dst + attn_score_elems;

    /* GQA broadcast: nev1 = Hi, neq1/nek1 are smaller heads counts. iv3 maps
     * to iq3 = iv3/rq3 etc. rq3 = nev3/neq3 = ns/neq3. */
    const int neq3_i = (int) neq3;
    const int nek3_i = (int) nek3;
    const int rq3    = ns / (neq3_i ? neq3_i : 1);
    const int rk3    = ns / (nek3_i ? nek3_i : 1);
    const int neq1_i = (int) neq1;
    const int nek1_i = (int) nek1;

    const float scale = 1.0f / sqrtf((float) Sv);
    const int shift = nt - Ki;

#pragma omp parallel
    {
        float work[GDN_MAX_SV * GDN_MAX_SV] __attribute__((aligned(64)));
        float delta[GDN_MAX_SV]              __attribute__((aligned(64)));

        const int nr = Hi * ns;  /* (head, seq) pairs */
#pragma omp for
        for (int ir = 0; ir < nr; ++ir) {
            const int iv1 = ir % Hi;            /* head */
            const int iv3 = ir / Hi;            /* sequence */
            const int iq1 = iv1 % (neq1_i ? neq1_i : 1);
            const int ik1 = iv1 % (nek1_i ? nek1_i : 1);
            const int iq3 = iv3 / (rq3 ? rq3 : 1);
            const int ik3 = iv3 / (rk3 ? rk3 : 1);

            const float * q_base = q + iq3 * qnb3_f + iq1 * qnb1_f;
            const float * k_base = k + ik3 * knb3_f + ik1 * knb1_f;
            const float * v_base = v + iv3 * vnb3_f + iv1 * vnb1_f;
            const float * g_base = g + iv3 * gnb3_f + iv1 * gnb1_f;
            const float * b_base = beta + iv3 * bnb3_f + iv1 * bnb1_f;

            const float * state_in_h = state_in
                + iv3 * state_seq_stride_f
                + iv1 * Sv * Sv;

            float * attn_out_h =
                attn_out_base + (iv3 * nt * Hi + iv1) * Sv;
            float * state_out_h = (Ki > 1)
                ? state_out_base + (iv3 * Hi + iv1) * Sv * Sv
                : state_out_base + (iv3 * Hi + iv1) * Sv * Sv;

            gated_delta_net_one_head(
                attn_out_h, state_out_h,
                q_base, k_base, v_base, g_base, b_base,
                state_in_h,
                work, delta,
                Sv, kda,
                nt, Ki, shift, state_size_per_snap,
                iv1, iv3, Hi,
                (int) qnb1_f, (int) qnb2_f,
                (int) knb1_f, (int) knb2_f,
                (int) vnb1_f, (int) vnb2_f,
                (int) gnb1_f, (int) gnb2_f,
                (int) bnb1_f, (int) bnb2_f,
                Sv * Hi,
                scale);
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* General F32 dup/cpy for differing logical shapes with matching total
 * element count. Mirrors ggml-cpu's dup_bytes path: walk the source in
 * row-major order, map each element's linear index to a dst position via
 * dst's own ne[]/nb[]. Required by Qwen3.5's conv_state slide where
 * src is 3D [d_conv-1, conv_dim, n_seqs] and dst is 2D
 * [(d_conv-1)*conv_dim, n_seqs] (a re-shape on the way back into the
 * recurrent cache).
 *
 * One launch per call, scalar per element. The conv_state slides for
 * Qwen3.5 are tiny (~12k floats per layer per token); launch overhead
 * dominates so we don't bother with a vectorised fast path here.
 */
uint64_t ve_copy_bytes_f32_hbm(
    VEDAdeviceptr dst_vptr,
    VEDAdeviceptr src_vptr,
    uint64_t s_ne0, uint64_t s_ne1, uint64_t s_ne2, uint64_t s_ne3,
    uint64_t s_nb1_f, uint64_t s_nb2_f, uint64_t s_nb3_f,
    uint64_t d_ne0, uint64_t d_ne1, uint64_t d_ne2, uint64_t d_ne3,
    uint64_t d_nb1_f, uint64_t d_nb2_f, uint64_t d_nb3_f) {

    float * dst;
    const float * src;
    if (vedaMemPtr((void **)&dst, dst_vptr) != 0) return 1;
    if (vedaMemPtr((void **)(void*)&src, src_vptr) != 0) return 2;

    const long sN0 = (long) s_ne0, sN1 = (long) s_ne1, sN2 = (long) s_ne2, sN3 = (long) s_ne3;
    const long dN0 = (long) d_ne0, dN1 = (long) d_ne1, dN2 = (long) d_ne2;
    (void) d_ne3;  /* d_ne3 implied by total count */

    const long s1 = (long) s_nb1_f, s2 = (long) s_nb2_f, s3 = (long) s_nb3_f;
    const long d1 = (long) d_nb1_f, d2 = (long) d_nb2_f, d3 = (long) d_nb3_f;

    const long total = sN0 * sN1 * sN2 * sN3;

#pragma omp parallel for
    for (long k = 0; k < total; ++k) {
        /* Source: row-major decomposition by src's ne. */
        long i0 = k % sN0;
        long i1 = (k / sN0) % sN1;
        long i2 = (k / (sN0 * sN1)) % sN2;
        long i3 =  k / (sN0 * sN1 * sN2);
        const long s_off = i0 + i1 * s1 + i2 * s2 + i3 * s3;

        /* Destination: row-major decomposition by dst's ne. */
        long j0 = k % dN0;
        long j1 = (k / dN0) % dN1;
        long j2 = (k / (dN0 * dN1)) % dN2;
        long j3 =  k / (dN0 * dN1 * dN2);
        const long d_off = j0 + j1 * d1 + j2 * d2 + j3 * d3;

        dst[d_off] = src[s_off];
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* F32 -> "BF16 precision" in-place truncation.
 *
 * Matches what CPU's GGML does for MUL_MAT with src0=BF16, src1=F32:
 *   vec_dot_type[BF16] = BF16, so GGML converts src1 from F32 to BF16
 *   first (top 16 bits, round-to-nearest-even via from_float), then runs
 *   ggml_vec_dot_bf16. We mimic the same precision loss by masking the
 *   low 16 bits of each F32 to zero (truncate-toward-zero — not RTNE,
 *   but close enough for the Qwen3.5 precision-matching experiment).
 *
 * Operates in-place on the F32 buffer.
 */
uint64_t ve_f32_truncate_to_bf16_precision_inplace(
    VEDAdeviceptr buf_vptr,
    uint64_t n) {
    uint32_t * buf;
    if (vedaMemPtr((void **)&buf, buf_vptr) != 0) return 1;
    const long N = (long) n;
#pragma omp parallel for
    for (long i = 0; i < N; ++i) {
        buf[i] &= 0xFFFF0000u;
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* SUM_ROWS:  dst[0, i1, i2, i3] = sum_i0 src[i0, i1, i2, i3]
 *
 * src strides passed in floats (sn1, sn2, sn3 — stride between successive
 * i1, i2, i3 slices). dst strides similarly (dn1, dn2, dn3). dst->ne[0]
 * must be 1.
 *
 * Used by Qwen3.5 gated-delta-net.
 */
uint64_t ve_sum_rows_f32_hbm(
    VEDAdeviceptr dst_vptr,
    VEDAdeviceptr src_vptr,
    uint64_t ne00,
    uint64_t ne01,
    uint64_t ne02,
    uint64_t ne03,
    uint64_t sn1_f, uint64_t sn2_f, uint64_t sn3_f,
    uint64_t dn1_f, uint64_t dn2_f, uint64_t dn3_f) {

    float * dst;
    const float * src;
    if (vedaMemPtr((void **)&dst, dst_vptr) != 0) return 1;
    if (vedaMemPtr((void **)(void*)&src, src_vptr) != 0) return 2;

    const int n0 = (int) ne00;
    const int n1 = (int) ne01;
    const int n2 = (int) ne02;
    const int n3 = (int) ne03;
    const int s1 = (int) sn1_f, s2 = (int) sn2_f, s3 = (int) sn3_f;
    const int d1 = (int) dn1_f, d2 = (int) dn2_f, d3 = (int) dn3_f;

#pragma omp parallel for collapse(3) if (n1*n2*n3 >= 8)
    for (int i3 = 0; i3 < n3; ++i3) {
        for (int i2 = 0; i2 < n2; ++i2) {
            for (int i1 = 0; i1 < n1; ++i1) {
                const float * s = src + i3 * s3 + i2 * s2 + i1 * s1;
                double acc = 0.0;
#pragma _NEC ivdep
                for (int i0 = 0; i0 < n0; ++i0) acc += s[i0];
                dst[i3 * d3 + i2 * d2 + i1 * d1] = (float) acc;
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* REPEAT:  tile src across all dims so dst->ne = nr * src->ne
 * Both contiguous. Innermost contiguous copy is ne00 floats per chunk.
 *
 * Used by Qwen3.5 GDN to expand k_in / v_in from [head_v_dim, 1, num_heads]
 * to [head_v_dim, head_v_dim, num_heads].
 */
uint64_t ve_repeat_f32_hbm(
    VEDAdeviceptr dst_vptr,
    VEDAdeviceptr src_vptr,
    uint64_t ne00, uint64_t ne01, uint64_t ne02, uint64_t ne03,
    uint64_t nr0,  uint64_t nr1,  uint64_t nr2,  uint64_t nr3) {

    float * dst;
    const float * src;
    if (vedaMemPtr((void **)&dst, dst_vptr) != 0) return 1;
    if (vedaMemPtr((void **)(void*)&src, src_vptr) != 0) return 2;

    const long n00 = (long) ne00, n01 = (long) ne01;
    const long n02 = (long) ne02, n03 = (long) ne03;
    const long r0  = (long) nr0,  r1  = (long) nr1;
    const long r2  = (long) nr2,  r3  = (long) nr3;

    const long dN0 = n00 * r0;
    const long dN1 = n01 * r1;
    const long dN2 = n02 * r2;
    /* dN3 = n03 * r3 — implied by total iteration. */

    /* Walk the dst index space. For each output coord (j0, j1, j2, j3)
     * the source coord is (j0 % n00, j1 % n01, j2 % n02, j3 % n03). */
#pragma omp parallel for collapse(3) if (dN1*dN2 >= 8)
    for (long j3 = 0; j3 < n03 * r3; ++j3) {
        for (long j2 = 0; j2 < dN2; ++j2) {
            for (long j1 = 0; j1 < dN1; ++j1) {
                const long k1 = j1 % n01;
                const long k2 = j2 % n02;
                const long k3 = j3 % n03;
                const float * s = src + k3 * (n00 * n01 * n02)
                                       + k2 * (n00 * n01)
                                       + k1 * n00;
                float * d = dst + j3 * (dN0 * dN1 * dN2)
                                + j2 * (dN0 * dN1)
                                + j1 * dN0;
                /* Tile the innermost ne00 floats r0 times. */
                for (long i = 0; i < r0; ++i) {
#pragma _NEC ivdep
                    for (long k0 = 0; k0 < n00; ++k0) d[i*n00 + k0] = s[k0];
                }
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* CONCAT along an arbitrary dim 0..3. Both sources F32, contiguous (the
 * common path; non-contiguous can be staged through CPY first if needed).
 *
 * dim       which axis to concatenate along
 * a_ne[i]   src0 dims
 * b_ne[i]   src1 dims (== a_ne[i] except for i==dim)
 * dst_ne[i] = a_ne[i] for i != dim,  a_ne[dim] + b_ne[dim] for i == dim
 *
 * All-contig fast path: walk dst row-major; the index space splits into
 * "from a" and "from b" regions along `dim`.
 */
uint64_t ve_concat_f32_hbm(
    VEDAdeviceptr dst_vptr,
    VEDAdeviceptr a_vptr,
    VEDAdeviceptr b_vptr,
    uint64_t dim,
    uint64_t a_ne0, uint64_t a_ne1, uint64_t a_ne2, uint64_t a_ne3,
    uint64_t b_ne0, uint64_t b_ne1, uint64_t b_ne2, uint64_t b_ne3) {

    float * dst;
    const float * a;
    const float * b;
    if (vedaMemPtr((void **)&dst,                 dst_vptr) != 0) return 1;
    if (vedaMemPtr((void **)(void*)&a,            a_vptr)   != 0) return 2;
    if (vedaMemPtr((void **)(void*)&b,            b_vptr)   != 0) return 3;

    const long aN[4] = { (long) a_ne0, (long) a_ne1, (long) a_ne2, (long) a_ne3 };
    const long bN[4] = { (long) b_ne0, (long) b_ne1, (long) b_ne2, (long) b_ne3 };
    long dN[4];
    for (int i = 0; i < 4; ++i) {
        dN[i] = aN[i];
        if (i == (int) dim) dN[i] = aN[i] + bN[i];
    }
    const long aOff[4] = { 0, 0, 0, 0 };  /* a starts at 0 along every dim */
    long bOff[4] = { 0, 0, 0, 0 };
    bOff[dim] = aN[dim];  /* b starts where a ends along the concat axis */

    /* Strides in float units (contiguous). */
    const long aS[4] = { 1, aN[0], aN[0]*aN[1], aN[0]*aN[1]*aN[2] };
    const long bS[4] = { 1, bN[0], bN[0]*bN[1], bN[0]*bN[1]*bN[2] };
    const long dS[4] = { 1, dN[0], dN[0]*dN[1], dN[0]*dN[1]*dN[2] };

#pragma omp parallel for collapse(2) if (dN[2]*dN[3] >= 4)
    for (long j3 = 0; j3 < dN[3]; ++j3) {
        for (long j2 = 0; j2 < dN[2]; ++j2) {
            for (long j1 = 0; j1 < dN[1]; ++j1) {
                /* Vectorise the innermost loop. */
                for (long j0 = 0; j0 < dN[0]; ++j0) {
                    int from_b = 0;
                    long i0 = j0, i1 = j1, i2 = j2, i3 = j3;
                    switch ((int) dim) {
                        case 0: if (j0 >= aN[0]) { from_b = 1; i0 = j0 - aN[0]; } break;
                        case 1: if (j1 >= aN[1]) { from_b = 1; i1 = j1 - aN[1]; } break;
                        case 2: if (j2 >= aN[2]) { from_b = 1; i2 = j2 - aN[2]; } break;
                        case 3: if (j3 >= aN[3]) { from_b = 1; i3 = j3 - aN[3]; } break;
                        default: break;
                    }
                    float v;
                    if (from_b) v = b[i0*bS[0] + i1*bS[1] + i2*bS[2] + i3*bS[3]];
                    else        v = a[i0*aS[0] + i1*aS[1] + i2*aS[2] + i3*aS[3]];
                    dst[j0*dS[0] + j1*dS[1] + j2*dS[2] + j3*dS[3]] = v;
                }
            }
        }
    }
    /* silence aOff unused-warning */
    (void) aOff[0];
    return 0;
}
