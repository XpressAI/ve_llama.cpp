/*
 * Column-major Flash Attention (Stage 2).
 *
 * Layout (the only difference from flash_attn_bf16_intrinsics.c):
 *   K_col[kv_head][head_dim][seq] — seq is the unit-stride dim, BF16
 *   V_col[kv_head][head_dim][seq] — same shape, BF16
 *   Q[h][d]                       — FP32
 *   out[h][d]                     — FP32
 *
 * Pattern (per codex review):
 *   KQ:  for each query head:
 *          for each s_block in chunks of 2*VLEN seq positions:
 *            S_vec = 0  (kept in vector registers across all d)
 *            for d in [0, head_dim):
 *              q_vec = broadcast(Q[h][d])           — packed FP32
 *              k_vec = packed_load(K_col[kv_h][d][s_block..s_block+2*VLEN])
 *              S_vec = pvfmad(S_vec, q_vec, k_vec)
 *            store S[s_block..s_block+2*VLEN]
 *   Mask+softmax: scalar two-pass (max-reduce, then exp/normalize).
 *   SV:  for d in [0, head_dim):
 *          out[h][d] = dot(S, V_col[kv_h][d][:])     — reuses the existing
 *                                                      seq-dense BF16·FP32 dot
 *
 * Built with LLVM-VE-RV (clang --target=ve-linux), as for the other
 * intrinsics kernels. NCC cannot emit velintrin.
 */

#include <velintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#define VLEN 256
typedef uint16_t bf16;

/* Reused from flash_attn_bf16_intrinsics.c — these macros are stateless,
 * just inlined intrinsics. Duplicated here so this file compiles
 * standalone. */
#define load_bf16_to_packed_fp32(wv, wp, vl)                                    \
    do {                                                                        \
        wv = _vel_vldunc_vssl(4, (void *)(wp), vl);                             \
        __vr wr = _vel_vsrl_vvsl(wv, 16, vl);                                   \
        wr = _vel_vand_vvvl(wr, bf16mskl, vl);                                  \
        wv = _vel_vor_vvvl(wv, wr, vl);                                         \
    } while (0)

#define sumup_packed_fp32(tv, result, VL)                                       \
    do {                                                                        \
        __vr rv = _vel_vand_vvvl(tv, low32msk, VL);                             \
        __vr sv = _vel_vsll_vvsl(rv, 32, VL);                                   \
        tv = _vel_vfadds_vvvl(tv, sv, VL);                                      \
        tv = _vel_vfsums_vvl(tv, VL);                                           \
        result = _vel_lvss_svs(tv, 0);                                          \
    } while (0)

/* ----------------------------------------------------------------------
 * KQ pass: S[s] = sum_d Q[d] * K_col[d][s] for s in [0, seq_len).
 *
 * S is FP32, head_dim is the model's per-head depth (128 for Llama-3.2-3B).
 * k_h points at K_col[kv_h][0][0]; advancing by seq_max moves to the next
 * d-row. The pair (seq_max, seq_len) lets a single .so cover any decode
 * position: seq_max is the cache's allocated stride; seq_len is how much
 * we actually attend to this call.
 * ---------------------------------------------------------------------- */
/* Non-static so the NCC orchestrator can call it directly. */
void compute_KQ_colmajor(float * S,
                                       const float * qh,
                                       const bf16 * k_h,
                                       int head_dim,
                                       int seq_len,
                                       int seq_max) {
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);

    /* Iterate seq in chunks. For a chunk with `remaining` scalar positions
     * left, vl = number of packed PAIRS we'll process this iteration:
     *   - remaining >= 2*VLEN  → vl = VLEN  (full register)
     *   - else                 → vl = (remaining + 1) / 2  (partial, 1..VLEN-1)
     * The same shape the existing dot_bf16_fp32_intrinsics uses. */
    int s_block = 0;
    while (s_block < seq_len) {
        int remaining = seq_len - s_block;
        int vl = remaining < 2 * VLEN ? (remaining + 1) >> 1 : VLEN;
        if (vl <= 0) break;

        /* Zero packed accumulator — kept entirely in registers across d. */
        float zero[2] = { 0.0f, 0.0f };
        __vr S_vec = _vel_vld_vssl(0, &zero[0], vl);

        for (int d = 0; d < head_dim; d++) {
            /* Broadcast Q[d] into both halves of every 64-bit slot. */
            uint32_t qbits;
            memcpy(&qbits, &qh[d], 4);
            uint64_t qpacked = ((uint64_t) qbits << 32) | qbits;
            __vr q_vec = _vel_vbrdl_vsl((long) qpacked, vl);

            const bf16 * k_ds = k_h + (size_t) d * seq_max + s_block;
            __vr k_vec;
            load_bf16_to_packed_fp32(k_vec, k_ds, vl);

            S_vec = _vel_pvfmad_vvvvl(S_vec, q_vec, k_vec, vl);
        }

        /* Store S[s_block..s_block + 2*vl] back as packed FP32. The store
         * may write 2*vl elements; for odd remaining we wrote one extra
         * past seq_len. That's fine — S has at least 2*VLEN scratch slots
         * past seq_len in the caller's buffer (we allocate 8192). */
        if ((unsigned long)(S + s_block) & 0x7) {
            _vel_vstunc_vssl(S_vec, 8, (void *)(S + s_block + 1), vl);
            _vel_vstlnc_vssl(S_vec, 8, (void *)(S + s_block),     vl);
        } else {
            _vel_vstnc_vssl(S_vec, 8, (void *)(S + s_block), vl);
        }
        s_block += 2 * vl;
    }
}

/* ----------------------------------------------------------------------
 * SV pass: out[d] = sum_s S[s] * V_col[d][s] for d in [0, head_dim).
 *
 * Per codex's "treat V as BF16 SGEMV with probability vector as input":
 * process D_BLOCK output rows at once. S is loaded ONCE per s-block and
 * reused across D_BLOCK FMAs into D_BLOCK accumulators — cuts S memory
 * traffic by D_BLOCK. Each accumulator is held in a vector register
 * across all s-blocks; the VE has 64 vector regs so 4 fits easily.
 *
 * D_BLOCK must divide head_dim. Llama heads are usually 64/128/256 —
 * D_BLOCK=4 always works for those.
 * ---------------------------------------------------------------------- */
#define D_BLOCK 4

/* Non-static so the NCC orchestrator can call it directly. */
void compute_SV_colmajor(float * out_h,
                                       const float * S,
                                       const bf16 * v_h,
                                       int head_dim,
                                       int seq_len,
                                       int seq_max) {
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);
    __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffffUL, VLEN);
    float zero[2] = { 0.0f, 0.0f };

    int d_base;
    for (d_base = 0; d_base + D_BLOCK <= head_dim; d_base += D_BLOCK) {
        /* D_BLOCK accumulators in registers, alive across all s_blocks. */
        __vr acc0 = _vel_vld_vssl(0, &zero[0], VLEN);
        __vr acc1 = _vel_vld_vssl(0, &zero[0], VLEN);
        __vr acc2 = _vel_vld_vssl(0, &zero[0], VLEN);
        __vr acc3 = _vel_vld_vssl(0, &zero[0], VLEN);

        int j = 0;
        while (j < seq_len) {
            int remaining = seq_len - j;
            int vl = remaining < 2 * VLEN ? (remaining + 1) >> 1 : VLEN;
            if (vl <= 0) break;

            /* Load S[j..j + 2*vl] ONCE — reused across D_BLOCK below. */
            __vr sv;
            if (((unsigned long)(S + j)) & 0x7) {
                sv = _vel_vldu_vssl(8, (void *)(S + j + 1), vl);
                __vr slv = _vel_vldlzx_vssl(8, (void *)(S + j), vl);
                sv = _vel_pvor_vvvl(sv, slv, vl);
            } else {
                sv = _vel_vld_vssl(8, (void *)(S + j), vl);
            }

            const bf16 * v0 = v_h + (size_t) (d_base + 0) * seq_max + j;
            const bf16 * v1 = v_h + (size_t) (d_base + 1) * seq_max + j;
            const bf16 * v2 = v_h + (size_t) (d_base + 2) * seq_max + j;
            const bf16 * v3 = v_h + (size_t) (d_base + 3) * seq_max + j;
            __vr vv0, vv1, vv2, vv3;
            load_bf16_to_packed_fp32(vv0, v0, vl);
            load_bf16_to_packed_fp32(vv1, v1, vl);
            load_bf16_to_packed_fp32(vv2, v2, vl);
            load_bf16_to_packed_fp32(vv3, v3, vl);

            acc0 = _vel_pvfmad_vvvvl(acc0, sv, vv0, vl);
            acc1 = _vel_pvfmad_vvvvl(acc1, sv, vv1, vl);
            acc2 = _vel_pvfmad_vvvvl(acc2, sv, vv2, vl);
            acc3 = _vel_pvfmad_vvvvl(acc3, sv, vv3, vl);

            j += 2 * vl;
        }

        float r0, r1, r2, r3;
        sumup_packed_fp32(acc0, r0, VLEN);
        sumup_packed_fp32(acc1, r1, VLEN);
        sumup_packed_fp32(acc2, r2, VLEN);
        sumup_packed_fp32(acc3, r3, VLEN);
        out_h[d_base + 0] = r0;
        out_h[d_base + 1] = r1;
        out_h[d_base + 2] = r2;
        out_h[d_base + 3] = r3;
    }

    /* d-tail: any leftover heads after D_BLOCK chunks. For Llama-class
     * head_dim ∈ {64, 128, 256}, D_BLOCK=4 always divides; this loop is
     * dead. Kept for safety on odd head sizes. */
    for (int d = d_base; d < head_dim; d++) {
        __vr tv = _vel_vld_vssl(0, &zero[0], VLEN);
        const bf16 * v_d = v_h + (size_t) d * seq_max;
        int j = 0;
        while (j < seq_len) {
            int remaining = seq_len - j;
            int vl = remaining < 2 * VLEN ? (remaining + 1) >> 1 : VLEN;
            if (vl <= 0) break;
            __vr sv;
            if (((unsigned long)(S + j)) & 0x7) {
                sv = _vel_vldu_vssl(8, (void *)(S + j + 1), vl);
                __vr slv = _vel_vldlzx_vssl(8, (void *)(S + j), vl);
                sv = _vel_pvor_vvvl(sv, slv, vl);
            } else {
                sv = _vel_vld_vssl(8, (void *)(S + j), vl);
            }
            __vr vv;
            load_bf16_to_packed_fp32(vv, v_d + j, vl);
            tv = _vel_pvfmad_vvvvl(tv, sv, vv, vl);
            j += 2 * vl;
        }
        float r;
        sumup_packed_fp32(tv, r, VLEN);
        out_h[d] = r;
    }
}

/* ----------------------------------------------------------------------
 * Single-head FA in column-major. Two-pass softmax as per codex review.
 *
 * Exported (non-static) so the OMP wrapper can call it as the per-head
 * body. The wrapper itself lives in an NCC-compiled translation unit
 * because LLVM-VE-RV's `#pragma omp` lowers to __kmpc_* which is
 * incompatible with NCC's OpenMP runtime — the same reason the existing
 * flash_attn_single_head_intrinsics is no-OMP and ve_sgemv_wrapper.c
 * supplies the OMP `for`.
 * ---------------------------------------------------------------------- */
void flash_attn_colmajor_single_head(float *      out_h,
                                     const float * qh,
                                     const bf16 * k_h,
                                     const bf16 * v_h,
                                     const float * mask,
                                     int           head_dim,
                                     int           seq_len,
                                     int           seq_max,
                                     float         scale,
                                     float         slope) {
    /* Stack-allocated probability vector. Max ctx ≈ 4096-8192 means
     * 16-32 KB; comfortably fits the VE's stack. */
    float S[8192] __attribute__((aligned(64)));

    /* Step 1: S[s] = Q · K[s] for every s. */
    compute_KQ_colmajor(S, qh, k_h, head_dim, seq_len, seq_max);

    /* Step 2: apply scale + mask, find max. */
    float maxv = -INFINITY;
    if (mask) {
        for (int s = 0; s < seq_len; s++) {
            float v = S[s] * scale + mask[s] * slope;
            S[s] = v;
            if (v > maxv) maxv = v;
        }
    } else {
        for (int s = 0; s < seq_len; s++) {
            float v = S[s] * scale;
            S[s] = v;
            if (v > maxv) maxv = v;
        }
    }

    /* Step 3: exp(S - max), sum, normalize. */
    float sumv = 0.0f;
    for (int s = 0; s < seq_len; s++) {
        float e = expf(S[s] - maxv);
        S[s] = e;
        sumv += e;
    }
    float inv = (sumv > 0.0f) ? (1.0f / sumv) : 0.0f;
    for (int s = 0; s < seq_len; s++) {
        S[s] *= inv;
    }

    /* Step 4: out[d] = S · V[d][:] — D_BLOCK-wide SGEMV pattern. */
    compute_SV_colmajor(out_h, S, v_h, head_dim, seq_len, seq_max);
}

/* The OMP-parallel wrapper attention_f32q_bf16kv_colmajor_inner lives
 * in an NCC-compiled TU because LLVM-VE-RV's #pragma omp uses an
 * incompatible runtime. See ve_sgemv_wrapper.c (or the bench file
 * during Stage 2 testing) for the wrapper body. */
