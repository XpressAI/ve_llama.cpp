/*
 * Multi-Q tiled Flash Attention prefill kernel — LLVM-VE-RV intrinsics.
 *
 * What this gets us over the NCC tile kernel (ve_flash_attn_ext_f32q_bf16kv_tile_hbm):
 *
 *  - K and V decoded BF16 -> packed FP32 ONCE per ic into stack-local
 *    F32 buffers using packed pvfmad-friendly layout, then reused across
 *    nq queries. NCC's tile kernel already decodes once per ic, but it
 *    can't pack the layout — its per-Q dot ends up doing unpacked
 *    256-lane reductions, which on VE2 is roughly the same throughput as
 *    packed at d=128 but loses on instruction pipelining and emits more
 *    address computation.
 *
 *  - Per-Q dot, scale, and accumulate are explicit packed-FP32 intrinsic
 *    sequences (pvfmad chains). This bypasses NCC's reduction-idiom
 *    codegen for each query, which spends cycles on partial-tree reduce
 *    and address fixup. By using vfsums on the packed accumulator we
 *    do the reduce in 4 vector ops regardless of nq.
 *
 *  - Branched online softmax (1 expf per Q in the common path) avoids
 *    the unconditional-form's doubled VKQ FMA tax that crippled the
 *    vexp variant.
 *
 * Caller in ve_sgemv_wrapper.c handles OMP outer parallelism. This
 * function processes one (batch, head, q_tile).
 */

#include <velintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define VLEN 256
typedef uint16_t bf16;

/* Decode a BF16 array of n elements to F32 written to `out`, using the
 * standard packed-FP32 unpacking pattern. `out` must be 8-byte aligned. */
static inline void decode_bf16_array(float * __restrict out,
                                     const bf16 * __restrict in,
                                     int n)
{
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);

    for (int chunk = 0; chunk < n; chunk += 2 * VLEN) {
        int remaining = n - chunk;
        int vl = remaining >> 1;
        if (vl <= 0) break;
        if (vl > VLEN) vl = VLEN;

        /* Load 4-byte words from BF16 stream, then unpack the low BF16
         * into the lower 32 bits of each packed slot. */
        __vr wv = _vel_vldunc_vssl(4, (void *)(in + chunk), vl);
        __vr wr = _vel_vsrl_vvsl(wv, 16, vl);
        wr = _vel_vand_vvvl(wr, bf16mskl, vl);
        wv = _vel_vor_vvvl(wv, wr, vl);

        /* Store as packed 8-byte (2 FP32 per slot) to out. We require
         * out 8-byte aligned (asserted by caller). */
        _vel_vstnc_vssl(wv, 8, (void *)(out + chunk), vl);
    }
}

/* F32 dot product Q . K via packed-FP32 FMA. Both q and k are F32 (k
 * must be 8-byte aligned; q is read with the unaligned-pair pattern if
 * needed). Returns scalar sum. */
static inline float dot_f32_packed(const float * __restrict q,
                                   const float * __restrict k,
                                   int n)
{
    __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffffUL, VLEN);
    float zero[2] = { 0.0f, 0.0f };
    __vr acc = _vel_vld_vssl(0, &zero[0], VLEN);

    int acc_vl = 0;
    for (int chunk = 0; chunk < n; chunk += 2 * VLEN) {
        int remaining = n - chunk;
        int vl = remaining >> 1;
        if (vl <= 0) break;
        if (vl > VLEN) vl = VLEN;
        acc_vl = vl;

        __vr qv;
        if (((uintptr_t)(q + chunk)) & 7) {
            qv = _vel_vldu_vssl(8, (void *)(q + chunk + 1), vl);
            __vr qlv = _vel_vldlzx_vssl(8, (void *)(q + chunk), vl);
            qv = _vel_pvor_vvvl(qv, qlv, vl);
        } else {
            qv = _vel_vld_vssl(8, (void *)(q + chunk), vl);
        }

        __vr kv = _vel_vld_vssl(8, (void *)(k + chunk), vl);

        acc = _vel_pvfmad_vvvvl(acc, qv, kv, vl);
    }

    /* Reduce packed accumulator: split upper/lower lanes, sum, then
     * vfsums collapses across the vector. */
    __vr rv = _vel_vand_vvvl(acc, low32msk, acc_vl);
    __vr sv = _vel_vsll_vvsl(rv, 32, acc_vl);
    acc = _vel_vfadds_vvvl(acc, sv, acc_vl);
    acc = _vel_vfsums_vvl(acc, acc_vl);
    return _vel_lvss_svs(acc, 0);
}

/* result[i] *= scale, F32, packed. */
static inline void scale_f32_packed(float * __restrict result, float scale, int n)
{
    uint32_t s32; __builtin_memcpy(&s32, &scale, sizeof(s32));
    uint64_t sbits = ((uint64_t)s32 << 32) | s32;
    __vr s_packed = _vel_vbrdl_vsl(sbits, VLEN);

    for (int chunk = 0; chunk < n; chunk += 2 * VLEN) {
        int remaining = n - chunk;
        int vl = remaining >> 1;
        if (vl <= 0) break;
        if (vl > VLEN) vl = VLEN;

        __vr rv;
        if (((uintptr_t)(result + chunk)) & 7) {
            rv = _vel_vldu_vssl(8, (void *)(result + chunk + 1), vl);
            __vr rlv = _vel_vldlzx_vssl(8, (void *)(result + chunk), vl);
            rv = _vel_pvor_vvvl(rv, rlv, vl);
        } else {
            rv = _vel_vld_vssl(8, (void *)(result + chunk), vl);
        }

        rv = _vel_pvfmul_vvvl(rv, s_packed, vl);

        if (((uintptr_t)(result + chunk)) & 7) {
            _vel_vstunc_vssl(rv, 8, (void *)(result + chunk + 1), vl);
            _vel_vstlnc_vssl(rv, 8, (void *)(result + chunk), vl);
        } else {
            _vel_vstnc_vssl(rv, 8, (void *)(result + chunk), vl);
        }
    }
}

/* result[i] += weight * v[i], F32 result + F32 v, packed FMA. */
static inline void accumulate_f32_packed(float * __restrict result,
                                         const float * __restrict v,
                                         float weight, int n)
{
    uint32_t w32; __builtin_memcpy(&w32, &weight, sizeof(w32));
    uint64_t wbits = ((uint64_t)w32 << 32) | w32;
    __vr w_packed = _vel_vbrdl_vsl(wbits, VLEN);

    for (int chunk = 0; chunk < n; chunk += 2 * VLEN) {
        int remaining = n - chunk;
        int vl = remaining >> 1;
        if (vl <= 0) break;
        if (vl > VLEN) vl = VLEN;

        __vr rv;
        if (((uintptr_t)(result + chunk)) & 7) {
            rv = _vel_vldu_vssl(8, (void *)(result + chunk + 1), vl);
            __vr rlv = _vel_vldlzx_vssl(8, (void *)(result + chunk), vl);
            rv = _vel_pvor_vvvl(rv, rlv, vl);
        } else {
            rv = _vel_vld_vssl(8, (void *)(result + chunk), vl);
        }

        /* v is F32 (we decoded it from BF16 to packed FP32 layout). */
        __vr vv = _vel_vld_vssl(8, (void *)(v + chunk), vl);

        rv = _vel_pvfmad_vvvvl(rv, w_packed, vv, vl);

        if (((uintptr_t)(result + chunk)) & 7) {
            _vel_vstunc_vssl(rv, 8, (void *)(result + chunk + 1), vl);
            _vel_vstlnc_vssl(rv, 8, (void *)(result + chunk), vl);
        } else {
            _vel_vstnc_vssl(rv, 8, (void *)(result + chunk), vl);
        }
    }
}

/* F16 -> F32 conversion (same bit-twiddle as the NCC kernels). */
static inline float f16_to_f32_inline(uint16_t mf16)
{
    uint32_t sign  = (mf16 >> 15) & 0x1;
    uint32_t exp_b = (mf16 >> 10) & 0x1F;
    uint32_t mant  = mf16 & 0x3FF;
    uint32_t f32;
    if (exp_b == 0) {
        f32 = (mant == 0) ? (sign << 31) : 0;
    } else if (exp_b == 31) {
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp_b + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    __builtin_memcpy(&out, &f32, sizeof(out));
    return out;
}

/* === pack-2-queries variant ===========================================
 *
 * The per-Q intrinsic kernel above (flash_attn_tile_inner_intrinsics)
 * is correct and bit-identical to the reference, but at d=128 it loses
 * to NCC's auto-vec: packed VL=d_pairs=64 has the same throughput as
 * unpacked VL=128, and packed emits more total vector instructions.
 *
 * This variant fixes that by packing TWO queries per packed slot. K is
 * decoded once per ic, then DUPLICATED so each packed slot holds K[i]
 * in both upper and lower lanes. Q_pair is built with Q[j][i] in the
 * upper lane and Q[j+1][i] in the lower lane. One pvfmad chain then
 * accumulates Q[j].K in the upper lane and Q[j+1].K in the lower lane
 * simultaneously — the chain still issues at vl=d=128, but does 256
 * useful FMAs (2 lanes × 128 slots) where unpacked would do 128.
 *
 *   dot_j   = vfsums(extract_upper(acc))
 *   dot_j+1 = vfsums(extract_lower(acc))
 *
 * Same trick for the VKQ accumulate: V_dup has V[i] duplicated to both
 * lanes, weight_pair has (vs_j, vs_j+1) broadcast per slot, one pvfmad
 * does both accumulates.
 *
 * The rare "scale VKQ by exp(M_old - M_new)" step is handled per-query
 * via scale_f32_packed — that branch fires ~log(N_kv) times so it does
 * not need pack-2 treatment.
 */

/* Build K_dup: each packed slot[i] holds K[i] in both upper and lower
 * lanes. Input k_f32 is the decoded F32 K row; output is one __vr
 * register valid for vl = d slots. */
static inline __vr build_dup_vector(const float * x_f32, int n)
{
    __vr xu = _vel_vldu_vssl(4, (void *)x_f32, n);       /* upper = x[i] */
    __vr xl = _vel_vsrl_vvsl(xu, 32, n);                 /* lower = x[i] */
    return _vel_vor_vvvl(xu, xl, n);
}

/* Compute two dot products at once. Q is loaded with Q[j0][i] in the
 * upper lane and Q[j1][i] in the lower lane of slot i. K_dup_v is the
 * pre-built K-duplicated vector. Returns dot(Q[j0], K) in *s0_out and
 * dot(Q[j1], K) in *s1_out. */
static inline void dot_pair_packed(const float * Q_j0,
                                   const float * Q_j1,
                                   __vr K_dup_v,
                                   int d,
                                   float * s0_out, float * s1_out)
{
    /* Q_pair: upper = Q[j0][i], lower = Q[j1][i] */
    __vr qu = _vel_vldu_vssl  (4, (void *)Q_j0, d);
    __vr ql = _vel_vldlzx_vssl(4, (void *)Q_j1, d);
    __vr Q_pair = _vel_pvor_vvvl(qu, ql, d);

    float zero[2] = { 0.0f, 0.0f };
    __vr acc = _vel_vld_vssl(0, &zero[0], d);
    acc = _vel_pvfmad_vvvvl(acc, Q_pair, K_dup_v, d);

    /* VE bit numbering is big-endian: bit 0 = MSB. The "upper" FP32 of
     * a packed slot lives in bits [0:31]; "lower" in [32:63]. vfsums
     * (single-precision reduction) reads only bits [0:31] of each
     * element. So:
     *   - sum of uppers = vfsums(acc) directly (uppers already in [0:31])
     *   - sum of lowers = vsll(acc, 32) shifts lowers from [32:63] up
     *     into [0:31], then vfsums. */
    __vr sum_u = _vel_vfsums_vvl(acc, d);
    __vr acc_lower_up = _vel_vsll_vvsl(acc, 32, d);
    __vr sum_l = _vel_vfsums_vvl(acc_lower_up, d);
    *s0_out = _vel_lvss_svs(sum_u, 0);
    *s1_out = _vel_lvss_svs(sum_l, 0);
}

/* VKQ pair accumulate: VKQ[j0] += vs0 * V, VKQ[j1] += vs1 * V.
 * V_dup_v has V[i] duplicated in both lanes of slot i. */
static inline void accumulate_pair_packed(float * VKQ_j0, float * VKQ_j1,
                                          float vs0, float vs1,
                                          __vr V_dup_v, int dv)
{
    /* Load VKQ pair: upper = VKQ[j0][i], lower = VKQ[j1][i] */
    __vr vu = _vel_vldu_vssl  (4, (void *)VKQ_j0, dv);
    __vr vl = _vel_vldlzx_vssl(4, (void *)VKQ_j1, dv);
    __vr VKQ_pair = _vel_pvor_vvvl(vu, vl, dv);

    /* Build weight pair: upper = vs0, lower = vs1, broadcast per slot. */
    uint32_t b0; __builtin_memcpy(&b0, &vs0, 4);
    uint32_t b1; __builtin_memcpy(&b1, &vs1, 4);
    uint64_t weight_bits = ((uint64_t)b0 << 32) | (uint64_t)b1;
    __vr w_packed = _vel_vbrdl_vsl(weight_bits, dv);

    VKQ_pair = _vel_pvfmad_vvvvl(VKQ_pair, w_packed, V_dup_v, dv);

    /* Store upper -> VKQ[j0], lower -> VKQ[j1] with stride-4 stores. */
    _vel_vstu_vssl(VKQ_pair, 4, (void *)VKQ_j0, dv);
    _vel_vstl_vssl(VKQ_pair, 4, (void *)VKQ_j1, dv);
}

/* Single-Q variants for the leftover when nq is odd. */
static inline float dot_single_packed(const float * Q_j, __vr K_dup_v, int d)
{
    __vr qu = _vel_vldu_vssl(4, (void *)Q_j, d);  /* upper = Q[j][i], lower = 0 */
    float zero[2] = { 0.0f, 0.0f };
    __vr acc = _vel_vld_vssl(0, &zero[0], d);
    acc = _vel_pvfmad_vvvvl(acc, qu, K_dup_v, d);
    /* Only upper lane has useful data (lower was zeroed by vldu, so
     * lower acc = 0 * K = 0). vfsums reads bits [0:31] = upper = our
     * accumulated dot. */
    __vr sum_u = _vel_vfsums_vvl(acc, d);
    return _vel_lvss_svs(sum_u, 0);
}

static inline void accumulate_single_packed(float * VKQ_j, float vs, __vr V_dup_v, int dv)
{
    __vr vu = _vel_vldu_vssl(4, (void *)VKQ_j, dv);
    /* Treat as upper-only by feeding 0 in lower; we still write only upper. */
    uint32_t b; __builtin_memcpy(&b, &vs, 4);
    uint64_t weight_bits = ((uint64_t)b << 32);
    __vr w_packed = _vel_vbrdl_vsl(weight_bits, dv);
    vu = _vel_pvfmad_vvvvl(vu, w_packed, V_dup_v, dv);
    _vel_vstu_vssl(vu, 4, (void *)VKQ_j, dv);
}

/* Reuse the existing per-Q packed scale for the rare exp(M_old-M_new)
 * rescale of VKQ. Forward-declare so it's visible here. */
/* scale_f32_packed already defined above */

/* Per-work-item kernel — pack-2 variant. Processes one (b, h, q_tile). */
void flash_attn_tile_inner_pack2_intrinsics(
    float * __restrict       VKQ,
    float * __restrict       M_state,
    float * __restrict       S_state,
    const float * __restrict Q_tile,
    const bf16 * __restrict  K_base,
    const bf16 * __restrict  V_base,
    const uint16_t * const * mask_rows,
    int nq, int n_kv, int d, int dv,
    int64_t nb_k1, int64_t nb_v1,
    float scale, float slope, float logit_softcap)
{
    float k_f32[256] __attribute__((aligned(8)));
    float v_f32[256] __attribute__((aligned(8)));

    for (int ic = 0; ic < n_kv; ic++) {
        const bf16 * k_row = (const bf16 *)((const char *)K_base + (int64_t)ic * nb_k1);
        const bf16 * v_row = (const bf16 *)((const char *)V_base + (int64_t)ic * nb_v1);

        decode_bf16_array(k_f32, k_row, d);
        decode_bf16_array(v_f32, v_row, dv);

        /* Build K_dup and V_dup once per ic — shared by all query pairs. */
        __vr K_dup_v = build_dup_vector(k_f32, d);
        __vr V_dup_v = build_dup_vector(v_f32, dv);

        /* Process pairs of queries. */
        int j = 0;
        for (; j + 1 < nq; j += 2) {
            const int j0 = j, j1 = j + 1;

            /* Per-query mask decode. */
            float mv0 = 0.0f, mv1 = 0.0f;
            if (mask_rows[j0]) mv0 = f16_to_f32_inline(mask_rows[j0][ic]) * slope;
            if (mask_rows[j1]) mv1 = f16_to_f32_inline(mask_rows[j1][ic]) * slope;

            const int skip0 = (mv0 == -INFINITY);
            const int skip1 = (mv1 == -INFINITY);
            if (skip0 && skip1) continue;

            /* Packed dot — even if one query is masked, dot is cheap;
             * we just discard the bad lane. */
            float s0 = 0.0f, s1 = 0.0f;
            dot_pair_packed(Q_tile + (size_t)j0 * d, Q_tile + (size_t)j1 * d,
                            K_dup_v, d, &s0, &s1);
            s0 *= scale; s1 *= scale;
            if (logit_softcap != 0.0f) {
                s0 = logit_softcap * tanhf(s0);
                s1 = logit_softcap * tanhf(s1);
            }
            s0 += mv0; s1 += mv1;

            /* Per-Q online softmax bookkeeping (branched). */
            float vs0 = 0.0f, vs1 = 0.0f;
            if (!skip0) {
                float M_old0 = M_state[j0];
                if (s0 > M_old0) {
                    M_state[j0] = s0;
                    float ms0 = expf(M_old0 - s0);
                    scale_f32_packed(VKQ + (size_t)j0 * dv, ms0, dv);
                    S_state[j0] *= ms0;
                    vs0 = 1.0f;
                } else {
                    vs0 = expf(s0 - M_state[j0]);
                }
                S_state[j0] += vs0;
            }
            if (!skip1) {
                float M_old1 = M_state[j1];
                if (s1 > M_old1) {
                    M_state[j1] = s1;
                    float ms1 = expf(M_old1 - s1);
                    scale_f32_packed(VKQ + (size_t)j1 * dv, ms1, dv);
                    S_state[j1] *= ms1;
                    vs1 = 1.0f;
                } else {
                    vs1 = expf(s1 - M_state[j1]);
                }
                S_state[j1] += vs1;
            }

            /* Packed VKQ accumulate — both queries with one pvfmad. */
            accumulate_pair_packed(VKQ + (size_t)j0 * dv,
                                   VKQ + (size_t)j1 * dv,
                                   vs0, vs1, V_dup_v, dv);
        }

        /* Odd leftover query (when nq is odd). */
        if (j < nq) {
            const int j0 = j;
            float mv = 0.0f;
            if (mask_rows[j0]) mv = f16_to_f32_inline(mask_rows[j0][ic]) * slope;
            if (mv == -INFINITY) continue;

            float s = dot_single_packed(Q_tile + (size_t)j0 * d, K_dup_v, d) * scale;
            if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
            s += mv;

            float M_old = M_state[j0];
            float vs = 0.0f;
            if (s > M_old) {
                M_state[j0] = s;
                float ms = expf(M_old - s);
                scale_f32_packed(VKQ + (size_t)j0 * dv, ms, dv);
                S_state[j0] *= ms;
                vs = 1.0f;
            } else {
                vs = expf(s - M_state[j0]);
            }
            S_state[j0] += vs;
            accumulate_single_packed(VKQ + (size_t)j0 * dv, vs, V_dup_v, dv);
        }
    }
}

/* Per-work-item kernel: processes one (batch, head, Q_tile). Decodes
 * K[ic] and V[ic] to F32 ONCE per ic into stack buffers; per-query work
 * runs against the decoded buffers using packed-FP32 intrinsics.
 *
 * Caller (NCC wrapper) supplies pointers + dims; it owns the OMP
 * parallel-for over (b, h, q_tile_idx) and the per-Q state arrays. */
void flash_attn_tile_inner_intrinsics(
    float * __restrict       VKQ,           /* [nq * dv] running output (zeroed by caller) */
    float * __restrict       M_state,       /* [nq] online softmax max state */
    float * __restrict       S_state,       /* [nq] online softmax denom state */
    const float * __restrict Q_tile,        /* [nq * d] Q rows for this tile, contiguous */
    const bf16 * __restrict  K_base,        /* base pointer to K rows for (ib, ik_head) */
    const bf16 * __restrict  V_base,        /* base pointer to V rows for (ib, iv_head) */
    const uint16_t * const * mask_rows,     /* [nq] mask row pointers or NULL */
    int nq, int n_kv, int d, int dv,
    int64_t nb_k1, int64_t nb_v1,
    float scale, float slope, float logit_softcap)
{
    /* Stack-local decoded buffers, reused across nq queries each ic. */
    float k_f32[256] __attribute__((aligned(8)));
    float v_f32[256] __attribute__((aligned(8)));

    for (int ic = 0; ic < n_kv; ic++) {
        const bf16 * k_row = (const bf16 *)((const char *)K_base + (int64_t)ic * nb_k1);
        const bf16 * v_row = (const bf16 *)((const char *)V_base + (int64_t)ic * nb_v1);

        decode_bf16_array(k_f32, k_row, d);
        decode_bf16_array(v_f32, v_row, dv);

        for (int j = 0; j < nq; j++) {
            float mv = 0.0f;
            if (mask_rows[j] != NULL) {
                mv = f16_to_f32_inline(mask_rows[j][ic]) * slope;
            }
            if (mv == -INFINITY) continue;

            float s = dot_f32_packed(Q_tile + (size_t)j * d, k_f32, d) * scale;
            if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
            s += mv;

            float M_old = M_state[j];
            float ms = 1.0f, vs = 1.0f;
            if (s > M_state[j]) {
                M_state[j] = s;
                ms = expf(M_old - M_state[j]);
                scale_f32_packed(VKQ + (size_t)j * dv, ms, dv);
                S_state[j] *= ms;
            } else {
                vs = expf(s - M_state[j]);
            }
            accumulate_f32_packed(VKQ + (size_t)j * dv, v_f32, vs, dv);
            S_state[j] += vs;
        }
    }
}
