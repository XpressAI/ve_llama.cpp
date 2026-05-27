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
