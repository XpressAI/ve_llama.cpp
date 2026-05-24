/*
 * Flash Attention with BF16 K/V using LLVM-VE Intrinsics
 * 
 * This provides vectorized BF16->FP32 conversion and operations.
 * All operations use PACKED mode (2 floats per 64-bit slot) for consistency
 * with the working BF16 SGEMV implementation.
 * 
 * Build with LLVM-VE:
 *   /usr/local/ve/llvm-ve-rv-2.2.0/bin/clang --target=ve-linux -O3 -fPIC -c flash_attn_bf16_intrinsics.c
 */

#include <velintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define VLEN 256
typedef uint16_t bf16;

/* 
 * Load BF16 and convert to packed FP32
 * Loads 2*vl BF16 values and packs them into vl vector elements
 * Memory: [bf16_0, bf16_1, bf16_2, bf16_3, ...] at 2 bytes each
 * After: slot[i] = (fp32(bf16[2i]) in upper, fp32(bf16[2i+1]) in lower)
 */
#define load_bf16_to_packed_fp32(wv, wp, vl) \
    do { \
        wv = _vel_vldunc_vssl(4, (void *)(wp), vl); \
        __vr wr = _vel_vsrl_vvsl(wv, 16, vl); \
        wr = _vel_vand_vvvl(wr, bf16mskl, vl); \
        wv = _vel_vor_vvvl(wv, wr, vl); \
    } while(0)

/* Sum up packed FP32 vector to scalar */
#define sumup_packed_fp32(tv, result, VL) \
    do { \
        __vr rv = _vel_vand_vvvl(tv, low32msk, VL); \
        __vr sv = _vel_vsll_vvsl(rv, 32, VL); \
        tv = _vel_vfadds_vvvl(tv, sv, VL); \
        tv = _vel_vfsums_vvl(tv, VL); \
        result = _vel_lvss_svs(tv, 0); \
    } while(0)

/* 
 * Vectorized dot product of BF16 vector with FP32 vector
 * Computes: sum(k[i] * q[i]) for i in [0, n)
 * Uses packed mode for maximum throughput
 */
float dot_bf16_fp32_intrinsics(const bf16* k_ptr, const float* q_ptr, int n) {
    float zero[2] = {0.0f, 0.0f};
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);
    __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffffUL, VLEN);
    
    __vr tv = _vel_vld_vssl(0, &zero[0], VLEN);
    
    for (int j = 0; j < n; j += 2 * VLEN) {
        int remaining = n - j;
        int vl = remaining < 2 * VLEN ? remaining >> 1 : VLEN;
        if (vl <= 0) break;
        
        /* Load FP32 Q as packed */
        __vr qv;
        if (((unsigned long)(q_ptr + j)) & 0x7) {
            qv = _vel_vldu_vssl(8, (void*)(q_ptr + j + 1), vl);
            __vr qlv = _vel_vldlzx_vssl(8, (void*)(q_ptr + j), vl);
            qv = _vel_pvor_vvvl(qv, qlv, vl);
        } else {
            qv = _vel_vld_vssl(8, (void*)(q_ptr + j), vl);
        }
        
        /* Load BF16 K and convert to packed FP32 */
        __vr kv;
        load_bf16_to_packed_fp32(kv, k_ptr + j, vl);
        
        /* Packed FMA */
        tv = _vel_pvfmad_vvvvl(tv, qv, kv, vl);
    }
    
    float result;
    sumup_packed_fp32(tv, result, VLEN);
    return result;
}

/*
 * Vectorized weighted accumulation: result[i] += weight * v[i]
 * Where v is BF16 and result is FP32
 * Uses packed mode for all operations
 */
void accumulate_bf16_intrinsics(float* result, const bf16* v_ptr, float weight, int n) {
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);
    
    /* Pack weight into both 32-bit slots */
    uint32_t w32;
    __builtin_memcpy(&w32, &weight, sizeof(w32));
    uint64_t weight_bits = ((uint64_t)w32 << 32) | w32;
    __vr weight_packed = _vel_vbrdl_vsl(weight_bits, VLEN);
    
    for (int j = 0; j < n; j += 2 * VLEN) {
        int remaining = n - j;
        int vl = remaining < 2 * VLEN ? remaining >> 1 : VLEN;
        if (vl <= 0) break;
        
        /* Load FP32 result as packed */
        __vr rv;
        if (((unsigned long)(result + j)) & 0x7) {
            /* Unaligned load: upper from odd indices, lower from even */
            rv = _vel_vldu_vssl(8, (void*)(result + j + 1), vl);
            __vr rlv = _vel_vldlzx_vssl(8, (void*)(result + j), vl);
            rv = _vel_pvor_vvvl(rv, rlv, vl);
        } else {
            rv = _vel_vld_vssl(8, (void*)(result + j), vl);
        }
        
        /* Load BF16 V and convert to packed FP32 */
        __vr vv;
        load_bf16_to_packed_fp32(vv, v_ptr + j, vl);
        
        /* Packed FMA: result += weight * v */
        rv = _vel_pvfmad_vvvvl(rv, weight_packed, vv, vl);
        
        /* Store back as packed */
        if (((unsigned long)(result + j)) & 0x7) {
            /* Unaligned store: upper to odd indices, lower to even indices */
            _vel_vstunc_vssl(rv, 8, (void*)(result + j + 1), vl);
            _vel_vstlnc_vssl(rv, 8, (void*)(result + j), vl);
        } else {
            _vel_vstnc_vssl(rv, 8, (void*)(result + j), vl);
        }
    }
}

/*
 * Vectorized scale: result[i] *= scale
 * Uses packed mode
 */
void scale_fp32_intrinsics(float* result, float scale, int n) {
    /* Pack scale into both 32-bit slots */
    uint32_t s32;
    __builtin_memcpy(&s32, &scale, sizeof(s32));
    uint64_t scale_bits = ((uint64_t)s32 << 32) | s32;
    __vr scale_packed = _vel_vbrdl_vsl(scale_bits, VLEN);
    
    for (int j = 0; j < n; j += 2 * VLEN) {
        int remaining = n - j;
        int vl = remaining < 2 * VLEN ? remaining >> 1 : VLEN;
        if (vl <= 0) break;
        
        /* Load as packed */
        __vr rv;
        if (((unsigned long)(result + j)) & 0x7) {
            rv = _vel_vldu_vssl(8, (void*)(result + j + 1), vl);
            __vr rlv = _vel_vldlzx_vssl(8, (void*)(result + j), vl);
            rv = _vel_pvor_vvvl(rv, rlv, vl);
        } else {
            rv = _vel_vld_vssl(8, (void*)(result + j), vl);
        }
        
        /* Packed multiply */
        rv = _vel_pvfmul_vvvl(rv, scale_packed, vl);
        
        /* Store back as packed */
        if (((unsigned long)(result + j)) & 0x7) {
            _vel_vstunc_vssl(rv, 8, (void*)(result + j + 1), vl);
            _vel_vstlnc_vssl(rv, 8, (void*)(result + j), vl);
        } else {
            _vel_vstnc_vssl(rv, 8, (void*)(result + j), vl);
        }
    }
}

/*
 * Process a single head of flash attention
 */
void flash_attn_single_head_intrinsics(
    float* out,
    const float* q,
    const bf16* k,
    const bf16* v,
    const float* mask_row,
    int D,
    int S,
    int64_t nb_k1,
    int64_t nb_v1,
    float scale,
    float logit_softcap,
    float slope)
{
    /* Work buffer for accumulating V - must be 8-byte aligned for packed ops */
    float VKQ[512] __attribute__((aligned(8)));
    
    for (int i = 0; i < D; i++) {
        VKQ[i] = 0.0f;
    }
    float M = -INFINITY;
    float S_sum = 0.0f;
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    for (int ic = 0; ic < S; ic++) {
        float mv = 0.0f;
        if (mask_row != NULL) {
            mv = mask_row[ic] * slope;
        }
        
        if (mv == -INFINITY) {
            continue;
        }
        
        const bf16* k_ic = (const bf16*)((const char*)k + ic * nb_k1);
        
        float s = dot_bf16_fp32_intrinsics(k_ic, q, D);
        s *= scale;
        if (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }
        s += mv;
        
        float M_old = M;
        float ms = 1.0f;
        float vs = 1.0f;
        
        if (s > M) {
            M = s;
            ms = expf(M_old - M);
            scale_fp32_intrinsics(VKQ, ms, D);
            S_sum *= ms;
        } else {
            vs = expf(s - M);
        }
        
        const bf16* v_ic = (const bf16*)((const char*)v + ic * nb_v1);
        accumulate_bf16_intrinsics(VKQ, v_ic, vs, D);
        S_sum += vs;
    }
    
    float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
    scale_fp32_intrinsics(VKQ, S_inv, D);
    
    for (int i = 0; i < D; i++) {
        out[i] = VKQ[i];
    }
}
