/*
 * Fused Flash Attention with BF16 K/V using LLVM-VE Intrinsics
 * 
 * This version FULLY INLINES all operations to avoid function call overhead.
 * The key optimizations:
 * 1. All masks and constants are initialized once per head
 * 2. No function calls in the inner (sequence) loop
 * 3. Assumes 8-byte aligned accumulators (VKQ array on stack is aligned)
 * 4. Q pointer is 8-byte aligned in practice (head_dim is multiple of 2)
 * 
 * Build with LLVM-VE:
 *   /usr/local/ve/llvm-ve-rv-2.2.0/bin/clang --target=ve-linux -O3 -fPIC -c flash_attn_bf16_fused.c
 */

#include <velintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define VLEN 256
typedef uint16_t bf16;

/*
 * Fused single-head flash attention: fully inlined for maximum performance
 * 
 * This processes one head completely with online softmax.
 * All intrinsic operations are inlined - no function calls in inner loop.
 */
void flash_attn_single_head_fused(
    float* __restrict__ out,           /* Output [D] F32 */
    const float* __restrict__ q,       /* Query [D] F32, assumed 8-byte aligned */
    const bf16* k_base,                /* Key base pointer for this head */
    const bf16* v_base,                /* Value base pointer for this head */
    int D,                             /* head_dim (typically 128) */
    int S,                             /* seq_len */
    int64_t nb_k1,                     /* K stride for seq dimension (bytes) */
    int64_t nb_v1,                     /* V stride for seq dimension (bytes) */
    float scale)
{
    /* Constants - initialize once */
    const __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);
    const __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffffUL, VLEN);
    
    /* Accumulator for weighted V - 8-byte aligned on stack */
    float VKQ[512] __attribute__((aligned(8)));
    
    /* Initialize VKQ to zero */
    for (int d = 0; d < D; d++) {
        VKQ[d] = 0.0f;
    }
    
    float M = -INFINITY;  /* Running max for online softmax */
    float S_sum = 0.0f;   /* Running sum of exp(s - M) */
    
    /* Pre-compute number of packed iterations for dot product and accumulation */
    const int D_packed_full = D / (2 * VLEN) * (2 * VLEN);  /* Full vector iterations */
    const int D_packed_tail = D - D_packed_full;            /* Remaining elements */
    const int tail_vl = D_packed_tail > 0 ? D_packed_tail / 2 : 0;
    
    /* Process each K/V position */
    for (int t = 0; t < S; t++) {
        const bf16* k_t = (const bf16*)((const char*)k_base + t * nb_k1);
        
        /* === DOT PRODUCT: s = sum(q[d] * k_t[d]) === */
        float zero_pair[2] = {0.0f, 0.0f};
        __vr tv = _vel_vld_vssl(0, &zero_pair[0], VLEN);
        
        /* Full vector iterations */
        for (int j = 0; j < D_packed_full; j += 2 * VLEN) {
            /* Load FP32 Q as packed (assumed aligned) */
            __vr qv = _vel_vld_vssl(8, (void*)(q + j), VLEN);
            
            /* Load BF16 K and convert to packed FP32 */
            __vr kv = _vel_vldunc_vssl(4, (void*)(k_t + j), VLEN);
            __vr kr = _vel_vsrl_vvsl(kv, 16, VLEN);
            kr = _vel_vand_vvvl(kr, bf16mskl, VLEN);
            kv = _vel_vor_vvvl(kv, kr, VLEN);
            
            /* Packed FMA */
            tv = _vel_pvfmad_vvvvl(tv, qv, kv, VLEN);
        }
        
        /* Tail elements if D is not multiple of 512 */
        if (tail_vl > 0) {
            __vr qv = _vel_vld_vssl(8, (void*)(q + D_packed_full), tail_vl);
            __vr kv = _vel_vldunc_vssl(4, (void*)(k_t + D_packed_full), tail_vl);
            __vr kr = _vel_vsrl_vvsl(kv, 16, tail_vl);
            kr = _vel_vand_vvvl(kr, bf16mskl, tail_vl);
            kv = _vel_vor_vvvl(kv, kr, tail_vl);
            tv = _vel_pvfmad_vvvvl(tv, qv, kv, tail_vl);
        }
        
        /* Reduce packed vector to scalar */
        __vr rv = _vel_vand_vvvl(tv, low32msk, VLEN);
        __vr sv = _vel_vsll_vvsl(rv, 32, VLEN);
        tv = _vel_vfadds_vvvl(tv, sv, VLEN);
        tv = _vel_vfsums_vvl(tv, VLEN);
        float s = _vel_lvss_svs(tv, 0) * scale;
        
        /* === ONLINE SOFTMAX UPDATE === */
        float M_old = M;
        float ms = 1.0f;
        float vs = 1.0f;
        
        if (s > M) {
            M = s;
            ms = expf(M_old - M);
            
            /* Scale VKQ by ms: VKQ[d] *= ms */
            uint32_t ms32;
            __builtin_memcpy(&ms32, &ms, sizeof(ms32));
            uint64_t ms_bits = ((uint64_t)ms32 << 32) | ms32;
            __vr ms_packed = _vel_vbrdl_vsl(ms_bits, VLEN);
            
            for (int j = 0; j < D_packed_full; j += 2 * VLEN) {
                __vr vkq = _vel_vld_vssl(8, (void*)(VKQ + j), VLEN);
                vkq = _vel_pvfmul_vvvl(vkq, ms_packed, VLEN);
                _vel_vstnc_vssl(vkq, 8, (void*)(VKQ + j), VLEN);
            }
            if (tail_vl > 0) {
                __vr vkq = _vel_vld_vssl(8, (void*)(VKQ + D_packed_full), tail_vl);
                vkq = _vel_pvfmul_vvvl(vkq, ms_packed, tail_vl);
                _vel_vstnc_vssl(vkq, 8, (void*)(VKQ + D_packed_full), tail_vl);
            }
            
            S_sum *= ms;
            vs = 1.0f;
        } else {
            vs = expf(s - M);
        }
        
        /* === ACCUMULATE V: VKQ[d] += vs * v_t[d] === */
        const bf16* v_t = (const bf16*)((const char*)v_base + t * nb_v1);
        
        uint32_t vs32;
        __builtin_memcpy(&vs32, &vs, sizeof(vs32));
        uint64_t vs_bits = ((uint64_t)vs32 << 32) | vs32;
        __vr vs_packed = _vel_vbrdl_vsl(vs_bits, VLEN);
        
        for (int j = 0; j < D_packed_full; j += 2 * VLEN) {
            /* Load current VKQ (aligned) */
            __vr vkq = _vel_vld_vssl(8, (void*)(VKQ + j), VLEN);
            
            /* Load BF16 V and convert to packed FP32 */
            __vr vv = _vel_vldunc_vssl(4, (void*)(v_t + j), VLEN);
            __vr vr = _vel_vsrl_vvsl(vv, 16, VLEN);
            vr = _vel_vand_vvvl(vr, bf16mskl, VLEN);
            vv = _vel_vor_vvvl(vv, vr, VLEN);
            
            /* Packed FMA: VKQ += vs * V */
            vkq = _vel_pvfmad_vvvvl(vkq, vs_packed, vv, VLEN);
            
            /* Store back (aligned, non-cached) */
            _vel_vstnc_vssl(vkq, 8, (void*)(VKQ + j), VLEN);
        }
        if (tail_vl > 0) {
            __vr vkq = _vel_vld_vssl(8, (void*)(VKQ + D_packed_full), tail_vl);
            __vr vv = _vel_vldunc_vssl(4, (void*)(v_t + D_packed_full), tail_vl);
            __vr vr = _vel_vsrl_vvsl(vv, 16, tail_vl);
            vr = _vel_vand_vvvl(vr, bf16mskl, tail_vl);
            vv = _vel_vor_vvvl(vv, vr, tail_vl);
            vkq = _vel_pvfmad_vvvvl(vkq, vs_packed, vv, tail_vl);
            _vel_vstnc_vssl(vkq, 8, (void*)(VKQ + D_packed_full), tail_vl);
        }
        
        S_sum += vs;
    }
    
    /* === NORMALIZE: out[d] = VKQ[d] / S_sum === */
    float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
    
    uint32_t sinv32;
    __builtin_memcpy(&sinv32, &S_inv, sizeof(sinv32));
    uint64_t sinv_bits = ((uint64_t)sinv32 << 32) | sinv32;
    __vr sinv_packed = _vel_vbrdl_vsl(sinv_bits, VLEN);
    
    for (int j = 0; j < D_packed_full; j += 2 * VLEN) {
        __vr vkq = _vel_vld_vssl(8, (void*)(VKQ + j), VLEN);
        vkq = _vel_pvfmul_vvvl(vkq, sinv_packed, VLEN);
        _vel_vstnc_vssl(vkq, 8, (void*)(out + j), VLEN);
    }
    if (tail_vl > 0) {
        __vr vkq = _vel_vld_vssl(8, (void*)(VKQ + D_packed_full), tail_vl);
        vkq = _vel_pvfmul_vvvl(vkq, sinv_packed, tail_vl);
        _vel_vstnc_vssl(vkq, 8, (void*)(out + D_packed_full), tail_vl);
    }
}

/*
 * Multi-head wrapper - processes all heads with OpenMP parallelization
 * This is compiled by NCC (not LLVM-VE) to get OpenMP support
 */
/* Note: This wrapper is in ve_sgemv_wrapper.c, not here */
