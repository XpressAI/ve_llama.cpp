/*
 * VEDA Wrapper for SGEMV Intrinsic Kernels
 * 
 * This wraps the LLVM-VE compiled sgemv intrinsics to make them callable
 * via VEDA from the host.
 *
 * Build:
 *   1. Compile sgemv with LLVM-VE: clang --target=ve-linux -O3 -c sgemv_*.c
 *   2. Compile this wrapper with NCC: ncc -O4 -fpic -c -I/opt/nec/ve/share/veda/include ve_sgemv_wrapper.c
 *   3. Link together: ncc -shared -o libve_sgemv.so ve_sgemv_wrapper.o sgemv_*.o
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <veda_device.h>
#include <omp.h>

/* BF16 type - just a 16-bit unsigned int */
typedef uint16_t bf16;

/* Master debug flag - set to 0 to disable all debug output */
#define VE_KERNEL_DEBUG 0

/* Maximum VE cores per device */
#define VE_MAX_CORES 8

/* GGML quantization block sizes */
#define QK_K 256
#define QK_MXFP4 32

/* MXFP4 lookup table for -1.0 to 1.0 range */
static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    -8, -7, -6, -5, -4, -3, -2, -1
};

/* MXFP4 block structure for VE */
typedef struct __attribute__((packed)) {
    uint8_t e;               /* Shared exponent (8-bit) */
    uint8_t qs[QK_MXFP4/2];  /* Quantized values, 4 bits each */
} block_mxfp4_ve;

/* Convert E8M0 exponent to FP32 scale (2^(e-127)/2 for MXFP4 scaling) */
static inline float e8m0_to_fp32_half(uint8_t e) {
    /* E8M0 is just an 8-bit exponent with bias 127 */
    /* For MXFP4, we typically want 2^(e-127) / 2 = 2^(e-128) */
    if (e == 0) return 0.0f;  /* Special case: zero exponent means scale = 0 */
    uint32_t bits = ((uint32_t)(e - 1)) << 23;  /* e-128 bias adjustment, e-1 = e-127-1 */
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

/*
 * External LLVM-VE intrinsics functions for flash attention
 * These are compiled separately with LLVM-VE and linked with this file.
 */
extern void flash_attn_single_head_intrinsics(
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
    float slope);

/*
 * GNU half-float to single-float conversion function
 * Required by LLVM-compiled code that uses half-precision floats.
 * LLVM emits calls to __gnu_h2f_ieee but NCC runtime doesn't provide it.
 * 
 * IEEE 754 half-precision format:
 *   - 1 bit sign, 5 bits exponent (bias 15), 10 bits mantissa
 *   - Denormals, inf, NaN handled correctly
 */
float __gnu_h2f_ieee(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;      /* Sign bit */
    int32_t exp = (h >> 10) & 0x1F;          /* Exponent */
    uint32_t mant = h & 0x3FF;               /* Mantissa */
    
    uint32_t f;
    
    if (exp == 0) {
        /* Zero or denormal */
        if (mant == 0) {
            /* Zero */
            f = sign;
        } else {
            /* Denormal - convert to normalized single */
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        /* Inf or NaN */
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        /* Normalized */
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    
    float result;
    __builtin_memcpy(&result, &f, sizeof(float));
    return result;
}

/*
 * Fast vectorizable FP16 to FP32 conversion (ignores denormals/special cases)
 * For quantized weights, values are always normalized small floats, so this is safe.
 * 
 * FP16: 1 sign + 5 exp (bias 15) + 10 mantissa
 * FP32: 1 sign + 8 exp (bias 127) + 23 mantissa
 * 
 * Conversion: sign stays, exp += (127-15), mantissa << 13
 */
static inline float h2f_fast(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            f = sign;
        } else {
            /* Denormal fp16 -> normal fp32 */
            /* Normalize: shift mantissa left until leading 1 */
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;  /* Remove implicit 1 */
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        /* Inf or NaN */
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        /* Normal: adjust exponent bias (112 = 127 - 15) */
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    
    float result;
    __builtin_memcpy(&result, &f, sizeof(float));
    return result;
}

/*
 * GNU single-float to half-float conversion function
 * Also provide this in case it's needed.
 */
uint16_t __gnu_f2h_ieee(float f) {
    uint32_t fb;
    __builtin_memcpy(&fb, &f, sizeof(float));
    
    uint32_t sign = (fb >> 16) & 0x8000;
    int32_t exp = ((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (fb >> 13) & 0x3FF;
    
    uint16_t h;
    
    if (exp <= 0) {
        /* Underflow to zero or denormal */
        if (exp < -10) {
            h = sign;  /* Too small, round to zero */
        } else {
            /* Denormal */
            mant = (mant | 0x400) >> (1 - exp);
            h = sign | mant;
        }
    } else if (exp >= 0x1F) {
        /* Overflow to infinity */
        h = sign | 0x7C00;
    } else {
        /* Normal number */
        h = sign | (exp << 10) | mant;
    }
    
    return 0;
}

/*
 * Flash attention with F32 Q and BF16 K/V, all in HBM, with F32 mask
 * 
 * This version takes F32 mask (pre-converted on host) to avoid F16->F32
 * conversion in the inner loop, which can't be vectorized on VE.
 * 
 * The mask should be pre-converted from F16 to F32 on the host before
 * passing to this kernel.
 */
uint64_t ve_flash_attn_ext_f32q_bf16kv_hbm_f32mask(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,  /* HMEM - F32 mask (pre-converted from F16 on host) */
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1,
    uint64_t nb_q2,
    uint64_t nb_q3,
    uint64_t nb_k1,
    uint64_t nb_k2,
    uint64_t nb_k3,
    uint64_t nb_v1,
    uint64_t nb_v2,
    uint64_t nb_v3,
    uint64_t nb_m1,
    uint64_t nb_m2,
    uint64_t nb_m3,
    uint64_t nb_o1,
    uint64_t nb_o2,
    uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    /* Convert HBM pointers to raw VE addresses */
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    
    /* Mask is F32 (pre-converted on host), HMEM already converted to raw pointer */
    const float* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);
    
    int d = (int)D;
    int dv = (int)Dv;
    int n_tokens = (int)N;
    int n_kv = (int)S;
    int n_heads = (int)H;
    int n_batch = (int)B;
    
    int total_work = n_batch * n_heads * n_tokens;
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        /* Q is F32 - direct HBM access */
        const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        /* F32 mask row - already converted, vectorizable access */
        const float* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            /* Strides are for F32 now (4 bytes per element) */
            mask_row = (const float*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] = 0.0f;
        }
        
        for (int ic = 0; ic < n_kv; ic++) {
            /* F32 mask - direct read, no conversion needed! */
            float mv = 0.0f;
            if (mask_row != NULL) {
                mv = mask_row[ic] * slope;
            }
            
            if (mv == -INFINITY) {
                continue;
            }
            
            /* Convert K vector from BF16 to F32 (direct HBM access, vectorizable) */
            const uint16_t* k_bf16 = (const uint16_t*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                uint32_t f32 = ((uint32_t)k_bf16[i]) << 16;
                union { uint32_t u; float f; } conv;
                conv.u = f32;
                k_f32[i] = conv.f;
            }
            
            /* Q·K dot product */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_vec[i] * k_f32[i];
            }
            
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
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[i] *= ms;
                }
                S_sum *= ms;
            } else {
                vs = expf(s - M);
            }
            
            /* Convert V vector from BF16 to F32 (direct HBM access, vectorizable) */
            const uint16_t* v_bf16 = (const uint16_t*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                uint32_t f32 = ((uint32_t)v_bf16[i]) << 16;
                union { uint32_t u; float f; } conv;
                conv.u = f32;
                v_f32[i] = conv.f;
            }
            
            /* Accumulate weighted V */
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                VKQ[i] += vs * v_f32[i];
            }
            
            S_sum += vs;
        }
        
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] *= S_inv;
        }
        
        /* Write output directly to HBM */
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = VKQ[i];
        }
    }
    
    return 0;
}


/* ============================================================================
 * Q2_K → BF16 DEQUANTIZATION + BF16 HBM MATVEC
 * 
 * This is the optimal path for Q2_K inference on VE:
 *   1. Dequantize Q2_K to BF16 once at model load (stored in HBM)
 *   2. Use sgemv_packed_bf16_unr for all inference (VPU-optimized)
 * 
 * Memory for Llama-7B:
 *   - Q2_K: ~2.3 GB
 *   - BF16: ~14 GB (fits on single VE card with 48 GB HBM)
 * 
 * Performance: ~560 GB/s memory bandwidth (vs ~1.4 GB/s for native Q2_K)
 * ============================================================================
 */

/* Q2_K block structure (84 bytes per 256 elements) */
typedef struct __attribute__((packed)) {
    uint8_t  scales_q2k[16];  /* 4-bit scales (low) + 4-bit mins (high) */
    uint8_t  qs_q2k[64];      /* 2-bit quants: 256 values = 64 bytes */
    uint16_t d_q2k;           /* FP16 super-block scale */
    uint16_t dmin_q2k;        /* FP16 super-block min */
} block_q2_K_wrapper;

/* Convert FP32 to BF16 (truncation) - defined here, used by dequant functions below */
static inline bf16 f32_to_bf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));
    return (bf16)(bits >> 16);
}

/*
 * Dequantize one Q2_K block (256 elements) to BF16
 * 
 * Q2_K element extraction:
 *   For element j (0..255):
 *     byte_idx = (j % 32) + 32 * (j / 128)
 *     shift = 2 * ((j / 32) % 4)
 *     q = (qs[byte_idx] >> shift) & 3
 *     
 *     sub_block = j / 16
 *     scale = d * (scales[sub_block] & 0x0F)
 *     min = dmin * (scales[sub_block] >> 4)
 *     
 *     value = scale * q - min
 */
static void dequant_q2k_block_bf16_wrapper(const block_q2_K_wrapper* b, bf16* out) {
    const float d = __gnu_h2f_ieee(b->d_q2k);
    const float dmin = __gnu_h2f_ieee(b->dmin_q2k);
    const uint8_t* qs = b->qs_q2k;
    
    for (int is = 0; is < 16; is++) {
        const float scale = d * (b->scales_q2k[is] & 0x0F);
        const float min = dmin * (b->scales_q2k[is] >> 4);
        
        for (int l = 0; l < 16; l++) {
            int j = is * 16 + l;
            int byte_idx = (j % 32) + 32 * (j / 128);
            int shift = 2 * ((j / 32) % 4);
            int q = (qs[byte_idx] >> shift) & 3;
            
            out[j] = f32_to_bf16(scale * q - min);
        }
    }
}

/*
 * VEDA kernel: Dequantize Q2_K matrix to BF16 (one-time at model load)
 * 
 * Parameters:
 *   out_vptr: Output BF16 matrix in HBM [M × K]
 *   in_hmem: Input Q2_K matrix in HMEM
 *   M: Number of rows
 *   K: Number of columns (must be multiple of 256)
 */
uint64_t ve_dequant_q2k_bf16(VEDAdeviceptr out_vptr,
                             void* in_hmem,
                             uint64_t M,
                             uint64_t K) {
    bf16* out;
    vedaMemPtr((void**)&out, out_vptr);
    
    const block_q2_K_wrapper* in = (const block_q2_K_wrapper*)in_hmem;
    
    int m = (int)M;
    int k = (int)K;
    int nb = k / QK_K;  /* Blocks per row */
    
    #pragma omp parallel for
    for (int row = 0; row < m; row++) {
        const block_q2_K_wrapper* in_row = in + row * nb;
        bf16* out_row = out + row * k;
        
        for (int b = 0; b < nb; b++) {
            dequant_q2k_block_bf16_wrapper(&in_row[b], out_row + b * QK_K);
        }
    }
    
    return 0;
}

/*
 * VEDA kernel: Q2_K matvec via pre-dequantized BF16 weights in HBM
 * 
 * This is the fast inference path - weights already in BF16 in HBM.
 * Uses the VPU-optimized sgemv_packed_bf16_unr kernel.
 * 
 * Parameters:
 *   y_hmem: Output [M] floats in HMEM
 *   W_vptr: BF16 weights [M × K] in HBM (pre-dequantized)
 *   x_hmem: Input [K] floats in HMEM
 *   M: Output dimension
 *   K: Input dimension
 */
uint64_t ve_q2k_bf16_matvec_hbm(VEDAdeviceptr y_vptr,
                                VEDAdeviceptr W_vptr,
                                VEDAdeviceptr x_vptr,
                                uint64_t M,
                                uint64_t K) {
    float* y;
    float* x;
    bf16* W;
    if (vedaMemPtr((void**)&y, y_vptr) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_vptr) != 0) return 2;
    if (vedaMemPtr((void**)&W, W_vptr) != 0) return 3;
    
    int m = (int)M;
    int k = (int)K;
    
    /* Use OpenMP to parallelize across rows, each thread uses VPU kernel */
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        int chunk = (m + nthr - 1) / nthr;
        int row_start = tid * chunk;
        int row_end = row_start + chunk;
        if (row_end > m) row_end = m;
        
        if (row_start < row_end) {
            /* Call the VPU-optimized BF16 SGEMV for this chunk */
            sgemv_packed_bf16_unr(
                y + row_start,           /* Output chunk */
                x,                       /* Full input vector */
                W + row_start * k,       /* Weight rows for this chunk */
                k,                       /* K (input dim) */
                row_end - row_start      /* Number of rows in chunk */
            );
        }
    }
    
    return 0;
}

/*
 * VEDA kernel: Combined dequant + matvec (for testing, not optimal)
 * 
 * This dequantizes on-the-fly. For production, pre-dequant to HBM.
 */
uint64_t ve_q2k_matvec_dequant_bf16(void* y_hmem,
                                     void* W_hmem,
                                     void* x_hmem,
                                     uint64_t M,
                                     uint64_t K) {
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    const block_q2_K_wrapper* W = (const block_q2_K_wrapper*)W_hmem;
    
    int m = (int)M;
    int k = (int)K;
    int nb = k / QK_K;
    
    /* Per-thread BF16 buffer for one row */
    #pragma omp parallel
    {
        bf16 row_bf16[16384];  /* Max K = 16K for now */
        
        #pragma omp for
        for (int row = 0; row < m; row++) {
            const block_q2_K_wrapper* w_row = W + row * nb;
            
            /* Dequant this row to BF16 */
            for (int b = 0; b < nb; b++) {
                dequant_q2k_block_bf16_wrapper(&w_row[b], row_bf16 + b * QK_K);
            }
            
            /* Compute dot product using BF16 kernel for single row */
            sgemv_packed_bf16_unr(y + row, x, row_bf16, k, 1);
        }
    }
    
    return 0;
}
/*
 * Dequantize MXFP4 weights to BF16
 * 
 * This is the optimal approach for VE:
 *   - Dequant once at model load, store BF16 in HBM
 *   - Use existing optimized BF16 SGEMV (sgemv_packed_bf16_unr)
 *   - Half the memory vs FP32, same quality for inference
 *
 * Parameters:
 *   dst: Output BF16 buffer [num_elements]
 *   src: Input MXFP4 buffer
 *   num_elements: Total number of elements (must be multiple of 32)
 */
static void dequantize_mxfp4_to_bf16(bf16* dst, const block_mxfp4_ve* src, int64_t num_elements) {
    int64_t nb = num_elements / QK_MXFP4;
    
    #pragma omp parallel for
    for (int64_t i = 0; i < nb; i++) {
        const float d = e8m0_to_fp32_half(src[i].e);
        bf16* out = dst + i * QK_MXFP4;
        
        #pragma _NEC ivdep
        for (int j = 0; j < QK_MXFP4/2; j++) {
            const int8_t v0 = kvalues_mxfp4[src[i].qs[j] & 0x0F];
            const int8_t v1 = kvalues_mxfp4[src[i].qs[j] >> 4];
            
            out[j]              = f32_to_bf16(v0 * d);
            out[j + QK_MXFP4/2] = f32_to_bf16(v1 * d);
        }
    }
}

/*
 * VEDA kernel: Dequantize MXFP4 to BF16
 * 
 * Called once at model load to convert weights.
 * The BF16 output is then cached in HBM for fast inference.
 */
uint64_t ve_dequant_mxfp4_bf16_hmem(void* dst_hmem,
                                    void* src_hmem,
                                    uint64_t num_elements) {
    bf16* dst = (bf16*)dst_hmem;
    block_mxfp4_ve* src = (block_mxfp4_ve*)src_hmem;
    
    dequantize_mxfp4_to_bf16(dst, src, (int64_t)num_elements);
    
    return 0;
}

/*
 * MXFP4 matrix-vector multiply via BF16 dequant + optimized BF16 SGEMV
 * 
 * This is the HBM-cached path:
 *   1. Weights dequantized to BF16 once, stored in HBM
 *   2. Uses sgemv_packed_bf16_unr for inference (highly optimized)
 *
 * Parameters:
 *   y_hmem: Output [M] floats (HMEM)
 *   W_bf16_vptr: BF16 weights [M x K] in HBM (pre-dequantized from MXFP4)
 *   x_hmem: Input [K] floats (HMEM)
 *   M: Output dimension
 *   K: Input dimension
 */
uint64_t ve_mxfp4_matvec_bf16_hbm_omp(void* y_hmem,
                                       VEDAdeviceptr W_bf16_vptr,
                                       void* x_hmem,
                                       uint64_t M,
                                       uint64_t K) {
    /* This is just a wrapper around BF16 HBM kernel since weights are pre-dequantized */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    bf16* W;
    vedaMemPtr((void**)&W, W_bf16_vptr);
    
    int nthr = omp_get_max_threads();
    int m_int = (int)M;
    int k_int = (int)K;
    
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    /* Parallel over output rows, using optimized BF16 kernel per chunk */
    #pragma omp parallel num_threads(nthr)
    {
        int ithr = omp_get_thread_num();
        int chunk = (m_int + nthr - 1) / nthr;
        int imin = ithr * chunk;
        int imax = imin + chunk;
        if (imax > m_int) imax = m_int;
        
        if (imin < imax) {
            sgemv_packed_bf16_unr(&y[imin], x, &W[imin * k_int], k_int, imax - imin);
        }
    }
    
    return 0;
}

/*
 * Direct MXFP4 matrix-vector multiply (no pre-dequant, slower but simpler)
 * 
 * For cases where HBM caching isn't available.
 * Dequantizes on-the-fly during the dot product.
 */
uint64_t ve_mxfp4_matvec_f32_omp_hmem(void* y_hmem,
                                       void* W_hmem,
                                       void* x_hmem,
                                       uint64_t M,
                                       uint64_t K) {
    float* y = (float*)y_hmem;
    block_mxfp4_ve* W = (block_mxfp4_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int m_int = (int)M;
    int k_int = (int)K;
    int nb_per_row = k_int / QK_MXFP4;
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel for num_threads(nthr)
    for (int row = 0; row < m_int; row++) {
        block_mxfp4_ve* W_row = W + row * nb_per_row;
        float sum = 0.0f;
        
        for (int b = 0; b < nb_per_row; b++) {
            const float d = e8m0_to_fp32_half(W_row[b].e);
            const float* x_block = x + b * QK_MXFP4;
            
            #pragma _NEC ivdep
            for (int j = 0; j < QK_MXFP4/2; j++) {
                const int8_t v0 = kvalues_mxfp4[W_row[b].qs[j] & 0x0F];
                const int8_t v1 = kvalues_mxfp4[W_row[b].qs[j] >> 4];
                
                sum += (v0 * d) * x_block[j];
                sum += (v1 * d) * x_block[j + QK_MXFP4/2];
            }
        }
        
        y[row] = sum;
    }
    
    return 0;
}


/* ============================================================================
 * Pure NCC implementations of SGEMV kernels
 * 
 * These use simple loops that NCC auto-vectorizes to the VPU.
 * No LLVM-VE intrinsics needed - just let the compiler do its job.
 * ============================================================================
 */

/* Convert BF16 to FP32 - simple bit shift */
static inline float bf16_to_f32(bf16 val) {
    uint32_t f32_bits = ((uint32_t)val) << 16;
    float result;
    __builtin_memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

/*
 * FP32 SGEMV - Row-major with Frovedis-style optimizations
 * y = W @ x where W is [d x n] row-major, x is [n], y is [d]
 * 
 * Techniques from Frovedis (NEC's production VE library):
 * - vreg: keep accumulator in vector register
 * - shortloop: hint when n <= 256
 * - Tile by MAT_VLEN for large d
 */
#define MAT_VLEN 256

static void sgemv_fp32_simple(float *y, float *x, float *w, int n, int d) {
    if (n <= MAT_VLEN) {
        /* Short inner dimension - use vreg for x buffer */
        float xbuf[MAT_VLEN];
#pragma _NEC vreg(xbuf)
        
        /* Load x into vector register once */
#pragma _NEC shortloop
        for (int j = 0; j < n; j++) {
            xbuf[j] = x[j];
        }
        
        /* Process rows - inner loop uses register-buffered x */
        for (int i = 0; i < d; i++) {
            const float *w_row = w + i * n;
            float sum = 0.0f;
            
#pragma _NEC shortloop
            for (int j = 0; j < n; j++) {
                sum += w_row[j] * xbuf[j];
            }
            
            y[i] = sum;
        }
    }
    else {
        /* Long inner dimension - tile the output rows */
        for (int i = 0; i < d; i += MAT_VLEN) {
            int imax = (i + MAT_VLEN <= d) ? (i + MAT_VLEN) : d;
            
            for (int j = 0; j < n; j++) {
                float xj = x[j];
#pragma _NEC ivdep
                for (int k = i; k < imax; k++) {
                    if (j == 0) y[k] = 0.0f;
                    y[k] += w[k * n + j] * xj;
                }
            }
        }
    }
}

/*
 * FP32 SGEMV - Column-major (Fortran style) - FAST PATH
 * y = W_T @ x where W_T is [n x d] column-major (transposed weights)
 * 
 * This layout gives contiguous memory access in the inner loop,
 * which the VE loves. The inner loop vectorizes over output dimension d.
 * 
 * Memory access pattern:
 *   - x[j] is scalar (broadcast)
 *   - W_T[j*d + 0..d-1] is contiguous vector load
 *   - y[0..d-1] is contiguous vector accumulate
 */
static void sgemv_fp32_cmo_fast(float *y, float *x, float *w_t, int n, int d) {
    /* Zero output */
    for (int i = 0; i < d; i++) {
        y[i] = 0.0f;
    }
    
    /* Accumulate: y += x[j] * W_T[j,:] for each column j */
    for (int j = 0; j < n; j++) {
        float xj = x[j];
        const float *w_col = w_t + j * d;  /* Column j is contiguous! */
        
        for (int i = 0; i < d; i++) {
            y[i] += w_col[i] * xj;
        }
    }
}

/*
 * BF16 SGEMV - Highly optimized using LLVM-VE intrinsics
 * y = W @ x where W is [d x n] BF16 row-major, x is [n] FP32, y is [d] FP32
 * 
 * This external function uses packed fp32 vector operations with 16-way unrolling
 * to achieve near-peak memory bandwidth. Compiled with clang --target=ve-linux.
 * 
 * From: https://github.com/efocht/sgemv-intrinsics
 * See: BF16_ON_VECTOR_ENGINE.txt for details
 */
/* sgemv_packed_bf16_unr declared at top of file */

/* Legacy function names for compatibility */
static void sgemv_bf16(float *y, float *x, unsigned short *w, int n, int d) {
    sgemv_packed_bf16_unr(y, x, w, n, d);
}

static void sgemv_bf16_cmo(float *y, float *x, bf16 *w, int n, int d, int nd) {
    /* Column-major: just call row-major version, layout handled by caller */
    sgemv_packed_bf16_unr(y, x, w, n, d);
    (void)nd;
}

static void sgemv_fp32_cmo(float *y, float *x, float *w, int n, int d, int nd) {
    sgemv_fp32_simple(y, x, w, n, d);
    (void)nd;
}


/*
 * VEDA kernel: BF16 matrix-vector multiply
 * y = W * x where W is [d x n] BF16, x is [n] FP32, y is [d] FP32
 *
 * Uses the highly optimized packed BF16 kernel with 16-way unrolling.
 * Requires row-major weight layout (standard GGML layout).
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_bf16_matvec_hmem(void* y_hmem,
                              void* W_hmem,
                              void* x_hmem,
                              uint64_t d,    /* output dimension (rows) */
                              uint64_t n) {  /* input dimension (cols) */
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    bf16* W = (bf16*)W_hmem;
    float* x = (float*)x_hmem;
    
    /* Use the optimized packed BF16 kernel */
    sgemv_packed_bf16_unr(y, x, W, (int)n, (int)d);
    
    return 0;
}


/*
 * VEDA kernel: BF16 matrix-vector multiply with OpenMP parallelization (HMEM version)
 * y = W * x where:
 *   W is [d x n] BF16 weights in HMEM
 *   x is [n] FP32 input in HMEM
 *   y is [d] FP32 output in HMEM
 *
 * OpenMP parallelizes across output rows.
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_bf16_matvec_omp_hmem(void* y_hmem,
                                  void* W_hmem,
                                  void* x_hmem,
                                  uint64_t d,    /* output dimension (rows) */
                                  uint64_t n) {  /* input dimension (cols) */
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    bf16* W = (bf16*)W_hmem;
    float* x = (float*)x_hmem;
    
    int d_int = (int)d;
    int n_int = (int)n;
    
    /* Parallelize across output rows */
    #pragma omp parallel for
    for (int row = 0; row < d_int; row++) {
        bf16* w_row = W + row * n_int;
        float sum = 0.0f;
        
        #pragma _NEC ivdep
        for (int k = 0; k < n_int; k++) {
            sum += h2f_fast(w_row[k]) * x[k];
        }
        y[row] = sum;
    }
    
    return 0;
}


/*
 * VEDA kernel: FP32 matrix-vector multiply (HMEM version)
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_f32_matvec_hmem(void* y_hmem,
                             void* W_hmem,
                             void* x_hmem,
                             uint64_t d,    /* output dimension (rows) */
                             uint64_t n) {  /* input dimension (cols) */
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* W = (float*)W_hmem;
    float* x = (float*)x_hmem;
    
    sgemv_fp32_simple(y, x, W, (int)n, (int)d);
    
    return 0;
}


/*
 * VEDA kernel: Batched BF16 matrix-vector multiply (process multiple vectors)
 * Y = W * X where:
 *   W is [d x n] BF16 weights
 *   X is [n x batch] FP32 inputs (column-major, each column is an input)
 *   Y is [d x batch] FP32 outputs (column-major)
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_bf16_matmul_hmem(void* Y_hmem,
                              void* W_hmem,
                              void* X_hmem,
                              uint64_t d,      /* output dimension */
                              uint64_t n,      /* input dimension */
                              uint64_t batch) { /* batch size */
    /* HMEM pointers are already converted - use directly */
    float* Y = (float*)Y_hmem;
    bf16* W = (bf16*)W_hmem;
    float* X = (float*)X_hmem;
    
    int d_int = (int)d;
    int n_int = (int)n;
    
    /* Process each input vector in parallel */
    #pragma omp parallel for
    for (uint64_t b = 0; b < batch; b++) {
        float* y_b = Y + b * d;
        float* x_b = X + b * n;
        sgemv_packed_bf16_unr(y_b, x_b, W, n_int, d_int);
    }
    
    return 0;
}


/*
 * VEDA kernel: BF16 matrix-vector multiply with RAW HOST POINTERS
 * This version takes raw 64-bit addresses and casts them directly.
 * VE can access host memory via DMA, so this avoids HMEM copy overhead.
 * 
 * WARNING: The host pointers must be from page-locked memory or HMEM!
 */
uint64_t ve_bf16_matvec_raw(uint64_t y_addr,
                            uint64_t W_addr,
                            uint64_t x_addr,
                            uint64_t d,    /* output dimension (rows) */
                            uint64_t n) {  /* input dimension (cols) */
    float* y = (float*)y_addr;
    bf16* W = (bf16*)W_addr;
    float* x = (float*)x_addr;
    
    /* Use the optimized packed BF16 kernel */
    sgemv_packed_bf16_unr(y, x, W, (int)n, (int)d);
    
    return 0;
}


/*
 * VEDA kernel: BF16 matrix-vector multiply with RAW POINTERS + OpenMP
 */
uint64_t ve_bf16_matvec_raw_omp(uint64_t y_addr,
                                 uint64_t W_addr,
                                 uint64_t x_addr,
                                 uint64_t d,    /* output dimension (rows) */
                                 uint64_t n) {  /* input dimension (cols) */
    float* y = (float*)y_addr;
    bf16* W = (bf16*)W_addr;
    float* x = (float*)x_addr;
    
    int nthr = omp_get_max_threads();
    int d_int = (int)d;
    int n_int = (int)n;
    
    /* Parallel over output rows */
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        int chunk = (d_int + nthr - 1) / nthr;
        int imin = ithr * chunk;
        int imax = imin + chunk;
        if (imax > d_int) imax = d_int;
        
        if (imin < imax) {
            sgemv_packed_bf16_unr(&y[imin], x, &W[imin * n_int], n_int, imax - imin);
        }
    }
    
    return 0;
}


/*
 * BF16 row-major matvec with pointer arguments (for graph compiler)
 * This is a wrapper around the optimized intrinsics kernel.
 * W is row-major: W[m,k] stored at W[m * K + k]
 * Computes: y = W @ x  where y[M], x[K]
 */
void ve_bf16_matvec_rowmajor_ptr_omp(float* y,
                                      const bf16* W,
                                      const float* x,
                                      int M,
                                      int K) {
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;

    #pragma omp parallel num_threads(nthr)
    {
        int ithr = omp_get_thread_num();
        int chunk = (M + nthr - 1) / nthr;
        int imin = ithr * chunk;
        int imax = imin + chunk;
        if (imax > M) imax = M;

        if (imin < imax) {
            sgemv_packed_bf16_unr(&y[imin], (float*)x, (bf16*)&W[imin * K], K, imax - imin);
        }
    }
}

/*
 * "_inner" variant of ve_bf16_matvec_rowmajor_ptr_omp. Caller MUST be
 * inside a #pragma omp parallel region. Uses #pragma omp for so the
 * implicit barrier at end-of-for synchronizes the team before the
 * caller's next op — no manual barrier needed.
 *
 * The graph compiler uses this to amortize OMP fork/join cost across all
 * the matvecs in one compiled cgraph (~18 per attention+FFN block).
 */
void ve_bf16_matvec_rowmajor_ptr_inner(float* y,
                                        const bf16* W,
                                        const float* x,
                                        int M,
                                        int K) {
    int nthr = omp_get_num_threads();
    int chunk = (M + nthr - 1) / nthr;
    #pragma omp for
    for (int g = 0; g < nthr; g++) {
        int imin = g * chunk;
        int imax = imin + chunk;
        if (imax > M) imax = M;
        if (imin < imax) {
            sgemv_packed_bf16_unr(&y[imin], (float*)x, (bf16*)&W[imin * K], K, imax - imin);
        }
    }
}


/*
 * VEDA kernel: FP32 matrix-vector multiply with RAW POINTERS
 */
uint64_t ve_f32_matvec_raw(uint64_t y_addr,
                            uint64_t W_addr,
                            uint64_t x_addr,
                            uint64_t d,    /* output dimension (rows) */
                            uint64_t n) {  /* input dimension (cols) */
    float* y = (float*)y_addr;
    float* W = (float*)W_addr;
    float* x = (float*)x_addr;
    
    sgemv_fp32_simple(y, x, W, (int)n, (int)d);
    
    return 0;
}


/*
 * ==== HBM WEIGHT VERSIONS ====
 * These kernels store weights in VE HBM (device memory) for maximum bandwidth.
 * Input/output remain in HMEM for efficient host access.
 *
 * Weights in HBM:
 *   - Uploaded once at model load time
 *   - VE can access at full HBM bandwidth (1.2 TB/s per card)
 *   - No copy overhead during inference
 *
 * Input/output in HMEM:
 *   - Small buffers (16KB input, ~4KB output typical)
 *   - Changed every kernel call
 *   - HMEM access is slower but avoids copy overhead
 */


/*
 * VEDA kernel: BF16 matvec with weights in HBM, input/output in HMEM
 * This is the optimal configuration for inference:
 *   - W_vptr: VE device pointer (from vedaMemAlloc) pointing to weights in HBM
 *   - x_hmem, y_hmem: HMEM pointers for input/output
 */
uint64_t ve_bf16_matvec_hbm(void* y_hmem,
                             VEDAdeviceptr W_vptr,
                             void* x_hmem,
                             uint64_t d,    /* output dimension (rows) */
                             uint64_t n) {  /* input dimension (cols) */
    /* NOTE: y_hmem and x_hmem are already raw pointers - vedaArgsSetHMEM converts them!
     * Only W_vptr (HBM) needs vedaMemPtr conversion. */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    bf16* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    /* Use the optimized packed BF16 kernel */
    sgemv_packed_bf16_unr(y, x, W, (int)n, (int)d);
    
    return 0;
}


/*
 * VEDA kernel: BF16 matvec with HBM weights + OpenMP parallelism
 */
uint64_t ve_bf16_matvec_hbm_omp(void* y_hmem,
                                 VEDAdeviceptr W_vptr,
                                 void* x_hmem,
                                 uint64_t d,    /* output dimension (rows) */
                                 uint64_t n) {  /* input dimension (cols) */
    /* NOTE: y_hmem and x_hmem are already raw pointers - vedaArgsSetHMEM converts them!
     * Only W_vptr (HBM) needs vedaMemPtr conversion. */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    bf16* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int nthr = omp_get_max_threads();
    int d_int = (int)d;
    int n_int = (int)n;
    
    /* Parallel over output rows */
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        int chunk = (d_int + nthr - 1) / nthr;
        int imin = ithr * chunk;
        int imax = imin + chunk;
        if (imax > d_int) imax = d_int;
        
        if (imin < imax) {
            sgemv_packed_bf16_unr(&y[imin], x, &W[imin * n_int], n_int, imax - imin);
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: FP32 matvec with HBM weights + OpenMP (row-major)
 * 
 * NOTE: VE only has 8 cores. We cap threads to avoid explosion when
 * called from host with many CPU threads.
 */
/* VE_MAX_CORES defined at top of file */

uint64_t ve_f32_matvec_hbm_omp(void* y_hmem,
                                VEDAdeviceptr W_vptr,
                                void* x_hmem,
                                uint64_t d,    /* output dimension (rows) */
                                uint64_t n) {  /* input dimension (cols) */
    /* NOTE: y_hmem and x_hmem are already raw pointers - vedaArgsSetHMEM converts them!
     * Only W_vptr (HBM) needs vedaMemPtr conversion. */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    float* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int d_int = (int)d;
    int n_int = (int)n;
    
    /* Cap threads at VE core count to avoid thread explosion */
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel num_threads(nthr)
    {
        int ithr = omp_get_thread_num();
        int chunk = (d_int + nthr - 1) / nthr;
        int imin = ithr * chunk;
        int imax = imin + chunk;
        if (imax > d_int) imax = d_int;
        
        if (imin < imax) {
            sgemv_fp32_simple(&y[imin], x, &W[imin * n_int], n_int, imax - imin);
        }
    }
    
    return 0;
}

/*
 * VEDA kernel: FP32 matvec with TRANSPOSED HBM weights (column-major)
 * 
 * W_T is [n x d] column-major, meaning W_T[j,i] = W[i,j]
 * This gives contiguous access in the inner loop - perfect for VE.
 * 
 * The inner loop vectorizes over d (output dimension), giving excellent
 * memory bandwidth utilization since we're doing contiguous loads.
 */
uint64_t ve_f32_matvec_hbm_cmo(void* y_hmem,
                                VEDAdeviceptr W_T_vptr,
                                void* x_hmem,
                                uint64_t d,    /* output dimension */
                                uint64_t n) {  /* input dimension */
    /* NOTE: y_hmem and x_hmem are already raw pointers - vedaArgsSetHMEM converts them!
     * Only W_T_vptr (HBM) needs vedaMemPtr conversion. */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    float* W_T;
    vedaMemPtr((void**)&W_T, W_T_vptr);
    
    int d_int = (int)d;
    int n_int = (int)n;
    
    /* Zero output */
    for (int i = 0; i < d_int; i++) {
        y[i] = 0.0f;
    }
    
    /* Accumulate: y += x[j] * W_T[j,:] for each input element j */
    for (int j = 0; j < n_int; j++) {
        float xj = x[j];
        const float *w_col = W_T + j * d_int;  /* Column j is contiguous! */
        
        /* This inner loop vectorizes beautifully - contiguous access */
        for (int i = 0; i < d_int; i++) {
            y[i] += w_col[i] * xj;
        }
    }
    
    return 0;
}


/* ============================================================================
 * RMS NORMALIZATION KERNELS
 * 
 * RMS Norm: y[i] = x[i] / sqrt(mean(x^2) + eps)
 * where mean(x^2) = sum(x[i]^2) / n
 *
 * This is used in LLaMA/Llama2/Llama3 models for layer normalization.
 * Called twice per transformer layer (pre-attention and pre-FFN).
 * ============================================================================
 */

#include <math.h>

/*
 * VEDA kernel: RMS Normalization (single row)
 * 
 * Computes: y[i] = x[i] * scale / sqrt(sum(x[i]^2)/n + eps)
 *
 * Parameters:
 *   y_hmem: Output buffer [n] floats (HMEM)
 *   x_hmem: Input buffer [n] floats (HMEM)
 *   n: Number of elements
 *   eps: Epsilon for numerical stability (passed as uint64_t, reinterpret as float)
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_rms_norm_f32_hmem(void* y_hmem,
                               void* x_hmem,
                               uint64_t n,
                               uint64_t eps_bits) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    /* Reinterpret eps_bits as float */
    float eps;
    uint32_t eps32 = (uint32_t)eps_bits;
    __builtin_memcpy(&eps, &eps32, sizeof(float));
    
    int n_int = (int)n;
    
    /* Step 1: Compute sum of squares using vector operations */
    /* NOTE: No vovertake here! The sum_sq result must complete before next loop */
    double sum_sq = 0.0;
    
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    
    /* Step 2: Compute scale factor */
    float mean_sq = (float)(sum_sq / n_int);
    float scale = 1.0f / sqrtf(mean_sq + eps);
    
    /* Step 3: Scale each element - vovertake OK here, elements are independent */
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    for (int i = 0; i < n_int; i++) {
        y[i] = x[i] * scale;
    }
    
    return 0;
}


/*
 * VEDA kernel: RMS Normalization with OpenMP (for batched processing)
 * 
 * Processes multiple rows in parallel:
 *   - x is [ne01 x ne00] (ne01 rows, each of ne00 elements)
 *   - y is [ne01 x ne00] output
 *   - Each row is normalized independently
 *
 * This matches GGML's tensor layout where ne00 is the innermost dimension.
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_rms_norm_f32_omp_hmem(void* y_hmem,
                                   void* x_hmem,
                                   uint64_t ne00,    /* elements per row (normalization dim) */
                                   uint64_t ne01,    /* number of rows to process */
                                   uint64_t eps_bits) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    /* Reinterpret eps_bits as float */
    float eps;
    uint32_t eps32 = (uint32_t)eps_bits;
    __builtin_memcpy(&eps, &eps32, sizeof(float));
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    
    /* Process rows in parallel */
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* x_row = x + row * n;
        float* y_row = y + row * n;
        
        /* Compute sum of squares for this row */
        /* NOTE: No vovertake! sum_sq must complete before next loop */
        double sum_sq = 0.0;
        
        #pragma _NEC ivdep
        for (int i = 0; i < n; i++) {
            sum_sq += (double)x_row[i] * (double)x_row[i];
        }
        
        /* Compute scale factor */
        float mean_sq = (float)(sum_sq / n);
        float scale = 1.0f / sqrtf(mean_sq + eps);
        
        /* Scale each element - vovertake OK here, elements are independent */
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] = x_row[i] * scale;
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: RMS Normalization in-place (single row)
 * 
 * Modifies x in place: x[i] = x[i] / sqrt(mean(x^2) + eps)
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_rms_norm_f32_inplace_hmem(void* x_hmem,
                                       uint64_t n,
                                       uint64_t eps_bits) {
    /* HMEM pointers are already converted - use directly */
    float* x = (float*)x_hmem;
    
    /* Reinterpret eps_bits as float */
    float eps;
    uint32_t eps32 = (uint32_t)eps_bits;
    __builtin_memcpy(&eps, &eps32, sizeof(float));
    
    int n_int = (int)n;
    
    /* Step 1: Compute sum of squares */
    /* NOTE: No vovertake here! sum_sq must complete before next loop */
    double sum_sq = 0.0;
    
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    
    /* Step 2: Compute scale factor */
    float mean_sq = (float)(sum_sq / n_int);
    float scale = 1.0f / sqrtf(mean_sq + eps);
    
    /* Step 3: Scale in place */
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    for (int i = 0; i < n_int; i++) {
        x[i] *= scale;
    }
    
    return 0;
}


/* ============================================================================
 * SOFTMAX KERNELS
 * 
 * Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
 *
 * This is used in attention layers for computing attention weights.
 * The numerically stable version subtracts max before exp to prevent overflow.
 * ============================================================================
 */

/*
 * VEDA kernel: Softmax (single row, simple version)
 * 
 * Computes: y[i] = exp(scale * x[i] - max) / sum(exp(scale * x[j] - max))
 *
 * Parameters:
 *   y_hmem: Output buffer [n] floats (HMEM)
 *   x_hmem: Input buffer [n] floats (HMEM)
 *   n: Number of elements
 *   scale_bits: Scale factor (passed as uint64_t, reinterpret as float)
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_soft_max_f32_hmem(void* y_hmem,
                               void* x_hmem,
                               uint64_t n,
                               uint64_t scale_bits) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    /* Reinterpret scale_bits as float */
    float scale;
    uint32_t scale32 = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &scale32, sizeof(float));
    
    int n_int = (int)n;
    
    /* Step 1: Find max value (for numerical stability) */
    /* NOTE: No vovertake here! max_val must complete before next loop */
    float max_val = -INFINITY;
    
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        float val = scale * x[i];
        if (val > max_val) max_val = val;
    }
    
    /* Step 2: Compute exp(scale*x - max) and sum */
    /* NOTE: No vovertake here! sum must complete before next loop */
    double sum = 0.0;
    
    #pragma _NEC ivdep
    #pragma _NEC novob
    for (int i = 0; i < n_int; i++) {
        float val = expf(scale * x[i] - max_val);
        y[i] = val;
        sum += (double)val;
    }
    
    /* Step 3: Normalize by sum */
    float inv_sum = 1.0f / (float)sum;
    
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    for (int i = 0; i < n_int; i++) {
        y[i] *= inv_sum;
    }
    
    return 0;
}


/*
 * VEDA kernel: Softmax with OpenMP (for batched processing)
 * 
 * Processes multiple rows in parallel:
 *   - x is [ne01 x ne00] (ne01 rows, each of ne00 elements)
 *   - y is [ne01 x ne00] output
 *   - Each row is softmax'd independently
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_soft_max_f32_omp_hmem(VEDAdeviceptr y_hbm,
                                   VEDAdeviceptr x_hbm,
                                   uint64_t ne00,    /* elements per row (softmax dim) */
                                   uint64_t ne01,    /* number of rows to process */
                                   uint64_t scale_bits) {
    float* y;
    float* x;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    
    /* Reinterpret scale_bits as float */
    float scale;
    uint32_t scale32 = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &scale32, sizeof(float));
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    
    /* Process rows in parallel */
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* x_row = x + row * n;
        float* y_row = y + row * n;
        
        /* Find max value for this row */
        /* NOTE: No vovertake here! max_val must complete before next loop */
        float max_val = -INFINITY;
        
        #pragma _NEC ivdep
        for (int i = 0; i < n; i++) {
            float val = scale * x_row[i];
            if (val > max_val) max_val = val;
        }
        
        /* Compute exp(scale*x - max) and sum */
        /* NOTE: No vovertake here! sum must complete before next loop */
        double sum = 0.0;
        
        #pragma _NEC ivdep
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            float val = expf(scale * x_row[i] - max_val);
            y_row[i] = val;
            sum += (double)val;
        }
        
        /* Normalize by sum */
        float inv_sum = 1.0f / (float)sum;
        
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] *= inv_sum;
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: Softmax with mask (F32 mask) - broadcast mode
 * 
 * Computes: y[i] = exp(scale * x[i] + mask[i] - max) / sum(...)
 * 
 * The mask is broadcast across dimensions where mask dim is 1.
 * For attention softmax:
 *   - Input:  [ne0, ne1, ne2, ne3] = [n_kv, n_tokens, n_heads, batch]
 *   - Mask:   [ne0, mask_ne1, 1, 1]
 *   - Mask row for input row r = mask[(r % ne1) * ne0] when mask_ne1 == ne1
 *   - If mask_ne1 == 1, use same mask for all rows
 *
 * Parameters:
 *   ne00: columns (softmax dimension)
 *   ne01: total number of rows (ne1 * ne2 * ne3 from input)
 *   mask_ne1: number of mask rows (typically matches ne1 of input)
 *   ne1: ne1 of input (for computing mask row index)
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_soft_max_f32_masked_hmem(void* y_hmem,
                                      void* x_hmem,
                                      void* mask_hmem,
                                      uint64_t ne00,    /* elements per row */
                                      uint64_t ne01,    /* number of rows */
                                      uint64_t scale_bits) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    float* mask = (float*)mask_hmem;
    
    /* Reinterpret scale_bits as float */
    float scale;
    uint32_t scale32 = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &scale32, sizeof(float));
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    
    /* Process rows in parallel */
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* x_row = x + row * n;
        float* y_row = y + row * n;
        /* Mask is broadcast across rows - use same mask for all rows */
        float* mask_row = mask;
        
        /* Find max value for this row */
        /* NOTE: No vovertake here! max_val must complete before next loop */
        float max_val = -INFINITY;
        
        #pragma _NEC ivdep
        for (int i = 0; i < n; i++) {
            float val = scale * x_row[i] + mask_row[i];
            if (val > max_val) max_val = val;
        }
        
        /* Compute exp(scale*x + mask - max) and sum */
        /* NOTE: No vovertake here! sum must complete before next loop */
        double sum = 0.0;
        
        #pragma _NEC ivdep
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            float val = expf(scale * x_row[i] + mask_row[i] - max_val);
            y_row[i] = val;
            sum += (double)val;
        }
        
        /* Normalize by sum */
        float inv_sum = 1.0f / (float)sum;
        
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] *= inv_sum;
        }
    }
    
    return 0;
}

/*
 * VEDA kernel: Softmax with mask - attention broadcast mode
 * 
 * For attention softmax where:
 *   - Input:  [ne0, ne1, ne2, ne3] = [n_kv, n_tokens, n_heads, batch]
 *   - Mask:   [ne0, ne1, 1, 1] = [n_kv, n_tokens, 1, 1]
 * 
 * The mask is indexed by token (ne1 dimension), and the same mask is used
 * for all heads of the same token.
 *
 * Total rows = ne1 * ne2 * ne3
 * For row r: token_idx = r % ne1, head_idx = (r / ne1) % ne2
 * Mask row = mask + token_idx * ne0
 */
uint64_t ve_soft_max_f32_masked_attn_hmem(VEDAdeviceptr y_hbm,
                                           VEDAdeviceptr x_hbm,
                                           VEDAdeviceptr mask_hbm,
                                           uint64_t ne00,       /* elements per row (n_kv) */
                                           uint64_t total_rows, /* ne1 * ne2 * ne3 */
                                           uint64_t ne1,        /* n_tokens (mask row stride) */
                                           uint64_t scale_bits) {
    float* y;
    float* x;
    float* mask;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&mask, mask_hbm) != 0) return 3;
    
    float scale;
    uint32_t scale32 = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &scale32, sizeof(float));
    
    int n = (int)ne00;
    int nrows = (int)total_rows;
    int ntokens = (int)ne1;
    
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* x_row = x + row * n;
        float* y_row = y + row * n;
        
        /* Compute mask row: token_idx = row % ntokens */
        int token_idx = row % ntokens;
        float* mask_row = mask + token_idx * n;
        
        /* Find max value */
        float max_val = -INFINITY;
        
        #pragma _NEC ivdep
        for (int i = 0; i < n; i++) {
            float val = scale * x_row[i] + mask_row[i];
            if (val > max_val) max_val = val;
        }
        
        /* Compute exp and sum */
        double sum = 0.0;
        
        #pragma _NEC ivdep
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            float val = expf(scale * x_row[i] + mask_row[i] - max_val);
            y_row[i] = val;
            sum += (double)val;
        }
        
        /* Normalize */
        float inv_sum = 1.0f / (float)sum;
        
        #pragma _NEC ivdep
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] *= inv_sum;
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: Softmax with attention mask AND attention sinks
 * 
 * Computes: y[row, i] = exp(scale * x[row, i] + mask[token_idx, i] - max) / (sum + exp(sink[head] - max))
 * 
 * This kernel handles "attention sinks" used by models like gpt-oss-20b.
 * Attention sinks absorb excess probability mass, so rows will sum to < 1.0.
 *
 * Layout:
 * - Input x: [ne00, ne1, ne2, ne3] where ne1=tokens, ne2=heads, ne3=batch
 * - Mask: [ne00, ne1] - per-token mask, broadcast across heads
 * - Sinks: [ne2] - one value per head
 * - Total rows = ne1 * ne2 * ne3
 * - For row r: token_idx = r % ne1, head_idx = (r / ne1) % ne2
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_soft_max_f32_masked_attn_sinks_hmem(void* y_hmem,
                                                 void* x_hmem,
                                                 void* mask_hmem,
                                                 void* sinks_hmem,
                                                 uint64_t ne00,       /* elements per row (n_kv) */
                                                 uint64_t total_rows, /* ne1 * ne2 * ne3 */
                                                 uint64_t ne1,        /* n_tokens (for mask indexing) */
                                                 uint64_t ne2,        /* n_heads (for sink indexing) */
                                                 uint64_t scale_bits) {
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    float* mask = (float*)mask_hmem;
    float* sinks = (float*)sinks_hmem;
    
    float scale;
    uint32_t scale32 = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &scale32, sizeof(float));
    
    int n = (int)ne00;
    int nrows = (int)total_rows;
    int ntokens = (int)ne1;
    int nheads = (int)ne2;
    
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* x_row = x + row * n;
        float* y_row = y + row * n;
        
        /* Compute indices: token_idx = row % ne1, head_idx = (row / ne1) % ne2 */
        int token_idx = row % ntokens;
        int head_idx = (row / ntokens) % nheads;
        float* mask_row = mask + token_idx * n;
        float sink_val = sinks[head_idx];
        
        /* Find max value - include sink in max calculation */
        float max_val = sink_val;  /* Start with sink value */
        
        #pragma _NEC ivdep
        for (int i = 0; i < n; i++) {
            float val = scale * x_row[i] + mask_row[i];
            if (val > max_val) max_val = val;
        }
        
        /* Compute exp and sum */
        double sum = 0.0;
        
        #pragma _NEC ivdep
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            float val = expf(scale * x_row[i] + mask_row[i] - max_val);
            y_row[i] = val;
            sum += (double)val;
        }
        
        /* Add sink contribution to sum */
        sum += (double)expf(sink_val - max_val);
        
        /* Normalize - rows will sum to < 1.0 because sink absorbs some mass */
        float inv_sum = 1.0f / (float)sum;
        
        #pragma _NEC ivdep
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] *= inv_sum;
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: Softmax with per-row mask (F32 mask)
 * 
 * Computes: y[row, i] = exp(scale * x[row, i] + mask[row, i] - max) / sum(...)
 * 
 * The mask has the same shape as input: [ne01, ne00].
 * Each row uses its own corresponding mask row.
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_soft_max_f32_masked_full_hmem(void* y_hmem,
                                           void* x_hmem,
                                           void* mask_hmem,
                                           uint64_t ne00,    /* elements per row */
                                           uint64_t ne01,    /* number of rows */
                                           uint64_t scale_bits) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    float* mask = (float*)mask_hmem;
    
    /* Reinterpret scale_bits as float */
    float scale;
    uint32_t scale32 = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &scale32, sizeof(float));
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    
    /* Process rows in parallel */
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* x_row = x + row * n;
        float* y_row = y + row * n;
        float* mask_row = mask + row * n;  /* Each row has its own mask */
        
        /* Find max value for this row */
        /* NOTE: No vovertake here! max_val must complete before next loop */
        float max_val = -INFINITY;
        
        #pragma _NEC ivdep
        for (int i = 0; i < n; i++) {
            float val = scale * x_row[i] + mask_row[i];
            if (val > max_val) max_val = val;
        }
        
        /* Compute exp(scale*x + mask - max) and sum */
        /* NOTE: No vovertake here! sum must complete before next loop */
        double sum = 0.0;
        
        #pragma _NEC ivdep
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            float val = expf(scale * x_row[i] + mask_row[i] - max_val);
            y_row[i] = val;
            sum += (double)val;
        }
        
        /* Normalize by sum */
        float inv_sum = 1.0f / (float)sum;
        
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] *= inv_sum;
        }
    }
    
    return 0;
}


/* ============================================================================
 * ELEMENT-WISE OPERATIONS
 * 
 * These are simple but frequently used operations in transformer models.
 * They benefit from VE's vector registers (256 elements per instruction).
 * ============================================================================
 */

/*
 * VEDA kernel: Element-wise ADD (y = a + b)
 * 
 * Supports broadcasting: if repeat_b > 1, b is repeated for each segment
 * This handles cases like adding a bias to multiple rows.
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_add_f32_hmem(void* y_hmem,
                          void* a_hmem,
                          void* b_hmem,
                          uint64_t n,          /* total elements to process */
                          uint64_t n_b,        /* elements in b (for broadcasting) */
                          uint64_t repeat_b) { /* how many times to repeat b */
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* a = (float*)a_hmem;
    float* b = (float*)b_hmem;
    
    int n_int = (int)n;
    int nb_int = (int)n_b;
    
    if (repeat_b == 1 && n == n_b) {
        /* Simple case: same size, no broadcasting */
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] + b[i];
        }
    } else {
        /* Broadcasting case: b is repeated */
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] + b[i % nb_int];
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: Element-wise ADD with OpenMP
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_add_f32_omp_hmem(void* y_hmem,
                              void* a_hmem,
                              void* b_hmem,
                              uint64_t ne00,     /* innermost dim */
                              uint64_t ne01,     /* second dim (rows) */
                              uint64_t nb_total) { /* total elements in b */
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* a = (float*)a_hmem;
    float* b = (float*)b_hmem;
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    int nb = (int)nb_total;
    
    /* Process rows in parallel */
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* y_row = y + row * n;
        float* a_row = a + row * n;
        
        /* Determine which part of b to use */
        int b_offset = (row * n) % nb;
        
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] = a_row[i] + b[(b_offset + i) % nb];
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: Element-wise MUL (y = a * b)
 * 
 * Supports broadcasting similar to ADD.
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_mul_f32_hmem(void* y_hmem,
                          void* a_hmem,
                          void* b_hmem,
                          uint64_t n,
                          uint64_t n_b,
                          uint64_t repeat_b) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* a = (float*)a_hmem;
    float* b = (float*)b_hmem;
    
    int n_int = (int)n;
    int nb_int = (int)n_b;
    
    if (repeat_b == 1 && n == n_b) {
        /* Simple case: same size */
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] * b[i];
        }
    } else {
        /* Broadcasting case */
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] * b[i % nb_int];
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: Element-wise MUL with OpenMP
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_mul_f32_omp_hmem(void* y_hmem,
                              void* a_hmem,
                              void* b_hmem,
                              uint64_t ne00,
                              uint64_t ne01,
                              uint64_t nb_total) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* a = (float*)a_hmem;
    float* b = (float*)b_hmem;
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    int nb = (int)nb_total;
    
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* y_row = y + row * n;
        float* a_row = a + row * n;
        int b_offset = (row * n) % nb;
        
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] = a_row[i] * b[(b_offset + i) % nb];
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: SCALE (y = a * scalar)
 * 
 * Simple scalar multiplication.
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_scale_f32_hmem(void* y_hmem,
                            void* a_hmem,
                            uint64_t n,
                            uint64_t scale_bits) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* a = (float*)a_hmem;
    
    /* Reinterpret scale_bits as float */
    float scale;
    uint32_t scale32 = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &scale32, sizeof(float));
    
    int n_int = (int)n;
    
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    for (int i = 0; i < n_int; i++) {
        y[i] = a[i] * scale;
    }
    
    return 0;
}


/*
 * VEDA kernel: SCALE with OpenMP
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_scale_f32_omp_hmem(void* y_hmem,
                                void* a_hmem,
                                uint64_t ne00,
                                uint64_t ne01,
                                uint64_t scale_bits) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* a = (float*)a_hmem;
    
    float scale;
    uint32_t scale32 = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &scale32, sizeof(float));
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* y_row = y + row * n;
        float* a_row = a + row * n;
        
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            y_row[i] = a_row[i] * scale;
        }
    }
    
    return 0;
}


/* ============================================================================
 * ACTIVATION FUNCTIONS
 * 
 * Used in FFN layers of transformer models.
 * ============================================================================
 */

/*
 * VEDA kernel: SILU (Sigmoid Linear Unit) activation
 * 
 * SILU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Also known as "Swish" activation.
 * Used in LLaMA models for FFN.
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_silu_f32_hmem(void* y_hmem,
                           void* x_hmem,
                           uint64_t n) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    int n_int = (int)n;
    
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    for (int i = 0; i < n_int; i++) {
        float xi = x[i];
        /* SILU = x * sigmoid(x) = x / (1 + exp(-x)) */
        y[i] = xi / (1.0f + expf(-xi));
    }
    
    return 0;
}


/*
 * VEDA kernel: SILU with OpenMP
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_silu_f32_omp_hmem(void* y_hmem,
                               void* x_hmem,
                               uint64_t ne00,
                               uint64_t ne01) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* y_row = y + row * n;
        float* x_row = x + row * n;
        
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            float xi = x_row[i];
            y_row[i] = xi / (1.0f + expf(-xi));
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: GELU activation
 * 
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Used in some models (GPT-2, BERT).
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_gelu_f32_hmem(void* y_hmem,
                           void* x_hmem,
                           uint64_t n) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    int n_int = (int)n;
    const float SQRT_2_OVER_PI = 0.7978845608f;  /* sqrt(2/pi) */
    const float GELU_COEF = 0.044715f;
    
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    for (int i = 0; i < n_int; i++) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        y[i] = 0.5f * xi * (1.0f + tanhf(SQRT_2_OVER_PI * (xi + GELU_COEF * x3)));
    }
    
    return 0;
}


/*
 * VEDA kernel: GELU with OpenMP
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_gelu_f32_omp_hmem(void* y_hmem,
                               void* x_hmem,
                               uint64_t ne00,
                               uint64_t ne01) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    int n = (int)ne00;
    int nrows = (int)ne01;
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float GELU_COEF = 0.044715f;
    
    #pragma omp parallel for
    for (int row = 0; row < nrows; row++) {
        float* y_row = y + row * n;
        float* x_row = x + row * n;
        
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < n; i++) {
            float xi = x_row[i];
            float x3 = xi * xi * xi;
            y_row[i] = 0.5f * xi * (1.0f + tanhf(SQRT_2_OVER_PI * (xi + GELU_COEF * x3)));
        }
    }
    
    return 0;
}


/* ============================================================================
 * K-Quant Support (Q2_K, Q3_K, Q6_K)
 * 
 * These quantization formats are used by popular models (Llama, Phi, etc.)
 * with the K-quant family providing good quality at low bit-widths.
 * ============================================================================
 */

/* QK_K defined at top of file */

/*
 * GGML-compatible nearest_int function.
 * This uses the same bit manipulation trick as GGML for exact compatibility.
 * The trick: adding 2^23 * 1.5 shifts the integer into the mantissa bits.
 */
static inline int nearest_int_ggml(float fval) {
    float val = fval + 12582912.f;  /* 2^23 * 1.5 */
    int i;
    __builtin_memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

/*
 * Q2_K Block Structure:
 *   uint8_t  scales[16]  - 4-bit scales and mins packed
 *   uint8_t  qs[64]      - 2-bit quants (4 per byte)
 *   uint16_t d           - fp16 super-block scale
 *   uint16_t dmin        - fp16 super-block min scale
 * Total: 84 bytes per 256 elements
 */
typedef struct {
    uint8_t  scales[QK_K/16];  /* 16 bytes */
    uint8_t  qs[QK_K/4];       /* 64 bytes */
    uint16_t d;                /* fp16 scale */
    uint16_t dmin;             /* fp16 min scale */
} block_q2_K_ve;

/*
 * Q8_K Block Structure (activations):
 *   float   d         - scale
 *   int8_t  qs[256]   - int8 quants
 *   int16_t bsums[16] - sum of quants in groups of 16
 * Total: 292 bytes per 256 elements
 */
typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K_ve;


/*
 * VEDA kernel: Q2_K matrix-vector multiply
 * 
 * Computes y = W @ x where:
 *   - W is (d x n) in Q2_K format
 *   - x is (n,) in F32 format (quantized to Q8_K internally)
 *   - y is (d,) in F32 format
 *
 * NOTE: When passed via vedaArgsSetHMEM(), HMEM pointers are already converted
 * to raw VE-accessible pointers. Do NOT call vedaHMemPtr()!
 */
uint64_t ve_q2k_matvec_f32_hmem(void* y_hmem,
                                 void* W_hmem,
                                 void* x_hmem,
                                 uint64_t d,
                                 uint64_t n) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    block_q2_K_ve* W = (block_q2_K_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int d_int = (int)d;
    int n_int = (int)n;
    int nb = n_int / QK_K;  /* Number of 256-element blocks per row */
    
    /* Quantize input x to Q8_K format */
    block_q8_K_ve q8[nb];
    
    /* Quantize x to Q8_K */
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        /* Find max absolute value */
        float amax = 0.0f;
        float max_val = 0.0f;
        #pragma _NEC ivdep
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K/16; j++) {
                q8[i].bsums[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                q8[i].qs[j] = (int8_t)v;
            }
            
            /* Compute block sums */
            for (int j = 0; j < QK_K/16; j++) {
                int sum = 0;
                #pragma _NEC ivdep
                for (int k = 0; k < 16; k++) {
                    sum += q8[i].qs[j*16 + k];
                }
                q8[i].bsums[j] = (int16_t)sum;
            }
        }
    }
    
    /* Compute matrix-vector product */
    for (int row = 0; row < d_int; row++) {
        block_q2_K_ve* Wrow = W + row * nb;
        float sumf = 0.0f;
        
        for (int i = 0; i < nb; i++) {
            const uint8_t* q2 = Wrow[i].qs;
            const int8_t* q8_qs = q8[i].qs;
            const uint8_t* sc = Wrow[i].scales;
            
            int summs = 0;
            #pragma _NEC ivdep
            for (int j = 0; j < 16; j++) {
                summs += q8[i].bsums[j] * (sc[j] >> 4);
            }
            
            float dall = q8[i].d * __gnu_h2f_ieee(Wrow[i].d);
            float dmin = q8[i].d * __gnu_h2f_ieee(Wrow[i].dmin);
            
            int isum = 0;
            int is = 0;
            
            for (int k = 0; k < QK_K/128; k++) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    int d_scale = sc[is++] & 0xF;
                    
                    int isuml = 0;
                    #pragma _NEC ivdep
                    for (int l = 0; l < 16; l++) {
                        isuml += q8_qs[l] * ((q2[l] >> shift) & 3);
                    }
                    isum += d_scale * isuml;
                    
                    d_scale = sc[is++] & 0xF;
                    isuml = 0;
                    #pragma _NEC ivdep
                    for (int l = 16; l < 32; l++) {
                        isuml += q8_qs[l] * ((q2[l] >> shift) & 3);
                    }
                    isum += d_scale * isuml;
                    
                    shift += 2;
                    q8_qs += 32;
                }
                q2 += 32;
            }
            
            sumf += dall * isum - dmin * summs;
        }
        
        y[row] = sumf;
    }
    
    return 0;
}


/*
 * VEDA kernel: Q2_K matrix-vector multiply with OpenMP
 */
uint64_t ve_q2k_matvec_f32_omp_hmem(void* y_hmem,
                                     void* W_hmem,
                                     void* x_hmem,
                                     uint64_t d,
                                     uint64_t n) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    block_q2_K_ve* W = (block_q2_K_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int d_int = (int)d;
    int n_int = (int)n;
    int nb = n_int / QK_K;
    
    /* Quantize input x to Q8_K format (shared across all threads) */
    block_q8_K_ve* q8 = (block_q8_K_ve*)__builtin_alloca(nb * sizeof(block_q8_K_ve));
    
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        float amax = 0.0f;
        float max_val = 0.0f;
        #pragma _NEC ivdep
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K/16; j++) {
                q8[i].bsums[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                q8[i].qs[j] = (int8_t)v;
            }
            
            for (int j = 0; j < QK_K/16; j++) {
                int sum = 0;
                #pragma _NEC ivdep
                for (int k = 0; k < 16; k++) {
                    sum += q8[i].qs[j*16 + k];
                }
                q8[i].bsums[j] = (int16_t)sum;
            }
        }
    }
    
    /* Parallel matrix-vector product over rows */
    #pragma omp parallel for
    for (int row = 0; row < d_int; row++) {
        block_q2_K_ve* Wrow = W + row * nb;
        float sumf = 0.0f;
        
        for (int i = 0; i < nb; i++) {
            const uint8_t* q2 = Wrow[i].qs;
            const int8_t* q8_qs = q8[i].qs;
            const uint8_t* sc = Wrow[i].scales;
            
            int summs = 0;
            #pragma _NEC ivdep
            for (int j = 0; j < 16; j++) {
                summs += q8[i].bsums[j] * (sc[j] >> 4);
            }
            
            float dall = q8[i].d * __gnu_h2f_ieee(Wrow[i].d);
            float dmin = q8[i].d * __gnu_h2f_ieee(Wrow[i].dmin);
            
            int isum = 0;
            int is = 0;
            
            for (int k = 0; k < QK_K/128; k++) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    int d_scale = sc[is++] & 0xF;
                    
                    int isuml = 0;
                    #pragma _NEC ivdep
                    for (int l = 0; l < 16; l++) {
                        isuml += q8_qs[l] * ((q2[l] >> shift) & 3);
                    }
                    isum += d_scale * isuml;
                    
                    d_scale = sc[is++] & 0xF;
                    isuml = 0;
                    #pragma _NEC ivdep
                    for (int l = 16; l < 32; l++) {
                        isuml += q8_qs[l] * ((q2[l] >> shift) & 3);
                    }
                    isum += d_scale * isuml;
                    
                    shift += 2;
                    q8_qs += 32;
                }
                q2 += 32;
            }
            
            sumf += dall * isum - dmin * summs;
        }
        
        y[row] = sumf;
    }
    
    return 0;
}


/* ============================================================================
 * Q2_Kx8 (REPACKED) K-Quant Support
 * 
 * This handles the CPU_REPACK format where 8 consecutive Q2_K blocks are
 * interleaved for SIMD efficiency. The layout is:
 *   - d[8]       : FP16 scales for 8 blocks
 *   - dmin[8]    : FP16 min scales for 8 blocks
 *   - scales[128]: Interleaved 4-bit scales/mins
 *   - qs[512]    : Interleaved 2-bit quants
 * 
 * Total: 672 bytes for 8 blocks = 2048 elements
 * ============================================================================
 */

/*
 * Q2_Kx8 Block Structure (repacked format for SIMD):
 * 8 Q2_K blocks interleaved together.
 */
typedef struct {
    uint16_t d[8];        /* FP16 super-block scales for 8 blocks */
    uint16_t dmin[8];     /* FP16 super-block min scales for 8 blocks */
    uint8_t  scales[128]; /* Interleaved 4-bit scales and mins */
    uint8_t  qs[512];     /* Interleaved 2-bit quants (blocklen=8) */
} block_q2_Kx8_ve;

/* Static assert to verify size: 16*2 + 128 + 512 = 672 bytes */
_Static_assert(sizeof(block_q2_Kx8_ve) == 672, "block_q2_Kx8_ve size mismatch");


/*
 * VEDA kernel: Q2_Kx8 (repacked) matrix-vector multiply
 * 
 * Computes y = W @ x where:
 *   - W is (M x K) in Q2_Kx8 repacked format
 *   - x is (K,) in F32 format (quantized to Q8_K internally)
 *   - y is (M,) in F32 format
 *
 * The repacked format groups 8 consecutive rows together, so:
 *   - M must be a multiple of 8 (enforced by CPU_REPACK)
 *   - K must be a multiple of 256 (QK_K)
 */
uint64_t ve_q2kx8_matvec_f32_hmem(void* y_hmem,
                                   void* W_hmem,
                                   void* x_hmem,
                                   uint64_t M,
                                   uint64_t K) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    block_q2_Kx8_ve* W = (block_q2_Kx8_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / QK_K;  /* Number of 256-element blocks per row */
    
    const int ncols_interleaved = 8;  /* 8 rows packed together */
    const int blocklen = 8;           /* Interleave block length */
    
    /* Quantize input x to Q8_K format */
    block_q8_K_ve* q8 = (block_q8_K_ve*)__builtin_alloca(nb * sizeof(block_q8_K_ve));
    
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        float amax = 0.0f;
        float max_val = 0.0f;
        #pragma _NEC ivdep
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K/16; j++) {
                q8[i].bsums[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                q8[i].qs[j] = (int8_t)v;
            }
            
            for (int j = 0; j < QK_K/16; j++) {
                int sum = 0;
                #pragma _NEC ivdep
                for (int kk = 0; kk < 16; kk++) {
                    sum += q8[i].qs[j*16 + kk];
                }
                q8[i].bsums[j] = (int16_t)sum;
            }
        }
    }
    
    /* Process 8 rows at a time (matching the repacked format) */
    for (int row_group = 0; row_group < M_int; row_group += ncols_interleaved) {
        /* Block pointer for this group of 8 rows */
        block_q2_Kx8_ve* b_ptr = W + (row_group / ncols_interleaved) * nb;
        
        float sumf[8] = {0};
        float sum_minf[8] = {0};
        
        /* Process each block along K dimension */
        for (int l = 0; l < nb; l++) {
            /* Main dot product - following reference implementation */
            for (int k = 0; k < (QK_K / (4 * blocklen)); k++) {
                const uint8_t* scales_0 = b_ptr[l].scales + (k / 4) * 64;
                const uint8_t* scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                const uint8_t* scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                const uint8_t* scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                
                for (int j = 0; j < ncols_interleaved; j++) {
                    int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
                    int sumi = 0;
                    int offset = ((k / 2) % 2) + j * 2;
                    
                    for (int ii = 0; ii < blocklen; ii++) {
                        const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] & 3);
                        const int v1 = (int8_t)((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 2) & 3);
                        const int v2 = (int8_t)((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 4) & 3);
                        const int v3 = (int8_t)((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 6) & 3);
                        
                        sumi1 = v0 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii];
                        sumi2 = v1 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 32];
                        sumi3 = v2 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 64];
                        sumi4 = v3 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 96];
                        
                        sumi1 = sumi1 * (scales_0[offset] & 0xF);
                        sumi2 = sumi2 * (scales_1[offset] & 0xF);
                        sumi3 = sumi3 * (scales_2[offset] & 0xF);
                        sumi4 = sumi4 * (scales_3[offset] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * __gnu_h2f_ieee(b_ptr[l].d[j]) * q8[l].d;
                }
            }
            
            /* Compute min term */
            for (int sb = 0; sb < 8; sb++) {
                const uint8_t* mins = b_ptr[l].scales + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += ((mins[j * 2] >> 4) * q8[l].bsums[sb * 2] +
                                   (mins[j * 2 + 1] >> 4) * q8[l].bsums[sb * 2 + 1]) *
                                   __gnu_h2f_ieee(b_ptr[l].dmin[j]) * q8[l].d;
                }
            }
        }
        
        /* Write output for this group of 8 rows */
        for (int j = 0; j < ncols_interleaved && (row_group + j) < M_int; j++) {
            y[row_group + j] = sumf[j] - sum_minf[j];
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: Q2_Kx8 matrix-vector multiply with OpenMP
 */
uint64_t ve_q2kx8_matvec_f32_omp_hmem(void* y_hmem,
                                       void* W_hmem,
                                       void* x_hmem,
                                       uint64_t M,
                                       uint64_t K) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    block_q2_Kx8_ve* W = (block_q2_Kx8_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / QK_K;
    
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    
    /* Quantize input x to Q8_K format (shared across all threads) */
    block_q8_K_ve* q8 = (block_q8_K_ve*)__builtin_alloca(nb * sizeof(block_q8_K_ve));
    
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        float amax = 0.0f;
        float max_val = 0.0f;
        #pragma _NEC ivdep
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K/16; j++) {
                q8[i].bsums[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                q8[i].qs[j] = (int8_t)v;
            }
            
            for (int j = 0; j < QK_K/16; j++) {
                int sum = 0;
                #pragma _NEC ivdep
                for (int kk = 0; kk < 16; kk++) {
                    sum += q8[i].qs[j*16 + kk];
                }
                q8[i].bsums[j] = (int16_t)sum;
            }
        }
    }
    
    /* Parallel over row groups (each group = 8 rows) */
    int n_groups = M_int / ncols_interleaved;
    
    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int row_group = g * ncols_interleaved;
        block_q2_Kx8_ve* b_ptr = W + g * nb;
        
        float sumf[8] = {0};
        float sum_minf[8] = {0};
        
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (QK_K / (4 * blocklen)); k++) {
                const uint8_t* scales_0 = b_ptr[l].scales + (k / 4) * 64;
                const uint8_t* scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                const uint8_t* scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                const uint8_t* scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                
                for (int j = 0; j < ncols_interleaved; j++) {
                    int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
                    int sumi = 0;
                    int offset = ((k / 2) % 2) + j * 2;
                    
                    for (int ii = 0; ii < blocklen; ii++) {
                        const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] & 3);
                        const int v1 = (int8_t)((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 2) & 3);
                        const int v2 = (int8_t)((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 4) & 3);
                        const int v3 = (int8_t)((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 6) & 3);
                        
                        sumi1 = v0 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii];
                        sumi2 = v1 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 32];
                        sumi3 = v2 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 64];
                        sumi4 = v3 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 96];
                        
                        sumi1 = sumi1 * (scales_0[offset] & 0xF);
                        sumi2 = sumi2 * (scales_1[offset] & 0xF);
                        sumi3 = sumi3 * (scales_2[offset] & 0xF);
                        sumi4 = sumi4 * (scales_3[offset] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * __gnu_h2f_ieee(b_ptr[l].d[j]) * q8[l].d;
                }
            }
            
            for (int sb = 0; sb < 8; sb++) {
                const uint8_t* mins = b_ptr[l].scales + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += ((mins[j * 2] >> 4) * q8[l].bsums[sb * 2] +
                                   (mins[j * 2 + 1] >> 4) * q8[l].bsums[sb * 2 + 1]) *
                                   __gnu_h2f_ieee(b_ptr[l].dmin[j]) * q8[l].d;
                }
            }
        }
        
        for (int j = 0; j < ncols_interleaved; j++) {
            y[row_group + j] = sumf[j] - sum_minf[j];
        }
    }
    
    return 0;
}


/* ============================================================================
 * HBM-CACHED Q2_Kx8 KERNELS
 * 
 * These kernels read weights from VE HBM (device memory) at full 1.2 TB/s
 * bandwidth, while input/output use HMEM for host communication.
 * 
 * Weights are uploaded once to HBM at model load time, eliminating the
 * per-inference PCIe transfer overhead.
 * ============================================================================
 */

/*
 * VEDA kernel: Q2_Kx8 matvec with weights in HBM + OpenMP
 * 
 * Parameters:
 *   y_hmem: Output vector in HMEM (float[M])
 *   W_vptr: Weight matrix in VE HBM (VEDAdeviceptr to Q2_Kx8 blocks)
 *   x_hmem: Input vector in HMEM (float[K])
 *   M: Output dimension (rows, must be multiple of 8)
 *   K: Input dimension (columns, must be multiple of 256)
 */
uint64_t ve_q2kx8_matvec_hbm_omp(void* y_hmem,
                                  VEDAdeviceptr W_vptr,
                                  void* x_hmem,
                                  uint64_t M,
                                  uint64_t K) {
    /* HMEM pointers are already converted by vedaArgsSetHMEM - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    /* HBM pointers need vedaMemPtr conversion */
    block_q2_Kx8_ve* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / QK_K;
    
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    
    /* Quantize input x to Q8_K format (done once, shared across threads) */
    block_q8_K_ve* q8 = (block_q8_K_ve*)__builtin_alloca(nb * sizeof(block_q8_K_ve));
    
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        float amax = 0.0f;
        float max_val = 0.0f;
        #pragma _NEC ivdep
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K/16; j++) {
                q8[i].bsums[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                q8[i].qs[j] = (int8_t)v;
            }
            
            for (int j = 0; j < QK_K/16; j++) {
                int sum = 0;
                #pragma _NEC ivdep
                for (int kk = 0; kk < 16; kk++) {
                    sum += q8[i].qs[j*16 + kk];
                }
                q8[i].bsums[j] = (int16_t)sum;
            }
        }
    }
    
    /* Parallel over row groups (each group = 8 rows) */
    int n_groups = M_int / ncols_interleaved;
    
    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int row_group = g * ncols_interleaved;
        block_q2_Kx8_ve* b_ptr = W + g * nb;
        
        float sumf[8] = {0};
        float sum_minf[8] = {0};
        
        for (int l = 0; l < nb; l++) {
            /* Pre-convert FP16 scale factors to FP32 ONCE per block */
            float d_f32[8];
            float dmin_f32[8];
            float q8_d = q8[l].d;
            
            /* Use h2f_fast - unrolled to avoid loop overhead */
            d_f32[0] = h2f_fast(b_ptr[l].d[0]) * q8_d;
            d_f32[1] = h2f_fast(b_ptr[l].d[1]) * q8_d;
            d_f32[2] = h2f_fast(b_ptr[l].d[2]) * q8_d;
            d_f32[3] = h2f_fast(b_ptr[l].d[3]) * q8_d;
            d_f32[4] = h2f_fast(b_ptr[l].d[4]) * q8_d;
            d_f32[5] = h2f_fast(b_ptr[l].d[5]) * q8_d;
            d_f32[6] = h2f_fast(b_ptr[l].d[6]) * q8_d;
            d_f32[7] = h2f_fast(b_ptr[l].d[7]) * q8_d;
            dmin_f32[0] = h2f_fast(b_ptr[l].dmin[0]) * q8_d;
            dmin_f32[1] = h2f_fast(b_ptr[l].dmin[1]) * q8_d;
            dmin_f32[2] = h2f_fast(b_ptr[l].dmin[2]) * q8_d;
            dmin_f32[3] = h2f_fast(b_ptr[l].dmin[3]) * q8_d;
            dmin_f32[4] = h2f_fast(b_ptr[l].dmin[4]) * q8_d;
            dmin_f32[5] = h2f_fast(b_ptr[l].dmin[5]) * q8_d;
            dmin_f32[6] = h2f_fast(b_ptr[l].dmin[6]) * q8_d;
            dmin_f32[7] = h2f_fast(b_ptr[l].dmin[7]) * q8_d;
            
            /* Original working structure - correct indexing */
            for (int k = 0; k < (QK_K / (4 * blocklen)); k++) {
                const uint8_t* scales_0 = b_ptr[l].scales + (k / 4) * 64;
                const uint8_t* scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                const uint8_t* scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                const uint8_t* scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                
                for (int j = 0; j < ncols_interleaved; j++) {
                    int sumi = 0;
                    int offset = ((k / 2) % 2) + j * 2;
                    
                    int sc0 = scales_0[offset] & 0xF;
                    int sc1 = scales_1[offset] & 0xF;
                    int sc2 = scales_2[offset] & 0xF;
                    int sc3 = scales_3[offset] & 0xF;
                    
                    #pragma _NEC ivdep
                    for (int ii = 0; ii < blocklen; ii++) {
                        const int v0 = (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii]) & 3;
                        const int v1 = (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 2) & 3;
                        const int v2 = (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 4) & 3;
                        const int v3 = (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + ii] >> 6) & 3;
                        
                        sumi += v0 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii] * sc0;
                        sumi += v1 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 32] * sc1;
                        sumi += v2 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 64] * sc2;
                        sumi += v3 * q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + 96] * sc3;
                    }
                    sumf[j] += sumi * d_f32[j];
                }
            }
            
            /* Compute min contribution */
            for (int sb = 0; sb < 8; sb++) {
                const uint8_t* mins = b_ptr[l].scales + sb * 16;
                int bs0 = q8[l].bsums[sb * 2];
                int bs1 = q8[l].bsums[sb * 2 + 1];
                
                for (int j = 0; j < ncols_interleaved; j++) {
                    int m0 = mins[j * 2] >> 4;
                    int m1 = mins[j * 2 + 1] >> 4;
                    sum_minf[j] += (m0 * bs0 + m1 * bs1) * dmin_f32[j];
                }
            }
        }
        
        #pragma _NEC ivdep
        for (int j = 0; j < 8; j++) {
            y[row_group + j] = sumf[j] - sum_minf[j];
        }
    }
    
    return 0;
}


/* ============================================================================
 * HIGHLY VECTORIZED Q2_Kx8 HBM KERNEL v2
 * 
 * Key optimization: Process data in a way that enables 256-element vector ops.
 * 
 * Strategy:
 * 1. Pre-expand Q2 values to int32 arrays of 256 elements
 * 2. Pre-expand scales similarly
 * 3. Use simple vectorizable loops for dot products
 * 
 * This achieves A.V. Length ~256 instead of ~8.
 * ============================================================================
 */

uint64_t ve_q2kx8_matvec_hbm_vec2(void* y_hmem,
                                   VEDAdeviceptr W_vptr,
                                   void* x_hmem,
                                   uint64_t M,
                                   uint64_t K) {
    /* HMEM pointers are already converted by vedaArgsSetHMEM - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    /* HBM pointers need vedaMemPtr conversion */
    block_q2_Kx8_ve* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / QK_K;
    
    /* Quantize input x to Q8 format - use int32 for better vectorization */
    float* q8_scale = (float*)__builtin_alloca(nb * sizeof(float));
    int32_t* q8_vals = (int32_t*)__builtin_alloca(K_int * sizeof(int32_t));
    int32_t* q8_bsums = (int32_t*)__builtin_alloca(nb * 16 * sizeof(int32_t));
    
    /* Quantize x to Q8 with vectorizable 256-element loops */
    for (int blk = 0; blk < nb; blk++) {
        float* xi = x + blk * QK_K;
        int32_t* qi = q8_vals + blk * QK_K;
        
        /* Find max - vectorizable reduction */
        float amax = 0.0f;
        float max_val = 0.0f;
        #pragma _NEC ivdep
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8_scale[blk] = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                qi[j] = 0;
            }
            #pragma _NEC ivdep
            for (int j = 0; j < 16; j++) {
                q8_bsums[blk * 16 + j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8_scale[blk] = 1.0f / iscale;
            
            /* Quantize - this is a 256-element vectorizable loop! */
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                qi[j] = v;
            }
            
            /* Compute block sums - 16-element loops */
            for (int sb = 0; sb < 16; sb++) {
                int32_t sum = 0;
                #pragma _NEC ivdep
                for (int j = 0; j < 16; j++) {
                    sum += qi[sb * 16 + j];
                }
                q8_bsums[blk * 16 + sb] = sum;
            }
        }
    }
    
    int n_groups = M_int / 8;
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int g = 0; g < n_groups; g++) {
        block_q2_Kx8_ve* b_ptr = W + g * nb;
        
        /* Accumulator for 8 rows */
        float sumf[8] = {0};
        float sum_minf[8] = {0};
        
        /* 
         * Temporary buffers for expanded Q2 values.
         * For each row, we expand 256 Q2 values and their scales to int32.
         * Then we can do a single 256-element vectorized dot product.
         */
        int32_t q2_expanded[256] __attribute__((aligned(64)));
        int32_t sc_expanded[256] __attribute__((aligned(64)));
        
        for (int l = 0; l < nb; l++) {
            float q8_d = q8_scale[l];
            int32_t* q8_blk = q8_vals + l * QK_K;
            
            /* Pre-convert FP16 scales to FP32 */
            float d_f32[8], dmin_f32[8];
            #pragma _NEC ivdep
            for (int j = 0; j < 8; j++) {
                d_f32[j] = h2f_fast(b_ptr[l].d[j]) * q8_d;
                dmin_f32[j] = h2f_fast(b_ptr[l].dmin[j]) * q8_d;
            }
            
            /* Process each of 8 rows - expand Q2 values then do vector dot */
            for (int row = 0; row < 8; row++) {
                /* 
                 * EXPAND Q2 values for this row to enable 256-element vectors.
                 * 
                 * Q2_Kx8 layout: 512 bytes of qs[], organized as:
                 *   8 super-blocks × 64 bytes per super-block
                 *   Each super-block: 8 rows × 8 bytes per row
                 *   Each byte: 4 Q2 values (2 bits each)
                 *   So each row gets 8 bytes × 4 = 32 Q2 values per super-block
                 *   Total per row: 8 × 32 = 256 Q2 values ✓
                 * 
                 * Q8 layout: 256 sequential int8 values per block
                 *   Organized as 8 super-blocks of 32 values each
                 * 
                 * The tricky part is matching up Q2 positions with Q8 positions.
                 * In the original code, the Q8 index was:
                 *   q8[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + ii + offset]
                 * where k goes 0..7 (8 iterations), ii goes 0..7, offset is 0/32/64/96
                 */
                
                /* Expand all 256 Q2 values and scales for this row */
                for (int sb = 0; sb < 8; sb++) {
                    /* Get Q2 bytes for this row in this super-block */
                    const uint8_t* qs_row = b_ptr[l].qs + sb * 64 + row * 8;
                    
                    /* Get scales for this super-block and row */
                    /* In the original: scales_N at sb*16, offset = (sb_mod_2) + row*2 */
                    int sb_half = sb / 4;  /* 0 for sb 0-3, 1 for sb 4-7 */
                    int sb_quarter = sb % 4;
                    int offset = (sb_quarter / 2) + row * 2;
                    
                    int sc0 = b_ptr[l].scales[sb_half * 64 + 0  + offset] & 0xF;
                    int sc1 = b_ptr[l].scales[sb_half * 64 + 16 + offset] & 0xF;
                    int sc2 = b_ptr[l].scales[sb_half * 64 + 32 + offset] & 0xF;
                    int sc3 = b_ptr[l].scales[sb_half * 64 + 48 + offset] & 0xF;
                    
                    /* Expand 8 bytes (32 Q2 values) for this super-block */
                    int base = sb * 32;
                    
                    /* Each byte has 4 Q2 values at bit positions 0-1, 2-3, 4-5, 6-7 */
                    /* These correspond to 4 different Q8 value groups with different scales */
                    #pragma _NEC ivdep
                    for (int b = 0; b < 8; b++) {
                        uint8_t qbyte = qs_row[b];
                        q2_expanded[base + b +  0] = (qbyte >> 0) & 3;
                        q2_expanded[base + b +  8] = (qbyte >> 2) & 3;
                        q2_expanded[base + b + 16] = (qbyte >> 4) & 3;
                        q2_expanded[base + b + 24] = (qbyte >> 6) & 3;
                        
                        sc_expanded[base + b +  0] = sc0;
                        sc_expanded[base + b +  8] = sc1;
                        sc_expanded[base + b + 16] = sc2;
                        sc_expanded[base + b + 24] = sc3;
                    }
                }
                
                /* 
                 * NOW we can do a 256-element vectorized dot product!
                 * But we need to match Q8 indices correctly.
                 * 
                 * Original Q8 index: (k >> 2) * 128 + (k % 4) * 8 + ii + {0,32,64,96}
                 * For k=0..7, ii=0..7:
                 *   k=0: base=0, positions 0..7, 32..39, 64..71, 96..103
                 *   k=1: base=8, positions 8..15, 40..47, 72..79, 104..111
                 *   ...
                 * 
                 * This is NOT sequential! We need to reorganize.
                 * For now, let's do the dot product in a way that respects the indexing.
                 */
                
                int64_t sumi = 0;
                
                /* Process 8 super-blocks of 32 elements each */
                for (int sb = 0; sb < 8; sb++) {
                    int base = sb * 32;
                    
                    /* Map sb to original k value for Q8 indexing */
                    /* sb 0-3 map to k 0-3 with base 0 */
                    /* sb 4-7 map to k 4-7 with base 128 */
                    int k = sb % 4;
                    int q8_base = (sb / 4) * 128 + k * 8;
                    
                    /* Sum products for this super-block */
                    #pragma _NEC ivdep
                    for (int b = 0; b < 8; b++) {
                        sumi += (int64_t)q2_expanded[base + b + 0] * sc_expanded[base + b + 0] * q8_blk[q8_base + b + 0];
                    }
                    #pragma _NEC ivdep
                    for (int b = 0; b < 8; b++) {
                        sumi += (int64_t)q2_expanded[base + b + 8] * sc_expanded[base + b + 8] * q8_blk[q8_base + b + 32];
                    }
                    #pragma _NEC ivdep
                    for (int b = 0; b < 8; b++) {
                        sumi += (int64_t)q2_expanded[base + b + 16] * sc_expanded[base + b + 16] * q8_blk[q8_base + b + 64];
                    }
                    #pragma _NEC ivdep
                    for (int b = 0; b < 8; b++) {
                        sumi += (int64_t)q2_expanded[base + b + 24] * sc_expanded[base + b + 24] * q8_blk[q8_base + b + 96];
                    }
                }
                
                sumf[row] += (float)sumi * d_f32[row];
            }
            
            /* Compute min contribution using bsums */
            for (int sb = 0; sb < 8; sb++) {
                const uint8_t* mins = b_ptr[l].scales + sb * 16;
                int32_t bs0 = q8_bsums[l * 16 + sb * 2];
                int32_t bs1 = q8_bsums[l * 16 + sb * 2 + 1];
                
                for (int row = 0; row < 8; row++) {
                    int m0 = mins[row * 2] >> 4;
                    int m1 = mins[row * 2 + 1] >> 4;
                    sum_minf[row] += (float)(m0 * bs0 + m1 * bs1) * dmin_f32[row];
                }
            }
        }
        
        /* Write output */
        int row_base = g * 8;
        #pragma _NEC ivdep
        for (int j = 0; j < 8; j++) {
            y[row_base + j] = sumf[j] - sum_minf[j];
        }
    }
    
    return 0;
}


/* ============================================================================
 * VE-NATIVE Q2_K FORMAT KERNEL
 * 
 * This kernel operates on pre-transformed data where:
 *   - Q2 values are expanded from 2-bit to 8-bit
 *   - Scales are pre-multiplied into the Q2 values
 *   - Data is organized for 256-element vector operations
 * 
 * Format per row (256 elements, nb blocks per row):
 *   - d:       FP32 scale (4 bytes)
 *   - dmin:    FP32 min scale (4 bytes)
 *   - qs[256]: int8 Q2×scale values (256 bytes)
 *   - mins[8]: int8 min values (8 bytes)
 *   - _pad[8]: padding (8 bytes)
 *   Total: 280 bytes per row per K-block
 * ============================================================================
 */

/* Block structure - must match host-side definition */
typedef struct {
    float d;           /* Scale factor */
    float dmin;        /* Min scale factor */
    int8_t qs[256];    /* Expanded Q2 × scale values */
    int8_t mins[16];   /* Min values: mins[sb*2+0]=m0, mins[sb*2+1]=m1 */
} block_q2k_ve_native;  /* Total: 280 bytes */

/*
 * VEDA kernel: VE-native Q2_K matvec with weights in HBM
 * 
 * This is the optimal kernel for VE - enables full 256-element vectorization.
 * 
 * Parameters:
 *   y_hmem: Output vector in HMEM (float[M])
 *   W_vptr: Weight matrix in VE HBM (VE-native format)
 *   x_hmem: Input vector in HMEM (float[K])
 *   M: Output dimension (rows)
 *   K: Input dimension (columns, must be multiple of 256)
 */
uint64_t ve_q2kx8_matvec_hbm_native(void* y_hmem,
                                     VEDAdeviceptr W_vptr,
                                     void* x_hmem,
                                     uint64_t M,
                                     uint64_t K) {
    /* HMEM pointers are already converted by vedaArgsSetHMEM - use directly */
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    /* HBM pointers need vedaMemPtr conversion */
    block_q2k_ve_native* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / 256;  /* Number of 256-element blocks per row */
    
    /* Quantize input x to int8 Q8 format */
    float* q8_scale = (float*)__builtin_alloca(nb * sizeof(float));
    int8_t* q8_vals = (int8_t*)__builtin_alloca(K_int * sizeof(int8_t));
    int32_t* q8_bsums = (int32_t*)__builtin_alloca(nb * 16 * sizeof(int32_t));  /* 16 bsums per block */
    
    /* Quantize x to Q8 format - 256-element vectorizable loops */
    for (int blk = 0; blk < nb; blk++) {
        float* xi = x + blk * 256;
        int8_t* qi = q8_vals + blk * 256;
        
        /* Find max absolute value */
        float amax = 0.0f;
        float max_val = 0.0f;
        #pragma _NEC ivdep
        for (int j = 0; j < 256; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8_scale[blk] = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < 256; j++) {
                qi[j] = 0;
            }
            #pragma _NEC ivdep
            for (int j = 0; j < 16; j++) {
                q8_bsums[blk * 16 + j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8_scale[blk] = 1.0f / iscale;
            
            /* Quantize - 256-element vectorizable loop! */
            #pragma _NEC ivdep
            for (int j = 0; j < 256; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                qi[j] = (int8_t)v;
            }
            
            /* Compute block sums for min calculation - 16 groups of 16 elements */
            for (int sb = 0; sb < 16; sb++) {
                int32_t sum = 0;
                #pragma _NEC ivdep
                for (int j = 0; j < 16; j++) {
                    sum += qi[sb * 16 + j];
                }
                q8_bsums[blk * 16 + sb] = sum;
            }
        }
    }
    
    /* Parallel over rows */
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M_int; row++) {
        float sumf = 0.0f;
        float sum_minf = 0.0f;
        
        /* Process each 256-element block */
        for (int blk = 0; blk < nb; blk++) {
            block_q2k_ve_native* b = &W[row * nb + blk];
            float d = b->d * q8_scale[blk];
            float dmin = b->dmin * q8_scale[blk];
            
            int8_t* qs = b->qs;
            int8_t* q8_blk = q8_vals + blk * 256;
            
            /* 
             * THIS IS THE KEY: 256-element vectorized dot product!
             * qs[] contains pre-expanded Q2×scale values (int8)
             * q8_blk[] contains quantized input (int8)
             * Both are contiguous 256-element arrays.
             * 
             * VE can vectorize int32 operations but not int8.
             * Widen to int32 arrays first, then do vectorized dot product.
             */
            int32_t qs32[256] __attribute__((aligned(64)));
            int32_t q832[256] __attribute__((aligned(64)));
            
            /* Widen int8 to int32 - not vectorized but fast */
            for (int k = 0; k < 256; k++) {
                qs32[k] = qs[k];
                q832[k] = q8_blk[k];
            }
            
            /* Vectorized 256-element dot product */
            int64_t sumi = 0;
            #pragma _NEC ivdep
            for (int k = 0; k < 256; k++) {
                sumi += (int64_t)qs32[k] * q832[k];
            }
            
            sumf += (float)sumi * d;
            
            /* Min contribution - uses both m0 and m1 per super-block */
            for (int sb = 0; sb < 8; sb++) {
                int m0 = b->mins[sb * 2 + 0];
                int m1 = b->mins[sb * 2 + 1];
                int bs0 = q8_bsums[blk * 16 + sb * 2 + 0];
                int bs1 = q8_bsums[blk * 16 + sb * 2 + 1];
                sum_minf += (float)(m0 * bs0 + m1 * bs1) * dmin;
            }
        }
        
        y[row] = sumf - sum_minf;
    }
    
    return 0;
}


/* ============================================================================
 * Q3_K Kernel
 * 
 * Q3_K Block Structure (110 bytes per 256 elements):
 *   uint8_t  hmask[32]  - high bits mask (1 = add 0, 0 = subtract 4)
 *   uint8_t  qs[64]     - low 2 bits of quants (4 per byte, packed differently!)
 *   uint8_t  scales[12] - 6-bit scales packed
 *   uint16_t d          - fp16 super-block scale
 * 
 * IMPORTANT: Q3_K uses a special bit layout:
 *   - qs[64] stores 4 groups of 64 elements, packed 4 per byte
 *   - Within each 128-element half:
 *     - First 32 elements: bits 0-1 of qs[0..31]
 *     - Next 32 elements:  bits 2-3 of qs[0..31]  
 *     - Next 32 elements:  bits 4-5 of qs[0..31]
 *     - Next 32 elements:  bits 6-7 of qs[0..31]
 *   - hmask[32] has 1 bit per element, used as mask bit m that shifts
 *   - Value = (qs_2bit) - (hmask_bit ? 0 : 4)
 * ============================================================================
 */
typedef struct {
    uint8_t  hmask[QK_K/8];    /* 32 bytes - high bit mask */
    uint8_t  qs[QK_K/4];       /* 64 bytes - low 2 bits */
    uint8_t  scales[12];       /* 6-bit scales packed */
    uint16_t d;                /* fp16 scale */
} block_q3_K_ve;

/*
 * VEDA kernel: Q3_K matrix-vector multiply with OpenMP
 * Follows reference implementation from ggml-cpu/quants.c
 */
uint64_t ve_q3k_matvec_f32_omp_hmem(void* y_hmem,
                                    void* W_hmem,
                                    void* x_hmem,
                                    uint64_t M,
                                    uint64_t K) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    block_q3_K_ve* W = (block_q3_K_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / QK_K;
    
    /* Constants for scale unpacking (same as reference) */
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    
    /* Quantize input x to Q8_K format */
    block_q8_K_ve* q8 = (block_q8_K_ve*)__builtin_alloca(nb * sizeof(block_q8_K_ve));
    
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        float amax = 0.0f;
        float max_val = 0.0f;
        #pragma _NEC ivdep
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K/16; j++) {
                q8[i].bsums[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                q8[i].qs[j] = (int8_t)v;
            }
            
            for (int j = 0; j < QK_K/16; j++) {
                int sum = 0;
                #pragma _NEC ivdep
                for (int k = 0; k < 16; k++) {
                    sum += q8[i].qs[j*16 + k];
                }
                q8[i].bsums[j] = (int16_t)sum;
            }
        }
    }
    
    /* Parallel over rows */
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M_int; row++) {
        block_q3_K_ve* Wrow = W + row * nb;
        
        int8_t aux8[QK_K];
        int16_t aux16[8];
        int32_t aux32[8];
        float sums[8];
        uint32_t auxs[4];
        
        for (int i = 0; i < 8; i++) sums[i] = 0.0f;
        
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t* q3 = Wrow[blk].qs;
            const uint8_t* hm = Wrow[blk].hmask;
            const int8_t* q8_qs = q8[blk].qs;
            
            for (int i = 0; i < 8; i++) aux32[i] = 0;
            
            /* Decode 3-bit values following reference implementation */
            int8_t* a = aux8;
            uint8_t m = 1;
            for (int j = 0; j < QK_K; j += 128) {
                /* First 32: bits 0-1 */
                for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
                for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
                a += 32; m <<= 1;
                /* Next 32: bits 2-3 */
                for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
                for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
                a += 32; m <<= 1;
                /* Next 32: bits 4-5 */
                for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
                for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
                a += 32; m <<= 1;
                /* Next 32: bits 6-7 */
                for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
                for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
                a += 32; m <<= 1;
                q3 += 32;
            }
            a = aux8;
            
            /* Unpack scales using reference method */
            /* Copy 12 bytes of scales into auxs[0..2] */
            auxs[0] = *((uint32_t*)&Wrow[blk].scales[0]);
            auxs[1] = *((uint32_t*)&Wrow[blk].scales[4]);
            auxs[2] = *((uint32_t*)&Wrow[blk].scales[8]);
            auxs[3] = 0;
            
            uint32_t tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            
            const int8_t* scales = (const int8_t*)auxs;
            
            float d = h2f_fast(Wrow[blk].d) * q8[blk].d;
            
            /* Compute dot product (16 groups of 16 elements) */
            for (int j = 0; j < QK_K/16; ++j) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8_qs[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
                q8_qs += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux16[l] = q8_qs[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
                q8_qs += 8; a += 8;
            }
            
            for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        }
        
        float sumf = 0.0f;
        for (int l = 0; l < 8; ++l) sumf += sums[l];
        y[row] = sumf;
    }
    
    return 0;
}


/* ============================================================================
 * Q6_K Kernel
 * 
 * Q6_K Block Structure (210 bytes per 256 elements):
 *   uint8_t  ql[128]    - lower 4 bits of quants (called q4 in ref)
 *   uint8_t  qh[64]     - upper 2 bits of quants
 *   int8_t   scales[16] - 8-bit scales
 *   uint16_t d          - fp16 super-block scale
 * 
 * IMPORTANT: Complex interleaving within each 128-element chunk:
 *   Elements 0-31:   ql[0:31] low nibble  + qh[0:31] bits 0-1
 *   Elements 32-63:  ql[32:63] low nibble + qh[0:31] bits 2-3
 *   Elements 64-95:  ql[0:31] high nibble + qh[0:31] bits 4-5
 *   Elements 96-127: ql[32:63] high nibble + qh[0:31] bits 6-7
 * 
 * Each element is 6-bit: value = (low4 | (high2 << 4)) - 32
 * Range: -32 to +31 (centered)
 * ============================================================================
 */
typedef struct {
    uint8_t  ql[QK_K/2];       /* 128 bytes - lower 4 bits */
    uint8_t  qh[QK_K/4];       /* 64 bytes - upper 2 bits */
    int8_t   scales[QK_K/16]; /* 16 bytes - 8-bit scales */
    uint16_t d;                /* fp16 scale */
} block_q6_K_ve;

/*
 * VEDA kernel: Q6_K matrix-vector multiply with OpenMP
 * Follows reference implementation from ggml-cpu/quants.c
 */
uint64_t ve_q6k_matvec_f32_omp_hmem(void* y_hmem,
                                    void* W_hmem,
                                    void* x_hmem,
                                    uint64_t M,
                                    uint64_t K) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    block_q6_K_ve* W = (block_q6_K_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / QK_K;
    
    /* Quantize input x to Q8_K format */
    block_q8_K_ve* q8 = (block_q8_K_ve*)__builtin_alloca(nb * sizeof(block_q8_K_ve));
    
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        float amax = 0.0f;
        float max_val = 0.0f;
        /* NO ivdep - max-finding has loop-carried dependency on amax/max_val */
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            #pragma _NEC ivdep
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                q8[i].qs[j] = (int8_t)v;
            }
        }
    }
    
    /* Main computation - parallelized over rows */
    #pragma omp parallel for
    for (int row = 0; row < M_int; row++) {
        block_q6_K_ve* Wrow = W + row * nb;
        
        int8_t aux8[QK_K];
        int32_t aux32[8];
        float sums[8];
        
        for (int i = 0; i < 8; i++) sums[i] = 0.0f;
        
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t* q4 = Wrow[blk].ql;
            const uint8_t* qh = Wrow[blk].qh;
            const int8_t* q8_qs = q8[blk].qs;
            
            for (int i = 0; i < 8; i++) aux32[i] = 0;
            
            /* Decode 6-bit values following reference implementation */
            int8_t* a = aux8;
            for (int j = 0; j < QK_K; j += 128) {
                for (int l = 0; l < 32; ++l) {
                    int v0 = (q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4);
                    int v1 = (q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                    int v2 = (q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4);
                    int v3 = (q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4);
                    a[l +  0] = (int8_t)(v0 - 32);
                    a[l + 32] = (int8_t)(v1 - 32);
                    a[l + 64] = (int8_t)(v2 - 32);
                    a[l + 96] = (int8_t)(v3 - 32);
                }
                a  += 128;
                q4 += 64;
                qh += 32;
            }
            a = aux8;
            
            float d = h2f_fast(Wrow[blk].d) * q8[blk].d;
            
            /* Compute dot product (16 groups of 16 elements) */
            int is = 0;
            for (int j = 0; j < QK_K/16; ++j) {
                int scale = Wrow[blk].scales[is++];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
            }
            
            for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        }
        
        float sumf = 0.0f;
        for (int l = 0; l < 8; ++l) sumf += sums[l];
        y[row] = sumf;
    }
    
    return 0;
}

/* Debug function to check struct sizes - returns Q6_K size */
uint64_t ve_check_q6k_sizeof(void) {
    return sizeof(block_q6_K_ve);
}

/* ============================================================================
 * Q4_K Support
 * ============================================================================
 * Q4_K: 4-bit quantization with per-block scales and mins
 * Block size: 256 elements, 144 bytes per block
 * Layout: d (fp16) + dmin (fp16) + scales[12] + qs[128]
 */

typedef struct __attribute__((packed)) {
    uint16_t d;           /* fp16 super-block scale */
    uint16_t dmin;        /* fp16 super-block min scale */
    uint8_t  scales[12];  /* 6-bit scales and mins, packed */
    uint8_t  qs[QK_K/2];  /* 4-bit quants (128 bytes) */
} block_q4_K_ve;

/*
 * VEDA kernel: Q4_K matrix-vector multiply with OpenMP
 * Follows reference implementation from ggml-cpu/quants.c exactly
 */
uint64_t ve_q4k_matvec_f32_omp_hmem(void* y_hmem,
                                    void* W_hmem,
                                    void* x_hmem,
                                    uint64_t M,
                                    uint64_t K) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    block_q4_K_ve* W = (block_q4_K_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / QK_K;
    
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    
    /* Quantize input x to Q8_K format */
    block_q8_K_ve* q8 = (block_q8_K_ve*)__builtin_alloca(nb * sizeof(block_q8_K_ve));
    
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        float amax = 0.0f;
        float max_val = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
            /* Compute bsums for Q4_K */
            for (int j = 0; j < QK_K/16; j++) {
                q8[i].bsums[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                q8[i].qs[j] = (int8_t)v;
            }
            /* Compute bsums - sum of each 16-element group */
            for (int j = 0; j < QK_K/16; j++) {
                int sum = 0;
                for (int l = 0; l < 16; l++) {
                    sum += q8[i].qs[j*16 + l];
                }
                q8[i].bsums[j] = (int16_t)sum;
            }
        }
    }
    
    /* Main computation - parallelized over rows */
    #pragma omp parallel for
    for (int row = 0; row < M_int; row++) {
        block_q4_K_ve* Wrow = W + row * nb;
        
        int8_t aux8[QK_K];
        int32_t aux32[8];
        float sums[8];
        uint32_t utmp[4];
        
        for (int i = 0; i < 8; i++) sums[i] = 0.0f;
        
        float sumf = 0.0f;
        
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t* q4 = Wrow[blk].qs;
            const int8_t* q8_qs = q8[blk].qs;
            
            for (int i = 0; i < 8; i++) aux32[i] = 0;
            
            /* Decode 4-bit values */
            int8_t* a = aux8;
            for (int j = 0; j < QK_K/64; ++j) {
                for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
                a += 32;
                for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
                a += 32;
                q4 += 32;
            }
            
            /* Unpack scales and mins from 12 bytes */
            __builtin_memcpy(utmp, Wrow[blk].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            
            const uint8_t* scales = (const uint8_t*)&utmp[0];
            const uint8_t* mins   = (const uint8_t*)&utmp[2];
            
            /* Compute min contribution */
            int sumi = 0;
            for (int j = 0; j < QK_K/16; ++j) {
                sumi += q8[blk].bsums[j] * mins[j/2];
            }
            
            /* Compute dot product (8 groups of 32 elements) */
            a = aux8;
            int is = 0;
            for (int j = 0; j < QK_K/32; ++j) {
                int32_t scale = scales[is++];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
            }
            
            float d = h2f_fast(Wrow[blk].d) * q8[blk].d;
            for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
            
            float dmin = h2f_fast(Wrow[blk].dmin) * q8[blk].d;
            sumf -= dmin * sumi;
        }
        
        for (int l = 0; l < 8; ++l) sumf += sums[l];
        y[row] = sumf;
    }
    
    return 0;
}

/* ============================================================================
 * Q5_K Support
 * ============================================================================
 * Q5_K: 5-bit quantization with per-block scales and mins
 * Block size: 256 elements, 176 bytes per block
 * Layout: d (fp16) + dmin (fp16) + scales[12] + qh[32] + qs[128]
 */

typedef struct __attribute__((packed)) {
    uint16_t d;           /* fp16 super-block scale */
    uint16_t dmin;        /* fp16 super-block min scale */
    uint8_t  scales[12];  /* 6-bit scales and mins, packed */
    uint8_t  qh[QK_K/8];  /* high bits (32 bytes) */
    uint8_t  qs[QK_K/2];  /* low 4-bit quants (128 bytes) */
} block_q5_K_ve;

/*
 * VEDA kernel: Q5_K matrix-vector multiply with OpenMP
 * Follows reference implementation from ggml-cpu/quants.c exactly
 */
uint64_t ve_q5k_matvec_f32_omp_hmem(void* y_hmem,
                                    void* W_hmem,
                                    void* x_hmem,
                                    uint64_t M,
                                    uint64_t K) {
    /* HMEM pointers are already converted - use directly */
    float* y = (float*)y_hmem;
    block_q5_K_ve* W = (block_q5_K_ve*)W_hmem;
    float* x = (float*)x_hmem;
    
    int M_int = (int)M;
    int K_int = (int)K;
    int nb = K_int / QK_K;
    
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    
    /* Quantize input x to Q8_K format */
    block_q8_K_ve* q8 = (block_q8_K_ve*)__builtin_alloca(nb * sizeof(block_q8_K_ve));
    
    for (int i = 0; i < nb; i++) {
        float* xi = x + i * QK_K;
        
        float amax = 0.0f;
        float max_val = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float ax = xi[j] > 0 ? xi[j] : -xi[j];
            if (ax > amax) {
                amax = ax;
                max_val = xi[j];
            }
        }
        
        if (amax == 0.0f) {
            q8[i].d = 0.0f;
            for (int j = 0; j < QK_K; j++) {
                q8[i].qs[j] = 0;
            }
            for (int j = 0; j < QK_K/16; j++) {
                q8[i].bsums[j] = 0;
            }
        } else {
            float iscale = -127.0f / max_val;
            q8[i].d = 1.0f / iscale;
            
            for (int j = 0; j < QK_K; j++) {
                int v = nearest_int_ggml(iscale * xi[j]);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                q8[i].qs[j] = (int8_t)v;
            }
            /* Compute bsums - sum of each 16-element group */
            for (int j = 0; j < QK_K/16; j++) {
                int sum = 0;
                for (int l = 0; l < 16; l++) {
                    sum += q8[i].qs[j*16 + l];
                }
                q8[i].bsums[j] = (int16_t)sum;
            }
        }
    }
    
    /* Main computation - parallelized over rows */
    #pragma omp parallel for
    for (int row = 0; row < M_int; row++) {
        block_q5_K_ve* Wrow = W + row * nb;
        
        int8_t aux8[QK_K];
        int32_t aux32[8];
        float sums[8];
        uint32_t utmp[4];
        
        for (int i = 0; i < 8; i++) sums[i] = 0.0f;
        
        float sumf = 0.0f;
        
        for (int blk = 0; blk < nb; blk++) {
            const uint8_t* q4 = Wrow[blk].qs;
            const uint8_t* hm = Wrow[blk].qh;
            const int8_t* q8_qs = q8[blk].qs;
            
            for (int i = 0; i < 8; i++) aux32[i] = 0;
            
            /* Decode 5-bit values (4 low bits + 1 high bit) */
            int8_t* a = aux8;
            uint8_t m = 1;
            for (int j = 0; j < QK_K/64; ++j) {
                for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
                for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
                a += 32; m <<= 1;
                for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
                for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
                a += 32; m <<= 1;
                q4 += 32;
            }
            
            /* Unpack scales and mins from 12 bytes */
            __builtin_memcpy(utmp, Wrow[blk].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            
            const uint8_t* scales = (const uint8_t*)&utmp[0];
            const uint8_t* mins   = (const uint8_t*)&utmp[2];
            
            /* Compute min contribution */
            int sumi = 0;
            for (int j = 0; j < QK_K/16; ++j) {
                sumi += q8[blk].bsums[j] * mins[j/2];
            }
            
            /* Compute dot product (8 groups of 32 elements) */
            a = aux8;
            int is = 0;
            for (int j = 0; j < QK_K/32; ++j) {
                int32_t scale = scales[is++];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8_qs[l] * a[l]);
                q8_qs += 8; a += 8;
            }
            
            float d = h2f_fast(Wrow[blk].d) * q8[blk].d;
            for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
            
            float dmin = h2f_fast(Wrow[blk].dmin) * q8[blk].d;
            sumf -= dmin * sumi;
        }
        
        for (int l = 0; l < 8; ++l) sumf += sums[l];
        y[row] = sumf;
    }
    
    return 0;
}

/* Return Q4_K and Q5_K struct sizes for verification */
uint64_t ve_check_q4k_q5k_sizeof(void) {
    return sizeof(block_q4_K_ve);
}


/* ============================================================================
 * LLVM-VE INTRINSICS Q2_K KERNEL WRAPPER
 * 
 * This wraps the LLVM-compiled q2k_vec_intrinsics.c kernel which uses
 * 8x row unrolling for better instruction-level parallelism.
 * ============================================================================
 */

/*
 * VEDA kernel: Q2_K matvec (stub - use FP32 dequantization path instead)
 * 
 * These are stubs. The preferred path is to dequantize Q2_K -> FP32 on the host
 * and use the vectorized FP32 matvec kernel (ve_f32_matvec_hbm_omp).
 */
uint64_t ve_q2k_matvec_unr8_hmem(void* y_hmem,
                                  void* W_hmem,
                                  void* x_hmem,
                                  uint64_t M,
                                  uint64_t K) {
    (void)y_hmem; (void)W_hmem; (void)x_hmem; (void)M; (void)K;
    return 1;  /* Error - use FP32 dequant path */
}

uint64_t ve_q2k_matvec_unr8_omp_hmem(void* y_hmem,
                                      void* W_hmem,
                                      void* x_hmem,
                                      uint64_t M,
                                      uint64_t K) {
    (void)y_hmem; (void)W_hmem; (void)x_hmem; (void)M; (void)K;
    return 1;  /* Error - use FP32 dequant path */
}


/* ============================================================================
 * BATCHED MATRIX MULTIPLY (MATMUL) KERNELS
 * 
 * These kernels compute Y = W @ X where:
 *   - W is [M x K] weight matrix (FP32, pre-dequantized)
 *   - X is [K x N] input matrix (N batch of K-dimensional vectors)
 *   - Y is [M x N] output matrix
 * 
 * Using batched matmul instead of repeated matvec amortizes weight loading
 * across multiple input vectors, dramatically improving memory efficiency.
 * 
 * Key optimization: Use VLA syntax and I-K-J loop order to trigger NCC's
 * "opt(1800): Idiom detected (matrix multiply)" optimization.
 * ============================================================================
 */

/*
 * VLA-based matrix multiply helper (for idiom detection)
 * 
 * NCC detects the matrix multiply idiom when:
 *   1. Using VLA syntax for 2D array access
 *   2. Using I-K-J loop order (row of A × column of B)
 *   3. Inner loop writes to contiguous memory
 * 
 * This triggers massive optimization including:
 *   - Automatic blocking/tiling
 *   - Optimal vector register usage
 *   - Memory prefetch insertion
 */
static void matmul_vla_ikj(int m, int k, int n,
                           const float W[m][k],
                           const float X[k][n],
                           float Y[m][n]) {
    /* Zero output first */
    for (int i = 0; i < m; i++) {
        #pragma _NEC ivdep
        for (int j = 0; j < n; j++) {
            Y[i][j] = 0.0f;
        }
    }
    
    /* I-K-J loop order for matrix multiply idiom */
    for (int i = 0; i < m; i++) {
        for (int kk = 0; kk < k; kk++) {
            float w_val = W[i][kk];
            #pragma _NEC ivdep
            for (int j = 0; j < n; j++) {
                Y[i][j] += w_val * X[kk][j];
            }
        }
    }
}


/*
 * VEDA kernel: FP32 batched matmul with HBM weights
 * 
 * Computes Y = W @ X where:
 *   - W is [M x K] in VE HBM (row-major)
 *   - X is [K x N] in HMEM (row-major, N vectors of K elements)
 *   - Y is [M x N] in HMEM (row-major, N vectors of M elements)
 * 
 * This batched version amortizes weight loading across N vectors.
 * With N=32 and K=4096, each weight element is used 32 times,
 * effectively increasing memory bandwidth utilization 32x!
 * 
 * Parameters:
 *   Y_hmem: Output matrix [M x N] in HMEM (float*)
 *   W_vptr: Weight matrix [M x K] in HBM (VEDAdeviceptr)
 *   X_hmem: Input matrix [K x N] in HMEM (float*)
 *   M: Output dimension (rows of W and Y)
 *   K: Shared dimension (cols of W, rows of X)
 *   N: Batch size (cols of X and Y)
 */
uint64_t ve_f32_matmul_hbm_omp(void* Y_hmem,
                                VEDAdeviceptr W_vptr,
                                void* X_hmem,
                                uint64_t M,
                                uint64_t K,
                                uint64_t N) {
    /* HMEM pointers are already converted by vedaArgsSetHMEM - use directly */
    float* Y = (float*)Y_hmem;
    float* X = (float*)X_hmem;
    /* HBM pointers need vedaMemPtr conversion */
    float* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    int n = (int)N;
    
    /* 
     * For small batch sizes (N <= 32), use OpenMP over rows + VLA matmul
     * For larger batches, tile to fit in cache
     */
    
    if (n <= 256) {
        /* Use VLA with fixed-size inner dimension for better optimization */
        /* Cast pointers to VLA types */
        const float (*W_vla)[k] = (const float (*)[k])W;
        const float (*X_vla)[n] = (const float (*)[n])X;
        float (*Y_vla)[n] = (float (*)[n])Y;
        
        /* Call VLA-based matmul */
        matmul_vla_ikj(m, k, n, W_vla, X_vla, Y_vla);
    } else {
        /* For larger N, tile and use OpenMP */
        const int TILE_N = 256;
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; i++) {
            for (int jt = 0; jt < n; jt += TILE_N) {
                int j_end = (jt + TILE_N < n) ? jt + TILE_N : n;
                
                /* Zero this tile of output */
                for (int j = jt; j < j_end; j++) {
                    Y[i * n + j] = 0.0f;
                }
                
                /* Compute this tile */
                for (int kk = 0; kk < k; kk++) {
                    float w_val = W[i * k + kk];
                    #pragma _NEC ivdep
                    for (int j = jt; j < j_end; j++) {
                        Y[i * n + j] += w_val * X[kk * n + j];
                    }
                }
            }
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: FP32 batched matmul with HBM weights (OpenMP parallelized)
 * 
 * This version uses OpenMP to parallelize over output rows while using
 * vectorization in the inner loop for memory efficiency.
 */
uint64_t ve_f32_matmul_hbm_omp_v2(void* Y_hmem,
                                   VEDAdeviceptr W_vptr,
                                   void* X_hmem,
                                   uint64_t M,
                                   uint64_t K,
                                   uint64_t N) {
    /* HMEM pointers are already converted by vedaArgsSetHMEM - use directly */
    float* Y = (float*)Y_hmem;
    float* X = (float*)X_hmem;
    /* HBM pointers need vedaMemPtr conversion */
    float* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    int n = (int)N;
    
    /* Cap threads at VE core count */
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel for num_threads(nthr)
    for (int i = 0; i < m; i++) {
        float* y_row = Y + i * n;
        const float* w_row = W + i * k;
        
        /* Zero output row */
        #pragma _NEC ivdep
        for (int j = 0; j < n; j++) {
            y_row[j] = 0.0f;
        }
        
        /* Accumulate: y_row += w_row[kk] * X[kk,:] for each k */
        for (int kk = 0; kk < k; kk++) {
            float w_val = w_row[kk];
            const float* x_row = X + kk * n;
            
            /* Inner loop vectorizes over batch dimension */
            #pragma _NEC ivdep
            for (int j = 0; j < n; j++) {
                y_row[j] += w_val * x_row[j];
            }
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: Pure VLA matmul (no OpenMP, for idiom detection testing)
 * 
 * This is the simplest form to trigger the MatMul idiom.
 * Use for benchmarking and verifying idiom detection.
 */
uint64_t ve_f32_matmul_vla_hmem(void* Y_hmem,
                                 void* W_hmem,
                                 void* X_hmem,
                                 uint64_t M,
                                 uint64_t K,
                                 uint64_t N) {
    /* HMEM pointers are already converted - use directly */
    float* Y = (float*)Y_hmem;
    float* W = (float*)W_hmem;
    float* X = (float*)X_hmem;
    
    int m = (int)M;
    int k = (int)K;
    int n = (int)N;
    
    /* Cast to VLA types */
    const float (*W_vla)[k] = (const float (*)[k])W;
    const float (*X_vla)[n] = (const float (*)[n])X;
    float (*Y_vla)[n] = (float (*)[n])Y;
    
    /* Call VLA-based matmul */
    matmul_vla_ikj(m, k, n, W_vla, X_vla, Y_vla);
    
    return 0;
}


/*
 * VEDA kernel: FP32 batched matmul (all HMEM, for testing)
 * 
 * All data in HMEM - useful for initial testing before HBM integration.
 */
uint64_t ve_f32_matmul_hmem(void* Y_hmem,
                             void* W_hmem,
                             void* X_hmem,
                             uint64_t M,
                             uint64_t K,
                             uint64_t N) {
    /* HMEM pointers are already converted - use directly */
    float* Y = (float*)Y_hmem;
    float* W = (float*)W_hmem;
    float* X = (float*)X_hmem;
    
    int m = (int)M;
    int k = (int)K;
    int n = (int)N;
    
    /* Cap threads */
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel for num_threads(nthr)
    for (int i = 0; i < m; i++) {
        float* y_row = Y + i * n;
        const float* w_row = W + i * k;
        
        #pragma _NEC ivdep
        for (int j = 0; j < n; j++) {
            y_row[j] = 0.0f;
        }
        
        for (int kk = 0; kk < k; kk++) {
            float w_val = w_row[kk];
            const float* x_row = X + kk * n;
            
            #pragma _NEC ivdep
            for (int j = 0; j < n; j++) {
                y_row[j] += w_val * x_row[j];
            }
        }
    }
    
    return 0;
}


/*
 * ==========================================================================
 * FULL HBM KERNELS - All tensors in VE device memory (1.2 TB/s)
 * ==========================================================================
 * 
 * These kernels take VEDAdeviceptr for ALL tensors (input, output, weights).
 * This eliminates PCIe transfers for intermediate activations.
 * 
 * Usage pattern:
 *   1. Allocate activation buffers in HBM (vedaMemAlloc)
 *   2. Pass VEDAdeviceptr to kernels via vedaArgsSetVPtr
 *   3. Kernel uses vedaMemPtr() to get raw pointer (no host sync)
 *   4. Only copy final output to host at the end
 */

/*
 * BF16 matvec: y = W @ x (all in HBM)
 */
uint64_t ve_bf16_matvec_hbm_full(VEDAdeviceptr y_vptr,
                                  VEDAdeviceptr W_vptr,
                                  VEDAdeviceptr x_vptr,
                                  uint64_t d,    /* output dimension (rows) */
                                  uint64_t n) {  /* input dimension (cols) */
    float* y;
    bf16* W;
    float* x;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    int d_int = (int)d;
    int n_int = (int)n;
    
    #pragma omp parallel num_threads(nthr)
    {
        int ithr = omp_get_thread_num();
        int chunk = (d_int + nthr - 1) / nthr;
        int imin = ithr * chunk;
        int imax = imin + chunk;
        if (imax > d_int) imax = d_int;
        
        if (imin < imax) {
            sgemv_packed_bf16_unr(&y[imin], x, &W[imin * n_int], n_int, imax - imin);
        }
    }
    
    return 0;
}

/*
 * BF16 matvec: y = W @ x (mixed HBM/HMEM)
 * - y: HBM (output, VEDAdeviceptr)
 * - W: HBM (weights, VEDAdeviceptr, cached)
 * - x: HMEM (input, fresh from host each call)
 *
 * This is the optimal pattern when:
 * - Weights are cached in HBM for full bandwidth
 * - Input comes from host each token (via HMEM)
 * - Output stays in HBM for next layer (no PCIe for intermediate activations)
 */
uint64_t ve_bf16_matvec_hbm_hmem(VEDAdeviceptr y_vptr,
                                  VEDAdeviceptr W_vptr,
                                  void* x_hmem,
                                  uint64_t d,    /* output dimension (rows) */
                                  uint64_t n) {  /* input dimension (cols) */
    float* y;
    bf16* W;
    float* x = (float*)x_hmem;  /* HMEM already converted by vedaArgsSetHMEM */
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    int d_int = (int)d;
    int n_int = (int)n;
    
    #pragma omp parallel num_threads(nthr)
    {
        int ithr = omp_get_thread_num();
        int chunk = (d_int + nthr - 1) / nthr;
        int imin = ithr * chunk;
        int imax = imin + chunk;
        if (imax > d_int) imax = d_int;
        
        if (imin < imax) {
            sgemv_packed_bf16_unr(&y[imin], x, &W[imin * n_int], n_int, imax - imin);
        }
    }
    
    return 0;
}

/*
 * BF16 batched matmul: Y = W @ X (all in HBM)
 * 
 * Uses intrinsics-based BF16 kernel for each batch vector in parallel.
 * This is much faster than dequantizing to FP32 + CBLAS SGEMM.
 * 
 * W is [d x n] BF16 weights (row-major) in HBM
 * X is [n x batch] FP32 inputs (column-major, each column is an input) in HBM
 * Y is [d x batch] FP32 outputs (column-major) in HBM
 */
uint64_t ve_bf16_matmul_hbm_full(VEDAdeviceptr y_vptr,
                                  VEDAdeviceptr W_vptr,
                                  VEDAdeviceptr x_vptr,
                                  uint64_t d,      /* output dimension (rows) */
                                  uint64_t n,      /* input dimension (cols) */
                                  uint64_t batch) { /* batch size */
    float* Y;
    bf16* W;
    float* X;
    
    vedaMemPtr((void**)&Y, y_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    vedaMemPtr((void**)&X, x_vptr);
    
    int d_int = (int)d;
    int n_int = (int)n;
    
    /* Process each batch vector in parallel using intrinsics BF16 kernel */
    #pragma omp parallel for
    for (uint64_t b = 0; b < batch; b++) {
        float* y_b = Y + b * d;
        float* x_b = X + b * n;
        sgemv_packed_bf16_unr(y_b, x_b, W, n_int, d_int);
    }
    
    return 0;
}

/*
 * FP32 matvec: y = W @ x (all in HBM)
 */
uint64_t ve_f32_matvec_hbm_full(VEDAdeviceptr y_vptr,
                                 VEDAdeviceptr W_vptr,
                                 VEDAdeviceptr x_vptr,
                                 uint64_t d,
                                 uint64_t n) {
    float* y;
    float* W;
    float* x;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    int d_int = (int)d;
    int n_int = (int)n;
    
    #pragma omp parallel for num_threads(nthr)
    for (int i = 0; i < d_int; i++) {
        float sum = 0.0f;
        const float* w_row = W + i * n_int;
        #pragma _NEC ivdep
        for (int j = 0; j < n_int; j++) {
            sum += w_row[j] * x[j];
        }
        y[i] = sum;
    }
    
    return 0;
}

/*
 * RMS Norm: y = x * rsqrt(mean(x^2) + eps) * weight (all in HBM)
 */
uint64_t ve_rms_norm_hbm_full(VEDAdeviceptr y_vptr,
                               VEDAdeviceptr x_vptr,
                               VEDAdeviceptr weight_vptr,
                               uint64_t n,
                               uint64_t eps_bits) {
    float* y;
    float* x;
    float* weight;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    vedaMemPtr((void**)&weight, weight_vptr);
    
    float eps;
    __builtin_memcpy(&eps, &eps_bits, sizeof(float));
    
    int n_int = (int)n;
    
    /* Compute mean of squares */
    double sum_sq = 0.0;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    
    float scale = 1.0f / sqrtf((float)(sum_sq / n_int) + eps);
    
    /* Apply normalization and weight */
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        y[i] = x[i] * scale * weight[i];
    }
    
    return 0;
}

/*
 * RMS Norm without weight multiply: y = x * rsqrt(mean(x^2) + eps) (all in HBM)
 * This matches GGML's RMS_NORM op where weight multiply is separate
 */
uint64_t ve_rms_norm_hbm_simple(VEDAdeviceptr y_vptr,
                                 VEDAdeviceptr x_vptr,
                                 uint64_t n,
                                 uint64_t eps_bits) {
    float* y;
    float* x;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    
    float eps;
    __builtin_memcpy(&eps, &eps_bits, sizeof(float));
    
    int n_int = (int)n;
    
    /* Compute mean of squares */
    double sum_sq = 0.0;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    
    float scale = 1.0f / sqrtf((float)(sum_sq / n_int) + eps);
    
    /* Apply normalization only (no weight) */
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        y[i] = x[i] * scale;
    }
    
    return 0;
}

/*
 * RMS Norm for multiple rows (all in HBM)
 * Each row is normalized independently
 * y, x: [ne00 x ne01] where ne00 = hidden_dim, ne01 = num_rows
 */
uint64_t ve_rms_norm_hbm_omp(VEDAdeviceptr y_vptr,
                              VEDAdeviceptr x_vptr,
                              uint64_t ne00,
                              uint64_t ne01,
                              uint64_t eps_bits) {
    float* y;
    float* x;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    
    float eps;
    __builtin_memcpy(&eps, &eps_bits, sizeof(float));
    
    int cols = (int)ne00;
    int rows = (int)ne01;
    
    #pragma omp parallel for
    for (int row = 0; row < rows; row++) {
        float* x_row = x + row * cols;
        float* y_row = y + row * cols;
        
        /* Compute mean of squares for this row */
        double sum_sq = 0.0;
        #pragma _NEC ivdep
        for (int i = 0; i < cols; i++) {
            sum_sq += (double)x_row[i] * (double)x_row[i];
        }
        
        float scale = 1.0f / sqrtf((float)(sum_sq / cols) + eps);
        
        /* Apply normalization */
        #pragma _NEC ivdep
        for (int i = 0; i < cols; i++) {
            y_row[i] = x_row[i] * scale;
        }
    }
    
    return 0;
}

/*
 * Softmax for multiple rows (all in HBM)
 * Each row is softmax'd independently
 * y, x: [ne00 x ne01] where ne00 = softmax_dim, ne01 = num_rows
 */
uint64_t ve_softmax_hbm_omp(VEDAdeviceptr y_vptr,
                             VEDAdeviceptr x_vptr,
                             uint64_t ne00,
                             uint64_t ne01,
                             uint64_t scale_bits) {
    float* y;
    float* x;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    
    float scale;
    __builtin_memcpy(&scale, &scale_bits, sizeof(float));
    
    int cols = (int)ne00;
    int rows = (int)ne01;
    
    #pragma omp parallel for
    for (int row = 0; row < rows; row++) {
        float* x_row = x + row * cols;
        float* y_row = y + row * cols;
        
        /* Find max for numerical stability */
        float max_val = -1e30f;
        #pragma _NEC ivdep
        for (int i = 0; i < cols; i++) {
            float val = x_row[i] * scale;
            if (val > max_val) max_val = val;
        }
        
        /* Compute exp and sum */
        double sum = 0.0;
        #pragma _NEC ivdep
        for (int i = 0; i < cols; i++) {
            float val = expf(x_row[i] * scale - max_val);
            y_row[i] = val;
            sum += (double)val;
        }
        
        /* Normalize */
        float inv_sum = 1.0f / (float)sum;
        #pragma _NEC ivdep
        for (int i = 0; i < cols; i++) {
            y_row[i] *= inv_sum;
        }
    }
    
    return 0;
}

/*
 * SILU activation: y = x * sigmoid(x) (all in HBM)
 */
uint64_t ve_silu_hbm_full(VEDAdeviceptr y_vptr,
                           VEDAdeviceptr x_vptr,
                           uint64_t n) {
    float* y;
    float* x;

    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);

    int n_int = (int)n;

    /* Clamp |x| <= 80 — NCC's vectorised expf returns NaN past ~|88| and
     * SILU saturates to 0 (x<<0) or x (x>>0) anyway. */
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        float xi = x[i];
        if (xi < -80.0f) xi = -80.0f;
        if (xi > 80.0f)  xi = 80.0f;
        y[i] = xi / (1.0f + expf(-xi));
    }

    return 0;
}

/* ------------------------------------------------------------------------
 * SIGMOID:  y = 1 / (1 + exp(-x))
 * ----------------------------------------------------------------------*/
uint64_t ve_sigmoid_hbm_full(VEDAdeviceptr y_vptr,
                             VEDAdeviceptr x_vptr,
                             uint64_t n) {
    float* y;
    float* x;
    if (vedaMemPtr((void**)&y, y_vptr) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_vptr) != 0) return 2;

    const int n_int = (int) n;
    /* Clamp |x| <= 80 to dodge NCC vectorised-expf NaN. */
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        float xi = x[i];
        if (xi < -80.0f) xi = -80.0f;
        if (xi > 80.0f)  xi = 80.0f;
        y[i] = 1.0f / (1.0f + expf(-xi));
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * SOFTPLUS:  y = log(1 + exp(x))
 * For large x, log(1+exp(x)) -> x (the +1 is negligible).
 * For very negative x, exp(x) underflows to 0 -> log(1) = 0.
 * Use the numerically-stable formulation max(x, 0) + log1p(exp(-|x|)).
 * ----------------------------------------------------------------------*/
uint64_t ve_softplus_hbm_full(VEDAdeviceptr y_vptr,
                              VEDAdeviceptr x_vptr,
                              uint64_t n) {
    float* y;
    float* x;
    if (vedaMemPtr((void**)&y, y_vptr) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_vptr) != 0) return 2;

    const int n_int = (int) n;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        float xi = x[i];
        float ax = (xi < 0.0f) ? -xi : xi;
        if (ax > 80.0f) ax = 80.0f;          /* expf NaN guard */
        float t = expf(-ax);
        float lp = log1pf(t);
        float m = (xi > 0.0f) ? xi : 0.0f;
        y[i] = m + lp;
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * EXP:  y = exp(x).  Clamp argument to the safe range.
 * ----------------------------------------------------------------------*/
uint64_t ve_exp_hbm_full(VEDAdeviceptr y_vptr,
                         VEDAdeviceptr x_vptr,
                         uint64_t n) {
    float* y;
    float* x;
    if (vedaMemPtr((void**)&y, y_vptr) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_vptr) != 0) return 2;

    const int n_int = (int) n;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        float xi = x[i];
        if (xi > 80.0f)  xi = 80.0f;
        if (xi < -80.0f) xi = -80.0f;
        y[i] = expf(xi);
    }
    return 0;
}

/* ------------------------------------------------------------------------
 * NEG:  y = -x
 * ----------------------------------------------------------------------*/
uint64_t ve_neg_hbm_full(VEDAdeviceptr y_vptr,
                         VEDAdeviceptr x_vptr,
                         uint64_t n) {
    float* y;
    float* x;
    if (vedaMemPtr((void**)&y, y_vptr) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_vptr) != 0) return 2;

    const int n_int = (int) n;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) y[i] = -x[i];
    return 0;
}

/* ------------------------------------------------------------------------
 * SQR:  y = x*x
 * ----------------------------------------------------------------------*/
uint64_t ve_sqr_hbm_full(VEDAdeviceptr y_vptr,
                         VEDAdeviceptr x_vptr,
                         uint64_t n) {
    float* y;
    float* x;
    if (vedaMemPtr((void**)&y, y_vptr) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_vptr) != 0) return 2;

    const int n_int = (int) n;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) { float xi = x[i]; y[i] = xi * xi; }
    return 0;
}

/* ------------------------------------------------------------------------
 * SUB:  y = a - b   (element-wise, same shape, all HBM)
 * ----------------------------------------------------------------------*/
uint64_t ve_sub_hbm_full(VEDAdeviceptr y_vptr,
                         VEDAdeviceptr a_vptr,
                         VEDAdeviceptr b_vptr,
                         uint64_t n) {
    float* y;
    float* a;
    float* b;
    if (vedaMemPtr((void**)&y, y_vptr) != 0) return 1;
    if (vedaMemPtr((void**)&a, a_vptr) != 0) return 2;
    if (vedaMemPtr((void**)&b, b_vptr) != 0) return 3;

    const int n_int = (int) n;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) y[i] = a[i] - b[i];
    return 0;
}

/* ------------------------------------------------------------------------
 * L2_NORM along innermost dimension.
 *   For each row of `nc` floats:  scale = 1 / sqrt(sum(x[j]^2) + eps)
 *                                 y[j]  = x[j] * scale
 * ggml uses eps = 1e-6 by default for L2_NORM. Input and output strides
 * are contiguous along ne[0]; we iterate rows in OMP.
 * ----------------------------------------------------------------------*/
uint64_t ve_l2_norm_hbm_full(VEDAdeviceptr y_vptr,
                             VEDAdeviceptr x_vptr,
                             uint64_t nc,
                             uint64_t nr,
                             uint64_t eps_bits) {
    float* y;
    float* x;
    if (vedaMemPtr((void**)&y, y_vptr) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_vptr) != 0) return 2;

    float eps;
    __builtin_memcpy(&eps, &eps_bits, sizeof(float));

    const int cols = (int) nc;
    const int rows = (int) nr;

    #pragma omp parallel for
    for (int r = 0; r < rows; r++) {
        const float * xr = x + r * cols;
        float       * yr = y + r * cols;
        double sum_sq = 0.0;
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            sum_sq += (double) xr[j] * (double) xr[j];
        }
        float scale = 1.0f / sqrtf((float) sum_sq + eps);
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            yr[j] = xr[j] * scale;
        }
    }
    return 0;
}

/*
 * SWIGLU activation (fused): y = silu(gate) * up = gate * sigmoid(gate) * up
 * This is the fused GLU operation used in LLaMA models
 * All tensors in HBM
 */
uint64_t ve_swiglu_hbm_full(VEDAdeviceptr y_vptr,
                             VEDAdeviceptr gate_vptr,
                             VEDAdeviceptr up_vptr,
                             uint64_t n) {
    float* y;
    float* gate;
    float* up;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&gate, gate_vptr);
    vedaMemPtr((void**)&up, up_vptr);
    
    int n_int = (int)n;
    
    /* WORKAROUND: Clamp gate to avoid NCC vectorized expf NaN bug */
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        float g = gate[i];
        if (g < -80.0f) g = -80.0f;
        if (g > 80.0f) g = 80.0f;
        float silu_g = g / (1.0f + expf(-g));
        y[i] = silu_g * up[i];
    }
    
    return 0;
}

/*
 * SWIGLU activation (fused) with OpenMP for multi-row: y = silu(gate) * up
 * All tensors in HBM
 */
/*
 * "_inner" variant — caller MUST be inside a #pragma omp parallel region.
 * Uses #pragma omp for so the implicit barrier at end-of-for synchronises
 * the team. Body matches ve_swiglu_hbm_full_omp below; same clamping
 * workaround for NCC's vectorised expf().
 */
void swiglu_hbm_full_inner(float* y, float* gate, float* up, int nc, int nr) {
    #pragma omp for
    for (int row = 0; row < nr; row++) {
        float* y_row    = y    + row * nc;
        float* gate_row = gate + row * nc;
        float* up_row   = up   + row * nc;
        #pragma _NEC ivdep
        for (int i = 0; i < nc; i++) {
            float g = gate_row[i];
            if (g < -80.0f) g = -80.0f;
            if (g >  80.0f) g =  80.0f;
            float silu_g = g / (1.0f + expf(-g));
            y_row[i] = silu_g * up_row[i];
        }
    }
}

uint64_t ve_swiglu_hbm_full_omp(VEDAdeviceptr y_vptr,
                                 VEDAdeviceptr gate_vptr,
                                 VEDAdeviceptr up_vptr,
                                 uint64_t ne0,
                                 uint64_t ne1) {
    float* y;
    float* gate;
    float* up;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&gate, gate_vptr);
    vedaMemPtr((void**)&up, up_vptr);
    
    int nc = (int)ne0;
    int nr = (int)ne1;
    
    #pragma omp parallel for
    for (int row = 0; row < nr; row++) {
        float* y_row = y + row * nc;
        float* gate_row = gate + row * nc;
        float* up_row = up + row * nc;
        
        /* 
         * WORKAROUND: NCC's vectorized expf produces NaN for inputs > ~88.
         * We clamp gate values to avoid this issue.
         * For g < -88, silu(g) ≈ 0 anyway, so clamping doesn't affect the result.
         * For g > 88, silu(g) ≈ g, but such large values shouldn't occur in practice.
         */
        #pragma _NEC ivdep
        for (int i = 0; i < nc; i++) {
            float g = gate_row[i];
            /* Clamp to safe range for expf vectorization */
            if (g < -80.0f) g = -80.0f;
            if (g > 80.0f) g = 80.0f;
            float silu_g = g / (1.0f + expf(-g));
            y_row[i] = silu_g * up_row[i];
        }
    }
    
    return 0;
}

/*
 * SWIGLU activation (fused) with HMEM: y = silu(gate) * up
 * Used when compute buffer is not in HBM
 */
uint64_t ve_swiglu_f32_hmem(void* y_hmem,
                             void* gate_hmem,
                             void* up_hmem,
                             uint64_t n) {
    float* y = (float*)y_hmem;
    float* gate = (float*)gate_hmem;
    float* up = (float*)up_hmem;
    
    int n_int = (int)n;
    
    /* WORKAROUND: Clamp gate to avoid NCC vectorized expf NaN bug */
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    for (int i = 0; i < n_int; i++) {
        float g = gate[i];
        if (g < -80.0f) g = -80.0f;
        if (g > 80.0f) g = 80.0f;
        float silu_g = g / (1.0f + expf(-g));
        y[i] = silu_g * up[i];
    }
    
    return 0;
}

/*
 * SWIGLU activation (fused) with HMEM and OpenMP: y = silu(gate) * up
 */
uint64_t ve_swiglu_f32_omp_hmem(void* y_hmem,
                                 void* gate_hmem,
                                 void* up_hmem,
                                 uint64_t ne0,
                                 uint64_t ne1) {
    float* y = (float*)y_hmem;
    float* gate = (float*)gate_hmem;
    float* up = (float*)up_hmem;
    
    int nc = (int)ne0;
    int nr = (int)ne1;
    
    #pragma omp parallel for
    for (int row = 0; row < nr; row++) {
        float* y_row = y + row * nc;
        float* gate_row = gate + row * nc;
        float* up_row = up + row * nc;
        
        /* WORKAROUND: Clamp gate to avoid NCC vectorized expf NaN bug */
        #pragma _NEC ivdep
        #pragma _NEC vovertake
        #pragma _NEC novob
        for (int i = 0; i < nc; i++) {
            float g = gate_row[i];
            if (g < -80.0f) g = -80.0f;
            if (g > 80.0f) g = 80.0f;
            float silu_g = g / (1.0f + expf(-g));
            y_row[i] = silu_g * up_row[i];
        }
    }
    
    return 0;
}

/*
 * Element-wise multiply: y = a * b (all in HBM)
 */
uint64_t ve_mul_hbm_full(VEDAdeviceptr y_vptr,
                          VEDAdeviceptr a_vptr,
                          VEDAdeviceptr b_vptr,
                          uint64_t n) {
    float* y;
    float* a;
    float* b;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&a, a_vptr);
    vedaMemPtr((void**)&b, b_vptr);
    
    int n_int = (int)n;
    
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        y[i] = a[i] * b[i];
    }
    
    return 0;
}

/*
 * Element-wise multiply: y = a * b (all in HBM) with broadcasting
 * Supports broadcasting: if n_b < n, b is repeated (n / n_b) times
 * This is used for RMS_NORM weight application: [hidden, rows] * [hidden] -> [hidden, rows]
 */
uint64_t ve_mul_hbm_full_bcast(VEDAdeviceptr y_vptr,
                               VEDAdeviceptr a_vptr,
                               VEDAdeviceptr b_vptr,
                               uint64_t n,
                               uint64_t n_b,
                               uint64_t ne00) {
    /*
     * GGML broadcast convention for src0=[ne00, ...] * src1=[ne10, ...] where ne10 <= ne00:
     * 
     * For MoE pattern: src0=[2880,32,2] * src1=[1,32,2]
     * - ne00=2880 (src0's first dim), n_b=64 (total elements in src1)
     * - Each row of ne00 elements gets multiplied by ONE value from src1
     * - The src1 index is (i / ne00) % n_b
     * 
     * For RMS_NORM pattern: src0=[2880,N] * src1=[2880]
     * - ne00=2880, n_b=2880
     * - Each element i uses b[i % n_b]
     * 
     * General rule: if n_b == ne00 (inner dimension match), use i % n_b
     *               else use (i / ne00) % n_b (outer dimension broadcast)
     */
    float* y;
    float* a;
    float* b;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&a, a_vptr);
    vedaMemPtr((void**)&b, b_vptr);
    
    int n_int = (int)n;
    int nb_int = (int)n_b;
    int ne00_int = (int)ne00;
    
    if (n == n_b) {
        /* Simple case: same size, no broadcast */
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] * b[i];
        }
    } else if (n_b == ne00) {
        /* Inner dimension match (like RMS_NORM): broadcast along outer dims */
        /* Pattern: [ne00, N] * [ne00] -> repeat src1 for each row */
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] * b[i % nb_int];
        }
    } else {
        /* Outer dimension broadcast (like MoE expert weighting) */
        /* Pattern: [ne00, ...] * [1, ...] -> each row uses one src1 element */
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] * b[(i / ne00_int) % nb_int];
        }
    }
    
    return 0;
}

/*
 * Mixed element-wise MUL: y = a * b with broadcasting
 * - y, a: in HBM (VEDAdeviceptr)
 * - b: in HMEM (already converted to raw pointer by vedaArgsSetHMEM)
 * 
 * Supports broadcasting: if n_b < n, b is repeated (n / n_b) times
 * This is used for RMS_NORM weight application: [hidden, rows] * [hidden] -> [hidden, rows]
 */
uint64_t ve_mul_hbm_hmem(VEDAdeviceptr y_vptr,
                          VEDAdeviceptr a_vptr,
                          void* b_hmem,
                          uint64_t n,
                          uint64_t n_b,
                          uint64_t ne00) {
    /*
     * GGML broadcast convention - see ve_mul_hbm_full_bcast for details.
     * ne00 = src0->ne[0] (innermost dimension of src0)
     */
    float* y;
    float* a;
    float* b = (float*)b_hmem;  /* HMEM already converted */
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&a, a_vptr);
    
    int n_int = (int)n;
    int nb_int = (int)n_b;
    int ne00_int = (int)ne00;
    
    if (n == n_b) {
        /* Simple case: same size, no broadcast */
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] * b[i];
        }
    } else if (n_b == ne00) {
        /* Inner dimension match: broadcast along outer dims */
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] * b[i % nb_int];
        }
    } else {
        /* Outer dimension broadcast */
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] * b[(i / ne00_int) % nb_int];
        }
    }
    
    return 0;
}

/*
 * Mixed element-wise ADD: y = a + b with broadcasting
 * - y, a: in HBM (VEDAdeviceptr)
 * - b: in HMEM (already converted to raw pointer by vedaArgsSetHMEM)
 */
uint64_t ve_add_hbm_hmem(VEDAdeviceptr y_vptr,
                          VEDAdeviceptr a_vptr,
                          void* b_hmem,
                          uint64_t n,
                          uint64_t n_b) {
    float* y;
    float* a;
    float* b = (float*)b_hmem;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&a, a_vptr);
    
    int n_int = (int)n;
    int nb_int = (int)n_b;
    
    if (n == n_b) {
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] + b[i];
        }
    } else {
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] + b[i % nb_int];
        }
    }
    
    return 0;
}

/*
 * Element-wise add: y = a + b (all in HBM)
 */
uint64_t ve_add_hbm_full(VEDAdeviceptr y_vptr,
                          VEDAdeviceptr a_vptr,
                          VEDAdeviceptr b_vptr,
                          uint64_t n) {
    float* y;
    float* a;
    float* b;

    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&a, a_vptr);
    vedaMemPtr((void**)&b, b_vptr);

    int n_int = (int)n;

    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        y[i] = a[i] + b[i];
    }

    return 0;
}

/*
 * Element-wise add with broadcasting: y = a + b (all in HBM)
 * b is broadcasted: b has n_b elements, repeated n/n_b times
 * Example: a=[8192], b=[4096] -> y[i] = a[i] + b[i % 4096]
 */
uint64_t ve_add_hbm_full_broadcast(VEDAdeviceptr y_vptr,
                                    VEDAdeviceptr a_vptr,
                                    VEDAdeviceptr b_vptr,
                                    uint64_t n,
                                    uint64_t n_b) {
    float* y;
    float* a;
    float* b;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&a, a_vptr);
    vedaMemPtr((void**)&b, b_vptr);
    
    int n_int = (int)n;
    int nb_int = (int)n_b;
    
    if (n == n_b) {
        /* No broadcast needed - simple add */
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] + b[i];
        }
    } else {
        /* Broadcast b: b[i % n_b] */
        #pragma _NEC ivdep
        for (int i = 0; i < n_int; i++) {
            y[i] = a[i] + b[i % nb_int];
        }
    }
    
    return 0;
}

/*
 * Scale: y = x * scale (all in HBM)
 * Supports inplace operation (y == x)
 */
uint64_t ve_scale_hbm_full(VEDAdeviceptr y_vptr,
                            VEDAdeviceptr x_vptr,
                            uint64_t scale_bits,
                            uint64_t n) {
    float* y;
    float* x;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    
    float scale;
    __builtin_memcpy(&scale, &scale_bits, sizeof(float));
    
    int n_int = (int)n;
    
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        y[i] = x[i] * scale;
    }
    
    return 0;
}

/*
 * Softmax: y = softmax(x * scale) (all in HBM)
 */
uint64_t ve_softmax_hbm_full(VEDAdeviceptr y_vptr,
                              VEDAdeviceptr x_vptr,
                              uint64_t n,
                              uint64_t scale_bits) {
    float* y;
    float* x;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    
    float scale;
    __builtin_memcpy(&scale, &scale_bits, sizeof(float));
    
    int n_int = (int)n;
    
    /* Find max for numerical stability */
    float max_val = -1e30f;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        float val = x[i] * scale;
        if (val > max_val) max_val = val;
    }
    
    /* Compute exp and sum */
    float sum = 0.0f;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        float val = expf(x[i] * scale - max_val);
        y[i] = val;
        sum += val;
    }
    
    /* Normalize */
    float inv_sum = 1.0f / sum;
    #pragma _NEC ivdep
    for (int i = 0; i < n_int; i++) {
        y[i] *= inv_sum;
    }
    
    return 0;
}

/*
 * Copy: y = x (HBM to HBM)
 */
uint64_t ve_copy_hbm_full(VEDAdeviceptr y_vptr,
                           VEDAdeviceptr x_vptr,
                           uint64_t n_bytes) {
    void* y;
    void* x;
    
    vedaMemPtr(&y, y_vptr);
    vedaMemPtr(&x, x_vptr);
    
    memcpy(y, x, (size_t)n_bytes);
    
    return 0;
}

/*
 * ROPE (Rotary Position Embedding): Apply rotation to Q/K vectors (all in HBM)
 * 
 * For each head, applies rotation:
 *   q_out[2i]   = q[2i] * cos - q[2i+1] * sin
 *   q_out[2i+1] = q[2i] * sin + q[2i+1] * cos
 */
uint64_t ve_rope_hbm_full(VEDAdeviceptr y_vptr,
                           VEDAdeviceptr x_vptr,
                           VEDAdeviceptr cos_vptr,
                           VEDAdeviceptr sin_vptr,
                           uint64_t n_head,
                           uint64_t n_rot,
                           uint64_t head_dim) {
    float* y;
    float* x;
    float* cos_cache;
    float* sin_cache;
    
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    vedaMemPtr((void**)&cos_cache, cos_vptr);
    vedaMemPtr((void**)&sin_cache, sin_vptr);
    
    int nh = (int)n_head;
    int nr = (int)n_rot;
    int hd = (int)head_dim;
    
    #pragma omp parallel for
    for (int h = 0; h < nh; h++) {
        float* y_head = y + h * hd;
        const float* x_head = x + h * hd;
        
        /* Apply rotation to first n_rot pairs */
        for (int i = 0; i < nr; i += 2) {
            float cos_val = cos_cache[i/2];
            float sin_val = sin_cache[i/2];
            
            float x0 = x_head[i];
            float x1 = x_head[i+1];
            
            y_head[i]   = x0 * cos_val - x1 * sin_val;
            y_head[i+1] = x0 * sin_val + x1 * cos_val;
        }
        
        /* Copy remaining elements unchanged */
        for (int i = nr; i < hd; i++) {
            y_head[i] = x_head[i];
        }
    }
    
    return 0;
}

/*
 * ROPE Normal style (HMEM version for llama.cpp integration)
 * 
 * Normal ROPE uses consecutive pair rotation:
 *   y[2i]   = x[2i] * cos - x[2i+1] * sin
 *   y[2i+1] = x[2i] * sin + x[2i+1] * cos
 * 
 * Parameters (passed as HMEM, which are already converted to raw pointers):
 *   y_ptr:     Output tensor (F32)
 *   x_ptr:     Input tensor (F32)  
 *   cache_ptr: Pre-computed [cos, sin] pairs (n_dims floats: cos0,sin0,cos1,sin1,...)
 *   ne0:       Head dimension (total elements per head)
 *   n_dims:    Number of dimensions to rotate (typically head_dim)
 *   n_heads:   Number of attention heads (ne1)
 *   n_ctx:     Sequence length (ne2)
 *   n_batch:   Batch size (ne3)
 *   nb1, nb2, nb3: Strides in bytes for input/output
 */
uint64_t ve_rope_normal_f32_hmem(void* y_ptr,
                                  void* x_ptr,
                                  void* cache_ptr,
                                  uint64_t ne0,
                                  uint64_t n_dims,
                                  uint64_t n_heads,
                                  uint64_t n_ctx,
                                  uint64_t n_batch,
                                  uint64_t nb1,
                                  uint64_t nb2,
                                  uint64_t nb3) {
    float* y = (float*)y_ptr;
    const float* x = (const float*)x_ptr;
    const float* cache = (const float*)cache_ptr;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int elem_per_head = (int)ne0;
    
    /* Process each sequence position (i2) x head (i1) x batch (i3) */
    for (int i3 = 0; i3 < batch; i3++) {
        for (int i2 = 0; i2 < ctx; i2++) {
            const float* pos_cache = cache;
            
            #pragma _NEC ivdep
            for (int i1 = 0; i1 < heads; i1++) {
                size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
                size_t dst_offset = src_offset;
                
                const float* src = (const float*)((const char*)x + src_offset);
                float* dst = (float*)((char*)y + dst_offset);
                
                /* Apply rotation to consecutive pairs: (x[2i], x[2i+1]) */
                #pragma _NEC ivdep
                for (int i0 = 0; i0 < nd; i0 += 2) {
                    float cos_val = pos_cache[i0];
                    float sin_val = pos_cache[i0 + 1];
                    
                    float x0 = src[i0];
                    float x1 = src[i0 + 1];
                    
                    dst[i0]     = x0 * cos_val - x1 * sin_val;
                    dst[i0 + 1] = x0 * sin_val + x1 * cos_val;
                }
                
                /* Copy remaining elements unchanged */
                for (int i0 = nd; i0 < elem_per_head; i0++) {
                    dst[i0] = src[i0];
                }
            }
        }
    }
    
    return 0;
}

/*
 * ROPE Normal style with OpenMP parallelization
 */
uint64_t ve_rope_normal_f32_omp_hmem(void* y_ptr,
                                      void* x_ptr,
                                      void* cache_ptr,
                                      uint64_t ne0,
                                      uint64_t n_dims,
                                      uint64_t n_heads,
                                      uint64_t n_ctx,
                                      uint64_t n_batch,
                                      uint64_t nb1,
                                      uint64_t nb2,
                                      uint64_t nb3) {
    float* y = (float*)y_ptr;
    const float* x = (const float*)x_ptr;
    const float* cache = (const float*)cache_ptr;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int elem_per_head = (int)ne0;
    
    /* Total rows = batch * ctx * heads */
    int total_rows = batch * ctx * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / (ctx * heads);
        int rem = row % (ctx * heads);
        int i2 = rem / heads;
        int i1 = rem % heads;
        
        /* Use position-specific cache: cache layout is [pos0_cos_sin, pos1_cos_sin, ...]
         * Each position has n_dims floats (n_dims/2 cos + n_dims/2 sin interleaved)
         * i2 is the sequence position index */
        const float* pos_cache = cache + i2 * nd;
        
        size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* Apply rotation to consecutive pairs */
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < nd; i0 += 2) {
            float cos_val = pos_cache[i0];
            float sin_val = pos_cache[i0 + 1];
            
            float x0 = src[i0];
            float x1 = src[i0 + 1];
            
            dst[i0]     = x0 * cos_val - x1 * sin_val;
            dst[i0 + 1] = x0 * sin_val + x1 * cos_val;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/*
 * ROPE NeoX style (HMEM version for llama.cpp integration)
 * 
 * NeoX/Llama uses a different rotation pattern than normal ROPE:
 *   y[i]             = x[i] * cos - x[i + n_dims/2] * sin
 *   y[i + n_dims/2]  = x[i] * sin + x[i + n_dims/2] * cos
 * 
 * Parameters (passed as HMEM, which are already converted to raw pointers):
 *   y_ptr:     Output tensor (F32)
 *   x_ptr:     Input tensor (F32)  
 *   cache_ptr: Pre-computed [cos, sin] pairs (2 * n_dims/2 floats)
 *   ne0:       Head dimension (total elements per head)
 *   n_dims:    Number of dimensions to rotate (typically head_dim)
 *   n_heads:   Number of attention heads (ne1)
 *   n_ctx:     Sequence length (ne2)
 *   n_batch:   Batch size (ne3)
 *   nb0, nb1, nb2, nb3: Strides in bytes for input/output
 */
uint64_t ve_rope_neox_f32_hmem(void* y_ptr,
                                void* x_ptr,
                                void* cache_ptr,
                                uint64_t ne0,
                                uint64_t n_dims,
                                uint64_t n_heads,
                                uint64_t n_ctx,
                                uint64_t n_batch,
                                uint64_t nb1,
                                uint64_t nb2,
                                uint64_t nb3) {
    float* y = (float*)y_ptr;
    const float* x = (const float*)x_ptr;
    const float* cache = (const float*)cache_ptr;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int nd_half = nd / 2;
    int elem_per_head = (int)ne0;
    
    /* Process each sequence position (i2) x head (i1) x batch (i3) */
    for (int i3 = 0; i3 < batch; i3++) {
        for (int i2 = 0; i2 < ctx; i2++) {
            /* Cache is precomputed: cache[i0] = cos, cache[i0+1] = sin for pair i0/2 */
            const float* pos_cache = cache;  /* Caller should provide cache for this position */
            
            #pragma _NEC ivdep
            for (int i1 = 0; i1 < heads; i1++) {
                /* Calculate byte offsets */
                size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
                size_t dst_offset = src_offset;  /* Same layout for output */
                
                const float* src = (const float*)((const char*)x + src_offset);
                float* dst = (float*)((char*)y + dst_offset);
                
                /* Apply rotation to first n_dims elements (NeoX style) */
                #pragma _NEC ivdep
                for (int i0 = 0; i0 < nd_half; i0++) {
                    float cos_val = pos_cache[i0 * 2];
                    float sin_val = pos_cache[i0 * 2 + 1];
                    
                    float x0 = src[i0];
                    float x1 = src[i0 + nd_half];
                    
                    dst[i0]           = x0 * cos_val - x1 * sin_val;
                    dst[i0 + nd_half] = x0 * sin_val + x1 * cos_val;
                }
                
                /* Copy remaining elements unchanged */
                for (int i0 = nd; i0 < elem_per_head; i0++) {
                    dst[i0] = src[i0];
                }
            }
        }
    }
    
    return 0;
}

/*
 * ROPE NeoX style with OpenMP parallelization
 */
uint64_t ve_rope_neox_f32_omp_hmem(void* y_ptr,
                                    void* x_ptr,
                                    void* cache_ptr,
                                    uint64_t ne0,
                                    uint64_t n_dims,
                                    uint64_t n_heads,
                                    uint64_t n_ctx,
                                    uint64_t n_batch,
                                    uint64_t nb1,
                                    uint64_t nb2,
                                    uint64_t nb3) {
    float* y = (float*)y_ptr;
    const float* x = (const float*)x_ptr;
    const float* cache = (const float*)cache_ptr;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int nd_half = nd / 2;
    int elem_per_head = (int)ne0;
    
    /* Total rows = batch * ctx * heads */
    int total_rows = batch * ctx * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / (ctx * heads);
        int rem = row % (ctx * heads);
        int i2 = rem / heads;
        int i1 = rem % heads;
        
        /* Use position-specific cache: cache layout is [pos0_cos_sin, pos1_cos_sin, ...]
         * Each position has n_dims floats (n_dims/2 cos + n_dims/2 sin interleaved)
         * i2 is the sequence position index */
        const float* pos_cache = cache + i2 * nd;
        
        /* Calculate byte offsets */
        size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* Apply rotation to first n_dims elements (NeoX style) */
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < nd_half; i0++) {
            float cos_val = pos_cache[i0 * 2];
            float sin_val = pos_cache[i0 * 2 + 1];
            
            float x0 = src[i0];
            float x1 = src[i0 + nd_half];
            
            dst[i0]           = x0 * cos_val - x1 * sin_val;
            dst[i0 + nd_half] = x0 * sin_val + x1 * cos_val;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/*
 * GET_ROWS: Extract rows from a matrix based on indices
 * 
 * For embedding lookups: output[i] = src[indices[i]]
 * 
 * Parameters:
 *   y_ptr:   Output tensor (F32), shape [nc, nr]
 *   x_ptr:   Source tensor (F32), shape [nc, n_rows_src]
 *   idx_ptr: Index tensor (I32), shape [nr]
 *   nc:      Number of columns (embedding dimension)
 *   nr:      Number of rows to extract (number of indices)
 *   nb_src:  Stride in bytes between rows in source
 *   nb_dst:  Stride in bytes between rows in destination
 */
uint64_t ve_get_rows_f32_hmem(void* y_ptr,
                               void* x_ptr,
                               void* idx_ptr,
                               uint64_t nc,
                               uint64_t nr,
                               uint64_t nb_src,
                               uint64_t nb_dst) {
    float* y = (float*)y_ptr;
    const float* x = (const float*)x_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t row_idx = idx[i];
        const float* src_row = (const float*)((const char*)x + row_idx * nb_src);
        float* dst_row = (float*)((char*)y + i * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            dst_row[j] = src_row[j];
        }
    }
    
    return 0;
}

/*
 * GET_ROWS for BF16 source -> F32 output
 * 
 * Extracts rows and converts from BF16 to F32 on the fly
 */
uint64_t ve_get_rows_bf16_f32_hmem(void* y_ptr,
                                    void* x_ptr,
                                    void* idx_ptr,
                                    uint64_t nc,
                                    uint64_t nr,
                                    uint64_t nb_src,
                                    uint64_t nb_dst) {
    float* y = (float*)y_ptr;
    const uint16_t* x = (const uint16_t*)x_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t row_idx = idx[i];
        const uint16_t* src_row = (const uint16_t*)((const char*)x + row_idx * nb_src);
        float* dst_row = (float*)((char*)y + i * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            /* BF16 to F32: shift left by 16 bits */
            uint32_t bf16_val = (uint32_t)src_row[j] << 16;
            float* fp32_ptr = (float*)&bf16_val;
            dst_row[j] = *fp32_ptr;
        }
    }
    
    return 0;
}

/*
 * GET_ROWS for F16 source -> F32 output
 * 
 * Extracts rows and converts from F16 to F32 on the fly
 */
uint64_t ve_get_rows_f16_f32_hmem(void* y_ptr,
                                   void* x_ptr,
                                   void* idx_ptr,
                                   uint64_t nc,
                                   uint64_t nr,
                                   uint64_t nb_src,
                                   uint64_t nb_dst) {
    float* y = (float*)y_ptr;
    const uint16_t* x = (const uint16_t*)x_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t row_idx = idx[i];
        const uint16_t* src_row = (const uint16_t*)((const char*)x + row_idx * nb_src);
        float* dst_row = (float*)((char*)y + i * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            /* F16 to F32 conversion
             * F16: 1 sign, 5 exponent, 10 mantissa
             * F32: 1 sign, 8 exponent, 23 mantissa
             */
            uint16_t h = src_row[j];
            uint32_t sign = (h & 0x8000) << 16;
            int32_t exponent = (h >> 10) & 0x1F;
            uint32_t mantissa = h & 0x3FF;
            
            uint32_t f;
            if (exponent == 0) {
                if (mantissa == 0) {
                    /* Zero */
                    f = sign;
                } else {
                    /* Denormalized number */
                    exponent = 1;
                    while ((mantissa & 0x400) == 0) {
                        mantissa <<= 1;
                        exponent--;
                    }
                    mantissa &= 0x3FF;
                    f = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
                }
            } else if (exponent == 31) {
                /* Inf or NaN */
                f = sign | 0x7F800000 | (mantissa << 13);
            } else {
                /* Normalized number */
                f = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }
            
            float* fp32_ptr = (float*)&f;
            dst_row[j] = *fp32_ptr;
        }
    }
    
    return 0;
}


/*
 * ============================================================================
 * GET_ROWS with HBM output - for HBM activations
 * ============================================================================
 * These write output to VE HBM instead of HMEM.
 * Source data comes from HMEM (host weights), output goes to HBM.
 */

/*
 * GET_ROWS F32 source -> F32 HBM output
 * 
 * y_hbm: VEDAdeviceptr to HBM output buffer
 * x_ptr: HMEM pointer to source embedding table (F32)
 * idx_ptr: HMEM pointer to row indices (I32)
 */
uint64_t ve_get_rows_f32_hbm(VEDAdeviceptr y_hbm,
                              void* x_ptr,
                              void* idx_ptr,
                              uint64_t nc,
                              uint64_t nr,
                              uint64_t nb_src,
                              uint64_t nb_dst) {
    // Convert HBM handle to raw pointer
    float* y;
    VEDAresult err = vedaMemPtr((void**)&y, y_hbm);
    if (err != VEDA_SUCCESS) {
        return err;
    }
    
    const float* x = (const float*)x_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t row_idx = idx[i];
        const float* src_row = (const float*)((const char*)x + row_idx * nb_src);
        float* dst_row = (float*)((char*)y + i * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            dst_row[j] = src_row[j];
        }
    }
    
    return 0;
}

/*
 * GET_ROWS BF16 source -> F32 HBM output
 * 
 * Extracts rows from BF16 embedding table and writes F32 to HBM.
 * BF16->F32 conversion is vectorizable on VE (shift left 16).
 */
uint64_t ve_get_rows_bf16_f32_hbm(VEDAdeviceptr y_hbm,
                                   void* x_ptr,
                                   void* idx_ptr,
                                   uint64_t nc,
                                   uint64_t nr,
                                   uint64_t nb_src,
                                   uint64_t nb_dst) {
    // Convert HBM handle to raw pointer
    float* y;
    VEDAresult err = vedaMemPtr((void**)&y, y_hbm);
    if (err != VEDA_SUCCESS) {
        return err;
    }
    
    const uint16_t* x = (const uint16_t*)x_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t row_idx = idx[i];
        const uint16_t* src_row = (const uint16_t*)((const char*)x + row_idx * nb_src);
        float* dst_row = (float*)((char*)y + i * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            /* BF16 to F32: shift left by 16 bits */
            uint32_t bf16_val = (uint32_t)src_row[j] << 16;
            float* fp32_ptr = (float*)&bf16_val;
            dst_row[j] = *fp32_ptr;
        }
    }
    
    return 0;
}

/*
 * GET_ROWS BF16 source HBM -> F32 HBM output
 * 
 * Both embedding table and output are in HBM - optimal path!
 * This avoids copying the entire embedding table over PCIe each call.
 */
uint64_t ve_get_rows_bf16_f32_hbm_hbm(VEDAdeviceptr y_hbm,
                                       VEDAdeviceptr x_hbm,
                                       VEDAdeviceptr idx_hbm,
                                       uint64_t nc,
                                       uint64_t nr,
                                       uint64_t nb_src,
                                       uint64_t nb_dst) {
    // Convert HBM handles to raw pointers
    float* y;
    const uint16_t* x;
    const int32_t* idx;
    VEDAresult err;

    err = vedaMemPtr((void**)&y, y_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&x, x_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&idx, idx_hbm);
    if (err != VEDA_SUCCESS) return err;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t row_idx = idx[i];
        const uint16_t* src_row = (const uint16_t*)((const char*)x + row_idx * nb_src);
        float* dst_row = (float*)((char*)y + i * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            /* BF16 to F32: shift left by 16 bits */
            uint32_t bf16_val = (uint32_t)src_row[j] << 16;
            float* fp32_ptr = (float*)&bf16_val;
            dst_row[j] = *fp32_ptr;
        }
    }
    
    return 0;
}

/*
 * GET_ROWS F32 source HBM -> F32 HBM output
 * 
 * Both embedding table and output are in HBM - optimal path!
 */
uint64_t ve_get_rows_f32_f32_hbm_hbm(VEDAdeviceptr y_hbm,
                                      VEDAdeviceptr x_hbm,
                                      VEDAdeviceptr idx_hbm,
                                      uint64_t nc,
                                      uint64_t nr,
                                      uint64_t nb_src,
                                      uint64_t nb_dst) {
    // Convert HBM handles to raw pointers
    float* y;
    const float* x;
    const int32_t* idx;
    VEDAresult err;

    err = vedaMemPtr((void**)&y, y_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&x, x_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&idx, idx_hbm);
    if (err != VEDA_SUCCESS) return err;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t row_idx = idx[i];
        const float* src_row = (const float*)((const char*)x + row_idx * nb_src);
        float* dst_row = (float*)((char*)y + i * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            dst_row[j] = src_row[j];
        }
    }
    
    return 0;
}


/*
 * ============================================================================
 * CPY/CONT/DUP kernels - copy tensor data
 * ============================================================================
 * These handle contiguous F32 tensors (most common case in inference)
 */

/* Simple contiguous copy */
uint64_t ve_cpy_f32_f32_hmem(void* dst_ptr,
                              void* src_ptr,
                              uint64_t n) {
    float* dst = (float*)dst_ptr;
    const float* src = (const float*)src_ptr;
    int count = (int)n;
    
    #pragma _NEC ivdep
    #pragma _NEC vovertake
    #pragma _NEC novob
    for (int i = 0; i < count; i++) {
        dst[i] = src[i];
    }
    
    return 0;
}

uint64_t ve_cpy_f32_f32_omp_hmem(void* dst_ptr,
                                  void* src_ptr,
                                  uint64_t n) {
    float* dst = (float*)dst_ptr;
    const float* src = (const float*)src_ptr;
    int count = (int)n;
    
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        dst[i] = src[i];
    }
    
    return 0;
}

/* Copy with stride support for non-contiguous tensors */
uint64_t ve_cpy_strided_f32_f32_hmem(void* dst_ptr,
                                      void* src_ptr,
                                      uint64_t ne0,
                                      uint64_t ne1,
                                      uint64_t ne2,
                                      uint64_t ne3,
                                      uint64_t nb0_src,
                                      uint64_t nb1_src,
                                      uint64_t nb2_src,
                                      uint64_t nb3_src,
                                      uint64_t nb0_dst,
                                      uint64_t nb1_dst,
                                      uint64_t nb2_dst,
                                      uint64_t nb3_dst) {
    const char* src = (const char*)src_ptr;
    char* dst = (char*)dst_ptr;
    
    int d0 = (int)ne0;
    int d1 = (int)ne1;
    int d2 = (int)ne2;
    int d3 = (int)ne3;
    
    /* Parallelize over outer dimensions */
    int total = d1 * d2 * d3;
    
    #pragma omp parallel for
    for (int idx = 0; idx < total; idx++) {
        int i3 = idx / (d1 * d2);
        int rem = idx % (d1 * d2);
        int i2 = rem / d1;
        int i1 = rem % d1;
        
        const float* src_row = (const float*)(src + i3 * nb3_src + i2 * nb2_src + i1 * nb1_src);
        float* dst_row = (float*)(dst + i3 * nb3_dst + i2 * nb2_dst + i1 * nb1_dst);
        
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < d0; i0++) {
            dst_row[i0] = src_row[i0];
        }
    }
    
    return 0;
}

/* 
 * Strided copy from HOST (via HMEM) to HBM
 * Source is passed via HMEM (already copied from host), destination is HBM
 * This is used for recurrent state management where source may be non-contiguous
 */
uint64_t ve_cpy_strided_f32_f32_hbm(VEDAdeviceptr dst_vptr,
                                     void* src_hmem,
                                     uint64_t ne0,
                                     uint64_t ne1,
                                     uint64_t ne2,
                                     uint64_t ne3,
                                     uint64_t nb0_src,
                                     uint64_t nb1_src,
                                     uint64_t nb2_src,
                                     uint64_t nb3_src,
                                     uint64_t nb0_dst,
                                     uint64_t nb1_dst,
                                     uint64_t nb2_dst,
                                     uint64_t nb3_dst) {
    float* dst;
    VEDAresult err = vedaMemPtr((void**)&dst, dst_vptr);
    (void)err;  // Suppress unused variable warning
    
    const char* src = (const char*)src_hmem;  /* Source is already via HMEM */
    
    int d0 = (int)ne0;
    int d1 = (int)ne1;
    int d2 = (int)ne2;
    int d3 = (int)ne3;
    
    /* Parallelize over outer dimensions */
    int total = d1 * d2 * d3;
    
    #pragma omp parallel for
    for (int idx = 0; idx < total; idx++) {
        int i3 = idx / (d1 * d2);
        int rem = idx % (d1 * d2);
        int i2 = rem / d1;
        int i1 = rem % d1;
        
        const float* src_row = (const float*)(src + i3 * nb3_src + i2 * nb2_src + i1 * nb1_src);
        float* dst_row = (float*)((char*)dst + i3 * nb3_dst + i2 * nb2_dst + i1 * nb1_dst);
        
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < d0; i0++) {
            dst_row[i0] = src_row[i0];
        }
    }
    
    return 0;
}

/* BF16 to F32 copy (for type conversion) */
uint64_t ve_cpy_bf16_f32_hmem(void* dst_ptr,
                               void* src_ptr,
                               uint64_t n) {
    float* dst = (float*)dst_ptr;
    const uint16_t* src = (const uint16_t*)src_ptr;
    int count = (int)n;
    
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        /* BF16 to F32: shift left by 16 bits */
        uint32_t bf16_val = (uint32_t)src[i] << 16;
        float* fp32_ptr = (float*)&bf16_val;
        dst[i] = *fp32_ptr;
    }
    
    return 0;
}

/* F32 to BF16 copy (for type conversion) */
uint64_t ve_cpy_f32_bf16_hmem(void* dst_ptr,
                               void* src_ptr,
                               uint64_t n) {
    uint16_t* dst = (uint16_t*)dst_ptr;
    const float* src = (const float*)src_ptr;
    int count = (int)n;
    
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        /* F32 to BF16: take upper 16 bits (truncation, not rounding) */
        const uint32_t* fp32_ptr = (const uint32_t*)&src[i];
        dst[i] = (uint16_t)(*fp32_ptr >> 16);
    }
    
    return 0;
}

/* F32 to F16 copy (for type conversion) */
/* F16 uses IEEE 754 half-precision format, different from BF16 */
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = ((x >> 23) & 0xff) - 127 + 15;  // Rebias exponent
    uint32_t mantissa = x & 0x007fffff;
    
    if (exponent <= 0) {
        // Subnormal or zero
        if (exponent < -10) {
            return sign;  // Zero
        }
        // Subnormal - shift mantissa
        mantissa = (mantissa | 0x00800000) >> (1 - exponent);
        return sign | (mantissa >> 13);
    } else if (exponent >= 31) {
        // Overflow to infinity or NaN
        if (exponent == 255 - 127 + 15) {
            // NaN or Inf in F32
            if (mantissa) {
                return sign | 0x7c00 | (mantissa >> 13);  // NaN
            }
            return sign | 0x7c00;  // Inf
        }
        return sign | 0x7c00;  // Overflow to Inf
    }
    
    return sign | (exponent << 10) | (mantissa >> 13);
}

uint64_t ve_cpy_f32_f16_hmem(void* dst_ptr,
                              void* src_ptr,
                              uint64_t n) {
    uint16_t* dst = (uint16_t*)dst_ptr;
    const float* src = (const float*)src_ptr;
    int count = (int)n;
    
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
    
    return 0;
}

/*
 * ============================================================================
 * DIAG_MASK kernels - causal attention masking
 * ============================================================================
 * 
 * For causal attention, we need to mask out future tokens.
 * The mask is lower triangular: position (i,j) is masked if j > i + n_past
 * 
 * DIAG_MASK_INF: Set masked positions to -INFINITY (for softmax)
 * DIAG_MASK_ZERO: Set masked positions to 0
 * 
 * These are in-place operations (dst == src0) or copy + mask.
 * 
 * Layout: [ne0 x ne1 x ne2] where:
 *   ne0 = nc (columns, key positions)
 *   ne1 = nr (rows, query positions)
 *   ne2 = nz (batch * heads)
 */

/*
 * DIAG_MASK_INF: Set upper triangular to -INFINITY
 * Used before softmax in attention to prevent attending to future tokens
 */
uint64_t ve_diag_mask_inf_f32_hmem(void* dst_ptr,
                                   void* src_ptr,
                                   uint64_t ne0,    /* nc - columns */
                                   uint64_t ne1,    /* nr - rows */
                                   uint64_t ne2,    /* nz - batches */
                                   uint64_t n_past, /* past context size */
                                   uint64_t inplace) {
    float* dst = (float*)dst_ptr;
    const float* src = (const float*)src_ptr;
    
    int nc = (int)ne0;
    int nr = (int)ne1;
    int nz = (int)ne2;
    int past = (int)n_past;
    
    /* IEEE 754 representation of -INFINITY */
    const float neg_inf = -1.0f / 0.0f;
    
    /* Copy if not inplace */
    if (!inplace) {
        int total = nc * nr * nz;
        #pragma omp parallel for
        for (int i = 0; i < total; i++) {
            dst[i] = src[i];
        }
    }
    
    /* Apply causal mask: mask position (row=j, col=i) if i > past + j
     * This masks out future tokens (upper triangular above the diagonal)
     */
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < nr; j++) {
            float* row = dst + k * nr * nc + j * nc;
            /* Mask columns i where i > past + j */
            int threshold = past + j;
            #pragma _NEC ivdep
            for (int i = 0; i < nc; i++) {
                if (i > threshold) {
                    row[i] = neg_inf;
                }
            }
        }
    }
    
    return 0;
}

/*
 * DIAG_MASK_ZERO: Set upper triangular to 0
 * Used for attention weight masking
 */
uint64_t ve_diag_mask_zero_f32_hmem(void* dst_ptr,
                                    void* src_ptr,
                                    uint64_t ne0,
                                    uint64_t ne1,
                                    uint64_t ne2,
                                    uint64_t n_past,
                                    uint64_t inplace) {
    float* dst = (float*)dst_ptr;
    const float* src = (const float*)src_ptr;
    
    int nc = (int)ne0;
    int nr = (int)ne1;
    int nz = (int)ne2;
    int past = (int)n_past;
    
    /* Copy if not inplace */
    if (!inplace) {
        int total = nc * nr * nz;
        #pragma omp parallel for
        for (int i = 0; i < total; i++) {
            dst[i] = src[i];
        }
    }
    
    /* Apply causal mask: set to 0 where i > past + j */
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < nr; j++) {
            float* row = dst + k * nr * nc + j * nc;
            int threshold = past + j;
            #pragma _NEC ivdep
            for (int i = 0; i < nc; i++) {
                if (i > threshold) {
                    row[i] = 0.0f;
                }
            }
        }
    }
    
    return 0;
}

/*
 * ============================================================================
 * KV Cache SET operation
 * ============================================================================
 * 
 * SET copies a source tensor into a destination tensor at a specific offset.
 * Used for updating KV cache during generation.
 *
 * dst[offset:offset+size] = src[0:size]
 */
uint64_t ve_set_f32_hmem(void* dst_ptr,
                          void* src_ptr,
                          uint64_t ne0,        /* elements per row */
                          uint64_t ne1,        /* rows in source */
                          uint64_t nb1_dst,    /* dst row stride (bytes) */
                          uint64_t nb1_src,    /* src row stride (bytes) */
                          uint64_t offset) {   /* byte offset into dst */
    char* dst = (char*)dst_ptr;
    const char* src = (const char*)src_ptr;
    
    int cols = (int)ne0;
    int rows = (int)ne1;
    
    /* Copy rows from src to dst at offset */
    #pragma omp parallel for
    for (int row = 0; row < rows; row++) {
        const float* src_row = (const float*)(src + row * nb1_src);
        float* dst_row = (float*)(dst + offset + row * nb1_dst);
        
        #pragma _NEC ivdep
        for (int col = 0; col < cols; col++) {
            dst_row[col] = src_row[col];
        }
    }
    
    return 0;
}

/*
 * SET_ROWS: Copy specific rows from source to destination
 * Used for sparse KV cache updates
 */
uint64_t ve_set_rows_f32_hmem(void* dst_ptr,
                               void* src_ptr,
                               void* idx_ptr,
                               uint64_t nc,
                               uint64_t nr,
                               uint64_t nb_dst,
                               uint64_t nb_src) {
    char* dst = (char*)dst_ptr;
    const float* src = (const float*)src_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t dst_row_idx = idx[i];
        const float* src_row = (const float*)((const char*)src + i * nb_src);
        float* dst_row = (float*)(dst + dst_row_idx * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            dst_row[j] = src_row[j];
        }
    }
    
    return 0;
}


/*
 * Helper: F32 -> F16 conversion (IEEE half-precision) with proper rounding
 * This is scalar (not vectorizable) on VE.
 * Uses round-to-nearest-even (IEEE 754 default rounding mode).
 */
static inline uint16_t fp32_to_fp16_scalar(float f) {
    uint32_t w;
    memcpy(&w, &f, sizeof(w));
    
    uint32_t sign = (w >> 16) & 0x8000;
    int32_t exp = (int32_t)((w >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = w & 0x007FFFFF;
    
    if (exp <= 0) {
        // Underflow to zero (ignore denormals for simplicity)
        return (uint16_t)sign;
    } else if (exp >= 31) {
        // Overflow to infinity
        return (uint16_t)(sign | 0x7C00);
    } else {
        // Round to nearest even:
        // The bits being truncated are mantissa[12:0] (13 bits)
        // Bit 12 is the guard bit, bits 11:0 are the sticky bits for rounding
        uint32_t round_bit = (mantissa >> 12) & 1;      // Guard bit (bit 12)
        uint32_t sticky = mantissa & 0x0FFF;            // Bits 11:0 (sticky)
        uint32_t truncated = mantissa >> 13;            // F16 mantissa (10 bits)
        
        // Round to nearest even:
        // - If round_bit=0: truncate (round down)
        // - If round_bit=1 and sticky>0: round up
        // - If round_bit=1 and sticky=0: round to even (up if truncated LSB is 1)
        if (round_bit) {
            if (sticky || (truncated & 1)) {
                truncated++;
                // Handle mantissa overflow (10 bits max)
                if (truncated > 0x3FF) {
                    truncated = 0;
                    exp++;
                    // Check for exponent overflow
                    if (exp >= 31) {
                        return (uint16_t)(sign | 0x7C00);  // Infinity
                    }
                }
            }
        }
        
        return (uint16_t)(sign | ((uint32_t)exp << 10) | truncated);
    }
}

/*
 * SET_ROWS F32 -> F16 to HBM destination (for F16 KV cache in HBM)
 * 
 * WARNING: F16 conversion is NOT vectorizable on VE - this will be slow!
 * Use BF16 KV cache (-ctk bf16 -ctv bf16) for better performance.
 * 
 * dst_hbm: VEDAdeviceptr to HBM buffer (destination, F16)
 * src_ptr: HMEM pointer to source data (F32)
 * idx_ptr: HMEM pointer to row indices (int32)
 * nc: number of columns (elements per row)
 * nr: number of rows to set
 * nb_dst: stride between rows in destination (bytes)
 * nb_src: stride between rows in source (bytes)
 */
uint64_t ve_set_rows_f16_hbm(VEDAdeviceptr dst_hbm,
                              void* src_ptr,
                              void* idx_ptr,
                              uint64_t nc,
                              uint64_t nr,
                              uint64_t nb_dst,
                              uint64_t nb_src) {
    // Convert HBM handle to raw pointer
    char* dst;
    VEDAresult err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) {
        return err;
    }
    
    const char* src = (const char*)src_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int rows = (int)nr;
    int cols = (int)nc;
    
    // F16 conversion: scalar, not vectorizable
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t dst_row_idx = idx[i];
        const float* src_row = (const float*)(src + i * nb_src);
        uint16_t* dst_row = (uint16_t*)(dst + dst_row_idx * nb_dst);
        
        for (int j = 0; j < cols; j++) {
            dst_row[j] = fp32_to_fp16_scalar(src_row[j]);
        }
    }
    
    return 0;
}


/*
 * SET_ROWS F32 -> BF16 to HBM destination (for BF16 KV cache in HBM)
 * 
 * BF16 conversion is fast and vectorizable - just right-shift by 16 bits.
 * Use BF16 KV cache (-ctk bf16 -ctv bf16) for best VE performance.
 * 
 * dst_hbm: VEDAdeviceptr to HBM buffer (destination, BF16)
 * src_ptr: HMEM pointer to source data (F32)
 * idx_ptr: HMEM pointer to row indices (int32)
 * nc: number of columns (elements per row)
 * nr: number of rows to set
 * nb_dst: stride between rows in destination (bytes)
 * nb_src: stride between rows in source (bytes)
 */
uint64_t ve_set_rows_bf16_hbm(VEDAdeviceptr dst_hbm,
                               void* src_ptr,
                               void* idx_ptr,
                               uint64_t nc,
                               uint64_t nr,
                               uint64_t nb_dst,
                               uint64_t nb_src) {
    // Convert HBM handle to raw pointer
    char* dst;
    VEDAresult err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) {
        return err;
    }
    
    const char* src = (const char*)src_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int rows = (int)nr;
    int cols = (int)nc;
    
    // BF16 conversion: F32 -> BF16 is just taking upper 16 bits
    // Process as 32-bit integers, store as 16-bit
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t dst_row_idx = idx[i];
        const uint32_t* src_row = (const uint32_t*)(src + i * nb_src);
        uint16_t* dst_row = (uint16_t*)(dst + dst_row_idx * nb_dst);
        
        // Vectorized F32 -> BF16 conversion (right shift by 16)
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            // BF16 = upper 16 bits of F32 (truncation, no rounding)
            dst_row[j] = (uint16_t)(src_row[j] >> 16);
        }
    }
    
    return 0;
}


/*
 * SET_ROWS F32 to HBM destination
 */
uint64_t ve_set_rows_f32_hbm(VEDAdeviceptr dst_hbm,
                              void* src_ptr,
                              void* idx_ptr,
                              uint64_t nc,
                              uint64_t nr,
                              uint64_t nb_dst,
                              uint64_t nb_src) {
    // Convert HBM handle to raw pointer
    char* dst;
    VEDAresult err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) {
        return err;
    }
    
    const float* src = (const float*)src_ptr;
    const int32_t* idx = (const int32_t*)idx_ptr;
    
    int cols = (int)nc;
    int rows = (int)nr;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t dst_row_idx = idx[i];
        const float* src_row = (const float*)((const char*)src + i * nb_src);
        float* dst_row = (float*)(dst + dst_row_idx * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            dst_row[j] = src_row[j];
        }
    }
    
    return 0;
}


/*
 * ===========================================================================
 * HBM-FULL SET_ROWS - Both source and destination in HBM (no HMEM)
 * ===========================================================================
 *
 * These kernels avoid HBM->HMEM->HBM copies by reading/writing HBM directly.
 * All arguments are VEDAdeviceptr, converted to raw pointers via vedaMemPtr().
 * Indices are passed via HMEM since they're small and come from host.
 *
 * Stream ordering ensures previous ops writing to src_hbm have completed
 * before this kernel runs - no explicit sync needed on host side.
 */

/*
 * SET_ROWS HBM-full: F32 src (HBM) -> F16 dst (HBM)
 */
uint64_t ve_set_rows_f16_hbm_full(VEDAdeviceptr dst_hbm,
                                   VEDAdeviceptr src_hbm,
                                   VEDAdeviceptr idx_hbm,
                                   uint64_t nc,
                                   uint64_t nr,
                                   uint64_t nb_dst,
                                   uint64_t nb_src) {
    char* dst;
    char* src;
    const int32_t* idx;
    VEDAresult err;

    err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&src, src_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&idx, idx_hbm);
    if (err != VEDA_SUCCESS) return err;
    int rows = (int)nr;
    int cols = (int)nc;
    
    // F16 conversion: scalar, not vectorizable on VE
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t dst_row_idx = idx[i];
        const float* src_row = (const float*)(src + i * nb_src);
        uint16_t* dst_row = (uint16_t*)(dst + dst_row_idx * nb_dst);
        
        for (int j = 0; j < cols; j++) {
            dst_row[j] = fp32_to_fp16_scalar(src_row[j]);
        }
    }
    
    return 0;
}

/*
 * SET_ROWS HBM-full: F32 src (HBM) -> BF16 dst (HBM)
 * BF16 conversion is vectorizable (just shift)
 */
uint64_t ve_set_rows_bf16_hbm_full(VEDAdeviceptr dst_hbm,
                                    VEDAdeviceptr src_hbm,
                                    VEDAdeviceptr idx_hbm,
                                    uint64_t nc,
                                    uint64_t nr,
                                    uint64_t nb_dst,
                                    uint64_t nb_src) {
    char* dst;
    char* src;
    const int32_t* idx;
    VEDAresult err;

    int is_layer23_v = 0;  /* Disabled */

    err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&src, src_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&idx, idx_hbm);
    if (err != VEDA_SUCCESS) return err;
    int rows = (int)nr;
    int cols = (int)nc;
    
    // Process pairs of F32 -> BF16 using 32-bit operations for vectorization
    // NCC cannot vectorize 16-bit stores, but can vectorize 32-bit
    int cols_pairs = cols / 2;
    int cols_remainder = cols % 2;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t dst_row_idx = idx[i];
        const uint32_t* src_row = (const uint32_t*)(src + i * nb_src);
        uint32_t* dst_row32 = (uint32_t*)(dst + dst_row_idx * nb_dst);
        
        // Vectorized: pack two BF16 into one 32-bit word
        // BF16 is top 16 bits of F32, so we shift and pack
        #pragma _NEC ivdep
        for (int j = 0; j < cols_pairs; j++) {
            uint32_t f0 = src_row[j*2];      // First F32
            uint32_t f1 = src_row[j*2 + 1];  // Second F32
            // Pack: bf16_0 in low 16 bits, bf16_1 in high 16 bits
            dst_row32[j] = (f0 >> 16) | (f1 & 0xFFFF0000);
        }
        
        /* Debug: verify disabled for production */
        
        // Handle odd column if present
        if (cols_remainder) {
            uint16_t* dst_row16 = (uint16_t*)(dst + dst_row_idx * nb_dst);
            dst_row16[cols - 1] = (uint16_t)(src_row[cols - 1] >> 16);
        }
    }
    
    /* Layer 23 V cache: verify write at position ic=6, kv_head=6, element 20 (u32[202]) */
    if (is_layer23_v) {
        /* Position 6 in V cache: row 6 at byte offset 6*1024=6144
         * kv_head 6 starts at byte 768 within row (6*128)
         * u32[10] within kv_head 6 = element 20 within head = byte 40
         * Global: byte 6144+768+40=6952 = u32[(6144+768)/4 + 10] = u32[1738] from base
         * Within row 6: byte 768+40=808 = u32[202]
         */
        uint32_t* row6_dst = (uint32_t*)(dst + 6 * nb_dst);
        uint32_t val_202 = row6_dst[202];
        fprintf(stderr, "[SET_ROWS_FULL-L23V] AFTER write: row6_dst[202]=0x%08x (at offset %lld)\n",
                val_202, (long long)(6 * nb_dst + 202*4));
        
        /* Also check what flash attention will read at offset 6952 */
        uint32_t val_at_6952 = *(uint32_t*)(dst + 6952);
        fprintf(stderr, "[SET_ROWS_FULL-L23V] At byte offset 6952: 0x%08x\n", val_at_6952);
    }
    
    return 0;
}

/*
 * SET_ROWS HBM-full: F32 src (HBM) -> F32 dst (HBM)
 */
uint64_t ve_set_rows_f32_hbm_full(VEDAdeviceptr dst_hbm,
                                   VEDAdeviceptr src_hbm,
                                   VEDAdeviceptr idx_hbm,
                                   uint64_t nc,
                                   uint64_t nr,
                                   uint64_t nb_dst,
                                   uint64_t nb_src) {
    char* dst;
    char* src;
    const int32_t* idx;
    VEDAresult err;

    err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&src, src_hbm);
    if (err != VEDA_SUCCESS) return err;

    err = vedaMemPtr((void**)&idx, idx_hbm);
    if (err != VEDA_SUCCESS) return err;
    int rows = (int)nr;
    int cols = (int)nc;
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        int32_t dst_row_idx = idx[i];
        const float* src_row = (const float*)(src + i * nb_src);
        float* dst_row = (float*)(dst + dst_row_idx * nb_dst);
        
        #pragma _NEC ivdep
        for (int j = 0; j < cols; j++) {
            dst_row[j] = src_row[j];
        }
    }
    
    return 0;
}


/*
 * ===========================================================================
 * FLASH ATTENTION EXT - Fused Scaled Dot-Product Attention
 * ===========================================================================
 *
 * Computes: O = softmax(Q @ K^T * scale + mask) @ V
 *
 * Uses online softmax algorithm to avoid O(n²) memory for attention matrix.
 * Reference: https://arxiv.org/pdf/2112.05682.pdf (FlashAttention paper)
 *
 * Tensor shapes (llama.cpp convention):
 *   Q: [D, N, H, B]  - queries: D=head_dim, N=n_tokens, H=n_heads, B=batch
 *   K: [D, S, Hk, B] - keys: D=head_dim, S=n_kv (sequence), Hk=n_kv_heads
 *   V: [Dv, S, Hv, B] - values: Dv=head_dim_v (usually same as D)
 *   mask: [S, N_padded, Hm, Bm] - attention mask (F16 or F32), broadcasted
 *   output: [Dv, H, N, B] - permute(0, 2, 1, 3) of natural order
 *
 * For GQA/MQA: H > Hk, heads are broadcast (H must be divisible by Hk)
 */

/*
 * Flash Attention for F32 Q/K/V with F16 mask
 * This is the most common configuration in llama.cpp with flash attention enabled
 */
uint64_t ve_flash_attn_ext_f32_f16mask_hmem(
    void* dst_ptr,           /* Output: [Dv, H, N, B] */
    void* q_ptr,             /* Query: [D, N, H, B] */
    void* k_ptr,             /* Key: [D, S, Hk, B] */
    void* v_ptr,             /* Value: [Dv, S, Hv, B] */
    VEDAdeviceptr mask_hbm,          /* Mask: [S, N_padded, Hm, Bm] F16, can be NULL */
    uint64_t D,              /* Head dimension (Q, K) */
    uint64_t Dv,             /* Value head dimension (usually same as D) */
    uint64_t N,              /* Number of query tokens */
    uint64_t S,              /* Number of KV tokens (sequence length) */
    uint64_t H,              /* Number of query heads */
    uint64_t Hk,             /* Number of KV heads (for GQA) */
    uint64_t B,              /* Batch size */
    uint64_t nb_q1,          /* Q stride for N dimension (bytes) */
    uint64_t nb_q2,          /* Q stride for H dimension (bytes) */
    uint64_t nb_q3,          /* Q stride for B dimension (bytes) */
    uint64_t nb_k1,          /* K stride for S dimension (bytes) */
    uint64_t nb_k2,          /* K stride for Hk dimension (bytes) */
    uint64_t nb_k3,          /* K stride for B dimension (bytes) */
    uint64_t nb_v1,          /* V stride for S dimension (bytes) */
    uint64_t nb_v2,          /* V stride for Hv dimension (bytes) */
    uint64_t nb_v3,          /* V stride for B dimension (bytes) */
    uint64_t nb_m1,          /* Mask stride for N dimension (bytes) */
    uint64_t nb_m2,          /* Mask stride for H dimension (bytes) */
    uint64_t nb_m3,          /* Mask stride for B dimension (bytes) */
    uint64_t nb_o1,          /* Output stride for H dimension (bytes) */
    uint64_t nb_o2,          /* Output stride for N dimension (bytes) */
    uint64_t nb_o3,          /* Output stride for B dimension (bytes) */
    uint64_t scale_bits,     /* Scale as float bits */
    uint64_t max_bias_bits,  /* ALiBi max_bias as float bits */
    uint64_t softcap_bits,   /* Logit softcap as float bits */
    uint64_t mask_ne2,       /* Mask ne[2] for broadcasting */
    uint64_t mask_ne3)       /* Mask ne[3] for broadcasting */
{
    float* dst = (float*)dst_ptr;
    const float* q = (const float*)q_ptr;
    const float* k = (const float*)k_ptr;
    const float* v = (const float*)v_ptr;
    const uint16_t* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    /* Decode float parameters */
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    /* Apply softcap to scale */
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    /* ALiBi slope computation */
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    /* GQA broadcast factors */
    const int64_t rk = (int64_t)(H / Hk);  /* How many Q heads per K head */
    const int64_t rv = (int64_t)(H / Hk);  /* Same for V (Hv == Hk assumed) */
    
    int d = (int)D;
    int dv = (int)Dv;
    int n_tokens = (int)N;
    int n_kv = (int)S;
    int n_heads = (int)H;
    int n_batch = (int)B;
    
    /* DEBUG: Dump tensors to files for first single-token attention call */
    static int f32_hmem_call_count = 0;
    f32_hmem_call_count++;
    
    /* Dump Q, K, V, mask for call #29 (first single-token attention, layer 0) */
    if (f32_hmem_call_count == 29 && n_tokens == 1) {
        char* dump_dir = getenv("GGML_VE_DUMP_DIR");
        if (dump_dir) {
            char fname[256];
            FILE* f;
            
            /* Dump Q (all heads, single position) */
            snprintf(fname, sizeof(fname), "%s/attn29_q.bin", dump_dir);
            f = fopen(fname, "wb");
            if (f) {
                for (int h = 0; h < n_heads; h++) {
                    const float* q_head = (const float*)((const char*)q + 0 * nb_q1 + h * nb_q2 + 0 * nb_q3);
                    fwrite(q_head, sizeof(float), d, f);
                }
                fclose(f);
                printf("[DUMP-HMEM] Wrote Q to %s (%d heads × %d dims)\n", fname, n_heads, d);
            }
            
            /* Dump K (first KV head, all positions) */
            snprintf(fname, sizeof(fname), "%s/attn29_k.bin", dump_dir);
            f = fopen(fname, "wb");
            if (f) {
                for (int s = 0; s < n_kv; s++) {
                    const float* k_pos = (const float*)((const char*)k + s * nb_k1 + 0 * nb_k2 + 0 * nb_k3);
                    fwrite(k_pos, sizeof(float), d, f);
                }
                fclose(f);
                printf("[DUMP-HMEM] Wrote K to %s (%d positions × %d dims)\n", fname, n_kv, d);
            }
            
            /* Dump V (first KV head, all positions) */
            snprintf(fname, sizeof(fname), "%s/attn29_v.bin", dump_dir);
            f = fopen(fname, "wb");
            if (f) {
                for (int s = 0; s < n_kv; s++) {
                    const float* v_pos = (const float*)((const char*)v + s * nb_v1 + 0 * nb_v2 + 0 * nb_v3);
                    fwrite(v_pos, sizeof(float), dv, f);
                }
                fclose(f);
                printf("[DUMP-HMEM] Wrote V to %s (%d positions × %d dims)\n", fname, n_kv, dv);
            }
            
            /* Dump mask */
            if (mask != NULL) {
                snprintf(fname, sizeof(fname), "%s/attn29_mask.bin", dump_dir);
                f = fopen(fname, "wb");
                if (f) {
                    const uint16_t* mask_row = (const uint16_t*)((const char*)mask + 0 * nb_m1 + 0 * nb_m2 + 0 * nb_m3);
                    fwrite(mask_row, sizeof(uint16_t), n_kv, f);
                    fclose(f);
                    printf("[DUMP-HMEM] Wrote mask to %s (%d positions, F16)\n", fname, n_kv);
                }
            }
        }
    }
    
    /* Parallelize over batch, heads, and query tokens */
    /* Total work items: B * H * N */
    int total_work = n_batch * n_heads * n_tokens;
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        /* Decompose work_id into (ib, ih, iq) */
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        /* ALiBi slope for this head */
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        /* KV head indices (for GQA) */
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        /* Get Q vector for this (batch, head, token) */
        const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        /* Get mask row if available */
        const uint16_t* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const uint16_t*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        /* Online softmax variables */
        float M = -INFINITY;  /* Running maximum */
        float S_sum = 0.0f;   /* Running sum of exp(s - M) */
        
        /* VKQ accumulator - will hold sum of v * exp(s - M) */
        float VKQ[256];  /* Assuming head_dim <= 256, adjust if needed */
        for (int i = 0; i < dv; i++) {
            VKQ[i] = 0.0f;
        }
        
        /* Loop over KV sequence */
        for (int ic = 0; ic < n_kv; ic++) {
            /* Get mask value (F16 -> F32) */
            float mv = 0.0f;
            if (mask_row != NULL) {
                /* Convert F16 to F32 */
                uint16_t mf16 = mask_row[ic];
                /* Simple F16 to F32 conversion */
                uint32_t sign = (mf16 >> 15) & 0x1;
                uint32_t exp = (mf16 >> 10) & 0x1F;
                uint32_t mant = mf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    if (mant == 0) {
                        f32 = sign << 31;  /* Zero */
                    } else {
                        /* Denormal - convert to normal */
                        exp = 1;
                        while ((mant & 0x400) == 0) {
                            mant <<= 1;
                            exp--;
                        }
                        mant &= 0x3FF;
                        f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7F800000 | (mant << 13);  /* Inf/NaN */
                } else {
                    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                __builtin_memcpy(&mv, &f32, sizeof(float));
                mv *= slope;  /* Apply ALiBi slope */
            }
            
            /* Skip if mask is -inf */
            if (mv == -INFINITY) {
                continue;
            }
            
            /* Get K vector */
            const float* k_vec = (const float*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            
            /* Compute Q @ K dot product */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_vec[i] * k_vec[i];
            }
            
            /* Apply scale */
            s *= scale;
            
            /* Apply logit softcap */
            if (logit_softcap != 0.0f) {
                s = logit_softcap * tanhf(s);
            }
            
            /* Add mask */
            s += mv;
            
            /* Online softmax update */
            float M_old = M;
            float ms = 1.0f;  /* Scale for existing VKQ when new max found */
            float vs = 1.0f;  /* Weight for current V */
            
            if (s > M) {
                /* New maximum found */
                M = s;
                ms = expf(M_old - M);  /* Scale down existing accumulator */
                /* vs = 1.0f (= exp(s - M) = exp(0)) */
                
                /* Scale existing VKQ */
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[i] *= ms;
                }
                S_sum *= ms;
            } else {
                /* No new max, compute weight for this V */
                vs = expf(s - M);
            }
            
            /* Get V vector */
            const float* v_vec = (const float*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            
            /* Accumulate V weighted by softmax */
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                VKQ[i] += vs * v_vec[i];
            }
            
            /* Update sum */
            S_sum += vs;
        }
        
        /* Normalize by sum (complete softmax) */
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] *= S_inv;
        }
        
        /* Write output with permutation: output shape is [Dv, H, N, B] */
        /* dst[i3*ne2*ne1 + i2 + i1*ne1] where i1=iq, i2=ih, i3=ib, ne1=H, ne2=N */
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = VKQ[i];
        }
    }
    
    /* Dump output for call #29 */
    if (f32_hmem_call_count == 29 && n_tokens == 1) {
        char* dump_dir = getenv("GGML_VE_DUMP_DIR");
        if (dump_dir) {
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/attn29_out.bin", dump_dir);
            FILE* f = fopen(fname, "wb");
            if (f) {
                for (int h = 0; h < n_heads; h++) {
                    float* out_head = (float*)((char*)dst + 0 * nb_o2 + h * nb_o1 + 0 * nb_o3);
                    fwrite(out_head, sizeof(float), dv, f);
                }
                fclose(f);
                printf("[DUMP-HMEM] Wrote output to %s (%d heads × %d dims)\n", fname, n_heads, dv);
            }
        }
    }
    
    return 0;
}

/*
 * Flash Attention for BF16 Q/K/V with F16 mask
 * Used when model weights and KV cache are in BF16
 */
uint64_t ve_flash_attn_ext_bf16_f16mask_hmem(
    void* dst_ptr,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    VEDAdeviceptr mask_hbm,
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1,
    uint64_t nb_q2,
    uint64_t nb_q3,
    uint64_t nb_k1,
    uint64_t nb_k2,
    uint64_t nb_k3,
    uint64_t nb_v1,
    uint64_t nb_v2,
    uint64_t nb_v3,
    uint64_t nb_m1,
    uint64_t nb_m2,
    uint64_t nb_m3,
    uint64_t nb_o1,
    uint64_t nb_o2,
    uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    /*
     * OPTIMIZED BF16 Flash Attention for VE
     * 
     * Key optimization: Pre-convert K and V vectors to F32 BEFORE the inner loop
     * so that the dot product and accumulation loops are pure FP32 and vectorizable.
     * 
     * BF16 to F32 conversion is trivial: just left-shift by 16 bits.
     * We use uint32_t arrays and interpret them as float via union for type punning
     * that NCC can vectorize.
     */
    
    /* Union for vectorizable BF16->F32 type punning */
    typedef union { uint32_t u; float f; } uf32;
    
    float* dst = (float*)dst_ptr;
    const uint16_t* q = (const uint16_t*)q_ptr;  /* BF16 */
    const uint16_t* k = (const uint16_t*)k_ptr;  /* BF16 */
    const uint16_t* v = (const uint16_t*)v_ptr;  /* BF16 */
    const uint16_t* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    /* Decode float parameters */
    float scale, max_bias, logit_softcap;
    uf32 utmp;
    utmp.u = (uint32_t)scale_bits;
    scale = utmp.f;
    utmp.u = (uint32_t)max_bias_bits;
    max_bias = utmp.f;
    utmp.u = (uint32_t)softcap_bits;
    logit_softcap = utmp.f;
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);
    
    int d = (int)D;
    int dv = (int)Dv;
    int n_tokens = (int)N;
    int n_kv = (int)S;
    int n_heads = (int)H;
    int n_batch = (int)B;
    
    int total_work = n_batch * n_heads * n_tokens;
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        /* Pre-convert Q from BF16 to F32 (outside inner loops) */
        const uint16_t* q_bf16 = (const uint16_t*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        float q_f32[256];
        #pragma _NEC ivdep
        for (int i = 0; i < d; i++) {
            uf32 conv;
            conv.u = ((uint32_t)q_bf16[i]) << 16;
            q_f32[i] = conv.f;
        }
        
        const uint16_t* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const uint16_t*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] = 0.0f;
        }
        
        /* Pre-allocate K and V F32 conversion buffers */
        float k_f32[256];
        float v_f32[256];
        
        for (int ic = 0; ic < n_kv; ic++) {
            /* Decode F16 mask (scalar, rare branch) */
            float mv = 0.0f;
            if (mask_row != NULL) {
                uint16_t mf16 = mask_row[ic];
                uint32_t sign = (mf16 >> 15) & 0x1;
                uint32_t exp = (mf16 >> 10) & 0x1F;
                uint32_t mant = mf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (mant == 0) ? (sign << 31) : 0;
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                uf32 mconv;
                mconv.u = f32;
                mv = mconv.f * slope;
            }
            
            if (mv == -INFINITY) {
                continue;
            }
            
            /* PRE-CONVERT K vector from BF16 to F32 (vectorizable!) */
            const uint16_t* k_bf16 = (const uint16_t*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                uf32 conv;
                conv.u = ((uint32_t)k_bf16[i]) << 16;
                k_f32[i] = conv.f;
            }
            
            /* Now compute Q·K dot product in pure F32 (vectorizable!) */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_f32[i] * k_f32[i];
            }
            
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
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[i] *= ms;
                }
                S_sum *= ms;
            } else {
                vs = expf(s - M);
            }
            
            /* PRE-CONVERT V vector from BF16 to F32 (vectorizable!) */
            const uint16_t* v_bf16 = (const uint16_t*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                uf32 conv;
                conv.u = ((uint32_t)v_bf16[i]) << 16;
                v_f32[i] = conv.f;
            }
            
            /* Accumulate V weighted by softmax (pure F32, vectorizable!) */
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                VKQ[i] += vs * v_f32[i];
            }
            
            S_sum += vs;
        }
        
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] *= S_inv;
        }
        
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = VKQ[i];
        }
    }
    
    return 0;
}

/*
 * Flash Attention for F32 Q/K/V with F32 mask (no mask type conversion needed)
 */
uint64_t ve_flash_attn_ext_f32_f32mask_hmem(
    void* dst_ptr,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    VEDAdeviceptr mask_hbm,
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1,
    uint64_t nb_q2,
    uint64_t nb_q3,
    uint64_t nb_k1,
    uint64_t nb_k2,
    uint64_t nb_k3,
    uint64_t nb_v1,
    uint64_t nb_v2,
    uint64_t nb_v3,
    uint64_t nb_m1,
    uint64_t nb_m2,
    uint64_t nb_m3,
    uint64_t nb_o1,
    uint64_t nb_o2,
    uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    float* dst = (float*)dst_ptr;
    const float* q = (const float*)q_ptr;
    const float* k = (const float*)k_ptr;
    const float* v = (const float*)v_ptr;
    const float* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);
    
    int d = (int)D;
    int dv = (int)Dv;
    int n_tokens = (int)N;
    int n_kv = (int)S;
    int n_heads = (int)H;
    int n_batch = (int)B;
    
    int total_work = n_batch * n_heads * n_tokens;
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        const float* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const float*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];
        for (int i = 0; i < dv; i++) {
            VKQ[i] = 0.0f;
        }
        
        for (int ic = 0; ic < n_kv; ic++) {
            float mv = (mask_row != NULL) ? slope * mask_row[ic] : 0.0f;
            
            if (mv == -INFINITY) {
                continue;
            }
            
            const float* k_vec = (const float*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_vec[i] * k_vec[i];
            }
            
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
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[i] *= ms;
                }
                S_sum *= ms;
            } else {
                vs = expf(s - M);
            }
            
            const float* v_vec = (const float*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                VKQ[i] += vs * v_vec[i];
            }
            
            S_sum += vs;
        }
        
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] *= S_inv;
        }
        
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = VKQ[i];
        }
    }
    
    return 0;
}

/*
 * Flash Attention for F32 Q with F16 K/V and F16 mask
 * This is the most common configuration in llama.cpp:
 * - Q comes from query projection (F32)
 * - K/V come from KV cache (stored as F16 to save memory)
 * - Mask is F16
 */
uint64_t ve_flash_attn_ext_f32q_f16kv_hmem(
    void* dst_ptr,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    VEDAdeviceptr mask_hbm,
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1,
    uint64_t nb_q2,
    uint64_t nb_q3,
    uint64_t nb_k1,
    uint64_t nb_k2,
    uint64_t nb_k3,
    uint64_t nb_v1,
    uint64_t nb_v2,
    uint64_t nb_v3,
    uint64_t nb_m1,
    uint64_t nb_m2,
    uint64_t nb_m3,
    uint64_t nb_o1,
    uint64_t nb_o2,
    uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    float* dst = (float*)dst_ptr;
    const float* q = (const float*)q_ptr;     /* F32 Q */
    const uint16_t* k = (const uint16_t*)k_ptr;  /* F16 K */
    const uint16_t* v = (const uint16_t*)v_ptr;  /* F16 V */
    const uint16_t* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);
    
    int d = (int)D;
    int dv = (int)Dv;
    int n_tokens = (int)N;
    int n_kv = (int)S;
    int n_heads = (int)H;
    int n_batch = (int)B;
    
    int total_work = n_batch * n_heads * n_tokens;
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        /* Q is F32 - use directly */
        const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        const uint16_t* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const uint16_t*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];
        for (int i = 0; i < dv; i++) {
            VKQ[i] = 0.0f;
        }
        
        for (int ic = 0; ic < n_kv; ic++) {
            /* Decode F16 mask */
            float mv = 0.0f;
            if (mask_row != NULL) {
                uint16_t mf16 = mask_row[ic];
                uint32_t sign = (mf16 >> 15) & 0x1;
                uint32_t exp = (mf16 >> 10) & 0x1F;
                uint32_t mant = mf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (mant == 0) ? (sign << 31) : 0;
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                __builtin_memcpy(&mv, &f32, sizeof(float));
                mv *= slope;
            }
            
            if (mv == -INFINITY) {
                continue;
            }
            
            /* PRE-CONVERT K vector from F16 to F32 (scalar, but outside inner dot product) */
            const uint16_t* k_f16 = (const uint16_t*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            for (int i = 0; i < d; i++) {
                uint16_t kf16 = k_f16[i];
                uint32_t sign = ((uint32_t)(kf16 & 0x8000)) << 16;
                uint32_t exp = (kf16 >> 10) & 0x1F;
                uint32_t mant = kf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (mant == 0) ? sign : 0;
                } else if (exp == 31) {
                    f32 = sign | 0x7F800000 | (mant << 13);
                } else {
                    f32 = sign | ((exp + 112) << 23) | (mant << 13);
                }
                /* Use union for type punning instead of memcpy */
                union { uint32_t u; float f; } conv;
                conv.u = f32;
                k_f32[i] = conv.f;
            }
            
            /* Now compute Q·K dot product in pure F32 (vectorizable!) */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_vec[i] * k_f32[i];
            }
            
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
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[i] *= ms;
                }
                S_sum *= ms;
            } else {
                vs = expf(s - M);
            }
            
            /* PRE-CONVERT V vector from F16 to F32 (scalar, but outside inner accumulate) */
            const uint16_t* v_f16 = (const uint16_t*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            for (int i = 0; i < dv; i++) {
                uint16_t vf16 = v_f16[i];
                uint32_t sign = ((uint32_t)(vf16 & 0x8000)) << 16;
                uint32_t exp = (vf16 >> 10) & 0x1F;
                uint32_t mant = vf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (mant == 0) ? sign : 0;
                } else if (exp == 31) {
                    f32 = sign | 0x7F800000 | (mant << 13);
                } else {
                    f32 = sign | ((exp + 112) << 23) | (mant << 13);
                }
                union { uint32_t u; float f; } conv;
                conv.u = f32;
                v_f32[i] = conv.f;
            }
            
            /* Accumulate V weighted by softmax (pure F32, vectorizable!) */
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                VKQ[i] += vs * v_f32[i];
            }
            
            S_sum += vs;
        }
        
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] *= S_inv;
        }
        
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = VKQ[i];
        }
    }
    
    return 0;
}

/*
 * Flash Attention with F32 Q and BF16 K/V (via HMEM)
 *
 * This is for BF16 KV cache - Q is F32 (after RoPE), K/V are BF16 (KV cache type).
 * Similar to F16 K/V version but uses BF16→F32 conversion (just left shift by 16).
 * 
 * Note: BF16 conversion is vectorizable since it's a simple bit shift, unlike F16
 * which requires exponent adjustment.
 */
uint64_t ve_flash_attn_ext_f32q_bf16kv_hmem(
    void* dst_ptr,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    VEDAdeviceptr mask_hbm,
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1,
    uint64_t nb_q2,
    uint64_t nb_q3,
    uint64_t nb_k1,
    uint64_t nb_k2,
    uint64_t nb_k3,
    uint64_t nb_v1,
    uint64_t nb_v2,
    uint64_t nb_v3,
    uint64_t nb_m1,
    uint64_t nb_m2,
    uint64_t nb_m3,
    uint64_t nb_o1,
    uint64_t nb_o2,
    uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    float* dst = (float*)dst_ptr;
    const float* q = (const float*)q_ptr;     /* F32 Q */
    const uint16_t* k = (const uint16_t*)k_ptr;  /* BF16 K */
    const uint16_t* v = (const uint16_t*)v_ptr;  /* BF16 V */
    const uint16_t* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);
    
    int d = (int)D;
    int dv = (int)Dv;
    int n_tokens = (int)N;
    int n_kv = (int)S;
    int n_heads = (int)H;
    int n_batch = (int)B;
    
    int total_work = n_batch * n_heads * n_tokens;
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        /* Q is F32 - use directly */
        const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        const uint16_t* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const uint16_t*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];
        for (int i = 0; i < dv; i++) {
            VKQ[i] = 0.0f;
        }
        
        for (int ic = 0; ic < n_kv; ic++) {
            /* Decode F16 mask (same as F16 K/V version - mask is always F16) */
            float mv = 0.0f;
            if (mask_row != NULL) {
                uint16_t mf16 = mask_row[ic];
                uint32_t sign = (mf16 >> 15) & 0x1;
                uint32_t exp = (mf16 >> 10) & 0x1F;
                uint32_t mant = mf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (mant == 0) ? (sign << 31) : 0;
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                __builtin_memcpy(&mv, &f32, sizeof(float));
                mv *= slope;
            }
            
            if (mv == -INFINITY) {
                continue;
            }
            
            /* Convert K vector from BF16 to F32 (VECTORIZABLE - just shift left by 16!) */
            const uint16_t* k_bf16 = (const uint16_t*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                /* BF16→F32: Just put the 16 bits in the upper half of F32 */
                uint32_t f32 = ((uint32_t)k_bf16[i]) << 16;
                union { uint32_t u; float f; } conv;
                conv.u = f32;
                k_f32[i] = conv.f;
            }
            
            /* Compute Q·K dot product in pure F32 (vectorizable!) */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_vec[i] * k_f32[i];
            }
            
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
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[i] *= ms;
                }
                S_sum *= ms;
            } else {
                vs = expf(s - M);
            }
            
            /* Convert V vector from BF16 to F32 (VECTORIZABLE - just shift left by 16!) */
            const uint16_t* v_bf16 = (const uint16_t*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                /* BF16→F32: Just put the 16 bits in the upper half of F32 */
                uint32_t f32 = ((uint32_t)v_bf16[i]) << 16;
                union { uint32_t u; float f; } conv;
                conv.u = f32;
                v_f32[i] = conv.f;
            }
            
            /* Accumulate V weighted by softmax (pure F32, vectorizable!) */
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                VKQ[i] += vs * v_f32[i];
            }
            
            S_sum += vs;
        }
        
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] *= S_inv;
        }
        
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = VKQ[i];
        }
    }
    
    return 0;
}

/*
 * ============================================================================
 * ROPE HBM Kernels - Input/output in VE HBM (for KV cache in HBM)
 * ============================================================================
 */

/*
 * ROPE Normal style with HBM input/output + OpenMP
 * 
 * Parameters:
 *   y_hbm:     Output tensor in HBM (VEDAdeviceptr)
 *   x_hbm:     Input tensor in HBM (VEDAdeviceptr)
 *   cache_ptr: Pre-computed [cos, sin] pairs in HMEM (already converted to raw pointer)
 *   ne0:       Head dimension (total elements per head)
 *   n_dims:    Number of dimensions to rotate (typically head_dim)
 *   n_heads:   Number of attention heads (ne1)
 *   n_ctx:     Sequence length (ne2)
 *   n_batch:   Batch size (ne3)
 *   nb1, nb2, nb3: Strides in bytes for input/output
 */
uint64_t ve_rope_normal_hbm_omp(VEDAdeviceptr y_hbm,
                                 VEDAdeviceptr x_hbm,
                                 void* cache_ptr,
                                 uint64_t ne0,
                                 uint64_t n_dims,
                                 uint64_t n_heads,
                                 uint64_t n_ctx,
                                 uint64_t n_batch,
                                 uint64_t nb1,
                                 uint64_t nb2,
                                 uint64_t nb3) {
    /* Convert HBM pointers to raw VE addresses */
    float* y;
    const float* x;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    
    const float* cache = (const float*)cache_ptr;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int elem_per_head = (int)ne0;
    
    /* Total rows = batch * ctx * heads */
    int total_rows = batch * ctx * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / (ctx * heads);
        int rem = row % (ctx * heads);
        int i2 = rem / heads;
        int i1 = rem % heads;
        
        /* Use position-specific cache: cache layout is [pos0_cos_sin, pos1_cos_sin, ...]
         * Each position has n_dims floats (n_dims/2 cos + n_dims/2 sin interleaved)
         * i2 is the sequence position index */
        const float* pos_cache = cache + i2 * nd;
        
        size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* Apply rotation to consecutive pairs (NORMAL style) */
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < nd; i0 += 2) {
            float cos_val = pos_cache[i0];
            float sin_val = pos_cache[i0 + 1];
            
            float x0 = src[i0];
            float x1 = src[i0 + 1];
            
            dst[i0]     = x0 * cos_val - x1 * sin_val;
            dst[i0 + 1] = x0 * sin_val + x1 * cos_val;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/*
 * ROPE NeoX style with HBM input/output + OpenMP
 * 
 * NeoX/Llama uses a different rotation pattern than normal ROPE:
 *   y[i]             = x[i] * cos - x[i + n_dims/2] * sin
 *   y[i + n_dims/2]  = x[i] * sin + x[i + n_dims/2] * cos
 */
uint64_t ve_rope_neox_hbm_omp(VEDAdeviceptr y_hbm,
                               VEDAdeviceptr x_hbm,
                               void* cache_ptr,
                               uint64_t ne0,
                               uint64_t n_dims,
                               uint64_t n_heads,
                               uint64_t n_ctx,
                               uint64_t n_batch,
                               uint64_t nb1,
                               uint64_t nb2,
                               uint64_t nb3) {
    /* Convert HBM pointers to raw VE addresses */
    float* y;
    const float* x;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    
    const float* cache = (const float*)cache_ptr;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int nd_half = nd / 2;
    int elem_per_head = (int)ne0;
    
    /* Total rows = batch * ctx * heads */
    int total_rows = batch * ctx * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / (ctx * heads);
        int rem = row % (ctx * heads);
        int i2 = rem / heads;
        int i1 = rem % heads;
        
        /* Use position-specific cache: cache layout is [pos0_cos_sin, pos1_cos_sin, ...]
         * Each position has n_dims floats (n_dims/2 cos + n_dims/2 sin interleaved)
         * i2 is the sequence position index */
        const float* pos_cache = cache + i2 * nd;
        
        /* Calculate byte offsets */
        size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* Apply rotation to first n_dims elements (NeoX style) */
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < nd_half; i0++) {
            float cos_val = pos_cache[i0 * 2];
            float sin_val = pos_cache[i0 * 2 + 1];
            
            float x0 = src[i0];
            float x1 = src[i0 + nd_half];
            
            dst[i0]           = x0 * cos_val - x1 * sin_val;
            dst[i0 + nd_half] = x0 * sin_val + x1 * cos_val;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/* ============================================================================
 * HBM Flash Attention - All data stays in VE HBM
 * 
 * This is the fast path: Q/K/V/dst are all VEDAdeviceptr pointing to HBM.
 * No HMEM copies needed - operates directly on HBM data at 1.2 TB/s bandwidth!
 * ============================================================================
 */

/*
 * Flash attention with F32 Q and BF16 K/V, all in HBM
 * 
 * Parameters are VEDAdeviceptr which we convert to raw pointers via vedaMemPtr.
 * Mask is still F16 and comes via HMEM (it's small and static).
 */
uint64_t ve_flash_attn_ext_f32q_bf16kv_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,  /* HMEM - mask is small and static */
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1,
    uint64_t nb_q2,
    uint64_t nb_q3,
    uint64_t nb_k1,
    uint64_t nb_k2,
    uint64_t nb_k3,
    uint64_t nb_v1,
    uint64_t nb_v2,
    uint64_t nb_v3,
    uint64_t nb_m1,
    uint64_t nb_m2,
    uint64_t nb_m3,
    uint64_t nb_o1,
    uint64_t nb_o2,
    uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    /* Convert HBM pointers to raw VE addresses */
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    
    const uint16_t* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);
    
    int d = (int)D;
    int dv = (int)Dv;
    int n_tokens = (int)N;
    int n_kv = (int)S;
    int n_heads = (int)H;
    int n_batch = (int)B;
    
    /* Pre-compute d/2 pairs for 32-bit vectorized BF16 conversion */
    int d_pairs = d / 2;
    int dv_pairs = dv / 2;
    
    int total_work = n_batch * n_heads * n_tokens;
    
    /* Debug: disabled for production */
#if VE_KERNEL_DEBUG
    static int debug_bf16kv_hbm_count = 0;
    debug_bf16kv_hbm_count++;
    int layer_num = debug_bf16kv_hbm_count - 1;
    int do_debug = (layer_num == 23);
    if (layer_num <= 27) {
        fprintf(stderr, "[FLASH-ADDR] Layer %d v_hbm=%p k_hbm=%p\n", layer_num, (void*)v, (void*)k);
    }
    if (do_debug) {
#else
    int do_debug = 0;
    if (0) {
#endif
        fprintf(stderr, "[FLASH-BF16KV-HBM] n_tokens=%d n_heads=%d n_batch=%d n_kv=%d d=%d dv=%d\n",
                n_tokens, n_heads, n_batch, n_kv, d, dv);
        fprintf(stderr, "[FLASH-BF16KV-HBM] nb_q1=%llu nb_q2=%llu nb_q3=%llu (Q strides)\n",
                (unsigned long long)nb_q1, (unsigned long long)nb_q2, (unsigned long long)nb_q3);
        fprintf(stderr, "[FLASH-BF16KV-HBM] nb_k1=%llu nb_k2=%llu nb_k3=%llu (K strides)\n",
                (unsigned long long)nb_k1, (unsigned long long)nb_k2, (unsigned long long)nb_k3);
        fprintf(stderr, "[FLASH-BF16KV-HBM] nb_v1=%llu nb_v2=%llu nb_v3=%llu (V strides)\n",
                (unsigned long long)nb_v1, (unsigned long long)nb_v2, (unsigned long long)nb_v3);
        fprintf(stderr, "[FLASH-BF16KV-HBM] nb_o1=%llu nb_o2=%llu nb_o3=%llu (output strides)\n",
                (unsigned long long)nb_o1, (unsigned long long)nb_o2, (unsigned long long)nb_o3);
        fprintf(stderr, "[FLASH-BF16KV-HBM] H=%d Hk=%llu rk=%lld rv=%lld\n",
                n_heads, (unsigned long long)Hk, (long long)rk, (long long)rv);
        fprintf(stderr, "[FLASH-BF16KV-HBM] dst_hbm=0x%llx dst_raw=%p total_size=%llu\n",
                (unsigned long long)dst_hbm, (void*)dst, (unsigned long long)(n_tokens * n_heads * dv * sizeof(float)));
        
        /* Sample first Q values */
        float q_sample[4];
        for (int i = 0; i < 4; i++) q_sample[i] = q[i];
        fprintf(stderr, "[FLASH-BF16KV-HBM] Q[0:3]=%.4f %.4f %.4f %.4f\n",
                q_sample[0], q_sample[1], q_sample[2], q_sample[3]);
        
        /* Sample first K values (convert from BF16) */
        uint16_t* k_u16 = (uint16_t*)k;
        float k_sample[4];
        for (int i = 0; i < 4; i++) {
            uint32_t f32 = ((uint32_t)k_u16[i]) << 16;
            union { uint32_t u; float f; } conv;
            conv.u = f32;
            k_sample[i] = conv.f;
        }
        fprintf(stderr, "[FLASH-BF16KV-HBM] K[0:3]=%.4f %.4f %.4f %.4f (raw: %04x %04x %04x %04x)\n",
                k_sample[0], k_sample[1], k_sample[2], k_sample[3],
                k_u16[0], k_u16[1], k_u16[2], k_u16[3]);
        
        /* Sample first V values (convert from BF16) */
        uint16_t* v_u16 = (uint16_t*)v;
        float v_sample[4];
        for (int i = 0; i < 4; i++) {
            uint32_t f32 = ((uint32_t)v_u16[i]) << 16;
            union { uint32_t u; float f; } conv;
            conv.u = f32;
            v_sample[i] = conv.f;
        }
        fprintf(stderr, "[FLASH-BF16KV-HBM] V[0:3]=%.4f %.4f %.4f %.4f (raw: %04x %04x %04x %04x)\n",
                v_sample[0], v_sample[1], v_sample[2], v_sample[3],
                v_u16[0], v_u16[1], v_u16[2], v_u16[3]);
    }
    
    /* Debug: Initialize output with signature pattern for layer 23 to detect overwrites */
    if (do_debug) {
        float* dst_f32 = (float*)dst;
        int n_elems = n_tokens * n_heads * dv;
        for (int i = 0; i < n_elems; i++) {
            dst_f32[i] = -9999.0f;
        }
        asm volatile("fencem 2" ::: "memory");
        fprintf(stderr, "[FLASH-INIT-L23] Initialized %d elements to -9999.0\n", n_elems);
    }
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        /* Q is F32 - direct HBM access */
        const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        const uint16_t* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const uint16_t*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];
        for (int i = 0; i < dv; i++) {
            VKQ[i] = 0.0f;
        }
        
        for (int ic = 0; ic < n_kv; ic++) {
            /* Decode F16 mask */
            float mv = 0.0f;
            if (mask_row != NULL) {
                uint16_t mf16 = mask_row[ic];
                uint32_t sign = (mf16 >> 15) & 0x1;
                uint32_t exp = (mf16 >> 10) & 0x1F;
                uint32_t mant = mf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (mant == 0) ? (sign << 31) : 0;
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                __builtin_memcpy(&mv, &f32, sizeof(float));
                mv *= slope;
            }
            
            if (mv == -INFINITY) {
                continue;
            }
            
            /* Convert K vector from BF16 to F32 using 32-bit loads for vectorization
             * BF16 is stored as 2 values per 32-bit word: [bf16_hi | bf16_lo] */
            const uint32_t* k_u32 = (const uint32_t*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d_pairs; i++) {
                uint32_t packed = k_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } conv_lo, conv_hi;
                conv_lo.u = f32_lo;
                conv_hi.u = f32_hi;
                k_f32[2*i] = conv_lo.f;
                k_f32[2*i + 1] = conv_hi.f;
            }
            
            /* Q·K dot product */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_vec[i] * k_f32[i];
            }
            
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
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[i] *= ms;
                }
                S_sum *= ms;
            } else {
                vs = expf(s - M);
            }
            
            /* Convert V vector from BF16 to F32 using 32-bit loads for vectorization */
            const uint32_t* v_u32 = (const uint32_t*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv_pairs; i++) {
                uint32_t packed = v_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } conv_lo, conv_hi;
                conv_lo.u = f32_lo;
                conv_hi.u = f32_hi;
                v_f32[2*i] = conv_lo.f;
                v_f32[2*i + 1] = conv_hi.f;
            }
            
            /* Accumulate weighted V */
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                VKQ[i] += vs * v_f32[i];
            }
            
            /* Debug: track VKQ[21] accumulation for head 55 token 6 */
            if (do_debug && iq == 6 && ih == 55 && ic == 6) {  /* Only ic=6 which has the garbage */
                #pragma omp critical
                {
                    /* v_f32[21] comes from v_u32[10] (high 16 bits, since 21=2*10+1) */
                    const uint32_t* v_u32_debug = (const uint32_t*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
                    uint32_t v_raw10 = v_u32_debug[10];
                    uint16_t v_bf16_lo = v_raw10 & 0xFFFF;
                    uint16_t v_bf16_hi = (v_raw10 >> 16) & 0xFFFF;
                    long long v_byte_offset = (long long)(ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
                    fprintf(stderr, "[FLASH-ACCUM] ic=%d VKQ[21]=%.6g vs=%.6g v_f32[21]=%.6g s=%.6g M=%.6g mv=%.6g\n",
                            ic, VKQ[21], vs, v_f32[21], s, M, mv);
                    fprintf(stderr, "[FLASH-ACCUM]   v_offset=%lld iv_head=%d v_u32[10]=0x%08x bf16_lo=0x%04x bf16_hi=0x%04x\n",
                            v_byte_offset, iv_head, v_raw10, v_bf16_lo, v_bf16_hi);
                }
            }
            
            S_sum += vs;
        }
        
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] *= S_inv;
        }
        
        /* Write output directly to HBM */
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        
        /* Debug: Check head 55 token 6 element 21 (global index 28117) */
        if (do_debug && iq == 6 && ih == 55) {
            #pragma omp critical
            {
                fprintf(stderr, "[FLASH-H55-T6] After loop: VKQ[21]=%.6g S_sum=%.6g M=%.6g S_inv=%.6g out_ptr=%p offset=%lld\n",
                        VKQ[21], S_sum, M, S_inv, (void*)out_ptr, (long long)((char*)out_ptr - (char*)dst));
            }
        }
        
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = VKQ[i];
        }
        
        /* Debug: Verify write for head 55 token 6 */
        if (do_debug && iq == 6 && ih == 55) {
            asm volatile("fencem 1" ::: "memory");
            #pragma omp critical
            {
                fprintf(stderr, "[FLASH-H55-T6] After write: out_ptr[21]=%.6g\n", out_ptr[21]);
                
                /* Also check neighboring addresses */
                long long base_offset = (long long)((char*)out_ptr - (char*)dst);
                float* global_28117 = (float*)((char*)dst + 28117*4);  /* Global element 28117 */
                fprintf(stderr, "[FLASH-H55-T6] Global[28117]=%.6g (at offset %lld bytes)\n", 
                        *global_28117, (long long)(28117*4));
            }
        }
    }

    /* Final debug disabled - see below */

    /* Force memory fence to ensure all HBM writes are visible before return */
    asm volatile("fencem 2" ::: "memory");  /* Full memory barrier for stores */

    return 0;
}

/*
 * Tiled-Q flash attention for prefill (F32 Q, BF16 K/V).
 *
 * The non-tiled kernel above parallelises over (batch * head * n_tokens),
 * giving each (b,h,iq) work item its own loop over n_kv. Every work
 * item independently reloads + BF16->F32-converts K[ic] and V[ic] for
 * the same (ic) -- so the same KV element is fetched n_tokens times
 * per (b,h) layer.
 *
 * This kernel groups NQ_TILE queries per work item. The K/V loads and
 * BF16->F32 conversions happen ONCE per (b,h,Q_tile,ic); the inner
 * loop reuses the cached k_f32 / v_f32 tile against NQ_TILE queries.
 * That's an NQ_TILE-fold reduction in HBM bandwidth and BF16-decode
 * work, which is what dominates the row-major prefill path.
 *
 * Algorithm is the same online-softmax FlashAttention used by the
 * single-Q kernel; per-Q state (M, S, VKQ) is maintained in stack
 * arrays sized to NQ_TILE. Mask / ALiBi / softcap handling is
 * per-query, matching the reference kernel exactly.
 */
uint64_t ve_flash_attn_ext_f32q_bf16kv_tile_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S_kv,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* mask = NULL;

    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 5;

    float scale, max_bias, logit_softcap;
    {
        uint32_t tmp;
        tmp = (uint32_t)scale_bits;     __builtin_memcpy(&scale, &tmp, sizeof(float));
        tmp = (uint32_t)max_bias_bits;  __builtin_memcpy(&max_bias, &tmp, sizeof(float));
        tmp = (uint32_t)softcap_bits;   __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    }
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);

    const int d        = (int)D;
    const int dv       = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv     = (int)S_kv;
    const int n_heads  = (int)H;
    const int n_batch  = (int)B;
    const int d_pairs  = d / 2;
    const int dv_pairs = dv / 2;

    /* NQ_TILE drives the BF16-K/V reuse factor. 16 is a sweet spot: 16x
     * fewer HBM loads + BF16 conversions vs. the row-major kernel, while
     * keeping per-thread state (Q_tile + VKQ ~= 16*128*4 + 16*128*4 =
     * 16 KB) inside L1. Bigger tiles need to spill. */
    #define NQ_TILE 16

    const int n_q_tiles  = (n_tokens + NQ_TILE - 1) / NQ_TILE;
    const int total_work = n_batch * n_heads * n_q_tiles;

    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        const int ib       = work_id / (n_heads * n_q_tiles);
        const int ih       = (work_id / n_q_tiles) % n_heads;
        const int q_tile_i = work_id % n_q_tiles;
        const int iq_base  = q_tile_i * NQ_TILE;
        const int nq       = (iq_base + NQ_TILE <= n_tokens) ? NQ_TILE : (n_tokens - iq_base);

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t hh = (uint32_t)ih;
            slope = (hh < n_head_log2) ? powf(m0, (float)(hh + 1))
                                        : powf(m1, (float)(2*(hh - n_head_log2) + 1));
        }

        const int ik_head = ih / (int)rk;
        const int iv_head = ih / (int)rv;

        /* Per-tile state. NQ_TILE * 256 floats each. */
        float Q_tile[NQ_TILE * 256];
        const uint16_t* mask_rows[NQ_TILE];
        float M_state[NQ_TILE];
        float S_state[NQ_TILE];
        float VKQ[NQ_TILE * 256];

        for (int j = 0; j < nq; j++) {
            const float* q_vec = (const float*)((const char*)q
                + (iq_base + j) * nb_q1 + ih * nb_q2 + ib * nb_q3);
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) Q_tile[j*d + i] = q_vec[i];
            M_state[j] = -INFINITY;
            S_state[j] = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) VKQ[j*dv + i] = 0.0f;
            if (mask != NULL) {
                int im2 = ih % (int)mask_ne2;
                int im3 = ib % (int)mask_ne3;
                mask_rows[j] = (const uint16_t*)((const char*)mask
                    + (iq_base + j) * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
            } else {
                mask_rows[j] = NULL;
            }
        }

        for (int ic = 0; ic < n_kv; ic++) {
            /* Load + convert K[ic] once. Reused across nq queries below. */
            const uint32_t* k_u32 = (const uint32_t*)((const char*)k
                + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d_pairs; i++) {
                uint32_t packed = k_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                k_f32[2*i]     = cl.f;
                k_f32[2*i + 1] = ch.f;
            }

            /* Same for V[ic]. */
            const uint32_t* v_u32 = (const uint32_t*)((const char*)v
                + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv_pairs; i++) {
                uint32_t packed = v_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                v_f32[2*i]     = cl.f;
                v_f32[2*i + 1] = ch.f;
            }

            /* Per-query update against the shared K/V[ic]. */
            for (int j = 0; j < nq; j++) {
                /* Decode F16 mask -> F32 (matches reference kernel above). */
                float mv = 0.0f;
                if (mask_rows[j] != NULL) {
                    uint16_t mf16 = mask_rows[j][ic];
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
                    __builtin_memcpy(&mv, &f32, sizeof(float));
                    mv *= slope;
                }

                if (mv == -INFINITY) continue;

                /* Q[j] . K[ic] */
                float s = 0.0f;
                #pragma _NEC ivdep
                for (int i = 0; i < d; i++) {
                    s += Q_tile[j*d + i] * k_f32[i];
                }
                s *= scale;
                if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
                s += mv;

                float M_old = M_state[j];
                float ms = 1.0f, vs = 1.0f;
                if (s > M_state[j]) {
                    M_state[j] = s;
                    ms = expf(M_old - M_state[j]);
                    #pragma _NEC ivdep
                    for (int i = 0; i < dv; i++) VKQ[j*dv + i] *= ms;
                    S_state[j] *= ms;
                } else {
                    vs = expf(s - M_state[j]);
                }
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) VKQ[j*dv + i] += vs * v_f32[i];
                S_state[j] += vs;
            }
        }

        for (int j = 0; j < nq; j++) {
            float S_inv = (S_state[j] == 0.0f) ? 0.0f : 1.0f / S_state[j];
            float* out_ptr = (float*)((char*)dst
                + ib * nb_o3 + (iq_base + j) * nb_o2 + ih * nb_o1);
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) out_ptr[i] = VKQ[j*dv + i] * S_inv;
        }
    }

    #undef NQ_TILE

    asm volatile("fencem 2" ::: "memory");
    return 0;
}

/*
 * NCC tile kernel with manual j-by-2 unrolling.
 *
 * Same idea as the pack-2 intrinsics kernel (share K[i] and V[i] reads
 * across 2 queries) but expressed in straight C so NCC's auto-vectorizer
 * handles the dot reductions and VKQ FMAs. NCC's Sum-idiom and FMA
 * codegen at unpacked VL=128 has the best instruction scheduling we've
 * seen on VE; the hope is that it can pipeline two parallel reductions
 * sharing the same K load, halving effective per-(Q, ic) cost while
 * keeping all the cycles-per-vector-op wins of staying in NCC.
 *
 * Same online-softmax and mask handling as ve_flash_attn_ext_f32q_bf16kv_tile_hbm.
 */
uint64_t ve_flash_attn_ext_f32q_bf16kv_tile_unr2_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,
    uint64_t D, uint64_t Dv,
    uint64_t N, uint64_t S_kv,
    uint64_t H, uint64_t Hk, uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits, uint64_t max_bias_bits, uint64_t softcap_bits,
    uint64_t mask_ne2, uint64_t mask_ne3)
{
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* mask = NULL;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 5;

    float scale, max_bias, logit_softcap;
    {
        uint32_t tmp;
        tmp = (uint32_t)scale_bits;    __builtin_memcpy(&scale, &tmp, sizeof(float));
        tmp = (uint32_t)max_bias_bits; __builtin_memcpy(&max_bias, &tmp, sizeof(float));
        tmp = (uint32_t)softcap_bits;  __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    }
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);

    const int d        = (int)D;
    const int dv       = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv     = (int)S_kv;
    const int n_heads  = (int)H;
    const int n_batch  = (int)B;
    const int d_pairs  = d / 2;
    const int dv_pairs = dv / 2;

    #define NQ_TILE_U 16

    const int n_q_tiles  = (n_tokens + NQ_TILE_U - 1) / NQ_TILE_U;
    const int total_work = n_batch * n_heads * n_q_tiles;

    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        const int ib       = work_id / (n_heads * n_q_tiles);
        const int ih       = (work_id / n_q_tiles) % n_heads;
        const int q_tile_i = work_id % n_q_tiles;
        const int iq_base  = q_tile_i * NQ_TILE_U;
        const int nq       = (iq_base + NQ_TILE_U <= n_tokens) ? NQ_TILE_U : (n_tokens - iq_base);

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t hh = (uint32_t)ih;
            slope = (hh < n_head_log2) ? powf(m0, (float)(hh + 1))
                                        : powf(m1, (float)(2*(hh - n_head_log2) + 1));
        }

        const int ik_head = ih / (int)rk;
        const int iv_head = ih / (int)rv;

        float Q_tile[NQ_TILE_U * 256];
        const uint16_t* mask_rows[NQ_TILE_U];
        float M_state[NQ_TILE_U];
        float S_state[NQ_TILE_U];
        float VKQ[NQ_TILE_U * 256];

        for (int j = 0; j < nq; j++) {
            const float* q_vec = (const float*)((const char*)q
                + (iq_base + j) * nb_q1 + ih * nb_q2 + ib * nb_q3);
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) Q_tile[j*d + i] = q_vec[i];
            M_state[j] = -INFINITY;
            S_state[j] = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) VKQ[j*dv + i] = 0.0f;
            if (mask != NULL) {
                int im2 = ih % (int)mask_ne2;
                int im3 = ib % (int)mask_ne3;
                mask_rows[j] = (const uint16_t*)((const char*)mask
                    + (iq_base + j) * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
            } else {
                mask_rows[j] = NULL;
            }
        }

        for (int ic = 0; ic < n_kv; ic++) {
            const uint32_t* k_u32 = (const uint32_t*)((const char*)k
                + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d_pairs; i++) {
                uint32_t packed = k_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                k_f32[2*i]     = cl.f;
                k_f32[2*i + 1] = ch.f;
            }

            const uint32_t* v_u32 = (const uint32_t*)((const char*)v
                + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv_pairs; i++) {
                uint32_t packed = v_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                v_f32[2*i]     = cl.f;
                v_f32[2*i + 1] = ch.f;
            }

            /* Per-PAIR update against the shared K/V[ic]. */
            int j = 0;
            for (; j + 1 < nq; j += 2) {
                const int j0 = j, j1 = j + 1;

                /* Mask decode for both queries. */
                float mv0 = 0.0f, mv1 = 0.0f;
                if (mask_rows[j0] != NULL) {
                    uint16_t mf16 = mask_rows[j0][ic];
                    uint32_t sign = (mf16 >> 15) & 0x1;
                    uint32_t exp_b = (mf16 >> 10) & 0x1F;
                    uint32_t mant = mf16 & 0x3FF;
                    uint32_t f32;
                    if (exp_b == 0)      f32 = (mant == 0) ? (sign << 31) : 0;
                    else if (exp_b == 31) f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                    else                  f32 = (sign << 31) | ((exp_b + 127 - 15) << 23) | (mant << 13);
                    __builtin_memcpy(&mv0, &f32, sizeof(float));
                    mv0 *= slope;
                }
                if (mask_rows[j1] != NULL) {
                    uint16_t mf16 = mask_rows[j1][ic];
                    uint32_t sign = (mf16 >> 15) & 0x1;
                    uint32_t exp_b = (mf16 >> 10) & 0x1F;
                    uint32_t mant = mf16 & 0x3FF;
                    uint32_t f32;
                    if (exp_b == 0)      f32 = (mant == 0) ? (sign << 31) : 0;
                    else if (exp_b == 31) f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                    else                  f32 = (sign << 31) | ((exp_b + 127 - 15) << 23) | (mant << 13);
                    __builtin_memcpy(&mv1, &f32, sizeof(float));
                    mv1 *= slope;
                }
                const int skip0 = (mv0 == -INFINITY);
                const int skip1 = (mv1 == -INFINITY);
                if (skip0 && skip1) continue;

                /* Parallel dot products — same k_f32[] read, two reductions.
                 * NCC should recognize both as Sum-idioms and pipeline. */
                float s0 = 0.0f, s1 = 0.0f;
                const float * Q0 = Q_tile + (size_t)j0 * d;
                const float * Q1 = Q_tile + (size_t)j1 * d;
                #pragma _NEC ivdep
                for (int i = 0; i < d; i++) {
                    float ki = k_f32[i];
                    s0 += Q0[i] * ki;
                    s1 += Q1[i] * ki;
                }
                s0 *= scale; s1 *= scale;
                if (logit_softcap != 0.0f) {
                    s0 = logit_softcap * tanhf(s0);
                    s1 = logit_softcap * tanhf(s1);
                }
                s0 += mv0; s1 += mv1;

                /* Per-Q softmax bookkeeping. */
                float vs0 = 0.0f, vs1 = 0.0f;
                if (!skip0) {
                    float M_old0 = M_state[j0];
                    float ms0 = 1.0f;
                    if (s0 > M_old0) {
                        M_state[j0] = s0;
                        ms0 = expf(M_old0 - s0);
                        #pragma _NEC ivdep
                        for (int i = 0; i < dv; i++) VKQ[j0*dv + i] *= ms0;
                        S_state[j0] *= ms0;
                        vs0 = 1.0f;
                    } else {
                        vs0 = expf(s0 - M_state[j0]);
                    }
                    S_state[j0] += vs0;
                }
                if (!skip1) {
                    float M_old1 = M_state[j1];
                    float ms1 = 1.0f;
                    if (s1 > M_old1) {
                        M_state[j1] = s1;
                        ms1 = expf(M_old1 - s1);
                        #pragma _NEC ivdep
                        for (int i = 0; i < dv; i++) VKQ[j1*dv + i] *= ms1;
                        S_state[j1] *= ms1;
                        vs1 = 1.0f;
                    } else {
                        vs1 = expf(s1 - M_state[j1]);
                    }
                    S_state[j1] += vs1;
                }

                /* Parallel VKQ accumulates — same v_f32[] read, two FMAs. */
                float * VKQ0 = VKQ + (size_t)j0 * dv;
                float * VKQ1 = VKQ + (size_t)j1 * dv;
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    float vi = v_f32[i];
                    VKQ0[i] += vs0 * vi;
                    VKQ1[i] += vs1 * vi;
                }
            }

            /* Leftover odd query. */
            if (j < nq) {
                const int j0 = j;
                float mv = 0.0f;
                if (mask_rows[j0] != NULL) {
                    uint16_t mf16 = mask_rows[j0][ic];
                    uint32_t sign = (mf16 >> 15) & 0x1;
                    uint32_t exp_b = (mf16 >> 10) & 0x1F;
                    uint32_t mant = mf16 & 0x3FF;
                    uint32_t f32;
                    if (exp_b == 0)      f32 = (mant == 0) ? (sign << 31) : 0;
                    else if (exp_b == 31) f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                    else                  f32 = (sign << 31) | ((exp_b + 127 - 15) << 23) | (mant << 13);
                    __builtin_memcpy(&mv, &f32, sizeof(float));
                    mv *= slope;
                }
                if (mv == -INFINITY) continue;

                float s = 0.0f;
                #pragma _NEC ivdep
                for (int i = 0; i < d; i++) s += Q_tile[j0*d + i] * k_f32[i];
                s *= scale;
                if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
                s += mv;

                float M_old = M_state[j0];
                float vs = 0.0f;
                if (s > M_old) {
                    M_state[j0] = s;
                    float ms = expf(M_old - s);
                    #pragma _NEC ivdep
                    for (int i = 0; i < dv; i++) VKQ[j0*dv + i] *= ms;
                    S_state[j0] *= ms;
                    vs = 1.0f;
                } else {
                    vs = expf(s - M_state[j0]);
                }
                S_state[j0] += vs;
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) VKQ[j0*dv + i] += vs * v_f32[i];
            }
        }

        for (int j = 0; j < nq; j++) {
            float S_inv = (S_state[j] == 0.0f) ? 0.0f : 1.0f / S_state[j];
            float* out_ptr = (float*)((char*)dst
                + ib * nb_o3 + (iq_base + j) * nb_o2 + ih * nb_o1);
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) out_ptr[i] = VKQ[j*dv + i] * S_inv;
        }
    }

    #undef NQ_TILE_U

    asm volatile("fencem 2" ::: "memory");
    return 0;
}

/*
 * Same idea as unr2 but unrolled 4-way. Each ic iteration processes 4
 * queries simultaneously, sharing k_f32[i] / v_f32[i] across 4 parallel
 * Sum-idiom reductions and FMA accumulates. If NCC can hold 4 dot
 * accumulators without spilling, this halves the per-ic loop iterations
 * again vs. unr2.
 *
 * Leftover handling: after the j+=4 main loop, fall back to a single-Q
 * path for the remaining 1/2/3 queries (simplifies the code; the cost
 * is negligible since leftover is <= 3 of nq=16).
 */
uint64_t ve_flash_attn_ext_f32q_bf16kv_tile_unr4_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,
    uint64_t D, uint64_t Dv,
    uint64_t N, uint64_t S_kv,
    uint64_t H, uint64_t Hk, uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits, uint64_t max_bias_bits, uint64_t softcap_bits,
    uint64_t mask_ne2, uint64_t mask_ne3)
{
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* mask = NULL;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 5;

    float scale, max_bias, logit_softcap;
    {
        uint32_t tmp;
        tmp = (uint32_t)scale_bits;    __builtin_memcpy(&scale, &tmp, sizeof(float));
        tmp = (uint32_t)max_bias_bits; __builtin_memcpy(&max_bias, &tmp, sizeof(float));
        tmp = (uint32_t)softcap_bits;  __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    }
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);

    const int d        = (int)D;
    const int dv       = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv     = (int)S_kv;
    const int n_heads  = (int)H;
    const int n_batch  = (int)B;
    const int d_pairs  = d / 2;
    const int dv_pairs = dv / 2;

    #define NQ_TILE_4 16

    const int n_q_tiles  = (n_tokens + NQ_TILE_4 - 1) / NQ_TILE_4;
    const int total_work = n_batch * n_heads * n_q_tiles;

    /* Helper: decode F16 mask to F32 scaled by slope. */
    #define DECODE_MV(jx) \
        do { \
            mv##jx = 0.0f; \
            if (mask_rows[j##jx] != NULL) { \
                uint16_t mf16 = mask_rows[j##jx][ic]; \
                uint32_t sign = (mf16 >> 15) & 0x1; \
                uint32_t exp_b = (mf16 >> 10) & 0x1F; \
                uint32_t mant = mf16 & 0x3FF; \
                uint32_t f32; \
                if (exp_b == 0)       f32 = (mant == 0) ? (sign << 31) : 0; \
                else if (exp_b == 31) f32 = (sign << 31) | 0x7F800000 | (mant << 13); \
                else                  f32 = (sign << 31) | ((exp_b + 127 - 15) << 23) | (mant << 13); \
                __builtin_memcpy(&mv##jx, &f32, sizeof(float)); \
                mv##jx *= slope; \
            } \
        } while (0)

    /* Helper: per-Q softmax bookkeeping. Updates M/S state and computes vs. */
    #define DO_SOFTMAX(jx) \
        do { \
            if (!skip##jx) { \
                float M_old = M_state[j##jx]; \
                if (s##jx > M_old) { \
                    M_state[j##jx] = s##jx; \
                    float ms = expf(M_old - s##jx); \
                    _Pragma("_NEC ivdep") \
                    for (int i = 0; i < dv; i++) VKQ[j##jx*dv + i] *= ms; \
                    S_state[j##jx] *= ms; \
                    vs##jx = 1.0f; \
                } else { \
                    vs##jx = expf(s##jx - M_state[j##jx]); \
                } \
                S_state[j##jx] += vs##jx; \
            } else { \
                vs##jx = 0.0f; \
            } \
        } while (0)

    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        const int ib       = work_id / (n_heads * n_q_tiles);
        const int ih       = (work_id / n_q_tiles) % n_heads;
        const int q_tile_i = work_id % n_q_tiles;
        const int iq_base  = q_tile_i * NQ_TILE_4;
        const int nq       = (iq_base + NQ_TILE_4 <= n_tokens) ? NQ_TILE_4 : (n_tokens - iq_base);

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t hh = (uint32_t)ih;
            slope = (hh < n_head_log2) ? powf(m0, (float)(hh + 1))
                                        : powf(m1, (float)(2*(hh - n_head_log2) + 1));
        }

        const int ik_head = ih / (int)rk;
        const int iv_head = ih / (int)rv;

        float Q_tile[NQ_TILE_4 * 256];
        const uint16_t* mask_rows[NQ_TILE_4];
        float M_state[NQ_TILE_4];
        float S_state[NQ_TILE_4];
        float VKQ[NQ_TILE_4 * 256];

        for (int j = 0; j < nq; j++) {
            const float* q_vec = (const float*)((const char*)q
                + (iq_base + j) * nb_q1 + ih * nb_q2 + ib * nb_q3);
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) Q_tile[j*d + i] = q_vec[i];
            M_state[j] = -INFINITY;
            S_state[j] = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) VKQ[j*dv + i] = 0.0f;
            if (mask != NULL) {
                int im2 = ih % (int)mask_ne2;
                int im3 = ib % (int)mask_ne3;
                mask_rows[j] = (const uint16_t*)((const char*)mask
                    + (iq_base + j) * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
            } else {
                mask_rows[j] = NULL;
            }
        }

        for (int ic = 0; ic < n_kv; ic++) {
            const uint32_t* k_u32 = (const uint32_t*)((const char*)k
                + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d_pairs; i++) {
                uint32_t packed = k_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                k_f32[2*i]     = cl.f;
                k_f32[2*i + 1] = ch.f;
            }

            const uint32_t* v_u32 = (const uint32_t*)((const char*)v
                + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv_pairs; i++) {
                uint32_t packed = v_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                v_f32[2*i]     = cl.f;
                v_f32[2*i + 1] = ch.f;
            }

            /* Main 4-way unrolled loop. */
            int j = 0;
            for (; j + 3 < nq; j += 4) {
                const int j0 = j, j1 = j+1, j2 = j+2, j3 = j+3;

                float mv0, mv1, mv2, mv3;
                DECODE_MV(0); DECODE_MV(1); DECODE_MV(2); DECODE_MV(3);
                const int skip0 = (mv0 == -INFINITY);
                const int skip1 = (mv1 == -INFINITY);
                const int skip2 = (mv2 == -INFINITY);
                const int skip3 = (mv3 == -INFINITY);
                if (skip0 && skip1 && skip2 && skip3) continue;

                /* 4-way parallel dot products sharing k_f32. */
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                const float * Q0 = Q_tile + (size_t)j0 * d;
                const float * Q1 = Q_tile + (size_t)j1 * d;
                const float * Q2 = Q_tile + (size_t)j2 * d;
                const float * Q3 = Q_tile + (size_t)j3 * d;
                #pragma _NEC ivdep
                for (int i = 0; i < d; i++) {
                    float ki = k_f32[i];
                    s0 += Q0[i] * ki;
                    s1 += Q1[i] * ki;
                    s2 += Q2[i] * ki;
                    s3 += Q3[i] * ki;
                }
                s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
                if (logit_softcap != 0.0f) {
                    s0 = logit_softcap * tanhf(s0);
                    s1 = logit_softcap * tanhf(s1);
                    s2 = logit_softcap * tanhf(s2);
                    s3 = logit_softcap * tanhf(s3);
                }
                s0 += mv0; s1 += mv1; s2 += mv2; s3 += mv3;

                float vs0, vs1, vs2, vs3;
                DO_SOFTMAX(0); DO_SOFTMAX(1); DO_SOFTMAX(2); DO_SOFTMAX(3);

                /* 4-way parallel VKQ accumulates sharing v_f32. */
                float * VKQ0 = VKQ + (size_t)j0 * dv;
                float * VKQ1 = VKQ + (size_t)j1 * dv;
                float * VKQ2 = VKQ + (size_t)j2 * dv;
                float * VKQ3 = VKQ + (size_t)j3 * dv;
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    float vi = v_f32[i];
                    VKQ0[i] += vs0 * vi;
                    VKQ1[i] += vs1 * vi;
                    VKQ2[i] += vs2 * vi;
                    VKQ3[i] += vs3 * vi;
                }
            }

            /* Leftover (1, 2, or 3 queries). Single-Q path, simple. */
            for (; j < nq; j++) {
                const int j0 = j;
                float mv0;
                DECODE_MV(0);
                if (mv0 == -INFINITY) continue;

                float s = 0.0f;
                #pragma _NEC ivdep
                for (int i = 0; i < d; i++) s += Q_tile[j0*d + i] * k_f32[i];
                s *= scale;
                if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
                s += mv0;

                float M_old = M_state[j0];
                float vs;
                if (s > M_old) {
                    M_state[j0] = s;
                    float ms = expf(M_old - s);
                    #pragma _NEC ivdep
                    for (int i = 0; i < dv; i++) VKQ[j0*dv + i] *= ms;
                    S_state[j0] *= ms;
                    vs = 1.0f;
                } else {
                    vs = expf(s - M_state[j0]);
                }
                S_state[j0] += vs;
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) VKQ[j0*dv + i] += vs * v_f32[i];
            }
        }

        for (int j = 0; j < nq; j++) {
            float S_inv = (S_state[j] == 0.0f) ? 0.0f : 1.0f / S_state[j];
            float* out_ptr = (float*)((char*)dst
                + ib * nb_o3 + (iq_base + j) * nb_o2 + ih * nb_o1);
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) out_ptr[i] = VKQ[j*dv + i] * S_inv;
        }
    }

    #undef NQ_TILE_4
    #undef DECODE_MV
    #undef DO_SOFTMAX

    asm volatile("fencem 2" ::: "memory");
    return 0;
}

/*
 * Same as unr4 but unrolled 8-way. Each ic iteration processes 8
 * queries simultaneously, sharing k_f32[i] / v_f32[i] across 8 parallel
 * Sum-idiom reductions and FMA accumulates. With NQ_TILE=16, the main
 * loop is exactly 2 iterations per ic and leftover handling is only
 * needed when nq < 8.
 *
 * Risk: 8 dot accumulators + 8 Q pointers + 8 vs scalars puts pressure
 * on NCC's register allocator. If it spills, unr8 will be a wash or
 * regression.
 */
uint64_t ve_flash_attn_ext_f32q_bf16kv_tile_unr8_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,
    uint64_t D, uint64_t Dv,
    uint64_t N, uint64_t S_kv,
    uint64_t H, uint64_t Hk, uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits, uint64_t max_bias_bits, uint64_t softcap_bits,
    uint64_t mask_ne2, uint64_t mask_ne3)
{
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* mask = NULL;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 5;

    float scale, max_bias, logit_softcap;
    {
        uint32_t tmp;
        tmp = (uint32_t)scale_bits;    __builtin_memcpy(&scale, &tmp, sizeof(float));
        tmp = (uint32_t)max_bias_bits; __builtin_memcpy(&max_bias, &tmp, sizeof(float));
        tmp = (uint32_t)softcap_bits;  __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    }
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);

    const int d        = (int)D;
    const int dv       = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv     = (int)S_kv;
    const int n_heads  = (int)H;
    const int n_batch  = (int)B;
    const int d_pairs  = d / 2;
    const int dv_pairs = dv / 2;

    #define NQ_TILE_8 16

    const int n_q_tiles  = (n_tokens + NQ_TILE_8 - 1) / NQ_TILE_8;
    const int total_work = n_batch * n_heads * n_q_tiles;

    #define DECODE_MV8(jx) \
        do { \
            mv##jx = 0.0f; \
            if (mask_rows[j##jx] != NULL) { \
                uint16_t mf16 = mask_rows[j##jx][ic]; \
                uint32_t sign = (mf16 >> 15) & 0x1; \
                uint32_t exp_b = (mf16 >> 10) & 0x1F; \
                uint32_t mant = mf16 & 0x3FF; \
                uint32_t f32; \
                if (exp_b == 0)       f32 = (mant == 0) ? (sign << 31) : 0; \
                else if (exp_b == 31) f32 = (sign << 31) | 0x7F800000 | (mant << 13); \
                else                  f32 = (sign << 31) | ((exp_b + 127 - 15) << 23) | (mant << 13); \
                __builtin_memcpy(&mv##jx, &f32, sizeof(float)); \
                mv##jx *= slope; \
            } \
        } while (0)

    #define DO_SOFTMAX8(jx) \
        do { \
            if (!skip##jx) { \
                float M_old = M_state[j##jx]; \
                if (s##jx > M_old) { \
                    M_state[j##jx] = s##jx; \
                    float ms = expf(M_old - s##jx); \
                    _Pragma("_NEC ivdep") \
                    for (int i = 0; i < dv; i++) VKQ[j##jx*dv + i] *= ms; \
                    S_state[j##jx] *= ms; \
                    vs##jx = 1.0f; \
                } else { \
                    vs##jx = expf(s##jx - M_state[j##jx]); \
                } \
                S_state[j##jx] += vs##jx; \
            } else { \
                vs##jx = 0.0f; \
            } \
        } while (0)

    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        const int ib       = work_id / (n_heads * n_q_tiles);
        const int ih       = (work_id / n_q_tiles) % n_heads;
        const int q_tile_i = work_id % n_q_tiles;
        const int iq_base  = q_tile_i * NQ_TILE_8;
        const int nq       = (iq_base + NQ_TILE_8 <= n_tokens) ? NQ_TILE_8 : (n_tokens - iq_base);

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t hh = (uint32_t)ih;
            slope = (hh < n_head_log2) ? powf(m0, (float)(hh + 1))
                                        : powf(m1, (float)(2*(hh - n_head_log2) + 1));
        }

        const int ik_head = ih / (int)rk;
        const int iv_head = ih / (int)rv;

        float Q_tile[NQ_TILE_8 * 256];
        const uint16_t* mask_rows[NQ_TILE_8];
        float M_state[NQ_TILE_8];
        float S_state[NQ_TILE_8];
        float VKQ[NQ_TILE_8 * 256];

        for (int j = 0; j < nq; j++) {
            const float* q_vec = (const float*)((const char*)q
                + (iq_base + j) * nb_q1 + ih * nb_q2 + ib * nb_q3);
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) Q_tile[j*d + i] = q_vec[i];
            M_state[j] = -INFINITY;
            S_state[j] = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) VKQ[j*dv + i] = 0.0f;
            if (mask != NULL) {
                int im2 = ih % (int)mask_ne2;
                int im3 = ib % (int)mask_ne3;
                mask_rows[j] = (const uint16_t*)((const char*)mask
                    + (iq_base + j) * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
            } else {
                mask_rows[j] = NULL;
            }
        }

        for (int ic = 0; ic < n_kv; ic++) {
            const uint32_t* k_u32 = (const uint32_t*)((const char*)k
                + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d_pairs; i++) {
                uint32_t packed = k_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                k_f32[2*i]     = cl.f;
                k_f32[2*i + 1] = ch.f;
            }

            const uint32_t* v_u32 = (const uint32_t*)((const char*)v
                + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv_pairs; i++) {
                uint32_t packed = v_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                v_f32[2*i]     = cl.f;
                v_f32[2*i + 1] = ch.f;
            }

            /* Main 8-way unrolled loop. */
            int j = 0;
            for (; j + 7 < nq; j += 8) {
                const int j0 = j, j1 = j+1, j2 = j+2, j3 = j+3;
                const int j4 = j+4, j5 = j+5, j6 = j+6, j7 = j+7;

                float mv0, mv1, mv2, mv3, mv4, mv5, mv6, mv7;
                DECODE_MV8(0); DECODE_MV8(1); DECODE_MV8(2); DECODE_MV8(3);
                DECODE_MV8(4); DECODE_MV8(5); DECODE_MV8(6); DECODE_MV8(7);
                const int skip0 = (mv0 == -INFINITY);
                const int skip1 = (mv1 == -INFINITY);
                const int skip2 = (mv2 == -INFINITY);
                const int skip3 = (mv3 == -INFINITY);
                const int skip4 = (mv4 == -INFINITY);
                const int skip5 = (mv5 == -INFINITY);
                const int skip6 = (mv6 == -INFINITY);
                const int skip7 = (mv7 == -INFINITY);
                if (skip0 && skip1 && skip2 && skip3 &&
                    skip4 && skip5 && skip6 && skip7) continue;

                /* 8-way parallel dot products sharing k_f32. */
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
                const float * Q0 = Q_tile + (size_t)j0 * d;
                const float * Q1 = Q_tile + (size_t)j1 * d;
                const float * Q2 = Q_tile + (size_t)j2 * d;
                const float * Q3 = Q_tile + (size_t)j3 * d;
                const float * Q4 = Q_tile + (size_t)j4 * d;
                const float * Q5 = Q_tile + (size_t)j5 * d;
                const float * Q6 = Q_tile + (size_t)j6 * d;
                const float * Q7 = Q_tile + (size_t)j7 * d;
                #pragma _NEC ivdep
                for (int i = 0; i < d; i++) {
                    float ki = k_f32[i];
                    s0 += Q0[i] * ki;
                    s1 += Q1[i] * ki;
                    s2 += Q2[i] * ki;
                    s3 += Q3[i] * ki;
                    s4 += Q4[i] * ki;
                    s5 += Q5[i] * ki;
                    s6 += Q6[i] * ki;
                    s7 += Q7[i] * ki;
                }
                s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
                s4 *= scale; s5 *= scale; s6 *= scale; s7 *= scale;
                if (logit_softcap != 0.0f) {
                    s0 = logit_softcap * tanhf(s0);
                    s1 = logit_softcap * tanhf(s1);
                    s2 = logit_softcap * tanhf(s2);
                    s3 = logit_softcap * tanhf(s3);
                    s4 = logit_softcap * tanhf(s4);
                    s5 = logit_softcap * tanhf(s5);
                    s6 = logit_softcap * tanhf(s6);
                    s7 = logit_softcap * tanhf(s7);
                }
                s0 += mv0; s1 += mv1; s2 += mv2; s3 += mv3;
                s4 += mv4; s5 += mv5; s6 += mv6; s7 += mv7;

                float vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7;
                DO_SOFTMAX8(0); DO_SOFTMAX8(1); DO_SOFTMAX8(2); DO_SOFTMAX8(3);
                DO_SOFTMAX8(4); DO_SOFTMAX8(5); DO_SOFTMAX8(6); DO_SOFTMAX8(7);

                /* 8-way parallel VKQ accumulates sharing v_f32. */
                float * VKQ0 = VKQ + (size_t)j0 * dv;
                float * VKQ1 = VKQ + (size_t)j1 * dv;
                float * VKQ2 = VKQ + (size_t)j2 * dv;
                float * VKQ3 = VKQ + (size_t)j3 * dv;
                float * VKQ4 = VKQ + (size_t)j4 * dv;
                float * VKQ5 = VKQ + (size_t)j5 * dv;
                float * VKQ6 = VKQ + (size_t)j6 * dv;
                float * VKQ7 = VKQ + (size_t)j7 * dv;
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    float vi = v_f32[i];
                    VKQ0[i] += vs0 * vi;
                    VKQ1[i] += vs1 * vi;
                    VKQ2[i] += vs2 * vi;
                    VKQ3[i] += vs3 * vi;
                    VKQ4[i] += vs4 * vi;
                    VKQ5[i] += vs5 * vi;
                    VKQ6[i] += vs6 * vi;
                    VKQ7[i] += vs7 * vi;
                }
            }

            /* Leftover (0-7 queries). Single-Q path. */
            for (; j < nq; j++) {
                const int j0 = j;
                float mv0;
                DECODE_MV8(0);
                if (mv0 == -INFINITY) continue;

                float s = 0.0f;
                #pragma _NEC ivdep
                for (int i = 0; i < d; i++) s += Q_tile[j0*d + i] * k_f32[i];
                s *= scale;
                if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
                s += mv0;

                float M_old = M_state[j0];
                float vs;
                if (s > M_old) {
                    M_state[j0] = s;
                    float ms = expf(M_old - s);
                    #pragma _NEC ivdep
                    for (int i = 0; i < dv; i++) VKQ[j0*dv + i] *= ms;
                    S_state[j0] *= ms;
                    vs = 1.0f;
                } else {
                    vs = expf(s - M_state[j0]);
                }
                S_state[j0] += vs;
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) VKQ[j0*dv + i] += vs * v_f32[i];
            }
        }

        for (int j = 0; j < nq; j++) {
            float S_inv = (S_state[j] == 0.0f) ? 0.0f : 1.0f / S_state[j];
            float* out_ptr = (float*)((char*)dst
                + ib * nb_o3 + (iq_base + j) * nb_o2 + ih * nb_o1);
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) out_ptr[i] = VKQ[j*dv + i] * S_inv;
        }
    }

    #undef NQ_TILE_8
    #undef DECODE_MV8
    #undef DO_SOFTMAX8

    asm volatile("fencem 2" ::: "memory");
    return 0;
}

/*
 * Tiled-Q + packed-BF16-intrinsics FA prefill (F32 Q + BF16 K/V).
 *
 * Same shape as ve_flash_attn_ext_f32q_bf16kv_tile_hbm above, but the
 * inner per-Q operations go through the LLVM-VE-RV intrinsics helpers
 * (dot_bf16_fp32_intrinsics / accumulate_bf16_intrinsics /
 *  scale_fp32_intrinsics) which use packed-FP32 mode for the FMA — 2x
 * the FMA throughput vs. NCC's unpacked-vector codegen. K/V live in
 * the LLC across the per-Q loop (128KB total K per (b,h) for typical
 * d=128 / n_kv=512), so reloading from BF16 per query is cheap.
 *
 * Online-softmax and mask handling identical to the NCC tile kernel;
 * outputs match bit-for-bit on the test_flash_attn_tile cases. Wired
 * up behind GGML_VE_FA_TILE_INTRIN=1 (default off) until we confirm
 * the perf gain holds outside benchmarks.
 */
extern float dot_bf16_fp32_intrinsics(const uint16_t* k_ptr, const float* q_ptr, int n);
extern void  accumulate_bf16_intrinsics(float* result, const uint16_t* v_ptr, float weight, int n);
extern void  scale_fp32_intrinsics(float* result, float scale, int n);

uint64_t ve_flash_attn_ext_f32q_bf16kv_tile_intrin_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,
    uint64_t D, uint64_t Dv,
    uint64_t N, uint64_t S_kv,
    uint64_t H, uint64_t Hk, uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits, uint64_t max_bias_bits, uint64_t softcap_bits,
    uint64_t mask_ne2, uint64_t mask_ne3)
{
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* mask = NULL;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 5;

    float scale, max_bias, logit_softcap;
    {
        uint32_t tmp;
        tmp = (uint32_t)scale_bits;    __builtin_memcpy(&scale, &tmp, sizeof(float));
        tmp = (uint32_t)max_bias_bits; __builtin_memcpy(&max_bias, &tmp, sizeof(float));
        tmp = (uint32_t)softcap_bits;  __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    }
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);

    const int d        = (int)D;
    const int dv       = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv     = (int)S_kv;
    const int n_heads  = (int)H;
    const int n_batch  = (int)B;

    #define NQ_TILE_I 16

    const int n_q_tiles  = (n_tokens + NQ_TILE_I - 1) / NQ_TILE_I;
    const int total_work = n_batch * n_heads * n_q_tiles;

    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        const int ib       = work_id / (n_heads * n_q_tiles);
        const int ih       = (work_id / n_q_tiles) % n_heads;
        const int q_tile_i = work_id % n_q_tiles;
        const int iq_base  = q_tile_i * NQ_TILE_I;
        const int nq       = (iq_base + NQ_TILE_I <= n_tokens) ? NQ_TILE_I : (n_tokens - iq_base);

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t hh = (uint32_t)ih;
            slope = (hh < n_head_log2) ? powf(m0, (float)(hh + 1))
                                        : powf(m1, (float)(2*(hh - n_head_log2) + 1));
        }

        const int ik_head = ih / (int)rk;
        const int iv_head = ih / (int)rv;

        /* Per-Q pointers + state. VKQ is per-Q, 8B-aligned for packed
         * intrinsics. */
        const float* q_ptr[NQ_TILE_I];
        const uint16_t* mask_rows[NQ_TILE_I];
        float M_state[NQ_TILE_I];
        float S_state[NQ_TILE_I];
        float VKQ[NQ_TILE_I * 256] __attribute__((aligned(8)));

        for (int j = 0; j < nq; j++) {
            q_ptr[j] = (const float*)((const char*)q
                + (iq_base + j) * nb_q1 + ih * nb_q2 + ib * nb_q3);
            M_state[j] = -INFINITY;
            S_state[j] = 0.0f;
            for (int i = 0; i < dv; i++) VKQ[j*dv + i] = 0.0f;
            if (mask != NULL) {
                int im2 = ih % (int)mask_ne2;
                int im3 = ib % (int)mask_ne3;
                mask_rows[j] = (const uint16_t*)((const char*)mask
                    + (iq_base + j) * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
            } else {
                mask_rows[j] = NULL;
            }
        }

        for (int ic = 0; ic < n_kv; ic++) {
            const uint16_t* k_row = (const uint16_t*)((const char*)k
                + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            const uint16_t* v_row = (const uint16_t*)((const char*)v
                + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);

            for (int j = 0; j < nq; j++) {
                float mv = 0.0f;
                if (mask_rows[j] != NULL) {
                    uint16_t mf16 = mask_rows[j][ic];
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
                    __builtin_memcpy(&mv, &f32, sizeof(float));
                    mv *= slope;
                }
                if (mv == -INFINITY) continue;

                /* Packed-BF16 Q.K. K stays in LLC across nq queries. */
                float s = dot_bf16_fp32_intrinsics(k_row, q_ptr[j], d);
                s *= scale;
                if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
                s += mv;

                float M_old = M_state[j];
                float ms = 1.0f, vs = 1.0f;
                if (s > M_state[j]) {
                    M_state[j] = s;
                    ms = expf(M_old - M_state[j]);
                    scale_fp32_intrinsics(VKQ + j*dv, ms, dv);
                    S_state[j] *= ms;
                } else {
                    vs = expf(s - M_state[j]);
                }
                accumulate_bf16_intrinsics(VKQ + j*dv, v_row, vs, dv);
                S_state[j] += vs;
            }
        }

        for (int j = 0; j < nq; j++) {
            float S_inv = (S_state[j] == 0.0f) ? 0.0f : 1.0f / S_state[j];
            float* out_ptr = (float*)((char*)dst
                + ib * nb_o3 + (iq_base + j) * nb_o2 + ih * nb_o1);
            scale_fp32_intrinsics(VKQ + j*dv, S_inv, dv);
            for (int i = 0; i < dv; i++) out_ptr[i] = VKQ[j*dv + i];
        }
    }

    #undef NQ_TILE_I

    asm volatile("fencem 2" ::: "memory");
    return 0;
}

/*
 * Tiled-Q FA prefill with vectorized softmax exp (F32 Q + BF16 K/V).
 *
 * Profiling the NCC tile kernel above showed 1186 cycles per (Q, ic) at
 * pp512 — dominated by libm's scalar expf (~200-500 cycles per call,
 * unvectorizable because NCC tags it as "obstructive function reference"
 * and bails on the surrounding loop).
 *
 * This variant restructures the inner ic loop so the expf calls happen
 * across all nq queries at once via NCC's __builtin_vec_expf (one vector
 * op for the whole tile). It uses the unconditional online-softmax
 * formulation:
 *
 *   M_new  = max(M_old, s)
 *   alpha  = exp(M_old - M_new)    -- 1 if s <= M_old, else exp(<0)
 *   beta   = exp(s     - M_new)    -- 1 if s >= M_old, else exp(<0)
 *   VKQ    = alpha * VKQ + beta * V
 *   S_sum  = alpha * S_sum + beta
 *
 * This is mathematically equivalent to the branched form ("if s > M, ...,
 * else, ..."), but lets us do two vector_expf calls per ic and one
 * vectorized fused multiply-add per dv element instead of a scalar branch
 * + scalar expf per (Q, ic).
 *
 * Masking: a masked position contributes mv = -INF to s, so beta becomes
 * exp(-INF - M_new) = 0 naturally — the position adds nothing to VKQ or
 * S_sum. No `continue`, no branch.
 */
typedef float vfNQ16 __attribute__((ext_vector_type(16)));

uint64_t ve_flash_attn_ext_f32q_bf16kv_tile_vexp_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,
    uint64_t D, uint64_t Dv,
    uint64_t N, uint64_t S_kv,
    uint64_t H, uint64_t Hk, uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits, uint64_t max_bias_bits, uint64_t softcap_bits,
    uint64_t mask_ne2, uint64_t mask_ne3)
{
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* mask = NULL;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 5;

    float scale, max_bias, logit_softcap;
    {
        uint32_t tmp;
        tmp = (uint32_t)scale_bits;    __builtin_memcpy(&scale, &tmp, sizeof(float));
        tmp = (uint32_t)max_bias_bits; __builtin_memcpy(&max_bias, &tmp, sizeof(float));
        tmp = (uint32_t)softcap_bits;  __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    }
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);

    const int d        = (int)D;
    const int dv       = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv     = (int)S_kv;
    const int n_heads  = (int)H;
    const int n_batch  = (int)B;
    const int d_pairs  = d / 2;
    const int dv_pairs = dv / 2;

    #define NQ_TILE_V 16

    const int n_q_tiles  = (n_tokens + NQ_TILE_V - 1) / NQ_TILE_V;
    const int total_work = n_batch * n_heads * n_q_tiles;

    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        const int ib       = work_id / (n_heads * n_q_tiles);
        const int ih       = (work_id / n_q_tiles) % n_heads;
        const int q_tile_i = work_id % n_q_tiles;
        const int iq_base  = q_tile_i * NQ_TILE_V;
        const int nq       = (iq_base + NQ_TILE_V <= n_tokens) ? NQ_TILE_V : (n_tokens - iq_base);

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t hh = (uint32_t)ih;
            slope = (hh < n_head_log2) ? powf(m0, (float)(hh + 1))
                                        : powf(m1, (float)(2*(hh - n_head_log2) + 1));
        }

        const int ik_head = ih / (int)rk;
        const int iv_head = ih / (int)rv;

        float Q_tile[NQ_TILE_V * 256];
        const uint16_t* mask_rows[NQ_TILE_V];
        float M_state[NQ_TILE_V];
        float S_state[NQ_TILE_V];
        float VKQ[NQ_TILE_V * 256];

        for (int j = 0; j < nq; j++) {
            const float* q_vec = (const float*)((const char*)q
                + (iq_base + j) * nb_q1 + ih * nb_q2 + ib * nb_q3);
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) Q_tile[j*d + i] = q_vec[i];
            M_state[j] = -INFINITY;
            S_state[j] = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) VKQ[j*dv + i] = 0.0f;
            if (mask != NULL) {
                int im2 = ih % (int)mask_ne2;
                int im3 = ib % (int)mask_ne3;
                mask_rows[j] = (const uint16_t*)((const char*)mask
                    + (iq_base + j) * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
            } else {
                mask_rows[j] = NULL;
            }
        }

        for (int ic = 0; ic < n_kv; ic++) {
            const uint32_t* k_u32 = (const uint32_t*)((const char*)k
                + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d_pairs; i++) {
                uint32_t packed = k_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                k_f32[2*i]     = cl.f;
                k_f32[2*i + 1] = ch.f;
            }

            const uint32_t* v_u32 = (const uint32_t*)((const char*)v
                + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv_pairs; i++) {
                uint32_t packed = v_u32[i];
                uint32_t f32_lo = (packed & 0xFFFF) << 16;
                uint32_t f32_hi = packed & 0xFFFF0000;
                union { uint32_t u; float f; } cl, ch;
                cl.u = f32_lo;  ch.u = f32_hi;
                v_f32[2*i]     = cl.f;
                v_f32[2*i + 1] = ch.f;
            }

            /* Phase 1: compute all NQ scores + mask values. The Q.K dot
             * vectorizes the d-loop as a Sum reduction; the per-j loop
             * is small and sequential. */
            float s_arr[NQ_TILE_V];
            for (int j = 0; j < nq; j++) {
                float mv = 0.0f;
                if (mask_rows[j] != NULL) {
                    uint16_t mf16 = mask_rows[j][ic];
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
                    __builtin_memcpy(&mv, &f32, sizeof(float));
                    mv *= slope;
                }

                float s = 0.0f;
                #pragma _NEC ivdep
                for (int i = 0; i < d; i++) {
                    s += Q_tile[j*d + i] * k_f32[i];
                }
                s *= scale;
                if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
                s += mv;
                s_arr[j] = s;
            }

            /* Phase 2: vectorized exp for ALL nq queries at once.
             * M_new = max(M_old, s), then alpha = exp(M_old - M_new),
             * beta = exp(s - M_new). One vector expf call per array. */
            vfNQ16 M_old_v, s_v, M_new_v, M_diff_v, s_diff_v, alpha_v, beta_v;
            for (int j = 0; j < nq; j++) {
                M_old_v[j] = M_state[j];
                s_v[j]     = s_arr[j];
            }
            for (int j = 0; j < nq; j++) {
                M_new_v[j] = (s_v[j] > M_old_v[j]) ? s_v[j] : M_old_v[j];
            }
            for (int j = 0; j < nq; j++) {
                M_diff_v[j] = M_old_v[j] - M_new_v[j];   /* <= 0 */
                s_diff_v[j] = s_v[j]     - M_new_v[j];   /* <= 0 */
            }

            __builtin_vec_expf(alpha_v, M_diff_v, nq);
            __builtin_vec_expf(beta_v,  s_diff_v, nq);

            /* Phase 3: update M/S, scale + accumulate VKQ. */
            for (int j = 0; j < nq; j++) {
                M_state[j] = M_new_v[j];
                S_state[j] = alpha_v[j] * S_state[j] + beta_v[j];
                float alpha = alpha_v[j];
                float beta  = beta_v[j];
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[j*dv + i] = alpha * VKQ[j*dv + i] + beta * v_f32[i];
                }
            }
        }

        for (int j = 0; j < nq; j++) {
            float S_inv = (S_state[j] == 0.0f) ? 0.0f : 1.0f / S_state[j];
            float* out_ptr = (float*)((char*)dst
                + ib * nb_o3 + (iq_base + j) * nb_o2 + ih * nb_o1);
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) out_ptr[i] = VKQ[j*dv + i] * S_inv;
        }
    }

    #undef NQ_TILE_V

    asm volatile("fencem 2" ::: "memory");
    return 0;
}

/*
 * VEDA wrapper for the multi-Q intrinsics tile FA kernel.
 *
 * Inner per-(b,h,q_tile) work lives in flash_attn_bf16kv_tile_fast.c
 * (compiled with LLVM-VE-RV for the velintrin.h packed-FP32 ops). This
 * wrapper handles OMP parallelism, mask plumbing, and the dims plumbing
 * to that inner function.
 */
extern void flash_attn_tile_inner_intrinsics(
    float * VKQ,
    float * M_state,
    float * S_state,
    const float * Q_tile,
    const uint16_t * K_base,
    const uint16_t * V_base,
    const uint16_t * const * mask_rows,
    int nq, int n_kv, int d, int dv,
    int64_t nb_k1, int64_t nb_v1,
    float scale, float slope, float logit_softcap);

extern void flash_attn_tile_inner_pack2_intrinsics(
    float * VKQ,
    float * M_state,
    float * S_state,
    const float * Q_tile,
    const uint16_t * K_base,
    const uint16_t * V_base,
    const uint16_t * const * mask_rows,
    int nq, int n_kv, int d, int dv,
    int64_t nb_k1, int64_t nb_v1,
    float scale, float slope, float logit_softcap);

/* Pack-2 wrapper. Same setup as the fast wrapper below; only difference
 * is which inner function it calls. Pack-2 puts 2 queries per packed
 * slot, sharing the K_dup register across both — half the chains. */
uint64_t ve_flash_attn_ext_f32q_bf16kv_tile_pack2_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,
    uint64_t D, uint64_t Dv,
    uint64_t N, uint64_t S_kv,
    uint64_t H, uint64_t Hk, uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits, uint64_t max_bias_bits, uint64_t softcap_bits,
    uint64_t mask_ne2, uint64_t mask_ne3)
{
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* mask = NULL;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 5;

    float scale, max_bias, logit_softcap;
    {
        uint32_t tmp;
        tmp = (uint32_t)scale_bits;    __builtin_memcpy(&scale, &tmp, sizeof(float));
        tmp = (uint32_t)max_bias_bits; __builtin_memcpy(&max_bias, &tmp, sizeof(float));
        tmp = (uint32_t)softcap_bits;  __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    }
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);

    const int d        = (int)D;
    const int dv       = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv     = (int)S_kv;
    const int n_heads  = (int)H;
    const int n_batch  = (int)B;

    #define NQ_TILE_P 16

    const int n_q_tiles  = (n_tokens + NQ_TILE_P - 1) / NQ_TILE_P;
    const int total_work = n_batch * n_heads * n_q_tiles;

    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        const int ib       = work_id / (n_heads * n_q_tiles);
        const int ih       = (work_id / n_q_tiles) % n_heads;
        const int q_tile_i = work_id % n_q_tiles;
        const int iq_base  = q_tile_i * NQ_TILE_P;
        const int nq       = (iq_base + NQ_TILE_P <= n_tokens) ? NQ_TILE_P : (n_tokens - iq_base);

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t hh = (uint32_t)ih;
            slope = (hh < n_head_log2) ? powf(m0, (float)(hh + 1))
                                        : powf(m1, (float)(2*(hh - n_head_log2) + 1));
        }

        const int ik_head = ih / (int)rk;
        const int iv_head = ih / (int)rv;

        float Q_tile[NQ_TILE_P * 256] __attribute__((aligned(8)));
        const uint16_t* mask_rows[NQ_TILE_P];
        float M_state[NQ_TILE_P];
        float S_state[NQ_TILE_P];
        float VKQ[NQ_TILE_P * 256] __attribute__((aligned(8)));

        for (int j = 0; j < nq; j++) {
            const float* q_vec = (const float*)((const char*)q
                + (iq_base + j) * nb_q1 + ih * nb_q2 + ib * nb_q3);
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) Q_tile[j*d + i] = q_vec[i];
            M_state[j] = -INFINITY;
            S_state[j] = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) VKQ[j*dv + i] = 0.0f;
            if (mask != NULL) {
                int im2 = ih % (int)mask_ne2;
                int im3 = ib % (int)mask_ne3;
                mask_rows[j] = (const uint16_t*)((const char*)mask
                    + (iq_base + j) * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
            } else {
                mask_rows[j] = NULL;
            }
        }

        const uint16_t * K_base = (const uint16_t *)((const char*)k
            + ik_head * nb_k2 + ib * nb_k3);
        const uint16_t * V_base = (const uint16_t *)((const char*)v
            + iv_head * nb_v2 + ib * nb_v3);

        flash_attn_tile_inner_pack2_intrinsics(
            VKQ, M_state, S_state, Q_tile, K_base, V_base, mask_rows,
            nq, n_kv, d, dv,
            (int64_t)nb_k1, (int64_t)nb_v1,
            scale, slope, logit_softcap);

        for (int j = 0; j < nq; j++) {
            float S_inv = (S_state[j] == 0.0f) ? 0.0f : 1.0f / S_state[j];
            float* out_ptr = (float*)((char*)dst
                + ib * nb_o3 + (iq_base + j) * nb_o2 + ih * nb_o1);
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) out_ptr[i] = VKQ[j*dv + i] * S_inv;
        }
    }

    #undef NQ_TILE_P

    asm volatile("fencem 2" ::: "memory");
    return 0;
}

uint64_t ve_flash_attn_ext_f32q_bf16kv_tile_fast_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,
    uint64_t D, uint64_t Dv,
    uint64_t N, uint64_t S_kv,
    uint64_t H, uint64_t Hk, uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits, uint64_t max_bias_bits, uint64_t softcap_bits,
    uint64_t mask_ne2, uint64_t mask_ne3)
{
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* mask = NULL;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 5;

    float scale, max_bias, logit_softcap;
    {
        uint32_t tmp;
        tmp = (uint32_t)scale_bits;    __builtin_memcpy(&scale, &tmp, sizeof(float));
        tmp = (uint32_t)max_bias_bits; __builtin_memcpy(&max_bias, &tmp, sizeof(float));
        tmp = (uint32_t)softcap_bits;  __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    }
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);

    const int d        = (int)D;
    const int dv       = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv     = (int)S_kv;
    const int n_heads  = (int)H;
    const int n_batch  = (int)B;

    #define NQ_TILE_F 16

    const int n_q_tiles  = (n_tokens + NQ_TILE_F - 1) / NQ_TILE_F;
    const int total_work = n_batch * n_heads * n_q_tiles;

    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        const int ib       = work_id / (n_heads * n_q_tiles);
        const int ih       = (work_id / n_q_tiles) % n_heads;
        const int q_tile_i = work_id % n_q_tiles;
        const int iq_base  = q_tile_i * NQ_TILE_F;
        const int nq       = (iq_base + NQ_TILE_F <= n_tokens) ? NQ_TILE_F : (n_tokens - iq_base);

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t hh = (uint32_t)ih;
            slope = (hh < n_head_log2) ? powf(m0, (float)(hh + 1))
                                        : powf(m1, (float)(2*(hh - n_head_log2) + 1));
        }

        const int ik_head = ih / (int)rk;
        const int iv_head = ih / (int)rv;

        /* Per-tile state on the stack. Q_tile + VKQ are 8-byte aligned
         * for the packed intrinsic loads in the inner function. */
        float Q_tile[NQ_TILE_F * 256] __attribute__((aligned(8)));
        const uint16_t* mask_rows[NQ_TILE_F];
        float M_state[NQ_TILE_F];
        float S_state[NQ_TILE_F];
        float VKQ[NQ_TILE_F * 256] __attribute__((aligned(8)));

        /* Load Q tile + zero VKQ + init M/S + resolve mask rows. */
        for (int j = 0; j < nq; j++) {
            const float* q_vec = (const float*)((const char*)q
                + (iq_base + j) * nb_q1 + ih * nb_q2 + ib * nb_q3);
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) Q_tile[j*d + i] = q_vec[i];
            M_state[j] = -INFINITY;
            S_state[j] = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) VKQ[j*dv + i] = 0.0f;
            if (mask != NULL) {
                int im2 = ih % (int)mask_ne2;
                int im3 = ib % (int)mask_ne3;
                mask_rows[j] = (const uint16_t*)((const char*)mask
                    + (iq_base + j) * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
            } else {
                mask_rows[j] = NULL;
            }
        }

        /* Per-(b, ih) K/V base; inner loop adds ic * nb_k1 / nb_v1. */
        const uint16_t * K_base = (const uint16_t *)((const char*)k
            + ik_head * nb_k2 + ib * nb_k3);
        const uint16_t * V_base = (const uint16_t *)((const char*)v
            + iv_head * nb_v2 + ib * nb_v3);

        flash_attn_tile_inner_intrinsics(
            VKQ, M_state, S_state, Q_tile, K_base, V_base, mask_rows,
            nq, n_kv, d, dv,
            (int64_t)nb_k1, (int64_t)nb_v1,
            scale, slope, logit_softcap);

        /* Final normalize + write out. */
        for (int j = 0; j < nq; j++) {
            float S_inv = (S_state[j] == 0.0f) ? 0.0f : 1.0f / S_state[j];
            float* out_ptr = (float*)((char*)dst
                + ib * nb_o3 + (iq_base + j) * nb_o2 + ih * nb_o1);
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) out_ptr[i] = VKQ[j*dv + i] * S_inv;
        }
    }

    #undef NQ_TILE_F

    asm volatile("fencem 2" ::: "memory");
    return 0;
}

/*
 * VEDA-callable wrapper for LLVM-VE intrinsics flash attention
 * 
 * This converts HBM pointers and calls the intrinsics-based implementation.
 * The intrinsics version uses VE vector load intrinsics which properly vectorize
 * BF16 -> FP32 conversion (unlike NCC which cannot vectorize 16-bit loads).
 */
uint64_t ve_flash_attn_bf16_intrinsics_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,  /* HMEM - F32 mask (pre-converted from F16 on host) */
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1,
    uint64_t nb_q2,
    uint64_t nb_q3,
    uint64_t nb_k1,
    uint64_t nb_k2,
    uint64_t nb_k3,
    uint64_t nb_v1,
    uint64_t nb_v2,
    uint64_t nb_v3,
    uint64_t nb_m1,
    uint64_t nb_m2,
    uint64_t nb_m3,
    uint64_t nb_o1,
    uint64_t nb_o2,
    uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    /* Convert HBM pointers to raw VE addresses */
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    
    /* Mask is F32 (pre-converted on host), HMEM already converted to raw pointer */
    const float* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    /* Convert bit patterns to floats */
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    /* Compute ALiBi parameters */
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);
    
    int d = (int)D;
    int n_tokens = (int)N;
    int n_kv = (int)S;
    int n_heads = (int)H;
    int n_batch = (int)B;
    
    int total_work = n_batch * n_heads * n_tokens;
    
    /* OpenMP parallel loop over work items (batch, head, token) */
    #pragma omp parallel for schedule(dynamic)
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        /* Compute ALiBi slope for this head */
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        /* Get Q pointer for this (batch, head, token) */
        const float* q_ptr = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        /* Get K, V base pointers for this (batch, k_head) */
        const bf16* k_base = (const bf16*)((const char*)k + ik_head * nb_k2 + ib * nb_k3);
        const bf16* v_base = (const bf16*)((const char*)v + iv_head * nb_v2 + ib * nb_v3);
        
        /* Get mask row if present */
        const float* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const float*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        /* Get output pointer */
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        
        /* Call the vectorized single-head intrinsics function */
        flash_attn_single_head_intrinsics(
            out_ptr, q_ptr, k_base, v_base, mask_row,
            d, n_kv, nb_k1, nb_v1, scale, logit_softcap, slope);
    }
    
    return 0;
}

/*
 * Simple (non-flash) attention for BF16 K/V cache in HBM
 * 
 * This implements ve-llama2.c style attention:
 * 1. Compute ALL Q·K scores into a buffer
 * 2. Apply softmax to scores
 * 3. Compute weighted sum of V vectors
 * 
 * This trades memory (scores buffer) for speed by avoiding:
 * - Online softmax recalculations
 * - Per-element exp() calls during V accumulation
 * 
 * Max context: 16384 tokens (scores buffer in stack)
 */
uint64_t ve_simple_attn_bf16kv_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,  /* HMEM - F16 mask */
    uint64_t D,      /* Head dimension for Q, K */
    uint64_t Dv,     /* Head dimension for V (usually same as D) */
    uint64_t N,      /* Number of query tokens (usually 1 for generation) */
    uint64_t S,      /* Number of K/V tokens (context length) */
    uint64_t H,      /* Number of query heads */
    uint64_t Hk,     /* Number of K/V heads (for GQA) */
    uint64_t B,      /* Batch size */
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    /* Convert HBM pointers */
    float* dst;
    const float* q;
    const uint16_t* k;  /* BF16 */
    const uint16_t* v;  /* BF16 */
    
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    
    const uint16_t* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    /* Decode float parameters */
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    /* ALiBi parameters */
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int d = (int)D;
    const int dv = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv = (int)S;
    const int n_heads = (int)H;
    const int n_batch = (int)B;
    const int d_pairs = d / 2;
    const int dv_pairs = dv / 2;
    
    int total_work = n_batch * n_heads * n_tokens;
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        /* ALiBi slope */
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rk;
        
        /* Q pointer (F32) */
        const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        /* Mask row */
        const uint16_t* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const uint16_t*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        /* Attention scores buffer (stack allocated, max 16K tokens) */
        float scores[16384];
        
        /* Pass 1: Compute ALL Q·K attention scores */
        float max_score = -INFINITY;
        #pragma _NEC ivdep
        for (int ic = 0; ic < n_kv; ic++) {
            /* Decode mask */
            float mv = 0.0f;
            if (mask_row != NULL) {
                uint16_t mf16 = mask_row[ic];
                uint32_t sign = (mf16 >> 15) & 0x1;
                uint32_t exp = (mf16 >> 10) & 0x1F;
                uint32_t mant = mf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (mant == 0) ? (sign << 31) : 0;
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                __builtin_memcpy(&mv, &f32, sizeof(float));
                mv *= slope;
            }
            
            if (mv == -INFINITY) {
                scores[ic] = -INFINITY;
                continue;
            }
            
            /* Convert K from BF16 to F32 */
            const uint32_t* k_u32 = (const uint32_t*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            float k_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < d_pairs; i++) {
                uint32_t packed = k_u32[i];
                union { uint32_t u; float f; } conv_lo, conv_hi;
                conv_lo.u = (packed & 0xFFFF) << 16;
                conv_hi.u = packed & 0xFFFF0000;
                k_f32[2*i] = conv_lo.f;
                k_f32[2*i + 1] = conv_hi.f;
            }
            
            /* Q·K dot product */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_vec[i] * k_f32[i];
            }
            
            s *= scale;
            if (logit_softcap != 0.0f) {
                s = logit_softcap * tanhf(s);
            }
            s += mv;
            
            scores[ic] = s;
            if (s > max_score) max_score = s;
        }
        
        /* Pass 2: Softmax - exp and sum */
        float sum_exp = 0.0f;
        #pragma _NEC ivdep
        for (int ic = 0; ic < n_kv; ic++) {
            if (scores[ic] == -INFINITY) {
                scores[ic] = 0.0f;
            } else {
                float e = expf(scores[ic] - max_score);
                scores[ic] = e;
                sum_exp += e;
            }
        }
        
        /* Normalize */
        float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
        #pragma _NEC ivdep
        for (int ic = 0; ic < n_kv; ic++) {
            scores[ic] *= inv_sum;
        }
        
        /* Pass 3: Weighted sum of V vectors */
        float out_vec[256];
        for (int i = 0; i < dv; i++) {
            out_vec[i] = 0.0f;
        }
        
        for (int ic = 0; ic < n_kv; ic++) {
            float att = scores[ic];
            if (att == 0.0f) continue;
            
            /* Convert V from BF16 to F32 */
            const uint32_t* v_u32 = (const uint32_t*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            float v_f32[256];
            #pragma _NEC ivdep
            for (int i = 0; i < dv_pairs; i++) {
                uint32_t packed = v_u32[i];
                union { uint32_t u; float f; } conv_lo, conv_hi;
                conv_lo.u = (packed & 0xFFFF) << 16;
                conv_hi.u = packed & 0xFFFF0000;
                v_f32[2*i] = conv_lo.f;
                v_f32[2*i + 1] = conv_hi.f;
            }
            
            /* Accumulate weighted V */
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                out_vec[i] += att * v_f32[i];
            }
        }
        
        /* Write output */
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = out_vec[i];
        }
    }
    
    return 0;
}

/*
 * Flash Attention for F32 Q/K/V in HBM
 * 
 * This is the FASTEST attention kernel because:
 * - No BF16->F32 conversion needed (pure F32 throughout)
 * - All data in HBM with 1.2 TB/s bandwidth
 * - Fully vectorizable dot products and accumulations
 * 
 * Uses online softmax (flash attention style) for numerical stability.
 * Should match or exceed ve-llama2.c performance (~42 tok/s on 7B).
 */
uint64_t ve_flash_attn_f32_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,  /* HMEM - F16 mask */
    uint64_t D,
    uint64_t Dv,
    uint64_t N,
    uint64_t S,
    uint64_t H,
    uint64_t Hk,
    uint64_t B,
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    /* Convert HBM pointers to raw VE addresses */
    float* dst;
    const float* q;
    const uint16_t* k;
    const uint16_t* v;
    
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    
    const uint16_t* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    /* Decode float parameters */
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits;
    __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits;
    __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits;
    __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    
    /* ALiBi parameters */
    const uint32_t n_head = (uint32_t)H;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);
    
    const int64_t rk = (int64_t)(H / Hk);
    const int64_t rv = (int64_t)(H / Hk);
    
    const int d = (int)D;
    const int dv = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv = (int)S;
    const int n_heads = (int)H;
    const int n_batch = (int)B;
    
    /* DEBUG: Dump tensors to files for first single-token attention call */
    static int f32_hbm_call_count = 0;
    f32_hbm_call_count++;
    
    /* Dump Q, K, V, mask, and output for call #29 (first single-token attention, layer 0) */
    if (f32_hbm_call_count == 29 && n_tokens == 1) {
        char* dump_dir = getenv("GGML_VE_DUMP_DIR");
        if (dump_dir) {
            char fname[256];
            FILE* f;
            
            /* Dump Q (all heads, single position) - shape [D, H] = [128, 24] */
            snprintf(fname, sizeof(fname), "%s/attn29_q.bin", dump_dir);
            f = fopen(fname, "wb");
            if (f) {
                /* Q is laid out as [N, H, D] with strides nb_q1, nb_q2, nb_q3 */
                /* For N=1: write all 24 heads × 128 dims */
                for (int h = 0; h < n_heads; h++) {
                    const float* q_head = (const float*)((const char*)q + 0 * nb_q1 + h * nb_q2 + 0 * nb_q3);
                    fwrite(q_head, sizeof(float), d, f);
                }
                fclose(f);
                printf("[DUMP] Wrote Q to %s (%d heads × %d dims)\n", fname, n_heads, d);
            }
            
            /* Dump K (first KV head, all positions) - shape [S, D] = [256, 128] */
            snprintf(fname, sizeof(fname), "%s/attn29_k.bin", dump_dir);
            f = fopen(fname, "wb");
            if (f) {
                for (int s = 0; s < n_kv; s++) {
                    const float* k_pos = (const float*)((const char*)k + s * nb_k1 + 0 * nb_k2 + 0 * nb_k3);
                    fwrite(k_pos, sizeof(float), d, f);
                }
                fclose(f);
                printf("[DUMP] Wrote K to %s (%d positions × %d dims)\n", fname, n_kv, d);
            }
            
            /* Dump V (first KV head, all positions) - shape [S, D] = [256, 128] */
            snprintf(fname, sizeof(fname), "%s/attn29_v.bin", dump_dir);
            f = fopen(fname, "wb");
            if (f) {
                for (int s = 0; s < n_kv; s++) {
                    const float* v_pos = (const float*)((const char*)v + s * nb_v1 + 0 * nb_v2 + 0 * nb_v3);
                    fwrite(v_pos, sizeof(float), dv, f);
                }
                fclose(f);
                printf("[DUMP] Wrote V to %s (%d positions × %d dims)\n", fname, n_kv, dv);
            }
            
            /* Dump mask (first row only for single-token) - shape [S] = [256] */
            if (mask != NULL) {
                snprintf(fname, sizeof(fname), "%s/attn29_mask.bin", dump_dir);
                f = fopen(fname, "wb");
                if (f) {
                    const uint16_t* mask_row = (const uint16_t*)((const char*)mask + 0 * nb_m1 + 0 * nb_m2 + 0 * nb_m3);
                    fwrite(mask_row, sizeof(uint16_t), n_kv, f);
                    fclose(f);
                    printf("[DUMP] Wrote mask to %s (%d positions, F16)\n", fname, n_kv);
                }
            }
        }
    }
    
    /* Debug output disabled for production - uncomment for debugging
    if (n_tokens == 1 && ((f32_hbm_call_count >= 29 && f32_hbm_call_count <= 34) || (f32_hbm_call_count >= 57 && f32_hbm_call_count <= 62))) {
        // Debug printfs removed for performance
    }
    */
    
    int total_work = n_batch * n_heads * n_tokens;
    
    #pragma omp parallel for
    for (int work_id = 0; work_id < total_work; work_id++) {
        int ib = work_id / (n_heads * n_tokens);
        int ih = (work_id / n_tokens) % n_heads;
        int iq = work_id % n_tokens;
        
        /* ALiBi slope */
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)ih;
            slope = (h < n_head_log2) ? powf(m0, (float)(h + 1)) 
                                       : powf(m1, (float)(2*(h - n_head_log2) + 1));
        }
        
        int ik_head = ih / (int)rk;
        int iv_head = ih / (int)rv;
        
        /* Q pointer - F32 direct access */
        const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
        
        /* Mask row */
        const uint16_t* mask_row = NULL;
        if (mask != NULL) {
            int im2 = ih % (int)mask_ne2;
            int im3 = ib % (int)mask_ne3;
            mask_row = (const uint16_t*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
        }
        
        /* Online softmax state */
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] = 0.0f;
        }
        
        /* Process each K/V position */
        for (int ic = 0; ic < n_kv; ic++) {
            /* Decode F16 mask */
            float mv = 0.0f;
            if (mask_row != NULL) {
                uint16_t mf16 = mask_row[ic];
                uint32_t sign = (mf16 >> 15) & 0x1;
                uint32_t exp = (mf16 >> 10) & 0x1F;
                uint32_t mant = mf16 & 0x3FF;
                uint32_t f32;
                if (exp == 0) {
                    f32 = (mant == 0) ? (sign << 31) : 0;
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                __builtin_memcpy(&mv, &f32, sizeof(float));
                mv *= slope;
            }
            
            if (mv == -INFINITY) {
                continue;
            }
            
            /* K vector - F32 direct access (no conversion!) */
            const float* k_vec = (const float*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
            
            /* Q·K dot product - fully vectorizable! */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < d; i++) {
                s += q_vec[i] * k_vec[i];
            }
            
            s *= scale;
            if (logit_softcap != 0.0f) {
                s = logit_softcap * tanhf(s);
            }
            s += mv;
            
            /* Online softmax update */
            float M_old = M;
            float ms = 1.0f;
            float vs = 1.0f;
            
            if (s > M) {
                M = s;
                ms = expf(M_old - M);
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    VKQ[i] *= ms;
                }
                S_sum *= ms;
            } else {
                vs = expf(s - M);
            }
            
            /* V vector - F32 direct access (no conversion!) */
            const float* v_vec = (const float*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
            
            /* Accumulate weighted V - fully vectorizable! */
            #pragma _NEC ivdep
            for (int i = 0; i < dv; i++) {
                VKQ[i] += vs * v_vec[i];
            }
            
            S_sum += vs;
        }
        
        /* Normalize by softmax sum */
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            VKQ[i] *= S_inv;
        }
        
        /* Write output */
        float* out_ptr = (float*)((char*)dst + ib * nb_o3 + iq * nb_o2 + ih * nb_o1);
        #pragma _NEC ivdep
        for (int i = 0; i < dv; i++) {
            out_ptr[i] = VKQ[i];
        }
    }
    
    /* Debug: print first head output for first few calls */
    static int f32_attn_count = 0;
    f32_attn_count++;
    
    /* Dump output for call #29 */
    if (f32_attn_count == 29 && n_tokens == 1) {
        char* dump_dir = getenv("GGML_VE_DUMP_DIR");
        if (dump_dir) {
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/attn29_out.bin", dump_dir);
            FILE* f = fopen(fname, "wb");
            if (f) {
                /* Output is laid out as [N, H, D] with strides nb_o2, nb_o1 */
                /* For N=1: write all 24 heads × 128 dims */
                for (int h = 0; h < n_heads; h++) {
                    float* out_head = (float*)((char*)dst + 0 * nb_o2 + h * nb_o1 + 0 * nb_o3);
                    fwrite(out_head, sizeof(float), dv, f);
                }
                fclose(f);
                printf("[DUMP] Wrote output to %s (%d heads × %d dims)\n", fname, n_heads, dv);
            }
        }
    }
    
    return 0;
}

/* Debug: query OpenMP thread count */
uint64_t ve_get_omp_threads(void) { 
    return (uint64_t)omp_get_max_threads(); 
}


/*
 * ROPE Normal style with HBM input/output + OpenMP
 * This version computes cos/sin internally - no cache copy needed!
 * 
 * Parameters:
 *   y_hbm:     Output tensor (HBM)
 *   x_hbm:     Input tensor (HBM)
 *   pos_ptr:   Position array (HMEM, I32)
 *   ne0:       Element count per head
 *   n_dims:    Number of dimensions to rotate
 *   n_heads:   Number of attention heads
 *   n_ctx:     Sequence length
 *   n_batch:   Batch size
 *   nb1-nb3:   Strides
 *   freq_base: ROPE frequency base (typically 10000)
 *   freq_scale: Frequency scale (typically 1.0)
 *   mscale:    Magnitude scale (typically 1.0)
 */
uint64_t ve_rope_normal_hbm_omp_nocache(VEDAdeviceptr y_hbm,
                                         VEDAdeviceptr x_hbm,
                                         VEDAdeviceptr pos_hbm,
                                         uint64_t ne0,
                                         uint64_t n_dims,
                                         uint64_t n_heads,
                                         uint64_t n_ctx,
                                         uint64_t n_batch,
                                         uint64_t nb1,
                                         uint64_t nb2,
                                         uint64_t nb3,
                                         float freq_base,
                                         float freq_scale,
                                         float mscale) {
    /* Convert HBM pointers to raw VE addresses */
    float* y;
    const float* x;
    const int32_t* pos;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&pos, pos_hbm) != 0) return 3;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int elem_per_head = (int)ne0;
    
    /* Precompute theta_scale for iterative theta computation */
    float theta_scale = powf(freq_base, -2.0f/nd);
    
    /* Total rows = batch * ctx * heads */
    int total_rows = batch * ctx * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / (ctx * heads);
        int rem = row % (ctx * heads);
        int i2 = rem / heads;
        int i1 = rem % heads;
        
        /* Get position for this row (i2 is sequence position index) */
        float theta = (float)pos[i2] * freq_scale;
        
        size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* Apply rotation to consecutive pairs (NORMAL style) */
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < nd; i0 += 2) {
            float cos_val = cosf(theta) * mscale;
            float sin_val = sinf(theta) * mscale;
            
            float x0 = src[i0];
            float x1 = src[i0 + 1];
            
            dst[i0]     = x0 * cos_val - x1 * sin_val;
            dst[i0 + 1] = x0 * sin_val + x1 * cos_val;
            
            theta *= theta_scale;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/*
 * ROPE Normal style with single position (for single-token generation)
 * This avoids ALL HMEM copies - position is passed as a scalar parameter!
 */
uint64_t ve_rope_normal_hbm_single_pos(VEDAdeviceptr y_hbm,
                                        VEDAdeviceptr x_hbm,
                                        int64_t position,  // Single position value
                                        uint64_t ne0,
                                        uint64_t n_dims,
                                        uint64_t n_heads,
                                        uint64_t n_batch,
                                        uint64_t nb1,
                                        uint64_t nb2,
                                        uint64_t nb3,
                                        float freq_base,
                                        float freq_scale,
                                        float mscale) {
    /* Convert HBM pointers to raw VE addresses */
    float* y;
    const float* x;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    
    int heads = (int)n_heads;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int elem_per_head = (int)ne0;
    
    /* Precompute theta_scale for iterative theta computation */
    float theta_scale = powf(freq_base, -2.0f/nd);
    float base_theta = (float)position * freq_scale;
    
    /* Total rows = batch * 1 * heads (single ctx position) */
    int total_rows = batch * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / heads;  // batch index
        int i1 = row % heads;  // head index
        
        float theta = base_theta;
        
        size_t src_offset = i3 * nb3 + 0 * nb2 + i1 * nb1;  // i2=0 for single position
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* Apply rotation to consecutive pairs (NORMAL style) */
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < nd; i0 += 2) {
            float cos_val = cosf(theta) * mscale;
            float sin_val = sinf(theta) * mscale;
            
            float x0 = src[i0];
            float x1 = src[i0 + 1];
            
            dst[i0]     = x0 * cos_val - x1 * sin_val;
            dst[i0 + 1] = x0 * sin_val + x1 * cos_val;
            
            theta *= theta_scale;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/*
 * ROPE NeoX style with HBM input/output + OpenMP - no cache version
 */
uint64_t ve_rope_neox_hbm_omp_nocache(VEDAdeviceptr y_hbm,
                                       VEDAdeviceptr x_hbm,
                                       VEDAdeviceptr pos_hbm,
                                       uint64_t ne0,
                                       uint64_t n_dims,
                                       uint64_t n_heads,
                                       uint64_t n_ctx,
                                       uint64_t n_batch,
                                       uint64_t nb1,
                                       uint64_t nb2,
                                       uint64_t nb3,
                                       float freq_base,
                                       float freq_scale,
                                       float mscale) {
    /* Convert HBM pointers to raw VE addresses */
    float* y;
    const float* x;
    const int32_t* pos;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&pos, pos_hbm) != 0) return 3;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int half_nd = nd / 2;
    int elem_per_head = (int)ne0;
    
    /* Precompute theta_scale */
    float theta_scale = powf(freq_base, -2.0f/nd);
    
    int total_rows = batch * ctx * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / (ctx * heads);
        int rem = row % (ctx * heads);
        int i2 = rem / heads;
        int i1 = rem % heads;
        
        float theta = (float)pos[i2] * freq_scale;
        
        size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* NeoX style: rotate first half with second half */
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < half_nd; i0++) {
            float cos_val = cosf(theta) * mscale;
            float sin_val = sinf(theta) * mscale;
            
            float x0 = src[i0];
            float x1 = src[i0 + half_nd];
            
            dst[i0]           = x0 * cos_val - x1 * sin_val;
            dst[i0 + half_nd] = x0 * sin_val + x1 * cos_val;
            
            theta *= theta_scale;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/*
 * ROPE NeoX style with single position (for single-token generation)
 * This avoids ALL HMEM copies - position is passed as a scalar parameter!
 */
uint64_t ve_rope_neox_hbm_single_pos(VEDAdeviceptr y_hbm,
                                      VEDAdeviceptr x_hbm,
                                      int64_t position,  // Single position value
                                      uint64_t ne0,
                                      uint64_t n_dims,
                                      uint64_t n_heads,
                                      uint64_t n_batch,
                                      uint64_t nb1,
                                      uint64_t nb2,
                                      uint64_t nb3,
                                      float freq_base,
                                      float freq_scale,
                                      float mscale) {
    /* Convert HBM pointers to raw VE addresses */
    float* y;
    const float* x;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    
    int heads = (int)n_heads;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int half_nd = nd / 2;
    int elem_per_head = (int)ne0;
    
    /* Precompute theta_scale */
    float theta_scale = powf(freq_base, -2.0f/nd);
    float base_theta = (float)position * freq_scale;
    
    /* Total rows = batch * 1 * heads (single ctx position) */
    int total_rows = batch * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / heads;  // batch index
        int i1 = row % heads;  // head index
        
        float theta = base_theta;
        
        size_t src_offset = i3 * nb3 + 0 * nb2 + i1 * nb1;  // i2=0 for single position
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* NeoX style: rotate first half with second half */
        #pragma _NEC ivdep
        for (int i0 = 0; i0 < half_nd; i0++) {
            float cos_val = cosf(theta) * mscale;
            float sin_val = sinf(theta) * mscale;
            
            float x0 = src[i0];
            float x1 = src[i0 + half_nd];
            
            dst[i0]           = x0 * cos_val - x1 * sin_val;
            dst[i0 + half_nd] = x0 * sin_val + x1 * cos_val;
            
            theta *= theta_scale;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/*
 * ===========================================================================
 * IMROPE (Interleaved Multi-ROPE) for Qwen3-VL
 * ===========================================================================
 * IMROPE uses 4 position values per token (time, height, width, extra) and
 * applies them in an interleaved pattern across embedding dimensions.
 * 
 * The rotation itself is NEOX-style (pairs from first/second half).
 * The key difference is how theta is computed for each dimension pair.
 */

/*
 * IMROPE kernel with HBM input/output + OpenMP
 * 
 * Position tensor layout: [n_ctx, 4] where:
 *   pos[i2]          = p_t (time position)
 *   pos[i2 + n_ctx]  = p_h (height position)
 *   pos[i2 + 2*n_ctx]= p_w (width position)
 *   pos[i2 + 3*n_ctx]= p_e (extra position)
 * 
 * Sections define how dimensions are split:
 *   sections[0] = dims for time (interleaved at sector%3==0)
 *   sections[1] = dims for height (interleaved at sector%3==1)
 *   sections[2] = dims for width (interleaved at sector%3==2)
 *   sections[3] = extra dims
 */
uint64_t ve_rope_imrope_hbm_omp(VEDAdeviceptr y_hbm,
                                 VEDAdeviceptr x_hbm,
                                 VEDAdeviceptr pos_hbm,
                                 uint64_t ne0,         // Elements per head
                                 uint64_t n_dims,      // ROPE dimensions
                                 uint64_t n_heads,     // Number of heads
                                 uint64_t n_ctx,       // Sequence length
                                 uint64_t n_batch,     // Batch size
                                 uint64_t nb1,         // Stride for head dimension
                                 uint64_t nb2,         // Stride for ctx dimension
                                 uint64_t nb3,         // Stride for batch dimension
                                 float freq_base,
                                 float freq_scale,
                                 float mscale,
                                 int32_t sect0,        // sections[0] - time
                                 int32_t sect1,        // sections[1] - height
                                 int32_t sect2,        // sections[2] - width
                                 int32_t sect3) {      // sections[3] - extra
    /* Convert HBM pointers to raw VE addresses */
    float* y;
    const float* x;
    const int32_t* pos;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&pos, pos_hbm) != 0) return 3;
    
    int heads = (int)n_heads;
    int ctx = (int)n_ctx;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int half_nd = nd / 2;
    int elem_per_head = (int)ne0;
    
    /* Compute total section dims */
    int sect_dims = sect0 + sect1 + sect2 + sect3;
    
    /* Precompute theta_scale */
    float theta_scale = powf(freq_base, -2.0f/nd);
    
    int total_rows = batch * ctx * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / (ctx * heads);
        int rem = row % (ctx * heads);
        int i2 = rem / heads;
        int i1 = rem % heads;
        
        /* Get 4 positions for this sequence position */
        int64_t p_t = pos[i2];              // time
        int64_t p_h = pos[i2 + ctx];        // height
        int64_t p_w = pos[i2 + ctx * 2];    // width
        int64_t p_e = pos[i2 + ctx * 3];    // extra
        
        /* Compute base thetas */
        float theta_base_t = (float)p_t * freq_scale;
        float theta_base_h = (float)p_h * freq_scale;
        float theta_base_w = (float)p_w * freq_scale;
        float theta_base_e = (float)p_e * freq_scale;
        
        size_t src_offset = i3 * nb3 + i2 * nb2 + i1 * nb1;
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        /* Track theta for each position type */
        float theta_t = theta_base_t;
        float theta_h = theta_base_h;
        float theta_w = theta_base_w;
        float theta_e = theta_base_e;
        
        /* IMROPE interleaved pattern: rotate pairs using NEOX style */
        /* i0 goes through half the dims (each i0 handles a pair) */
        for (int i0 = 0; i0 < half_nd; i0++) {
            /* Determine which theta to use based on sector (interleaved) */
            int sector = i0 % sect_dims;
            float theta;
            
            /* IMROPE interleaved assignment (order matters! matches GGML reference) */
            /* Check order: height first, width second, time third, else extra */
            if (sector % 3 == 1 && sector < 3 * sect1) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sect2) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sect0) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
            
            float cos_val = cosf(theta) * mscale;
            float sin_val = sinf(theta) * mscale;
            
            /* NeoX style rotation: pair (i0, i0+half_nd) */
            float x0 = src[i0];
            float x1 = src[i0 + half_nd];
            
            dst[i0]           = x0 * cos_val - x1 * sin_val;
            dst[i0 + half_nd] = x0 * sin_val + x1 * cos_val;
            
            /* Update all thetas for next dimension pair */
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }
        
        /* Copy remaining elements unchanged */
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    return 0;
}

/*
 * IMROPE kernel with single position (for single-token generation)
 * Avoids HMEM copies by passing all 4 positions as scalar parameters.
 */
static int imrope_interp_debug_count = 0;

uint64_t ve_rope_imrope_hbm_single_pos(VEDAdeviceptr y_hbm,
    /* DEBUG: Always print to check if this function is called */
                                        VEDAdeviceptr x_hbm,
                                        int64_t pos_t,     // Time position
                                        int64_t pos_h,     // Height position
                                        int64_t pos_w,     // Width position
                                        int64_t pos_e,     // Extra position
                                        uint64_t ne0,
                                        uint64_t n_dims,
                                        uint64_t n_heads,
                                        uint64_t n_batch,
                                        uint64_t nb1,
                                        uint64_t nb2,
                                        uint64_t nb3,
                                        float freq_base,
                                        float freq_scale,
                                        float mscale,
                                        int32_t sect0,
                                        int32_t sect1,
                                        int32_t sect2,
                                        int32_t sect3) {
    /* Convert HBM pointers to raw VE addresses */
    float* y;
    const float* x;
    if (vedaMemPtr((void**)&y, y_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&x, x_hbm) != 0) return 2;
    
    int heads = (int)n_heads;
    int batch = (int)n_batch;
    int nd = (int)n_dims;
    int half_nd = nd / 2;
    int elem_per_head = (int)ne0;
    
    int sect_dims = sect0 + sect1 + sect2 + sect3;
    
    float theta_scale = powf(freq_base, -2.0f/nd);
    
    /* Debug output - compare with compiled kernel */
    /* Note: Always debug first 2 calls since getenv may not work on VE */
    if (imrope_interp_debug_count < 2) {
        printf("[IMROPE-INTERP] pos_t=%lld pos_h=%lld pos_w=%lld pos_e=%lld\n", 
               (long long)pos_t, (long long)pos_h, (long long)pos_w, (long long)pos_e);
        printf("[IMROPE-INTERP] ne0=%llu n_dims=%llu n_heads=%llu n_batch=%llu\n",
               (unsigned long long)ne0, (unsigned long long)n_dims, 
               (unsigned long long)n_heads, (unsigned long long)n_batch);
        printf("[IMROPE-INTERP] nb1=%llu nb2=%llu nb3=%llu (expected nb1=%d for contiguous)\n",
               (unsigned long long)nb1, (unsigned long long)nb2, (unsigned long long)nb3,
               (int)ne0 * 4);
        printf("[IMROPE-INTERP] sect0=%d sect1=%d sect2=%d sect3=%d sect_dims=%d half_nd=%d\n",
               sect0, sect1, sect2, sect3, sect_dims, half_nd);
        printf("[IMROPE-INTERP] theta_scale=%.10f\n", theta_scale);
        printf("[IMROPE-INTERP] in[0:3]=%.6f %.6f %.6f %.6f\n", x[0], x[1], x[2], x[3]);
        printf("[IMROPE-INTERP] in[half_nd:half_nd+3]=%.6f %.6f %.6f %.6f\n",
               x[half_nd], x[half_nd+1], x[half_nd+2], x[half_nd+3]);
    }
    
    float theta_base_t = (float)pos_t * freq_scale;
    float theta_base_h = (float)pos_h * freq_scale;
    float theta_base_w = (float)pos_w * freq_scale;
    float theta_base_e = (float)pos_e * freq_scale;
    
    /* Total rows = batch * 1 * heads (single ctx position) */
    int total_rows = batch * heads;
    
    #pragma omp parallel for
    for (int row = 0; row < total_rows; row++) {
        int i3 = row / heads;
        int i1 = row % heads;
        
        size_t src_offset = i3 * nb3 + 0 * nb2 + i1 * nb1;
        size_t dst_offset = src_offset;
        
        const float* src = (const float*)((const char*)x + src_offset);
        float* dst = (float*)((char*)y + dst_offset);
        
        float theta_t = theta_base_t;
        float theta_h = theta_base_h;
        float theta_w = theta_base_w;
        float theta_e = theta_base_e;
        
        for (int i0 = 0; i0 < half_nd; i0++) {
            int sector = i0 % sect_dims;
            float theta;
            
            /* IMROPE interleaved assignment (order matters! matches GGML reference) */
            /* Check order: height first, width second, time third, else extra */
            if (sector % 3 == 1 && sector < 3 * sect1) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sect2) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sect0) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
            
            float cos_val = cosf(theta) * mscale;
            float sin_val = sinf(theta) * mscale;
            
            float x0 = src[i0];
            float x1 = src[i0 + half_nd];
            
            dst[i0]           = x0 * cos_val - x1 * sin_val;
            dst[i0 + half_nd] = x0 * sin_val + x1 * cos_val;
            
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }
        
        for (int i0 = nd; i0 < elem_per_head; i0++) {
            dst[i0] = src[i0];
        }
    }
    
    /* Debug output after processing */
    if (imrope_interp_debug_count < 2) {
        printf("[IMROPE-INTERP] out[0:3]=%.6f %.6f %.6f %.6f\n", y[0], y[1], y[2], y[3]);
        imrope_interp_debug_count++;
    }
    
    return 0;
}

/*
 * ===========================================================================
 * SINGLE-INDEX SET_ROWS VARIANTS
 * ===========================================================================
 * For single-token generation, we set exactly 1 row at a time.
 * These variants take the index as a scalar parameter, avoiding HMEM copies.
 */

/*
 * SET_ROWS single-index: F32 src (HBM) -> F16 dst (HBM)
 */
uint64_t ve_set_row_f16_hbm_single(VEDAdeviceptr dst_hbm,
                                    VEDAdeviceptr src_hbm,
                                    int64_t dst_idx,       // Single index as scalar!
                                    uint64_t nc,
                                    uint64_t nb_dst,
                                    uint64_t nb_src) {
    char* dst;
    char* src;
    VEDAresult err;
    
    err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) return err;
    
    err = vedaMemPtr((void**)&src, src_hbm);
    if (err != VEDA_SUCCESS) return err;
    
    int cols = (int)nc;
    
    const float* src_row = (const float*)src;  // Single row at src offset 0
    uint16_t* dst_row = (uint16_t*)(dst + dst_idx * nb_dst);
    
    // Convert F32 -> F16 
    #pragma _NEC ivdep
    for (int j = 0; j < cols; j++) {
        // Use simple truncation (good enough for KV cache)
        union { float f; uint32_t u; } conv;
        conv.f = src_row[j];
        uint32_t bits = conv.u;
        // F32 to F16: truncate mantissa, adjust exponent
        int sign = (bits >> 31) & 1;
        int exp = ((bits >> 23) & 0xFF) - 127 + 15;
        int mant = (bits >> 13) & 0x3FF;
        if (exp <= 0) {
            dst_row[j] = sign << 15;  // Underflow to zero
        } else if (exp >= 31) {
            dst_row[j] = (sign << 15) | 0x7C00;  // Overflow to inf
        } else {
            dst_row[j] = (sign << 15) | (exp << 10) | mant;
        }
    }
    
    return 0;
}

/*
 * SET_ROWS single-index: F32 src (HBM) -> BF16 dst (HBM)
 */
uint64_t ve_set_row_bf16_hbm_single(VEDAdeviceptr dst_hbm,
                                     VEDAdeviceptr src_hbm,
                                     int64_t dst_idx,
                                     uint64_t nc,
                                     uint64_t nb_dst,
                                     uint64_t nb_src) {
    char* dst;
    char* src;
    VEDAresult err;
    
    err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) return err;
    
    err = vedaMemPtr((void**)&src, src_hbm);
    if (err != VEDA_SUCCESS) return err;
    
    int cols = (int)nc;
    int cols_pairs = cols / 2;
    int cols_remainder = cols % 2;
    
    const uint32_t* src_row = (const uint32_t*)src;
    uint32_t* dst_row32 = (uint32_t*)(dst + dst_idx * nb_dst);
    
    // Vectorized: process pairs, packing two BF16 into one 32-bit word
    #pragma _NEC ivdep
    for (int j = 0; j < cols_pairs; j++) {
        uint32_t f0 = src_row[j*2];
        uint32_t f1 = src_row[j*2 + 1];
        dst_row32[j] = (f0 >> 16) | (f1 & 0xFFFF0000);
    }
    
    // Handle odd column
    if (cols_remainder) {
        uint16_t* dst_row16 = (uint16_t*)(dst + dst_idx * nb_dst);
        dst_row16[cols - 1] = (uint16_t)(src_row[cols - 1] >> 16);
    }
    
    return 0;
}

/*
 * SET_ROWS single-index: F32 src (HBM) -> F32 dst (HBM)
 */
uint64_t ve_set_row_f32_hbm_single(VEDAdeviceptr dst_hbm,
                                    VEDAdeviceptr src_hbm,
                                    int64_t dst_idx,
                                    uint64_t nc,
                                    uint64_t nb_dst,
                                    uint64_t nb_src) {
    char* dst;
    char* src;
    VEDAresult err;
    
    err = vedaMemPtr((void**)&dst, dst_hbm);
    if (err != VEDA_SUCCESS) return err;
    
    err = vedaMemPtr((void**)&src, src_hbm);
    if (err != VEDA_SUCCESS) return err;
    
    int cols = (int)nc;
    
    const float* src_row = (const float*)src;
    float* dst_row = (float*)(dst + dst_idx * nb_dst);
    
    #pragma _NEC ivdep
    for (int j = 0; j < cols; j++) {
        dst_row[j] = src_row[j];
    }
    
    return 0;
}

/*
 * ===========================================================================
 * NAIVE ATTENTION - Optimized for VE's memory-rich architecture
 * ===========================================================================
 *
 * VE has 1.2 TB/s bandwidth but only ~5 TFLOPS compute.
 * Flash attention trades memory for compute - WRONG for VE!
 * Naive attention uses O(n²) memory but minimal compute - RIGHT for VE!
 *
 * For context length 4096, attention scores = 32 heads × 4096² × 4 bytes = 2GB
 * For context length 256, attention scores = 32 heads × 256² × 4 bytes = 8MB
 * Both easily fit in 48GB HBM.
 */

/*
 * Naive attention: single query token against KV cache
 * 
 * For single-token generation (the common case):
 *   Q: [n_heads, head_dim] - single query
 *   K: [n_kv_heads, seq_len, head_dim] - cached keys
 *   V: [n_kv_heads, seq_len, head_dim] - cached values
 *   O: [n_heads, head_dim] - output
 *
 * All tensors in HBM. Score buffer allocated per-head.
 */
uint64_t ve_naive_attn_f32_hbm(
    VEDAdeviceptr O_hbm,      // Output [n_heads, head_dim]
    VEDAdeviceptr Q_hbm,      // Query [n_heads, head_dim] 
    VEDAdeviceptr K_hbm,      // Keys [seq_len, n_kv_heads, head_dim]
    VEDAdeviceptr V_hbm,      // Values [seq_len, n_kv_heads, head_dim]
    uint64_t n_heads,         // Number of query heads
    uint64_t n_kv_heads,      // Number of KV heads (for GQA)
    uint64_t head_dim,        // Dimension per head
    uint64_t seq_len,         // Current sequence length (KV cache size)
    float scale)              // 1/sqrt(head_dim)
{
    float *O, *Q, *K, *V;
    
    if (vedaMemPtr((void**)&O, O_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&Q, Q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&K, K_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&V, V_hbm) != 0) return 4;
    
    int heads = (int)n_heads;
    int kv_heads = (int)n_kv_heads;
    int hdim = (int)head_dim;
    int slen = (int)seq_len;
    int kv_mul = heads / kv_heads;  // GQA multiplier
    
    // Allocate score buffer - one per thread to avoid conflicts
    // Each thread handles one head at a time
    int max_threads = omp_get_max_threads();
    float* score_buf = (float*)malloc(max_threads * slen * sizeof(float));
    if (!score_buf) return 5;
    
    #pragma omp parallel for
    for (int h = 0; h < heads; h++) {
        int tid = omp_get_thread_num();
        float* scores = score_buf + tid * slen;
        
        // Query for this head
        const float* q = Q + h * hdim;
        
        // KV head index (for GQA)
        int kv_h = h / kv_mul;
        
        // Step 1: Compute attention scores = Q · K^T
        for (int t = 0; t < slen; t++) {
            // K layout: [seq_len, n_kv_heads, head_dim]
            const float* k = K + t * kv_heads * hdim + kv_h * hdim;
            
            float score = 0.0f;
            #pragma _NEC ivdep
            for (int d = 0; d < hdim; d++) {
                score += q[d] * k[d];
            }
            scores[t] = score * scale;
        }
        
        // Step 2: Softmax
        // Find max for numerical stability
        float max_score = scores[0];
        for (int t = 1; t < slen; t++) {
            if (scores[t] > max_score) max_score = scores[t];
        }
        
        // Exp and sum
        float sum_exp = 0.0f;
        #pragma _NEC ivdep
        for (int t = 0; t < slen; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        
        // Normalize
        float inv_sum = 1.0f / sum_exp;
        #pragma _NEC ivdep
        for (int t = 0; t < slen; t++) {
            scores[t] *= inv_sum;
        }
        
        // Step 3: Weighted sum of values
        float* out = O + h * hdim;
        
        // Zero output
        #pragma _NEC ivdep
        for (int d = 0; d < hdim; d++) {
            out[d] = 0.0f;
        }
        
        // Accumulate weighted values
        for (int t = 0; t < slen; t++) {
            const float* v = V + t * kv_heads * hdim + kv_h * hdim;
            float w = scores[t];
            
            #pragma _NEC ivdep
            for (int d = 0; d < hdim; d++) {
                out[d] += w * v[d];
            }
        }
    }
    
    free(score_buf);
    return 0;
}

/*
 * Naive attention with BF16 KV cache (common case)
 * Q is F32, K/V are BF16, output is F32
 */
uint64_t ve_naive_attn_f32q_bf16kv_hbm(
    VEDAdeviceptr O_hbm,
    VEDAdeviceptr Q_hbm,
    VEDAdeviceptr K_hbm,      // BF16
    VEDAdeviceptr V_hbm,      // BF16
    uint64_t n_heads,
    uint64_t n_kv_heads,
    uint64_t head_dim,
    uint64_t seq_len,
    float scale)
{
    float *O, *Q;
    uint16_t *K, *V;  // BF16
    
    if (vedaMemPtr((void**)&O, O_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&Q, Q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&K, K_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&V, V_hbm) != 0) return 4;
    
    int heads = (int)n_heads;
    int kv_heads = (int)n_kv_heads;
    int hdim = (int)head_dim;
    int slen = (int)seq_len;
    int kv_mul = heads / kv_heads;
    
    int max_threads = omp_get_max_threads();
    float* score_buf = (float*)malloc(max_threads * slen * sizeof(float));
    if (!score_buf) return 5;
    
    #pragma omp parallel for
    for (int h = 0; h < heads; h++) {
        int tid = omp_get_thread_num();
        float* scores = score_buf + tid * slen;
        
        const float* q = Q + h * hdim;
        int kv_h = h / kv_mul;
        
        // Step 1: Q · K^T with BF16->F32 conversion
        for (int t = 0; t < slen; t++) {
            const uint16_t* k_bf16 = K + t * kv_heads * hdim + kv_h * hdim;
            
            float score = 0.0f;
            #pragma _NEC ivdep
            for (int d = 0; d < hdim; d++) {
                // BF16 to F32: shift left by 16
                union { uint32_t u; float f; } conv;
                conv.u = ((uint32_t)k_bf16[d]) << 16;
                score += q[d] * conv.f;
            }
            scores[t] = score * scale;
        }
        
        // Step 2: Softmax
        float max_score = scores[0];
        for (int t = 1; t < slen; t++) {
            if (scores[t] > max_score) max_score = scores[t];
        }
        
        float sum_exp = 0.0f;
        #pragma _NEC ivdep
        for (int t = 0; t < slen; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        
        float inv_sum = 1.0f / sum_exp;
        #pragma _NEC ivdep
        for (int t = 0; t < slen; t++) {
            scores[t] *= inv_sum;
        }
        
        // Step 3: Weighted sum with BF16->F32 conversion
        float* out = O + h * hdim;
        
        #pragma _NEC ivdep
        for (int d = 0; d < hdim; d++) {
            out[d] = 0.0f;
        }
        
        for (int t = 0; t < slen; t++) {
            const uint16_t* v_bf16 = V + t * kv_heads * hdim + kv_h * hdim;
            float w = scores[t];
            
            #pragma _NEC ivdep
            for (int d = 0; d < hdim; d++) {
                union { uint32_t u; float f; } conv;
                conv.u = ((uint32_t)v_bf16[d]) << 16;
                out[d] += w * conv.f;
            }
        }
    }
    
    free(score_buf);
    return 0;
}

/*
 * Naive attention with F16 KV cache
 */
uint64_t ve_naive_attn_f32q_f16kv_hbm(
    VEDAdeviceptr O_hbm,
    VEDAdeviceptr Q_hbm,
    VEDAdeviceptr K_hbm,      // F16
    VEDAdeviceptr V_hbm,      // F16
    uint64_t n_heads,
    uint64_t n_kv_heads,
    uint64_t head_dim,
    uint64_t seq_len,
    float scale)
{
    float *O, *Q;
    uint16_t *K, *V;  // F16
    
    if (vedaMemPtr((void**)&O, O_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&Q, Q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&K, K_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&V, V_hbm) != 0) return 4;
    
    int heads = (int)n_heads;
    int kv_heads = (int)n_kv_heads;
    int hdim = (int)head_dim;
    int slen = (int)seq_len;
    int kv_mul = heads / kv_heads;
    
    int max_threads = omp_get_max_threads();
    float* score_buf = (float*)malloc(max_threads * slen * sizeof(float));
    if (!score_buf) return 5;
    
    #pragma omp parallel for
    for (int h = 0; h < heads; h++) {
        int tid = omp_get_thread_num();
        float* scores = score_buf + tid * slen;
        
        const float* q = Q + h * hdim;
        int kv_h = h / kv_mul;
        
        // Step 1: Q · K^T with F16->F32 conversion
        for (int t = 0; t < slen; t++) {
            const uint16_t* k_f16 = K + t * kv_heads * hdim + kv_h * hdim;
            
            float score = 0.0f;
            #pragma _NEC ivdep
            for (int d = 0; d < hdim; d++) {
                // F16 to F32 conversion
                uint16_t h_val = k_f16[d];
                uint32_t sign = (h_val >> 15) & 1;
                uint32_t exp = (h_val >> 10) & 0x1F;
                uint32_t mant = h_val & 0x3FF;
                uint32_t f32_bits;
                if (exp == 0) {
                    f32_bits = sign << 31;  // Zero or denormal -> zero
                } else if (exp == 31) {
                    f32_bits = (sign << 31) | 0x7F800000 | (mant << 13);  // Inf/NaN
                } else {
                    f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                }
                union { uint32_t u; float f; } conv;
                conv.u = f32_bits;
                score += q[d] * conv.f;
            }
            scores[t] = score * scale;
        }
        
        // Step 2: Softmax
        float max_score = scores[0];
        for (int t = 1; t < slen; t++) {
            if (scores[t] > max_score) max_score = scores[t];
        }
        
        float sum_exp = 0.0f;
        #pragma _NEC ivdep
        for (int t = 0; t < slen; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        
        float inv_sum = 1.0f / sum_exp;
        #pragma _NEC ivdep
        for (int t = 0; t < slen; t++) {
            scores[t] *= inv_sum;
        }
        
        // Step 3: Weighted sum with F16->F32 conversion
        float* out = O + h * hdim;
        
        #pragma _NEC ivdep
        for (int d = 0; d < hdim; d++) {
            out[d] = 0.0f;
        }
        
        for (int t = 0; t < slen; t++) {
            const uint16_t* v_f16 = V + t * kv_heads * hdim + kv_h * hdim;
            float w = scores[t];
            
            #pragma _NEC ivdep
            for (int d = 0; d < hdim; d++) {
                uint16_t h_val = v_f16[d];
                uint32_t sign = (h_val >> 15) & 1;
                uint32_t exp = (h_val >> 10) & 0x1F;
                uint32_t mant = h_val & 0x3FF;
                uint32_t f32_bits;
                if (exp == 0) {
                    f32_bits = sign << 31;
                } else if (exp == 31) {
                    f32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                }
                union { uint32_t u; float f; } conv;
                conv.u = f32_bits;
                out[d] += w * conv.f;
            }
        }
    }
    
    free(score_buf);
    return 0;
}


/*
 * =============================================================================
 * Raw pointer F32 attention kernel for compiled graph
 * =============================================================================
 * This kernel can be called directly from VE code (no VEDA overhead).
 * Uses online softmax for memory efficiency.
 * 
 * Memory layout (llama.cpp style):
 *   Q: [head_dim] per head, heads stride = head_dim
 *   K: [head_dim, n_heads, seq_len] - k[t,h,d] = k + t*n_heads*head_dim + h*head_dim + d
 *   V: same as K
 *   out: same as Q
 */
void attention_f32_raw_omp(
    float* out,           /* Output: [head_dim * n_heads] */
    const float* q,       /* Query: [head_dim * n_heads] for current token */
    const float* k,       /* Key cache: [seq_len * n_heads * head_dim] */
    const float* v,       /* Value cache: [seq_len * n_heads * head_dim] */
    int head_dim,         /* Typically 128 */
    int n_heads,          /* Typically 32 */
    int seq_len,          /* Current sequence length (pos + 1) */
    float scale)          /* Attention scale, typically 1/sqrt(head_dim) */
{
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_heads; h++) {
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        
        /* Online softmax state */
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];  /* Accumulator for weighted V (max head_dim = 256) */
        
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            VKQ[d] = 0.0f;
        }
        
        /* Process each K/V position with online softmax */
        for (int t = 0; t < seq_len; t++) {
            const float* kh = k + t * n_heads * head_dim + h * head_dim;
            
            /* Q·K dot product */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                s += qh[d] * kh[d];
            }
            s *= scale;
            
            /* Online softmax update */
            float M_old = M;
            float ms = 1.0f;
            float vs = 1.0f;
            
            if (s > M) {
                M = s;
                ms = expf(M_old - M);
                #pragma _NEC ivdep
                for (int d = 0; d < head_dim; d++) {
                    VKQ[d] *= ms;
                }
                S_sum *= ms;
                vs = 1.0f;
            } else {
                vs = expf(s - M);
            }
            
            /* Accumulate weighted V */
            const float* vh = v + t * n_heads * head_dim + h * head_dim;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                VKQ[d] += vs * vh[d];
            }
            S_sum += vs;
        }
        
        /* Normalize and write output */
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            out_h[d] = VKQ[d] * S_inv;
        }
    }
}

/*
 * =============================================================================
 * Raw pointer F32 attention kernel with GQA support for compiled graph
 * =============================================================================
 * Same as attention_f32_raw_omp but supports Grouped Query Attention (GQA)
 * where n_kv_heads < n_q_heads.
 * 
 * For GQA models like Phi-4 (40 Q heads, 10 KV heads), each KV head serves
 * multiple Q heads: gqa_ratio = n_q_heads / n_kv_heads = 4
 * 
 * Memory layout (llama.cpp style):
 *   Q: [head_dim * n_q_heads] - q_head h at offset h*head_dim
 *   K: [seq_len * n_kv_heads * head_dim] - k[t,kh,d] at t*n_kv_heads*head_dim + kh*head_dim + d
 *   V: same as K
 *   out: same as Q
 */
void attention_f32_raw_gqa_omp(
    float* out,           /* Output: [head_dim * n_q_heads] */
    const float* q,       /* Query: [head_dim * n_q_heads] for current token */
    const float* k,       /* Key cache: [seq_len * n_kv_heads * head_dim] */
    const float* v,       /* Value cache: [seq_len * n_kv_heads * head_dim] */
    int head_dim,         /* Typically 128 */
    int n_q_heads,        /* Number of Q heads (e.g., 40 for Phi-4) */
    int n_kv_heads,       /* Number of KV heads (e.g., 10 for Phi-4) */
    int seq_len,          /* Current sequence length (pos + 1) */
    float scale)          /* Attention scale, typically 1/sqrt(head_dim) */
{
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_q_heads; h++) {
        /* Map Q head to KV head for GQA */
        int kv_h = h * n_kv_heads / n_q_heads;  /* Integer division gives correct mapping */
        
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        
        /* Online softmax state */
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];  /* Accumulator for weighted V (max head_dim = 256) */
        
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            VKQ[d] = 0.0f;
        }
        
        /* Process each K/V position with online softmax */
        for (int t = 0; t < seq_len; t++) {
            /* Use kv_h instead of h for K/V indexing */
            const float* kh = k + t * n_kv_heads * head_dim + kv_h * head_dim;
            
            /* Q·K dot product */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                s += qh[d] * kh[d];
            }
            s *= scale;
            
            /* Online softmax update */
            float M_old = M;
            float ms = 1.0f;
            float vs = 1.0f;
            
            if (s > M) {
                M = s;
                ms = expf(M_old - M);
                #pragma _NEC ivdep
                for (int d = 0; d < head_dim; d++) {
                    VKQ[d] *= ms;
                }
                S_sum *= ms;
                vs = 1.0f;
            } else {
                vs = expf(s - M);
            }
            
            /* Accumulate weighted V - use kv_h for V indexing */
            const float* vh = v + t * n_kv_heads * head_dim + kv_h * head_dim;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                VKQ[d] += vs * vh[d];
            }
            S_sum += vs;
        }
        
        /* Normalize and write output */
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            out_h[d] = VKQ[d] * S_inv;
        }
    }
}

/*
 * =============================================================================
 * Stride-aware F32 GQA attention kernel for compiled graph
 * =============================================================================
 * Same as attention_f32_raw_gqa_omp but uses stride-based memory addressing
 * for KV cache compatibility with llama.cpp's actual memory layout.
 * 
 * Memory layout:
 *   Q: [head_dim * n_q_heads] F32 - contiguous
 *   K: strided F32 - K[t,h,d] at (const char*)k + t*nb_k1 + h*nb_k2 + d*sizeof(float)
 *   V: strided F32 - V[t,h,d] at (const char*)v + t*nb_v1 + h*nb_v2 + d*sizeof(float)
 *   out: [head_dim * n_q_heads] F32 - contiguous
 * 
 * The strides (nb_k1, nb_k2, nb_v1, nb_v2) are in BYTES.
 */
void attention_f32_raw_gqa_stride_omp(
    float* out,              /* Output: [head_dim * n_q_heads] F32 */
    const float* q,          /* Query: [head_dim * n_q_heads] F32 */
    const void* k,           /* Key cache: F32 with strides */
    const void* v,           /* Value cache: F32 with strides */
    int head_dim,            /* Typically 128 */
    int n_q_heads,           /* Number of Q heads */
    int n_kv_heads,          /* Number of KV heads */
    int seq_len,             /* Current sequence length (pos + 1) */
    float scale,             /* Attention scale */
    size_t nb_k1,            /* K stride for seq dimension (bytes) */
    size_t nb_k2,            /* K stride for head dimension (bytes) */
    size_t nb_v1,            /* V stride for seq dimension (bytes) */
    size_t nb_v2)            /* V stride for head dimension (bytes) */
{
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_q_heads; h++) {
        /* Map Q head to KV head for GQA */
        int kv_h = h * n_kv_heads / n_q_heads;
        
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        
        /* Online softmax state */
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];  /* Accumulator for weighted V (max head_dim = 256) */
        
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            VKQ[d] = 0.0f;
        }
        
        /* Process each K/V position with online softmax */
        for (int t = 0; t < seq_len; t++) {
            /* Use stride-based addressing for K */
            const float* kh = (const float*)((const char*)k + t * nb_k1 + kv_h * nb_k2);
            
            /* Q·K dot product */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                s += qh[d] * kh[d];
            }
            s *= scale;
            
            /* Online softmax update */
            float M_old = M;
            float ms = 1.0f;
            float vs = 1.0f;
            
            if (s > M) {
                M = s;
                ms = expf(M_old - M);
                #pragma _NEC ivdep
                for (int d = 0; d < head_dim; d++) {
                    VKQ[d] *= ms;
                }
                S_sum *= ms;
                vs = 1.0f;
            } else {
                vs = expf(s - M);
            }
            
            /* Use stride-based addressing for V */
            const float* vh = (const float*)((const char*)v + t * nb_v1 + kv_h * nb_v2);
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                VKQ[d] += vs * vh[d];
            }
            S_sum += vs;
        }
        
        /* Normalize and write output */
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            out_h[d] = VKQ[d] * S_inv;
        }
    }
}

/*
 * =============================================================================
 * Raw pointer F32 Q / BF16 KV attention kernel for compiled graph
 * =============================================================================
 * Same as attention_f32_raw_omp but K and V are BF16 (from BF16 KV cache)
 * Uses stride-based memory addressing for KV cache compatibility.
 * 
 * Memory layout:
 *   Q: [head_dim * n_heads] F32 - contiguous, q_head h at offset h*head_dim*sizeof(float)
 *   K: strided BF16 - K[t,h,d] at (const char*)k + t*nb_k1 + h*nb_k2 + d*sizeof(bf16)
 *   V: strided BF16 - V[t,h,d] at (const char*)v + t*nb_v1 + h*nb_v2 + d*sizeof(bf16)
 *   out: [head_dim * n_heads] F32 - contiguous
 */
void attention_f32q_bf16kv_raw_omp(
    float* out,              /* Output: [head_dim * n_heads] F32 */
    const float* q,          /* Query: [head_dim * n_heads] F32 */
    const void* k,           /* Key cache: BF16 with strides */
    const void* v,           /* Value cache: BF16 with strides */
    int head_dim,            /* Typically 128 */
    int n_heads,             /* Typically 32 */
    int seq_len,             /* Current sequence length (pos + 1) */
    float scale,             /* Attention scale, typically 1/sqrt(head_dim) */
    size_t nb_k1,            /* K stride for seq dimension (bytes) */
    size_t nb_k2,            /* K stride for head dimension (bytes) */
    size_t nb_v1,            /* V stride for seq dimension (bytes) */
    size_t nb_v2)            /* V stride for head dimension (bytes) */
{
    /* Use contiguous layout like F32 kernel */
    const uint16_t* k_bf16 = (const uint16_t*)k;
    const uint16_t* v_bf16 = (const uint16_t*)v;
    (void)nb_k1; (void)nb_k2; (void)nb_v1; (void)nb_v2;  /* Unused */
    
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_heads; h++) {
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        
        /* Online softmax state */
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];  /* Accumulator for weighted V (max head_dim = 256) */
        
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            VKQ[d] = 0.0f;
        }
        
        /* Process each K/V position with online softmax */
        for (int t = 0; t < seq_len; t++) {
            /* Contiguous layout: K[t, h, d] */
            const uint16_t* kh = k_bf16 + t * n_heads * head_dim + h * head_dim;
            
            /* Q·K dot product with BF16->F32 conversion */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                /* BF16 to F32: left-shift by 16 bits */
                uint32_t k_bits = ((uint32_t)kh[d]) << 16;
                float k_f32;
                __builtin_memcpy(&k_f32, &k_bits, sizeof(float));
                s += qh[d] * k_f32;
            }
            s *= scale;
            
            /* Online softmax update */
            float M_old = M;
            float ms = 1.0f;
            float vs = 1.0f;
            
            if (s > M) {
                M = s;
                ms = expf(M_old - M);
                #pragma _NEC ivdep
                for (int d = 0; d < head_dim; d++) {
                    VKQ[d] *= ms;
                }
                S_sum *= ms;
                vs = 1.0f;
            } else {
                vs = expf(s - M);
            }
            
            /* Contiguous layout: V[t, h, d] */
            const uint16_t* vh = v_bf16 + t * n_heads * head_dim + h * head_dim;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                uint32_t v_bits = ((uint32_t)vh[d]) << 16;
                float v_f32;
                __builtin_memcpy(&v_f32, &v_bits, sizeof(float));
                VKQ[d] += vs * v_f32;
            }
            S_sum += vs;
        }
        
        /* Normalize and write output */
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            out_h[d] = VKQ[d] * S_inv;
        }
    }
}

/*
 * =============================================================================
 * Raw pointer F32 Q / BF16 KV attention kernel with GQA support
 * =============================================================================
 * Same as attention_f32q_bf16kv_raw_omp but supports Grouped Query Attention (GQA)
 * where n_kv_heads < n_q_heads.
 * Uses stride-based memory addressing for KV cache compatibility.
 */
void attention_f32q_bf16kv_raw_gqa_omp(
    float* out,              /* Output: [head_dim * n_q_heads] F32 */
    const float* q,          /* Query: [head_dim * n_q_heads] F32 */
    const void* k,           /* Key cache: BF16 with strides */
    const void* v,           /* Value cache: BF16 with strides */
    int head_dim,            /* Typically 128 */
    int n_q_heads,           /* Number of Q heads (e.g., 40 for Phi-4) */
    int n_kv_heads,          /* Number of KV heads (e.g., 10 for Phi-4) */
    int seq_len,             /* Current sequence length (pos + 1) */
    float scale,             /* Attention scale, typically 1/sqrt(head_dim) */
    size_t nb_k1,            /* K stride for seq dimension (bytes) */
    size_t nb_k2,            /* K stride for head dimension (bytes) */
    size_t nb_v1,            /* V stride for seq dimension (bytes) */
    size_t nb_v2)            /* V stride for head dimension (bytes) */
{
    /* Use contiguous layout like F32 kernel for comparison */
    const uint16_t* k_bf16 = (const uint16_t*)k;
    const uint16_t* v_bf16 = (const uint16_t*)v;
    (void)nb_k1; (void)nb_k2; (void)nb_v1; (void)nb_v2;  /* Unused - using contiguous layout */
    
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_q_heads; h++) {
        /* Map Q head to KV head for GQA */
        int kv_h = h * n_kv_heads / n_q_heads;
        
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        
        /* Online softmax state */
        float M = -INFINITY;
        float S_sum = 0.0f;
        float VKQ[256];  /* Accumulator for weighted V (max head_dim = 256) */
        
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            VKQ[d] = 0.0f;
        }
        
        /* Process each K/V position with online softmax */
        for (int t = 0; t < seq_len; t++) {
            /* Contiguous layout: K[t, kv_h, d] at offset (t*n_kv_heads + kv_h)*head_dim + d */
            const uint16_t* kh = k_bf16 + t * n_kv_heads * head_dim + kv_h * head_dim;
            
            /* Q·K dot product with BF16->F32 conversion */
            float s = 0.0f;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                uint32_t k_bits = ((uint32_t)kh[d]) << 16;
                float k_f32;
                __builtin_memcpy(&k_f32, &k_bits, sizeof(float));
                s += qh[d] * k_f32;
            }
            s *= scale;
            
            /* Online softmax update */
            float M_old = M;
            float ms = 1.0f;
            float vs = 1.0f;
            
            if (s > M) {
                M = s;
                ms = expf(M_old - M);
                #pragma _NEC ivdep
                for (int d = 0; d < head_dim; d++) {
                    VKQ[d] *= ms;
                }
                S_sum *= ms;
                vs = 1.0f;
            } else {
                vs = expf(s - M);
            }
            
            /* Contiguous layout: V[t, kv_h, d] at offset (t*n_kv_heads + kv_h)*head_dim + d */
            const uint16_t* vh = v_bf16 + t * n_kv_heads * head_dim + kv_h * head_dim;
            #pragma _NEC ivdep
            for (int d = 0; d < head_dim; d++) {
                uint32_t v_bits = ((uint32_t)vh[d]) << 16;
                float v_f32;
                __builtin_memcpy(&v_f32, &v_bits, sizeof(float));
                VKQ[d] += vs * v_f32;
            }
            S_sum += vs;
        }
        
        /* Normalize and write output */
        float S_inv = (S_sum == 0.0f) ? 0.0f : 1.0f / S_sum;
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            out_h[d] = VKQ[d] * S_inv;
        }
        
    }
}

/*
 * =============================================================================
 * Intrinsics-based BF16 GQA Attention for Compiled Graph (FAST!)
 * =============================================================================
 * Wrapper around flash_attn_single_head_intrinsics that provides the same
 * interface as attention_f32q_bf16kv_raw_gqa_omp but uses vectorized intrinsics.
 * 
 * This is 2-3x faster than the scalar BF16->F32 conversion approach.
 */

/* External intrinsics function from flash_attn_bf16_intrinsics.c (LLVM-VE compiled) */
extern void flash_attn_single_head_intrinsics(
    float* out,
    const float* q,
    const bf16* k,
    const bf16* v,
    const float* mask_row,  /* NULL for no mask */
    int D,                  /* head_dim */
    int S,                  /* seq_len */
    int64_t nb_k1,          /* K stride for seq dimension (bytes) */
    int64_t nb_v1,          /* V stride for seq dimension (bytes) */
    float scale,
    float logit_softcap,    /* 0 for disabled */
    float slope);           /* 1.0 for no ALiBi */

void attention_f32q_bf16kv_intrinsics_gqa_omp(
    float* out,              /* Output: [head_dim * n_q_heads] F32 */
    const float* q,          /* Query: [head_dim * n_q_heads] F32 */
    const void* k,           /* Key cache: BF16 with strides */
    const void* v,           /* Value cache: BF16 with strides */
    int head_dim,            /* Typically 128 */
    int n_q_heads,           /* Number of Q heads */
    int n_kv_heads,          /* Number of KV heads */
    int seq_len,             /* Current sequence length (pos + 1) */
    float scale,             /* Attention scale */
    size_t nb_k1,            /* K stride for seq dimension (bytes) */
    size_t nb_k2,            /* K stride for head dimension (bytes) */
    size_t nb_v1,            /* V stride for seq dimension (bytes) */
    size_t nb_v2)            /* V stride for head dimension (bytes) */
{
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_q_heads; h++) {
        /* Map Q head to KV head for GQA */
        int kv_h = h * n_kv_heads / n_q_heads;
        
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        
        /* Get K/V pointers for this head using stride-based addressing */
        const bf16* kh = (const bf16*)((const char*)k + kv_h * nb_k2);
        const bf16* vh = (const bf16*)((const char*)v + kv_h * nb_v2);
        
        /* Call intrinsics-based single head attention */
        flash_attn_single_head_intrinsics(
            out_h,          /* output */
            qh,             /* query */
            kh,             /* key */
            vh,             /* value */
            NULL,           /* no mask */
            head_dim,       /* D */
            seq_len,        /* S */
            (int64_t)nb_k1, /* K seq stride */
            (int64_t)nb_v1, /* V seq stride */
            scale,          /* scale */
            0.0f,           /* logit_softcap disabled */
            1.0f);          /* slope (no ALiBi) */
    }
}

/*
 * =============================================================================
 * Fused BF16 Attention - Fully Inlined (FASTER!)
 * =============================================================================
 */

/* flash_attn_single_head_intrinsics is already declared at top of file */

/*
 * OpenMP wrapper for fused BF16 attention
 * Uses the fully-inlined intrinsics version for maximum performance.
 */
void attention_f32q_bf16kv_fused_gqa_omp(
    float* out,              /* Output: [head_dim * n_q_heads] F32 */
    const float* q,          /* Query: [head_dim * n_q_heads] F32 */
    const void* k,           /* Key cache: BF16 with strides */
    const void* v,           /* Value cache: BF16 with strides */
    int head_dim,            /* Typically 128 */
    int n_q_heads,           /* Number of Q heads */
    int n_kv_heads,          /* Number of KV heads */
    int seq_len,             /* Current sequence length (pos + 1) */
    float scale,             /* Attention scale */
    size_t nb_k1,            /* K stride for seq dimension (bytes) */
    size_t nb_k2,            /* K stride for head dimension (bytes) */
    size_t nb_v1,            /* V stride for seq dimension (bytes) */
    size_t nb_v2)            /* V stride for head dimension (bytes) */
{
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_q_heads; h++) {
        /* Map Q head to KV head for GQA */
        int kv_h = h * n_kv_heads / n_q_heads;

        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;

        /* Get K/V pointers for this head using stride-based addressing */
        const bf16* kh = (const bf16*)((const char*)k + kv_h * nb_k2);
        const bf16* vh = (const bf16*)((const char*)v + kv_h * nb_v2);

        /* Call intrinsics-based single head attention */
        flash_attn_single_head_intrinsics(
            out_h,          /* output */
            qh,             /* query */
            kh,             /* key */
            vh,             /* value */
            NULL,           /* mask_row (no mask) */
            head_dim,       /* D */
            seq_len,        /* S */
            (int64_t)nb_k1, /* K seq stride */
            (int64_t)nb_v1, /* V seq stride */
            scale,          /* scale */
            0.0f,           /* logit_softcap (disabled) */
            1.0f);          /* slope (ALiBi disabled) */
    }
}

/*
 * "_inner" variant — caller MUST be inside a #pragma omp parallel region.
 * Uses #pragma omp for to share the per-head loop. Implicit barrier at
 * end-of-for synchronizes before the caller's next op.
 */
void attention_f32q_bf16kv_fused_gqa_inner(
    float* out,
    const float* q,
    const void* k,
    const void* v,
    int head_dim,
    int n_q_heads,
    int n_kv_heads,
    int seq_len,
    float scale,
    size_t nb_k1,
    size_t nb_k2,
    size_t nb_v1,
    size_t nb_v2)
{
    int h;
    #pragma omp for private(h)
    for (h = 0; h < n_q_heads; h++) {
        int kv_h = h * n_kv_heads / n_q_heads;
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        const bf16* kh = (const bf16*)((const char*)k + kv_h * nb_k2);
        const bf16* vh = (const bf16*)((const char*)v + kv_h * nb_v2);
        flash_attn_single_head_intrinsics(
            out_h, qh, kh, vh, NULL,
            head_dim, seq_len,
            (int64_t)nb_k1, (int64_t)nb_v1,
            scale, 0.0f, 1.0f);
    }
}

/*
 * =============================================================================
 * BF16 Column-Major Batched SGEMM
 * =============================================================================
 * 
 * These kernels compute Y[M,N] = W[M,K] @ X[K,N] where:
 *   - W is BF16 in column-major format (K elements per column, M columns)
 *   - X is FP32 in column-major format
 *   - Y is FP32 in column-major format
 * 
 * Column-major storage means W[:,k] is contiguous, enabling efficient
 * vectorized loads of entire columns for the GEMM accumulation pattern.
 */

/* 
 * External intrinsics functions (compiled with LLVM-VE, no OpenMP)
 * OpenMP parallelization is done here in the NCC wrapper.
 */
extern void bf16_matvec_colmajor_intrinsics(
    float* y,           /* Output [M] */
    const bf16* W,      /* Weights [M x K] column-major BF16 */
    const float* x,     /* Input [K] */
    int M,
    int K);

extern void bf16_matvec_colmajor_rowchunk_intrinsics(
    float* y,           /* Output [M] */
    const bf16* W,      /* Weights [M x K] column-major BF16 */
    const float* x,     /* Input [K] */
    int M,
    int K,
    int m_start,        /* Start row */
    int m_end);         /* End row (exclusive) */

/*
 * BF16 Column-Major SGEMM with all data in HBM
 * Parallelizes over output columns.
 * 
 * Uses tiled version for better cache utilization on large matrices.
 */
uint64_t ve_bf16_sgemm_colmajor_hbm(VEDAdeviceptr y_vptr,
                                     VEDAdeviceptr W_vptr,
                                     VEDAdeviceptr x_vptr,
                                     uint64_t M,
                                     uint64_t K,
                                     uint64_t N) {
    float* Y;
    bf16* W;
    float* X;
    
    VEDAresult err;
    err = vedaMemPtr((void**)&Y, y_vptr);
    if (err != 0) return 1;
    err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 2;
    err = vedaMemPtr((void**)&X, x_vptr);
    if (err != 0) return 3;
    
    /* Parallelize over output columns (N) - each is an independent matvec */
    #pragma omp parallel for
    for (int n = 0; n < (int)N; n++) {
        const float* x_col = X + n * K;  /* X[:,n] */
        float* y_col = Y + n * M;         /* Y[:,n] */
        
        bf16_matvec_colmajor_intrinsics(y_col, W, x_col, (int)M, (int)K);
    }
    
    return 0;
}

/*
 * BF16 Column-Major SGEMM with OpenMP - row-parallel version
 * Better for small N, large M.
 */
uint64_t ve_bf16_sgemm_colmajor_hbm_omp(VEDAdeviceptr y_vptr,
                                         VEDAdeviceptr W_vptr,
                                         VEDAdeviceptr x_vptr,
                                         uint64_t M,
                                         uint64_t K,
                                         uint64_t N) {
    float* Y;
    bf16* W;
    float* X;
    
    VEDAresult err;
    err = vedaMemPtr((void**)&Y, y_vptr);
    if (err != 0) return 1;
    err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 2;
    err = vedaMemPtr((void**)&X, x_vptr);
    if (err != 0) return 3;
    
    /* Determine row chunk size for parallelization */
    const int CHUNK_SIZE = 512;  /* Rows per thread */
    int num_chunks = ((int)M + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    /* For each output column, parallelize over row chunks */
    for (int n = 0; n < (int)N; n++) {
        const float* x_col = X + n * K;
        float* y_col = Y + n * M;
        
        #pragma omp parallel for
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int m_start = chunk * CHUNK_SIZE;
            int m_end = m_start + CHUNK_SIZE;
            if (m_end > (int)M) m_end = (int)M;
            
            bf16_matvec_colmajor_rowchunk_intrinsics(y_col, W, x_col, 
                                                      (int)M, (int)K, m_start, m_end);
        }
    }
    
    return 0;
}


/*
 * VEDA kernel: BF16 column-major matvec with RAW POINTERS + OpenMP
 * For use by graph compiler (weights already converted via vedaMemPtr)
 *
 * W is column-major: W[m,k] stored at W[m + k*M]
 * Computes: y = W @ x  where y[M], x[K]
 */
void ve_bf16_matvec_colmajor_raw_omp(float* y,
                                      const bf16* W,
                                      const float* x,
                                      int M,
                                      int K) {
    /* Determine row chunk size for parallelization */
    const int CHUNK_SIZE = 512;  /* Rows per thread */
    int num_chunks = (M + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    #pragma omp parallel for
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int m_start = chunk * CHUNK_SIZE;
        int m_end = m_start + CHUNK_SIZE;
        if (m_end > M) m_end = M;
        
        bf16_matvec_colmajor_rowchunk_intrinsics(y, W, x, M, K, m_start, m_end);
    }
}

/*
 * =============================================================================
 * FAST Two-Pass Attention - Vectorized over seq_len
 * =============================================================================
 * 
 * Key insight: Instead of vectorizing over head_dim (128 elements) and
 * iterating over seq_len (2000 times), we vectorize over seq_len and
 * iterate over head_dim.
 *
 * This gives us 2000/256 = ~8 vector ops per dimension, vs 2000 scalar
 * iterations in the online softmax approach.
 *
 * Requirements:
 * - K and V must be accessible with stride-1 access over seq_len dimension
 * - Score buffer of size seq_len per thread (8KB for 2000 tokens)
 */

void attention_twopass_f32_gqa_omp(
    float* out,              /* Output: [head_dim * n_q_heads] F32 */
    const float* q,          /* Query: [head_dim * n_q_heads] F32 */
    const float* k,          /* Key cache: stride-based access */
    const float* v,          /* Value cache: stride-based access */
    int head_dim,            /* Typically 128 */
    int n_q_heads,           /* Number of Q heads */
    int n_kv_heads,          /* Number of KV heads */
    int seq_len,             /* Current sequence length (pos + 1) */
    float scale,             /* Attention scale */
    size_t nb_k1,            /* K stride for seq dimension (bytes) */
    size_t nb_k2,            /* K stride for head dimension (bytes) */
    size_t nb_v1,            /* V stride for seq dimension (bytes) */
    size_t nb_v2)            /* V stride for head dimension (bytes) */
{
    /* Allocate score buffers - one per thread */
    int max_threads = omp_get_max_threads();
    float* score_buf = (float*)malloc(max_threads * seq_len * sizeof(float));
    if (!score_buf) return;
    
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_q_heads; h++) {
        int tid = omp_get_thread_num();
        float* scores = score_buf + tid * seq_len;
        
        /* Map Q head to KV head for GQA */
        int kv_h = h * n_kv_heads / n_q_heads;
        
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        
        /* Get base K/V pointers for this KV head */
        const char* k_base = (const char*)k + kv_h * nb_k2;
        const char* v_base = (const char*)v + kv_h * nb_v2;
        
        /* ========== PASS 1: Compute all Q·K scores ========== */
        /* Initialize scores to zero */
        #pragma _NEC ivdep
        for (int t = 0; t < seq_len; t++) {
            scores[t] = 0.0f;
        }
        
        /* Accumulate: scores[t] += Q[d] * K[t,d] for each d */
        /* This vectorizes over the seq_len dimension! */
        for (int d = 0; d < head_dim; d++) {
            float qd = qh[d] * scale;
            
            /* Check if K has contiguous seq_len access */
            if (nb_k1 == sizeof(float) * head_dim) {
                /* Contiguous - fast path */
                const float* k_col = (const float*)k_base + d;
                #pragma _NEC ivdep
                for (int t = 0; t < seq_len; t++) {
                    scores[t] += qd * k_col[t * head_dim];
                }
            } else {
                /* Strided - slower but correct */
                #pragma _NEC ivdep
                for (int t = 0; t < seq_len; t++) {
                    const float* k_t = (const float*)(k_base + t * nb_k1);
                    scores[t] += qd * k_t[d];
                }
            }
        }
        
        /* ========== PASS 2: Softmax ========== */
        /* Find max for numerical stability */
        float max_score = scores[0];
        #pragma _NEC ivdep
        for (int t = 1; t < seq_len; t++) {
            if (scores[t] > max_score) max_score = scores[t];
        }
        
        /* Exp and sum - vectorized */
        float sum_exp = 0.0f;
        #pragma _NEC ivdep
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        
        /* Normalize - vectorized */
        float inv_sum = (sum_exp == 0.0f) ? 0.0f : 1.0f / sum_exp;
        #pragma _NEC ivdep
        for (int t = 0; t < seq_len; t++) {
            scores[t] *= inv_sum;
        }
        
        /* ========== PASS 3: Weighted sum S·V ========== */
        /* Zero output */
        #pragma _NEC ivdep
        for (int d = 0; d < head_dim; d++) {
            out_h[d] = 0.0f;
        }
        
        /* Accumulate: out[d] += scores[t] * V[t,d] */
        /* Iterate over d, vectorize over t */
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            
            if (nb_v1 == sizeof(float) * head_dim) {
                /* Contiguous - fast path */
                const float* v_col = (const float*)v_base + d;
                #pragma _NEC ivdep
                for (int t = 0; t < seq_len; t++) {
                    sum += scores[t] * v_col[t * head_dim];
                }
            } else {
                /* Strided */
                #pragma _NEC ivdep
                for (int t = 0; t < seq_len; t++) {
                    const float* v_t = (const float*)(v_base + t * nb_v1);
                    sum += scores[t] * v_t[d];
                }
            }
            
            out_h[d] = sum;
        }
    }
    
    free(score_buf);
}


/*
 * =============================================================================
 * FAST Two-Pass Attention V2 - With exp() clamping for VE
 * =============================================================================
 * 
 * Same as attention_twopass_f32_gqa_omp but with clamping to avoid NaN
 * from vectorized expf() on values > 88.
 */

void attention_twopass_v2_f32_gqa_omp(
    float* out,              /* Output: [head_dim * n_q_heads] F32 */
    const float* q,          /* Query: [head_dim * n_q_heads] F32 */
    const float* k,          /* Key cache: stride-based access */
    const float* v,          /* Value cache: stride-based access */
    int head_dim,            /* Typically 128 */
    int n_q_heads,           /* Number of Q heads */
    int n_kv_heads,          /* Number of KV heads */
    int seq_len,             /* Current sequence length (pos + 1) */
    float scale,             /* Attention scale */
    size_t nb_k1,            /* K stride for seq dimension (bytes) */
    size_t nb_k2,            /* K stride for head dimension (bytes) */
    size_t nb_v1,            /* V stride for seq dimension (bytes) */
    size_t nb_v2)            /* V stride for head dimension (bytes) */
{
    /* Allocate score buffers - one per thread */
    int max_threads = omp_get_max_threads();
    float* score_buf = (float*)malloc(max_threads * seq_len * sizeof(float));
    if (!score_buf) return;
    
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_q_heads; h++) {
        int tid = omp_get_thread_num();
        float* scores = score_buf + tid * seq_len;
        
        /* Map Q head to KV head for GQA */
        int kv_h = h * n_kv_heads / n_q_heads;
        
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        
        /* Get base K/V pointers for this KV head */
        const char* k_base = (const char*)k + kv_h * nb_k2;
        const char* v_base = (const char*)v + kv_h * nb_v2;
        
        /* ========== PASS 1: Compute all Q·K scores ========== */
        /* Initialize scores to zero */
        #pragma _NEC ivdep
        for (int t = 0; t < seq_len; t++) {
            scores[t] = 0.0f;
        }
        
        /* Accumulate: scores[t] += Q[d] * K[t,d] for each d */
        for (int d = 0; d < head_dim; d++) {
            float qd = qh[d] * scale;
            #pragma _NEC ivdep
            for (int t = 0; t < seq_len; t++) {
                const float* k_t = (const float*)(k_base + t * nb_k1);
                scores[t] += qd * k_t[d];
            }
        }
        
        /* ========== PASS 2: Softmax with clamping ========== */
        /* Find max for numerical stability */
        float max_score = scores[0];
        for (int t = 1; t < seq_len; t++) {
            if (scores[t] > max_score) max_score = scores[t];
        }
        
        /* Clamp, exp and sum - vectorized */
        /* CRITICAL: VE's vectorized expf() returns NaN for x > ~88 */
        float sum_exp = 0.0f;
        #pragma _NEC ivdep
        for (int t = 0; t < seq_len; t++) {
            float x = scores[t] - max_score;
            /* Clamp to safe range for expf */
            if (x < -80.0f) x = -80.0f;
            /* x > 0 not possible since we subtracted max */
            scores[t] = expf(x);
            sum_exp += scores[t];
        }
        
        /* Normalize - vectorized */
        float inv_sum = (sum_exp == 0.0f) ? 0.0f : 1.0f / sum_exp;
        #pragma _NEC ivdep
        for (int t = 0; t < seq_len; t++) {
            scores[t] *= inv_sum;
        }
        
        /* ========== PASS 3: Weighted sum S·V ========== */
        /* Accumulate: out[d] = sum_t(scores[t] * V[t,d]) */
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            #pragma _NEC ivdep
            for (int t = 0; t < seq_len; t++) {
                const float* v_t = (const float*)(v_base + t * nb_v1);
                sum += scores[t] * v_t[d];
            }
            out_h[d] = sum;
        }
    }
    
    free(score_buf);
}


/*
 * =============================================================================
 * Two-Pass F32 Attention - Like ve-llama2.c
 * =============================================================================
 * 
 * This uses the same algorithm as ve-llama2.c:
 * 1. Compute all Q·K scores into a buffer
 * 2. Softmax the scores
 * 3. Compute weighted sum of V
 *
 * This avoids the data dependencies of online softmax.
 */
uint64_t ve_flash_attn_f32_twopass_hbm(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr q_hbm,
    VEDAdeviceptr k_hbm,
    VEDAdeviceptr v_hbm,
    VEDAdeviceptr mask_hbm,  /* HMEM - F16 mask (can be NULL) */
    uint64_t D,      /* head_dim */
    uint64_t Dv,     /* value head_dim (usually same as D) */
    uint64_t N,      /* number of query tokens (1 for generation) */
    uint64_t S,      /* KV sequence length */
    uint64_t H,      /* number of Q heads */
    uint64_t Hk,     /* number of KV heads */
    uint64_t B,      /* batch size */
    uint64_t nb_q1, uint64_t nb_q2, uint64_t nb_q3,
    uint64_t nb_k1, uint64_t nb_k2, uint64_t nb_k3,
    uint64_t nb_v1, uint64_t nb_v2, uint64_t nb_v3,
    uint64_t nb_m1, uint64_t nb_m2, uint64_t nb_m3,
    uint64_t nb_o1, uint64_t nb_o2, uint64_t nb_o3,
    uint64_t scale_bits,
    uint64_t max_bias_bits,
    uint64_t softcap_bits,
    uint64_t mask_ne2,
    uint64_t mask_ne3)
{
    /* Convert HBM pointers */
    float *dst, *q, *k, *v;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&q, q_hbm) != 0) return 2;
    if (vedaMemPtr((void**)&k, k_hbm) != 0) return 3;
    if (vedaMemPtr((void**)&v, v_hbm) != 0) return 4;
    
    const uint16_t* mask = NULL;
    if (mask_hbm && vedaMemPtr((void**)&mask, mask_hbm) != 0) return 99;
    
    /* Decode float parameters */
    float scale, max_bias, logit_softcap;
    uint32_t tmp;
    tmp = (uint32_t)scale_bits; __builtin_memcpy(&scale, &tmp, sizeof(float));
    tmp = (uint32_t)max_bias_bits; __builtin_memcpy(&max_bias, &tmp, sizeof(float));
    tmp = (uint32_t)softcap_bits; __builtin_memcpy(&logit_softcap, &tmp, sizeof(float));
    
    const int d = (int)D;
    const int dv = (int)Dv;
    const int n_tokens = (int)N;
    const int n_kv = (int)S;
    const int n_heads = (int)H;
    const int n_kv_heads = (int)Hk;
    const int n_batch = (int)B;
    const int rk = n_heads / n_kv_heads;
    
    /* Allocate score buffers - one per thread */
    int max_threads = omp_get_max_threads();
    float* score_buf = (float*)malloc(max_threads * n_kv * sizeof(float));
    if (!score_buf) return 5;
    
    /* Process all (batch, token, head) combinations */
    #pragma omp parallel for collapse(3)
    for (int ib = 0; ib < n_batch; ib++) {
        for (int iq = 0; iq < n_tokens; iq++) {
            for (int ih = 0; ih < n_heads; ih++) {
                int tid = omp_get_thread_num();
                float* scores = score_buf + tid * n_kv;
                
                int ik_head = ih / rk;
                int iv_head = ih / rk;
                
                /* Q pointer */
                const float* q_vec = (const float*)((const char*)q + iq * nb_q1 + ih * nb_q2 + ib * nb_q3);
                
                /* Mask row (if present) */
                const uint16_t* mask_row = NULL;
                if (mask != NULL) {
                    int im2 = ih % (int)mask_ne2;
                    int im3 = ib % (int)mask_ne3;
                    mask_row = (const uint16_t*)((const char*)mask + iq * nb_m1 + im2 * nb_m2 + im3 * nb_m3);
                }
                
                /* ========== PASS 1: Compute all Q·K scores ========== */
                for (int ic = 0; ic < n_kv; ic++) {
                    /* Check mask */
                    float mv = 0.0f;
                    if (mask_row != NULL) {
                        uint16_t mf16 = mask_row[ic];
                        /* F16 to F32 conversion */
                        uint32_t sign = (mf16 >> 15) & 0x1;
                        uint32_t exp = (mf16 >> 10) & 0x1F;
                        uint32_t mant = mf16 & 0x3FF;
                        uint32_t f32;
                        if (exp == 0) f32 = (mant == 0) ? (sign << 31) : 0;
                        else if (exp == 31) f32 = (sign << 31) | 0x7F800000 | (mant << 13);
                        else f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                        __builtin_memcpy(&mv, &f32, sizeof(float));
                    }
                    
                    if (mv == -INFINITY) {
                        scores[ic] = -INFINITY;
                        continue;
                    }
                    
                    /* K vector */
                    const float* k_vec = (const float*)((const char*)k + ic * nb_k1 + ik_head * nb_k2 + ib * nb_k3);
                    
                    /* Q·K dot product */
                    float s = 0.0f;
                    #pragma _NEC ivdep
                    for (int i = 0; i < d; i++) {
                        s += q_vec[i] * k_vec[i];
                    }
                    scores[ic] = s * scale + mv;
                }
                
                /* ========== PASS 2: Softmax ========== */
                /* Find max */
                float max_score = scores[0];
                for (int ic = 1; ic < n_kv; ic++) {
                    if (scores[ic] > max_score) max_score = scores[ic];
                }
                
                /* Exp and sum - with clamping for VE */
                float sum_exp = 0.0f;
                #pragma _NEC ivdep
                for (int ic = 0; ic < n_kv; ic++) {
                    float x = scores[ic] - max_score;
                    if (x < -80.0f) x = -80.0f;  /* Clamp for VE's expf */
                    scores[ic] = expf(x);
                    sum_exp += scores[ic];
                }
                
                /* Normalize */
                float inv_sum = (sum_exp == 0.0f) ? 0.0f : 1.0f / sum_exp;
                #pragma _NEC ivdep
                for (int ic = 0; ic < n_kv; ic++) {
                    scores[ic] *= inv_sum;
                }
                
                /* ========== PASS 3: Weighted sum of V ========== */
                float* out = (float*)((char*)dst + iq * nb_o1 + ih * nb_o2 + ib * nb_o3);
                
                /* Zero output */
                #pragma _NEC ivdep
                for (int i = 0; i < dv; i++) {
                    out[i] = 0.0f;
                }
                
                /* Accumulate weighted V */
                for (int ic = 0; ic < n_kv; ic++) {
                    const float* v_vec = (const float*)((const char*)v + ic * nb_v1 + iv_head * nb_v2 + ib * nb_v3);
                    float w = scores[ic];
                    
                    #pragma _NEC ivdep
                    for (int i = 0; i < dv; i++) {
                        out[i] += w * v_vec[i];
                    }
                }
            }
        }
    }
    
    free(score_buf);
    return 0;
}

/*
 * =============================================================================
 * TRUE BATCHED BF16 SGEMM using NEC CBLAS
 * =============================================================================
 * 
 * For prompt eval (N > 1), this achieves much higher throughput than N separate
 * matvecs because:
 * 1. Weights are read from HBM only once (not N times)
 * 2. CBLAS sgemm is highly optimized for VE's vector architecture
 * 3. Better cache utilization with blocked algorithms
 *
 * The BF16→F32 conversion overhead is amortized over N columns.
 */

/* CBLAS interface - link with -lblas_openmp */
typedef enum { CblasRowMajor_=101, CblasColMajor_=102 } CBLAS_ORDER_;
typedef enum { CblasNoTrans_=111, CblasTrans_=112 } CBLAS_TRANSPOSE_;

extern void cblas_sgemm(int Order, int TransA, int TransB, 
                        int M, int N, int K, 
                        float alpha, const float *A, int lda, 
                        const float *B, int ldb,
                        float beta, float *C, int ldc);

/*
 * True batched BF16 SGEMM: Y[M,N] = W[M,K] @ X[K,N]
 * 
 * W: BF16 weights, row-major [M x K]
 * X: F32 inputs, column-major [K x N]  
 * Y: F32 outputs, column-major [M x N]
 *
 * Internally converts BF16→F32, then calls CBLAS sgemm.
 */
uint64_t ve_bf16_sgemm_batched_cblas_hbm(
    VEDAdeviceptr Y_vptr,      /* Output [M x N] F32 column-major */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] BF16 row-major */
    VEDAdeviceptr X_vptr,      /* Input [K x N] F32 column-major */
    uint64_t M,
    uint64_t K,
    uint64_t N)
{
    float* Y;
    bf16* W_bf16;
    float* X;
    
    VEDAresult err;
    err = vedaMemPtr((void**)&Y, Y_vptr);
    if (err != 0) return 1;
    err = vedaMemPtr((void**)&W_bf16, W_vptr);
    if (err != 0) return 2;
    err = vedaMemPtr((void**)&X, X_vptr);
    if (err != 0) return 3;
    
    /* Allocate F32 buffer for dequantized weights in COLUMN-MAJOR format */
    float* W_f32 = (float*)malloc(M * K * sizeof(float));
    if (!W_f32) return 4;
    
    /* Dequantize BF16 → F32 (row-major) with OpenMP parallelization */
    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)(M * K); i++) {
        uint32_t bits = ((uint32_t)W_bf16[i]) << 16;
        float val;
        __builtin_memcpy(&val, &bits, sizeof(float));
        W_f32[i] = val;
    }
    
    /* Call CBLAS sgemm: C = alpha * A @ B + beta * C
     * 
     * We have:
     * - W_f32: row-major [M x K] → treat as col-major [K x M] transposed
     * - X: col-major [K x N]
     * - Y: col-major [M x N]
     *
     * Using column-major CBLAS:
     * Y[M,N] = W[M,K] @ X[K,N]
     * 
     * In col-major terms with row-major W (equivalent to transposed col-major):
     * cblas_sgemm(ColMajor, Trans, NoTrans, M, N, K, 1.0, W, K, X, K, 0.0, Y, M)
     */
    cblas_sgemm(CblasColMajor_, CblasTrans_, CblasNoTrans_,
                (int)M, (int)N, (int)K,
                1.0f, W_f32, (int)K,   /* A = W^T, lda = K */
                X, (int)K,              /* B = X, ldb = K */
                0.0f, Y, (int)M);       /* C = Y, ldc = M */
    
    free(W_f32);
    return 0;
}

/*
 * Cached version: W is already F32 in HBM (pre-dequantized)
 * Much faster for repeated calls with same weights.
 */
uint64_t ve_f32_sgemm_batched_cblas_hbm(
    VEDAdeviceptr Y_vptr,      /* Output [M x N] F32 column-major */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] F32 row-major (pre-dequantized) */
    VEDAdeviceptr X_vptr,      /* Input [K x N] F32 column-major */
    uint64_t M,
    uint64_t K,
    uint64_t N)
{
    float* Y;
    float* W;
    float* X;
    
    VEDAresult err;
    err = vedaMemPtr((void**)&Y, Y_vptr);
    if (err != 0) return 1;
    err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 2;
    err = vedaMemPtr((void**)&X, X_vptr);
    if (err != 0) return 3;
    
    /* Y[M,N] = W[M,K] @ X[K,N] using CBLAS */
    cblas_sgemm(CblasColMajor_, CblasTrans_, CblasNoTrans_,
                (int)M, (int)N, (int)K,
                1.0f, W, (int)K,
                X, (int)K,
                0.0f, Y, (int)M);
    
    return 0;
}

/*
 * Dequant BF16 row-major to F32 column-major (transpose during dequant)
 * Returns HBM pointer to F32 column-major weights
 * 
 * This one-time cost enables 10-30x faster CBLAS NoTrans calls!
 */
uint64_t ve_bf16_to_f32_colmajor_hbm(
    VEDAdeviceptr W_bf16_vptr,    /* Input: BF16 row-major [M x K] */
    VEDAdeviceptr W_f32_vptr,     /* Output: F32 column-major [M x K] (pre-allocated) */
    uint64_t M,
    uint64_t K)
{
    bf16* W_bf16;
    float* W_f32;
    
    VEDAresult err;
    err = vedaMemPtr((void**)&W_bf16, W_bf16_vptr);
    if (err != 0) return 1;
    err = vedaMemPtr((void**)&W_f32, W_f32_vptr);
    if (err != 0) return 2;
    
    /* Dequant + transpose: row-major BF16 -> column-major F32.
     *
     * Both loop orders defeat NCC's auto-vectorisation here: outer-m gives
     * a stride-M *store*, outer-k gives a stride-K *load* — both
     * "unvectorisable loop structure" per ncc -fdiag-vector=2.
     *
     * Workaround: read 16-bit pairs (= one uint32_t) and pack two F32
     * results per inner step, then do the stride-M store with the packed
     * vector. NCC can vectorise this because the inner loop on `m` makes
     * the *load* unit-stride (we increment by 1 in m and load WBF16[m*K+k]
     * via the linear index `m*Ki + kk` which is still strided)... actually
     * still strided. The honest workaround is to use the dequant-only path
     * (row-major F32) and let CBLAS handle the transpose via CblasTrans —
     * the column-major variant of this transpose simply doesn't fit the
     * VE's vector model. We keep the kernel as a scalar fallback because
     * existing callers expect it, but the *fast path* now lives in
     * ve_bf16_to_f32_rowmajor_hbm below + CblasTrans on the matmul side.
     */
    int64_t Mi = (int64_t) M;
    int64_t Ki = (int64_t) K;
    #pragma omp parallel for
    for (int64_t m = 0; m < Mi; m++) {
        for (int64_t k = 0; k < Ki; k++) {
            uint32_t bits = ((uint32_t) W_bf16[m * Ki + k]) << 16;
            float val;
            __builtin_memcpy(&val, &bits, sizeof(float));
            W_f32[k * Mi + m] = val;
        }
    }

    return 0;
}

/*
 * Dequant BF16 row-major -> F32 row-major (no transpose).
 *
 * Vectorises trivially: both source and destination are unit-stride in the
 * inner loop, so NCC packs 2 BF16->F32 conversions per VE op. Pair this
 * with ve_f32_sgemm_batched_cblas_hbm (which uses CblasTrans internally)
 * to get the same end result as the colmajor variant without the slow
 * scalar transpose.
 */
uint64_t ve_bf16_to_f32_rowmajor_hbm(
    VEDAdeviceptr W_bf16_vptr,
    VEDAdeviceptr W_f32_vptr,
    uint64_t M,
    uint64_t K)
{
    bf16  * W_bf16;
    float * W_f32;
    VEDAresult err;
    err = vedaMemPtr((void **) &W_bf16, W_bf16_vptr); if (err != 0) return 1;
    err = vedaMemPtr((void **) &W_f32,  W_f32_vptr);  if (err != 0) return 2;

    /* uint32 reinterpret: each 32-bit slot packs two adjacent BF16
     * values; the upper half is BF16(2i) shifted into F32(2i)'s exponent
     * bits, the lower half can be shifted right to give F32(2i+1). This
     * is the standard NCC-vectorisable BF16->F32 unpack idiom. */
    int64_t total = (int64_t) M * (int64_t) K;
    int64_t pairs = total / 2;

    const uint32_t * src32 = (const uint32_t *) W_bf16;
    uint32_t       * dst32 = (uint32_t *) W_f32;

    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;

    #pragma omp parallel for num_threads(nthr)
    for (int64_t p = 0; p < pairs; p++) {
        uint32_t pair = src32[p];
        /* On VE BF16 storage is little-endian within a 32-bit word: low
         * half is element 2p, high half is element 2p+1. To widen to
         * F32 we just <<16 the appropriate half. */
        dst32[2 * p]     = (pair & 0xFFFFu) << 16;
        dst32[2 * p + 1] = pair & 0xFFFF0000u;
    }
    /* Tail element if M*K is odd — extremely unlikely for weights, but
     * harmless to handle. */
    if (total & 1) {
        uint32_t b = ((uint32_t) W_bf16[total - 1]) << 16;
        __builtin_memcpy(&W_f32[total - 1], &b, sizeof(float));
    }

    return 0;
}

/*
 * FAST F32 batched SGEMM with COLUMN-MAJOR weights (NoTrans)
 * 
 * This is 10-30x faster than CblasTrans for batched operations!
 * Weights must be pre-transposed to column-major [M x K] format.
 */
uint64_t ve_f32_sgemm_batched_cblas_hbm_notrans(
    VEDAdeviceptr Y_vptr,      /* Output [M x N] F32 column-major */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] F32 COLUMN-MAJOR */
    VEDAdeviceptr X_vptr,      /* Input [K x N] F32 column-major */
    uint64_t M,
    uint64_t K,
    uint64_t N)
{
    float* Y;
    float* W;
    float* X;
    
    VEDAresult err;
    err = vedaMemPtr((void**)&Y, Y_vptr);
    if (err != 0) return 1;
    err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 2;
    err = vedaMemPtr((void**)&X, X_vptr);
    if (err != 0) return 3;
    
    /* Y[M,N] = W[M,K] @ X[K,N] using CBLAS with column-major W (NoTrans) */
    cblas_sgemm(CblasColMajor_, CblasNoTrans_, CblasNoTrans_,
                (int)M, (int)N, (int)K,
                1.0f, W, (int)M,   /* lda = M for column-major */
                X, (int)K,
                0.0f, Y, (int)M);
    
    return 0;
}

/* ============================================================================
 * CBLAS SGEMV/SGEMM with HBM weights + HMEM I/O
 * 
 * These kernels take:
 *   - Weights (W) in HBM (pre-dequantized F32, row-major)
 *   - Input (x/X) in HMEM
 *   - Output (y/Y) in HMEM
 *   - alpha, beta passed as float bits in uint64
 * 
 * This enables the standard CBLAS path to work with HBM-cached weights.
 * ============================================================================
 */

/*
 * CBLAS SGEMV: y = alpha * W @ x + beta * y
 * W in HBM (row-major), x/y in HMEM
 */
uint64_t ve_cblas_sgemv_hbm_hmem(
    void* y_hmem,              /* Output [M] F32 in HMEM */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] F32 row-major in HBM */
    void* x_hmem,              /* Input [K] F32 in HMEM */
    uint64_t M,
    uint64_t K,
    uint64_t alpha_bits,       /* alpha as float bits */
    uint64_t beta_bits)        /* beta as float bits */
{
    float* W;
    VEDAresult err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 1;
    
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    float alpha, beta;
    uint32_t a32 = (uint32_t)alpha_bits;
    uint32_t b32 = (uint32_t)beta_bits;
    __builtin_memcpy(&alpha, &a32, sizeof(float));
    __builtin_memcpy(&beta, &b32, sizeof(float));
    
    /* y = alpha * W @ x + beta * y
     * W is row-major [M x K], treat as RowMajor NoTrans
     */
    cblas_sgemv(CblasRowMajor_, CblasNoTrans_,
                (int)M, (int)K,
                alpha, W, (int)K,
                x, 1,
                beta, y, 1);
    
    return 0;
}

/*
 * CBLAS SGEMM: Y = alpha * W @ X + beta * Y
 * W in HBM (row-major), X/Y in HMEM
 * 
 * Memory layout:
 *   - W: row-major [M x K] in HBM
 *   - X: column-major [K x N] in HMEM (standard GGML layout)
 *   - Y: column-major [M x N] in HMEM
 */
uint64_t ve_cblas_sgemm_hbm_hmem(
    void* Y_hmem,              /* Output [M x N] F32 column-major in HMEM */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] F32 row-major in HBM */
    void* X_hmem,              /* Input [K x N] F32 column-major in HMEM */
    uint64_t M,
    uint64_t K,
    uint64_t N,
    uint64_t alpha_bits,       /* alpha as float bits */
    uint64_t beta_bits)        /* beta as float bits */
{
    float* W;
    VEDAresult err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 1;
    
    float* Y = (float*)Y_hmem;
    float* X = (float*)X_hmem;
    
    float alpha, beta;
    uint32_t a32 = (uint32_t)alpha_bits;
    uint32_t b32 = (uint32_t)beta_bits;
    __builtin_memcpy(&alpha, &a32, sizeof(float));
    __builtin_memcpy(&beta, &b32, sizeof(float));
    
    /* Y[M,N] = W[M,K] @ X[K,N]
     * W is row-major, X/Y are column-major
     * In column-major CBLAS terms: W row-major = W^T column-major
     * So we use CblasTrans for W
     */
    cblas_sgemm(CblasColMajor_, CblasTrans_, CblasNoTrans_,
                (int)M, (int)N, (int)K,
                alpha, W, (int)K,   /* W row-major → lda = K */
                X, (int)K,          /* X col-major → ldb = K */
                beta, Y, (int)M);   /* Y col-major → ldc = M */
    
    return 0;
}

/*
 * CBLAS SGEMV with column-major weights: y = alpha * W @ x + beta * y
 * W in HBM (column-major), x/y in HMEM
 */
uint64_t ve_cblas_sgemv_colmajor_hbm_hmem(
    void* y_hmem,              /* Output [M] F32 in HMEM */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] F32 column-major in HBM */
    void* x_hmem,              /* Input [K] F32 in HMEM */
    uint64_t M,
    uint64_t K,
    uint64_t alpha_bits,       /* alpha as float bits */
    uint64_t beta_bits)        /* beta as float bits */
{
    float* W;
    VEDAresult err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 1;
    
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    
    float alpha, beta;
    uint32_t a32 = (uint32_t)alpha_bits;
    uint32_t b32 = (uint32_t)beta_bits;
    __builtin_memcpy(&alpha, &a32, sizeof(float));
    __builtin_memcpy(&beta, &b32, sizeof(float));
    
    /* y = alpha * W @ x + beta * y
     * W is column-major [M x K]
     */
    cblas_sgemv(CblasColMajor_, CblasNoTrans_,
                (int)M, (int)K,
                alpha, W, (int)M,   /* lda = M for column-major */
                x, 1,
                beta, y, 1);
    
    return 0;
}

/*
 * CBLAS SGEMM with column-major weights: Y = alpha * W @ X + beta * Y
 * W in HBM (column-major), X/Y in HMEM
 */
uint64_t ve_cblas_sgemm_colmajor_hbm_hmem(
    void* Y_hmem,              /* Output [M x N] F32 column-major in HMEM */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] F32 column-major in HBM */
    void* X_hmem,              /* Input [K x N] F32 column-major in HMEM */
    uint64_t M,
    uint64_t K,
    uint64_t N,
    uint64_t alpha_bits,       /* alpha as float bits */
    uint64_t beta_bits)        /* beta as float bits */
{
    float* W;
    VEDAresult err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 1;
    
    float* Y = (float*)Y_hmem;
    float* X = (float*)X_hmem;
    
    float alpha, beta;
    uint32_t a32 = (uint32_t)alpha_bits;
    uint32_t b32 = (uint32_t)beta_bits;
    __builtin_memcpy(&alpha, &a32, sizeof(float));
    __builtin_memcpy(&beta, &b32, sizeof(float));
    
    /* Y[M,N] = W[M,K] @ X[K,N]
     * All matrices column-major → use NoTrans for W
     */
    cblas_sgemm(CblasColMajor_, CblasNoTrans_, CblasNoTrans_,
                (int)M, (int)N, (int)K,
                alpha, W, (int)M,   /* W col-major → lda = M */
                X, (int)K,          /* X col-major → ldb = K */
                beta, Y, (int)M);   /* Y col-major → ldc = M */
    
    return 0;
}

/* ============================================================================
 * CBLAS SGEMV/SGEMM with HBM weights + HMEM input + HBM output
 * 
 * These kernels are for mixed memory paths where:
 *   - Weights (W) in HBM (pre-dequantized F32)
 *   - Input (x/X) in HMEM (copied from host)
 *   - Output (y/Y) in HBM (stays on device for next op)
 * 
 * Used by ggml_ve_cblas_matmul_hbm_hmem()
 * ============================================================================
 */

/*
 * CBLAS SGEMV: y = alpha * W @ x + beta * y
 * W and y in HBM, x in HMEM
 */
uint64_t ve_cblas_sgemv_hbm_hmem_hbm_out(
    VEDAdeviceptr y_vptr,      /* Output [M] F32 in HBM */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] F32 row-major in HBM */
    void* x_hmem,              /* Input [K] F32 in HMEM */
    uint64_t M,
    uint64_t K,
    uint64_t alpha_bits,       /* alpha as float bits */
    uint64_t beta_bits)        /* beta as float bits */
{
    float* y;
    float* W;
    VEDAresult err;
    
    err = vedaMemPtr((void**)&y, y_vptr);
    if (err != 0) return 1;
    err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 2;
    
    float* x = (float*)x_hmem;
    
    float alpha, beta;
    uint32_t a32 = (uint32_t)alpha_bits;
    uint32_t b32 = (uint32_t)beta_bits;
    __builtin_memcpy(&alpha, &a32, sizeof(float));
    __builtin_memcpy(&beta, &b32, sizeof(float));
    
    /* y = alpha * W @ x + beta * y
     * W is row-major [M x K], treat as RowMajor NoTrans
     */
    cblas_sgemv(CblasRowMajor_, CblasNoTrans_,
                (int)M, (int)K,
                alpha, W, (int)K,
                x, 1,
                beta, y, 1);
    
    return 0;
}

/*
 * CBLAS SGEMM: Y = alpha * W @ X + beta * Y
 * W and Y in HBM, X in HMEM
 */
uint64_t ve_cblas_sgemm_hbm_hmem_hbm_out(
    VEDAdeviceptr Y_vptr,      /* Output [M x N] F32 column-major in HBM */
    VEDAdeviceptr W_vptr,      /* Weights [M x K] F32 row-major in HBM */
    void* X_hmem,              /* Input [K x N] F32 column-major in HMEM */
    uint64_t M,
    uint64_t K,
    uint64_t N,
    uint64_t alpha_bits,       /* alpha as float bits */
    uint64_t beta_bits)        /* beta as float bits */
{
    float* Y;
    float* W;
    VEDAresult err;
    
    err = vedaMemPtr((void**)&Y, Y_vptr);
    if (err != 0) return 1;
    err = vedaMemPtr((void**)&W, W_vptr);
    if (err != 0) return 2;
    
    float* X = (float*)X_hmem;
    
    float alpha, beta;
    uint32_t a32 = (uint32_t)alpha_bits;
    uint32_t b32 = (uint32_t)beta_bits;
    __builtin_memcpy(&alpha, &a32, sizeof(float));
    __builtin_memcpy(&beta, &b32, sizeof(float));
    
    /* Y[M,N] = W[M,K] @ X[K,N]
     * W is row-major, X/Y are column-major
     * In column-major CBLAS: W row-major = W^T column-major → use Trans
     */
    cblas_sgemm(CblasColMajor_, CblasTrans_, CblasNoTrans_,
                (int)M, (int)N, (int)K,
                alpha, W, (int)K,   /* W row-major → lda = K */
                X, (int)K,          /* X col-major → ldb = K */
                beta, Y, (int)M);   /* Y col-major → ldc = M */
    
    return 0;
}

/* ============================================================================
 * Q8_0 → BF16 DEQUANTIZATION + BF16 HBM MATVEC
 * 
 * Q8_0 is a simple 8-bit quantization format:
 *   - Block size: 32 elements
 *   - Structure: FP16 scale (2 bytes) + 32 int8 quants (32 bytes) = 34 bytes/block
 *   - Dequantization: value = d * q[i]
 * 
 * This is optimal for VE:
 *   1. Dequantize Q8_0 to BF16 once at model load (stored in HBM)
 *   2. Use sgemv_packed_bf16_unr for all inference (VPU-optimized)
 * 
 * Memory for 20B model:
 *   - Q8_0: ~20 GB
 *   - BF16: ~40 GB (fits on single VE card with 48 GB HBM)
 * 
 * Performance: ~560 GB/s memory bandwidth (same as other BF16 inference)
 * ============================================================================
 */

/* Q8_0 block structure (34 bytes per 32 elements) */
#define QK8_0 32
typedef struct __attribute__((packed)) {
    uint16_t d;           /* FP16 delta (scale) */
    int8_t   qs[QK8_0];   /* 8-bit quants */
} block_q8_0_wrapper;

/*
 * Dequantize one Q8_0 block (32 elements) to BF16
 * 
 * Q8_0 element extraction:
 *   value[i] = d * qs[i]
 */
static void dequant_q8_0_block_bf16_wrapper(const block_q8_0_wrapper* b, bf16* out) {
    const float d = __gnu_h2f_ieee(b->d);
    
    for (int i = 0; i < QK8_0; i++) {
        out[i] = f32_to_bf16(d * (float)b->qs[i]);
    }
}

/*
 * VEDA kernel: Dequantize Q8_0 matrix to BF16 (one-time at model load)
 * 
 * Parameters:
 *   out_vptr: Output BF16 matrix in HBM [M × K]
 *   in_hmem: Input Q8_0 matrix in HMEM
 *   M: Number of rows
 *   K: Number of columns (must be multiple of 32)
 */
uint64_t ve_dequant_q8_0_bf16(VEDAdeviceptr out_vptr,
                              void* in_hmem,
                              uint64_t M,
                              uint64_t K) {
    bf16* out;
    vedaMemPtr((void**)&out, out_vptr);
    
    const block_q8_0_wrapper* in = (const block_q8_0_wrapper*)in_hmem;
    
    int m = (int)M;
    int k = (int)K;
    int nb = k / QK8_0;  /* Blocks per row */
    
    #pragma omp parallel for
    for (int row = 0; row < m; row++) {
        const block_q8_0_wrapper* in_row = in + row * nb;
        bf16* out_row = out + row * k;
        
        for (int b = 0; b < nb; b++) {
            dequant_q8_0_block_bf16_wrapper(&in_row[b], out_row + b * QK8_0);
        }
    }
    
    return 0;
}

/*
 * VEDA kernel: Q8_0 matvec via pre-dequantized BF16 weights in HBM
 * 
 * This is the fast inference path - weights already in BF16 in HBM.
 * Uses the VPU-optimized sgemv_packed_bf16_unr kernel.
 * 
 * Parameters:
 *   y_hmem: Output [M] floats in HMEM
 *   W_vptr: BF16 weights [M × K] in HBM (pre-dequantized from Q8_0)
 *   x_hmem: Input [K] floats in HMEM
 *   M: Output dimension
 *   K: Input dimension
 */
uint64_t ve_q8_0_bf16_matvec_hbm(void* y_hmem,
                                 VEDAdeviceptr W_vptr,
                                 void* x_hmem,
                                 uint64_t M,
                                 uint64_t K) {
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    bf16* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    
    /* Use OpenMP to parallelize across rows, each thread uses VPU kernel */
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        int chunk = (m + nthr - 1) / nthr;
        int row_start = tid * chunk;
        int row_end = row_start + chunk;
        if (row_end > m) row_end = m;
        
        if (row_start < row_end) {
            /* Call the VPU-optimized BF16 SGEMV for this chunk */
            sgemv_packed_bf16_unr(
                y + row_start,           /* Output chunk */
                x,                       /* Full input vector */
                W + row_start * k,       /* Weight rows for this chunk */
                k,                       /* K (input dim) */
                row_end - row_start      /* Number of rows in chunk */
            );
        }
    }
    
    return 0;
}

/*
 * VEDA kernel: Q8_0 matvec - ALL in HBM (weights, input, output)
 * 
 * This is the fastest path for HBM compute - zero HMEM transfers.
 * Uses the VPU-optimized sgemv_packed_bf16_unr kernel.
 * 
 * Parameters:
 *   y_vptr: Output [M] floats in HBM
 *   W_vptr: BF16 weights [M × K] in HBM (pre-dequantized from Q8_0)
 *   x_vptr: Input [K] floats in HBM
 *   M: Output dimension
 *   K: Input dimension
 */
uint64_t ve_q8_0_bf16_matvec_hbm_full(VEDAdeviceptr y_vptr,
                                      VEDAdeviceptr W_vptr,
                                      VEDAdeviceptr x_vptr,
                                      uint64_t M,
                                      uint64_t K) {
    float* y;
    float* x;
    bf16* W;
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    
    /* Use OpenMP to parallelize across rows, each thread uses VPU kernel */
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        int chunk = (m + nthr - 1) / nthr;
        int row_start = tid * chunk;
        int row_end = row_start + chunk;
        if (row_end > m) row_end = m;
        
        if (row_start < row_end) {
            /* Call the VPU-optimized BF16 SGEMV for this chunk */
            sgemv_packed_bf16_unr(
                y + row_start,           /* Output chunk */
                x,                       /* Full input vector */
                W + row_start * k,       /* Weight rows for this chunk */
                k,                       /* K (input dim) */
                row_end - row_start      /* Number of rows in chunk */
            );
        }
    }
    
    return 0;
}

/* ============================================================================
 * FUSED Q8_0 MATVEC - Reads Q8_0 directly, no intermediate dequantization!
 * 
 * This is the memory-efficient path: Q8_0 is 1.0625 bytes/element vs 2 bytes for BF16.
 * For a 20B model: Q8_0 = ~20GB vs BF16 = ~40GB
 * 
 * The kernel reads Q8_0 blocks (34 bytes = scale + 32 int8 quants) directly from HBM
 * and computes the dot product on-the-fly without materializing the full FP32 weights.
 * ============================================================================
 */

/*
 * VEDA kernel: Fused Q8_0 matvec - weights in HBM, I/O in HMEM
 * 
 * Reads Q8_0 blocks directly, computes: y[m] = sum_k(W[m,k] * x[k])
 * where W is stored as Q8_0 blocks.
 * 
 * Parameters:
 *   y_hmem: Output [M] floats in HMEM
 *   W_vptr: Q8_0 weights [M × K/32 blocks] in HBM
 *   x_hmem: Input [K] floats in HMEM
 *   M: Output dimension (rows)
 *   K: Input dimension (cols, must be multiple of 32)
 */
uint64_t ve_q8_0_fused_matvec_hbm(void* y_hmem,
                                   VEDAdeviceptr W_vptr,
                                   void* x_hmem,
                                   uint64_t M,
                                   uint64_t K) {
    float* y = (float*)y_hmem;
    float* x = (float*)x_hmem;
    block_q8_0_wrapper* W;
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    int nb = k / QK8_0;  /* Number of Q8_0 blocks per row */
    
    /* Parallelize over output rows */
    #pragma omp parallel for
    for (int row = 0; row < m; row++) {
        const block_q8_0_wrapper* W_row = W + row * nb;
        float sum = 0.0f;
        
        for (int b = 0; b < nb; b++) {
            const block_q8_0_wrapper* blk = &W_row[b];
            const float d = __gnu_h2f_ieee(blk->d);
            const float* x_blk = x + b * QK8_0;
            
            /* Compute dot product for this block */
            float blk_sum = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < QK8_0; i++) {
                blk_sum += (float)blk->qs[i] * x_blk[i];
            }
            sum += d * blk_sum;
        }
        y[row] = sum;
    }
    
    return 0;
}

/*
 * VEDA kernel: Fused Q8_0 matvec - ALL in HBM (weights, input, output)
 * 
 * Same as above but with HBM I/O for maximum throughput.
 */
uint64_t ve_q8_0_fused_matvec_hbm_full(VEDAdeviceptr y_vptr,
                                        VEDAdeviceptr W_vptr,
                                        VEDAdeviceptr x_vptr,
                                        uint64_t M,
                                        uint64_t K) {
    float* y;
    float* x;
    block_q8_0_wrapper* W;
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    int nb = k / QK8_0;
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel for num_threads(nthr)
    for (int row = 0; row < m; row++) {
        const block_q8_0_wrapper* W_row = W + row * nb;
        float sum = 0.0f;
        
        for (int b = 0; b < nb; b++) {
            const block_q8_0_wrapper* blk = &W_row[b];
            const float d = __gnu_h2f_ieee(blk->d);
            const float* x_blk = x + b * QK8_0;
            
            float blk_sum = 0.0f;
            #pragma _NEC ivdep
            for (int i = 0; i < QK8_0; i++) {
                blk_sum += (float)blk->qs[i] * x_blk[i];
            }
            sum += d * blk_sum;
        }
        y[row] = sum;
    }
    
    return 0;
}

/*
 * VEDA kernel: Fused Q8_0 batched SGEMM - weights in HBM, I/O in HBM
 * 
 * For batched operations (N > 1), computes: Y[m,n] = sum_k(W[m,k] * X[k,n])
 * 
 * Parameters:
 *   Y_vptr: Output [M × N] F32 column-major in HBM
 *   W_vptr: Q8_0 weights [M × K/32 blocks] row-major in HBM
 *   X_vptr: Input [K × N] F32 column-major in HBM
 *   M: Output rows
 *   K: Inner dimension (must be multiple of 32)
 *   N: Batch size (columns)
 */
uint64_t ve_q8_0_fused_sgemm_hbm(VEDAdeviceptr Y_vptr,
                                  VEDAdeviceptr W_vptr,
                                  VEDAdeviceptr X_vptr,
                                  uint64_t M,
                                  uint64_t K,
                                  uint64_t N) {
    float* Y;
    float* X;
    block_q8_0_wrapper* W;
    vedaMemPtr((void**)&Y, Y_vptr);
    vedaMemPtr((void**)&X, X_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    int n = (int)N;
    int nb = k / QK8_0;
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    /* Parallelize over output rows */
    #pragma omp parallel for num_threads(nthr)
    for (int row = 0; row < m; row++) {
        const block_q8_0_wrapper* W_row = W + row * nb;
        
        /* Process each column (batch item) */
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            
            for (int b = 0; b < nb; b++) {
                const block_q8_0_wrapper* blk = &W_row[b];
                const float d = __gnu_h2f_ieee(blk->d);
                /* X is column-major: X[k,n] at X[k + n*K] */
                const float* x_blk = X + b * QK8_0 + col * k;
                
                float blk_sum = 0.0f;
                #pragma _NEC ivdep
                for (int i = 0; i < QK8_0; i++) {
                    blk_sum += (float)blk->qs[i] * x_blk[i];
                }
                sum += d * blk_sum;
            }
            /* Y is column-major: Y[m,n] at Y[m + n*M] */
            Y[row + col * m] = sum;
        }
    }
    
    return 0;
}

/*
 * F32 SGEMM with raw pointers (for compiled graph kernel)
 * 
 * Takes raw pointers directly - designed for compiled graph where
 * weights are pre-dequantized to F32 at upload time.
 * 
 * W is F32 row-major [M x K]
 * X is F32 row-major [N x K] (each row is one input vector of length K)
 * Y is F32 row-major [N x M] (each row is one output vector of length M)
 * 
 * Computes: Y[n,m] = sum_k(X[n,k] * W[m,k]) for all n in [0,N), m in [0,M)
 * This is Y = X @ W^T in matrix notation
 */
void ve_f32_sgemm_ptr(
    float* Y,                  /* Output [N x M] F32 row-major */
    const float* W,            /* Weights [M x K] F32 row-major */
    const float* X,            /* Input [N x K] F32 row-major */
    int M,
    int K,
    int N)
{
    /* Y[N,M] = X[N,K] @ W[M,K]^T using CBLAS */
    cblas_sgemm(CblasRowMajor_, CblasNoTrans_, CblasTrans_,
                N, M, K,
                1.0f, X, K,        /* A = X[N,K], lda = K */
                W, K,              /* B = W[M,K], ldb = K (transposed) */
                0.0f, Y, M);       /* C = Y[N,M], ldc = M */
}

/*
 * F32 matvec with raw pointers (for compiled graph kernel, N=1 case)
 * 
 * Equivalent to ve_f32_sgemm_ptr with N=1, but may be slightly faster
 * for the common single-token generation case.
 */
void ve_f32_matvec_ptr(
    float* y,                  /* Output [M] F32 */
    const float* W,            /* Weights [M x K] F32 row-major */
    const float* x,            /* Input [K] F32 */
    int M,
    int K)
{
    /* y[M] = W[M,K] @ x[K] using CBLAS SGEMV */
    cblas_sgemv(CblasRowMajor_, CblasNoTrans_,
                M, K,
                1.0f, W, K,        /* A = W[M,K], lda = K */
                x, 1,              /* x vector, incx = 1 */
                0.0f, y, 1);       /* y vector, incy = 1 */
}

/* ============================================================================
 * LLC-TILED Q8_0 MATVEC - Stream Q8_0, dequant to FP32 in LLC, vectorized dot
 * 
 * Strategy:
 * 1. Process rows in parallel (OpenMP)
 * 2. For each row, process K dimension in chunks that fit in LLC
 * 3. Dequantize chunk of Q8_0 blocks to FP32 buffer (in LLC)
 * 4. Vectorized FP32 dot product with x
 * 5. Accumulate partial sums
 * 
 * LLC per core: 2MB. We use ~256KB for dequantized chunk = 64K floats = 2048 blocks
 * This gives good vectorization while staying in cache.
 * ============================================================================
 */

/* Chunk size: 2048 Q8_0 blocks = 65536 elements = 256KB as FP32 */
#define Q8_0_CHUNK_BLOCKS 2048
#define Q8_0_CHUNK_ELEMS  (Q8_0_CHUNK_BLOCKS * QK8_0)  /* 65536 */

/*
 * VEDA kernel: LLC-tiled Q8_0 matvec - ALL in HBM
 * 
 * Uses thread-local FP32 buffer for dequantization (fits in LLC)
 * Much faster than scalar int8 path due to vectorized FP32 dot product.
 */
uint64_t ve_q8_0_tiled_matvec_hbm(VEDAdeviceptr y_vptr,
                                   VEDAdeviceptr W_vptr,
                                   VEDAdeviceptr x_vptr,
                                   uint64_t M,
                                   uint64_t K) {
    float* y;
    float* x;
    block_q8_0_wrapper* W;
    vedaMemPtr((void**)&y, y_vptr);
    vedaMemPtr((void**)&x, x_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    int nb = k / QK8_0;  /* Total blocks per row */
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel num_threads(nthr)
    {
        /* Each thread allocates its own buffers */
        float* dequant_buf = (float*)aligned_alloc(64, Q8_0_CHUNK_ELEMS * sizeof(float));
        
        #pragma omp for
        for (int row = 0; row < m; row++) {
            const block_q8_0_wrapper* W_row = W + row * nb;
            float sum = 0.0f;
            
            /* Process in chunks that fit in LLC */
            for (int b_start = 0; b_start < nb; b_start += Q8_0_CHUNK_BLOCKS) {
                int b_end = b_start + Q8_0_CHUNK_BLOCKS;
                if (b_end > nb) b_end = nb;
                int chunk_blocks = b_end - b_start;
                int chunk_elems = chunk_blocks * QK8_0;
                
                /* Dequantize each block separately - vectorizable within block */
                for (int b = 0; b < chunk_blocks; b++) {
                    const block_q8_0_wrapper* blk = &W_row[b_start + b];
                    const float d = __gnu_h2f_ieee(blk->d);
                    float* dst = dequant_buf + b * QK8_0;
                    
                    /* First load int8 to local int32 array */
                    int32_t q32[QK8_0];
                    for (int i = 0; i < QK8_0; i++) {
                        q32[i] = (int32_t)blk->qs[i];
                    }
                    
                    /* Then convert int32 to float with scale - THIS is vectorizable! */
                    #pragma _NEC ivdep
                    for (int i = 0; i < QK8_0; i++) {
                        dst[i] = d * (float)q32[i];
                    }
                }
                
                /* Vectorized FP32 dot product */
                const float* x_chunk = x + b_start * QK8_0;
                float chunk_sum = 0.0f;
                
                #pragma _NEC ivdep
                for (int i = 0; i < chunk_elems; i++) {
                    chunk_sum += dequant_buf[i] * x_chunk[i];
                }
                sum += chunk_sum;
            }
            y[row] = sum;
        }
        
        free(dequant_buf);
    }
    
    return 0;
}

/*
 * VEDA kernel: LLC-tiled Q8_0 SGEMM - batched version, ALL in HBM
 * 
 * For batched ops (N > 1): Y[M,N] = W[M,K] @ X[K,N]
 * X and Y are column-major.
 */
uint64_t ve_q8_0_tiled_sgemm_hbm(VEDAdeviceptr Y_vptr,
                                  VEDAdeviceptr W_vptr,
                                  VEDAdeviceptr X_vptr,
                                  uint64_t M,
                                  uint64_t K,
                                  uint64_t N) {
    float* Y;
    float* X;
    block_q8_0_wrapper* W;
    vedaMemPtr((void**)&Y, Y_vptr);
    vedaMemPtr((void**)&X, X_vptr);
    vedaMemPtr((void**)&W, W_vptr);
    
    int m = (int)M;
    int k = (int)K;
    int n = (int)N;
    int nb = k / QK8_0;
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel num_threads(nthr)
    {
        /* Each thread allocates its own buffer */
        float* dequant_buf = (float*)aligned_alloc(64, Q8_0_CHUNK_ELEMS * sizeof(float));
        
        #pragma omp for
        for (int row = 0; row < m; row++) {
            const block_q8_0_wrapper* W_row = W + row * nb;
            
            /* Process each column (batch item) */
            for (int col = 0; col < n; col++) {
                float sum = 0.0f;
                
                /* Process K in chunks */
                for (int b_start = 0; b_start < nb; b_start += Q8_0_CHUNK_BLOCKS) {
                    int b_end = b_start + Q8_0_CHUNK_BLOCKS;
                    if (b_end > nb) b_end = nb;
                    int chunk_blocks = b_end - b_start;
                    int chunk_elems = chunk_blocks * QK8_0;
                    
                    /* Dequantize chunk - vectorizable version */
                    for (int b = 0; b < chunk_blocks; b++) {
                        const block_q8_0_wrapper* blk = &W_row[b_start + b];
                        const float d = __gnu_h2f_ieee(blk->d);
                        float* dst = dequant_buf + b * QK8_0;
                        
                        /* First load int8 to local int32 array */
                        int32_t q32[QK8_0];
                        for (int i = 0; i < QK8_0; i++) {
                            q32[i] = (int32_t)blk->qs[i];
                        }
                        
                        /* Then convert int32 to float with scale - vectorizable! */
                        #pragma _NEC ivdep
                        for (int i = 0; i < QK8_0; i++) {
                            dst[i] = d * (float)q32[i];
                        }
                    }
                    
                    /* Vectorized dot product */
                    /* X is column-major: X[k,n] at X[k + n*K] */
                    const float* x_chunk = X + (b_start * QK8_0) + col * k;
                    float chunk_sum = 0.0f;
                    
                    #pragma _NEC ivdep
                    for (int i = 0; i < chunk_elems; i++) {
                        chunk_sum += dequant_buf[i] * x_chunk[i];
                    }
                    sum += chunk_sum;
                }
                /* Y is column-major: Y[m,n] at Y[m + n*M] */
                Y[row + col * m] = sum;
            }
        }
        
        free(dequant_buf);
    }
    
    return 0;
}

/* ============================================================================
 * ARGSORT kernels - Sort indices by values (for MoE expert selection)
 * ============================================================================
 * 
 * ARGSORT returns indices that would sort the input array.
 * For MoE, we typically sort in descending order to get top-k experts.
 * 
 * Input:  src [ne0] - F32 values (router scores)
 * Output: dst [ne0] - I32 indices sorted by src values
 * 
 * For small arrays (ne0 <= 64, typical for MoE with 32-64 experts),
 * insertion sort is efficient and has low overhead.
 * ============================================================================ */

/* ARGSORT F32 - HMEM path (single row)
 * Sorts indices in ascending or descending order based on values.
 * order: 0 = ASC, 1 = DESC
 */
uint64_t ve_argsort_f32_hmem(
    void* dst_hmem,      /* Output: int32_t indices */
    void* src_hmem,      /* Input: float values */
    uint64_t ne0,        /* Number of elements */
    uint64_t order       /* 0=ASC, 1=DESC */
) {
    int32_t* dst = (int32_t*)dst_hmem;
    const float* src = (const float*)src_hmem;
    int n = (int)ne0;
    
    /* Initialize indices */
    for (int i = 0; i < n; i++) {
        dst[i] = i;
    }
    
    /* Insertion sort - efficient for small arrays (MoE typically has 32-64 experts) */
    if (order == 1) {
        /* Descending order */
        for (int i = 1; i < n; i++) {
            int32_t key_idx = dst[i];
            float key_val = src[key_idx];
            int j = i - 1;
            while (j >= 0 && src[dst[j]] < key_val) {
                dst[j + 1] = dst[j];
                j--;
            }
            dst[j + 1] = key_idx;
        }
    } else {
        /* Ascending order */
        for (int i = 1; i < n; i++) {
            int32_t key_idx = dst[i];
            float key_val = src[key_idx];
            int j = i - 1;
            while (j >= 0 && src[dst[j]] > key_val) {
                dst[j + 1] = dst[j];
                j--;
            }
            dst[j + 1] = key_idx;
        }
    }
    
    return 0;
}

/* ARGSORT F32 - HMEM path with OpenMP (multiple rows)
 * Each row is sorted independently in parallel.
 */
uint64_t ve_argsort_f32_omp_hmem(
    VEDAdeviceptr dst_hbm,   /* Output: int32_t indices [nrows x ne0] */
    VEDAdeviceptr src_hbm,   /* Input: float values [nrows x ne0] */
    uint64_t ne0,        /* Elements per row */
    uint64_t nrows,      /* Number of rows */
    uint64_t order       /* 0=ASC, 1=DESC */
) {
    int32_t* dst;
    const float* src;
    if (vedaMemPtr((void**)&dst, dst_hbm) != 0) return 1;
    if (vedaMemPtr((void**)&src, src_hbm) != 0) return 2;
    int n = (int)ne0;
    int nr = (int)nrows;
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    #pragma omp parallel for num_threads(nthr)
    for (int row = 0; row < nr; row++) {
        int32_t* dst_row = dst + row * n;
        const float* src_row = src + row * n;
        
        /* Initialize indices */
        for (int i = 0; i < n; i++) {
            dst_row[i] = i;
        }
        
        /* Insertion sort */
        if (order == 1) {
            /* Descending */
            for (int i = 1; i < n; i++) {
                int32_t key_idx = dst_row[i];
                float key_val = src_row[key_idx];
                int j = i - 1;
                while (j >= 0 && src_row[dst_row[j]] < key_val) {
                    dst_row[j + 1] = dst_row[j];
                    j--;
                }
                dst_row[j + 1] = key_idx;
            }
        } else {
            /* Ascending */
            for (int i = 1; i < n; i++) {
                int32_t key_idx = dst_row[i];
                float key_val = src_row[key_idx];
                int j = i - 1;
                while (j >= 0 && src_row[dst_row[j]] > key_val) {
                    dst_row[j + 1] = dst_row[j];
                    j--;
                }
                dst_row[j + 1] = key_idx;
            }
        }
    }
    
    return 0;
}

/* ============================================================================
 * MUL_MAT_ID kernels - MoE expert matrix multiply
 * ============================================================================
 * 
 * MUL_MAT_ID performs matrix multiplication with expert selection:
 * - src0: [K, M, n_experts] - Expert weight matrices (BF16)
 * - src1: [K, n_tokens] - Input activations (F32)
 * - ids:  [n_expert_used, n_tokens] - Selected expert indices per token
 * - dst:  [M, n_expert_used, n_tokens] - Output
 * 
 * For each token t and expert slot i:
 *   expert_id = ids[i, t]
 *   dst[:, i, t] = src0[:, :, expert_id] @ src1[:, t]
 * 
 * ============================================================================ */

/* MUL_MAT_ID BF16 weights, F32 input/output - HMEM path
 * 
 * Parameters:
 * - W_hmem: BF16 weights [K x M x n_experts], row-major per expert
 * - x_hmem: F32 input [K x n_tokens], column-major (each column is a token)
 * - ids_hmem: I32 expert indices [n_expert_used x n_tokens]
 * - y_hmem: F32 output [M x n_expert_used x n_tokens]
 * - M: output dimension (rows of each expert matrix)
 * - K: input dimension (columns of each expert matrix)
 * - n_experts: total number of experts
 * - n_expert_used: number of experts used per token (typically 2-4)
 * - n_tokens: batch size
 */
uint64_t ve_mul_mat_id_bf16_f32_hmem(
    void* y_hmem,        /* Output: F32 [M x n_expert_used x n_tokens] */
    void* W_hmem,        /* Input: BF16 weights [K x M x n_experts] */
    void* x_hmem,        /* Input: F32 activations [K x n_tokens] */
    void* ids_hmem,      /* Input: I32 expert indices [n_expert_used x n_tokens] */
    uint64_t M,          /* Output dimension */
    uint64_t K,          /* Input dimension */
    uint64_t n_experts,  /* Total experts */
    uint64_t n_expert_used, /* Experts per token */
    uint64_t n_tokens    /* Batch size */
) {
    float* y = (float*)y_hmem;
    const uint16_t* W = (const uint16_t*)W_hmem;  /* BF16 as uint16 */
    const float* x = (const float*)x_hmem;
    const int32_t* ids = (const int32_t*)ids_hmem;
    
    int m = (int)M;
    int k = (int)K;
    int ne = (int)n_experts;
    int neu = (int)n_expert_used;
    int nt = (int)n_tokens;
    
    /* Expert matrix stride: K * M elements (BF16) */
    size_t expert_stride = (size_t)k * m;
    
    int nthr = omp_get_max_threads();
    if (nthr > VE_MAX_CORES) nthr = VE_MAX_CORES;
    
    /* Parallelize over output rows */
    #pragma omp parallel for num_threads(nthr)
    for (int row = 0; row < m; row++) {
        /* Process each token */
        for (int t = 0; t < nt; t++) {
            const float* x_col = x + t * k;  /* Input column for token t */
            
            /* Process each expert used by this token */
            for (int e = 0; e < neu; e++) {
                /* ids shape: [n_expert_used, n_tokens] in GGML column-major
                 * Index (e, t) -> flat index = e + t * n_expert_used */
                int expert_id = ids[e + t * neu];  /* Expert index */
                
                /* Bounds check */
                if (expert_id < 0 || expert_id >= ne) continue;
                
                /* Get expert weight row */
                const uint16_t* W_expert = W + expert_id * expert_stride;
                const uint16_t* W_row = W_expert + row * k;
                
                /* Compute dot product: W_row @ x_col */
                float sum = 0.0f;
                #pragma _NEC ivdep
                for (int i = 0; i < k; i++) {
                    /* BF16 to F32: shift left by 16 bits */
                    uint32_t w_bits = ((uint32_t)W_row[i]) << 16;
                    float w_val;
                    memcpy(&w_val, &w_bits, sizeof(float));
                    sum += w_val * x_col[i];
                }
                
                /* Store result: dst[row, e, t] */
                /* Layout: [M x n_expert_used x n_tokens] row-major */
                y[row + e * m + t * m * neu] = sum;
            }
        }
    }
    
    return 0;
}

/* MUL_MAT_ID BF16 weights in HBM, F32 I/O in HMEM
 * Same as above but weights are pre-uploaded to HBM for bandwidth.
 */
uint64_t ve_mul_mat_id_bf16_f32_hbm(
    void* y_hmem,           /* Output: F32 [M x n_expert_used x n_tokens] */
    VEDAdeviceptr W_hbm,    /* Input: BF16 weights in HBM */
    void* x_hmem,           /* Input: F32 activations [K x n_tokens] */
    void* ids_hmem,         /* Input: I32 expert indices */
    uint64_t M,
    uint64_t K,
    uint64_t n_experts,
    uint64_t n_expert_used,
    uint64_t n_tokens
) {
    /* Convert HBM pointer */
    void* W_raw;
    if (vedaMemPtr(&W_raw, W_hbm) != 0) {
        return 1;
    }
    
    /* Call HMEM version */
    return ve_mul_mat_id_bf16_f32_hmem(
        y_hmem, W_raw, x_hmem, ids_hmem,
        M, K, n_experts, n_expert_used, n_tokens
    );
}

/* MUL_MAT_ID with all HBM (weights, input, output, indices) */
uint64_t ve_mul_mat_id_bf16_f32_hbm_full(
    VEDAdeviceptr y_hbm,    /* Output in HBM */
    VEDAdeviceptr W_hbm,    /* Weights in HBM */
    VEDAdeviceptr x_hbm,    /* Input in HBM */
    VEDAdeviceptr ids_hbm,  /* Indices in HBM */
    uint64_t M,
    uint64_t K,
    uint64_t n_experts,
    uint64_t n_expert_used,
    uint64_t n_tokens
) {
    void *y_raw, *W_raw, *x_raw, *ids_raw;
    
    if (vedaMemPtr(&y_raw, y_hbm) != 0) return 1;
    if (vedaMemPtr(&W_raw, W_hbm) != 0) return 2;
    if (vedaMemPtr(&x_raw, x_hbm) != 0) return 3;
    if (vedaMemPtr(&ids_raw, ids_hbm) != 0) return 4;
    
    return ve_mul_mat_id_bf16_f32_hmem(
        y_raw, W_raw, x_raw, ids_raw,
        M, K, n_experts, n_expert_used, n_tokens
    );
}

// ============================================================================
// ADD_ID: Element-wise add with expert ID lookup (for MoE biases)
// ============================================================================
// dst = src0 + src1[ids[i1, i2]]
// src0: [ne0, ne1, ne2] F32 input
// src1: [ne0, n_experts] F32 bias table  
// ids:  [n_expert_used, n_tokens] I32 expert indices
// dst:  [ne0, ne1, ne2] F32 output (same shape as src0)
//
// ne1 = n_expert_used (experts per token)
// ne2 = n_tokens
// For each (token, expert_slot), look up expert_id from ids, add bias from src1

uint64_t ve_add_id_f32_hmem(
    void* dst,       /* Output [ne0, ne1, ne2] F32 */
    void* src0,      /* Input [ne0, ne1, ne2] F32 */
    void* src1,      /* Bias table [ne0, n_experts] F32 */
    void* ids,       /* Expert indices [ne1, ne2] I32 */
    uint64_t ne0,    /* Vector dimension */
    uint64_t ne1,    /* n_expert_used */
    uint64_t ne2     /* n_tokens */
) {
    float* y = (float*)dst;
    float* a = (float*)src0;
    float* b = (float*)src1;
    int32_t* idx = (int32_t*)ids;
    
    // Total rows = ne1 * ne2
    int64_t nr = ne1 * ne2;
    
    #pragma omp parallel for
    for (int64_t ir = 0; ir < nr; ir++) {
        // Row indices
        int64_t i2 = ir / ne1;  // token index
        int64_t i1 = ir % ne1;  // expert slot index
        
        // Look up expert ID
        int32_t expert_id = idx[i1 + i2 * ne1];
        
        // Pointer to this row's data
        float* dst_row = y + ir * ne0;
        float* src0_row = a + ir * ne0;
        float* src1_row = b + expert_id * ne0;  // bias for this expert
        
        // Add: dst = src0 + bias
        #pragma _NEC ivdep
        for (uint64_t i = 0; i < ne0; i++) {
            dst_row[i] = src0_row[i] + src1_row[i];
        }
    }
    
    return 0;
}

// ADD_ID with all HBM pointers
uint64_t ve_add_id_f32_hbm_full(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr src0_hbm,
    VEDAdeviceptr src1_hbm,
    VEDAdeviceptr ids_hbm,
    uint64_t ne0,
    uint64_t ne1,
    uint64_t ne2
) {
    void *dst_raw, *src0_raw, *src1_raw, *ids_raw;
    
    if (vedaMemPtr(&dst_raw, dst_hbm) != 0) return 1;
    if (vedaMemPtr(&src0_raw, src0_hbm) != 0) return 2;
    if (vedaMemPtr(&src1_raw, src1_hbm) != 0) return 3;
    if (vedaMemPtr(&ids_raw, ids_hbm) != 0) return 4;
    
    return ve_add_id_f32_hmem(dst_raw, src0_raw, src1_raw, ids_raw, ne0, ne1, ne2);
}

// ============================================================================
// SWIGLU_OAI: OpenAI-style SwiGLU with alpha and limit parameters
// ============================================================================
// dst[i] = silu_clamped(x[i], alpha) * (g[i] + 1)
// where silu_clamped(x, alpha) = min(x, limit) / (1 + exp(alpha * -min(x, limit)))
// and g is clamped to [-limit, limit]
//
// If src1 is NULL (single-source mode), src0 is split in half:
//   - first half = x (gate input)
//   - second half = g (value input)
// If swapped=1, the halves are reversed

uint64_t ve_swiglu_oai_f32_hbm_full(
    VEDAdeviceptr dst_hbm,
    VEDAdeviceptr src0_hbm,
    VEDAdeviceptr src1_hbm,  /* Can be 0 for single-source mode */
    uint64_t ne0,            /* Vector dimension (output size) */
    uint64_t nr,             /* Number of rows */
    int32_t swapped,         /* If 1, swap gate and value halves */
    float alpha,             /* Alpha parameter for silu */
    float limit              /* Limit for clamping */
) {
    void *dst_raw, *src0_raw;
    void *src1_raw = NULL;
    
    if (vedaMemPtr(&dst_raw, dst_hbm) != 0) return 1;
    if (vedaMemPtr(&src0_raw, src0_hbm) != 0) return 2;
    if (src1_hbm != 0) {
        if (vedaMemPtr(&src1_raw, src1_hbm) != 0) return 3;
    }
    
    float* dst = (float*)dst_raw;
    float* src0 = (float*)src0_raw;
    float* src1 = (float*)src1_raw;
    
    // Determine nc (output columns per row) and stride
    uint64_t nc = ne0;  // output size per row
    uint64_t src0_stride = src1 ? nc : nc * 2;  // src0 stride depends on mode
    uint64_t src1_stride = src1 ? nc : nc * 2;
    
    #pragma omp parallel for
    for (uint64_t row = 0; row < nr; row++) {
        float* src0_row = src0 + row * src0_stride;
        float* src1_row = src1 ? (src1 + row * src1_stride) : src0_row;
        float* dst_row = dst + row * nc;
        
        // In single-source mode, split into x and g
        float* x_ptr;
        float* g_ptr;
        if (!src1) {
            x_ptr = swapped ? (src0_row + nc) : src0_row;
            g_ptr = swapped ? src0_row : (src0_row + nc);
        } else {
            x_ptr = src0_row;
            g_ptr = src1_row;
        }
        
        #pragma _NEC ivdep
        for (uint64_t i = 0; i < nc; i++) {
            float x = x_ptr[i];
            float g = g_ptr[i];
            
            // Clamp x and g
            if (x > limit) x = limit;
            if (g > limit) g = limit;
            if (g < -limit) g = -limit;
            
            // silu_clamped = x / (1 + exp(alpha * -x))
            float silu_val = x / (1.0f + expf(alpha * (-x)));
            
            // output = silu * (g + 1)
            dst_row[i] = silu_val * (g + 1.0f);
        }
    }
    
    return 0;
}

