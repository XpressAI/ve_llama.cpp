/*
 * BF16 Column-Major Batched SGEMM using LLVM-VE Intrinsics
 * 
 * Computes: Y[M,N] = W[M,K] @ X[K,N]
 * 
 * Where:
 *   - W is BF16 in COLUMN-MAJOR format (K elements per column, M columns)
 *   - X is FP32 in column-major format
 *   - Y is FP32 in column-major format
 * 
 * Column-major layout means:
 *   W[m,k] = W_data[m + k*M]  (stride-M access for rows)
 *   X[k,n] = X_data[k + n*K]  (contiguous for each column)
 *   Y[m,n] = Y_data[m + n*M]  (contiguous for each column)
 * 
 * For each output column n, we compute:
 *   Y[:,n] = W @ X[:,n]  (this is a matvec!)
 * 
 * Build with LLVM-VE (NO OpenMP - that's handled by NCC wrapper):
 *   /usr/local/ve/llvm-ve-rv-2.2.0/bin/clang --target=ve-linux -O3 -fPIC -c bf16_sgemm_intrinsics.c
 * 
 * NOTE: OpenMP parallelization is done in the NCC wrapper (ve_sgemv_wrapper.c)
 *       because LLVM-VE's OpenMP runtime is incompatible with NCC's runtime.
 */

#include <velintrin.h>
#include <stdint.h>
#include <stddef.h>

#define VLEN 256
typedef uint16_t bf16;

/* 
 * Load BF16 and convert to packed FP32 (2 FP32 per vector slot)
 * Processes 2*vl BF16 values at once
 */
#define load_bf16_to_packed_fp32(wv, wp, bf16mskl, vl) \
    do { \
        wv = _vel_vldunc_vssl(4, (void *)(wp), vl); \
        __vr wr = _vel_vsrl_vvsl(wv, 16, vl); \
        wr = _vel_vand_vvvl(wr, bf16mskl, vl); \
        wv = _vel_vor_vvvl(wv, wr, vl); \
    } while(0)

/* Sum packed FP32 vector to scalar */
#define sumup_packed_fp32(tv, result, low32msk, VL) \
    do { \
        __vr rv = _vel_vand_vvvl(tv, low32msk, VL); \
        __vr sv = _vel_vsll_vvsl(rv, 32, VL); \
        tv = _vel_vfadds_vvvl(tv, sv, VL); \
        tv = _vel_vfsums_vvl(tv, VL); \
        result = _vel_lvss_svs(tv, 0); \
    } while(0)

/*
 * BF16 matvec for column-major weights - vectorized over output rows
 * 
 * For each k, we process all M output rows at once:
 *   y[0:M] += W[0:M, k] * x[k]
 * 
 * Since W is column-major, W[:,k] is contiguous! This is the key optimization.
 * 
 * This is a single-threaded function. OpenMP parallelization is done at the
 * SGEMM level (over output columns or row chunks) in the NCC wrapper.
 */
void bf16_matvec_colmajor_intrinsics(
    float* y,           /* Output [M] */
    const bf16* W,      /* Weights [M x K] column-major BF16 */
    const float* x,     /* Input [K] */
    int M,
    int K)
{
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);
    
    /* Zero output */
    for (int m = 0; m < M; m++) {
        y[m] = 0.0f;
    }
    
    /* For each input element k */
    for (int k = 0; k < K; k++) {
        /* Broadcast x[k] to packed format */
        uint32_t x32;
        __builtin_memcpy(&x32, &x[k], sizeof(x32));
        uint64_t x_bits = ((uint64_t)x32 << 32) | x32;
        __vr xv = _vel_vbrdl_vsl(x_bits, VLEN);
        
        /* Pointer to W[:,k] which is contiguous in column-major */
        const bf16* w_col = W + k * M;
        
        /* Process M output elements in chunks of 2*VLEN */
        for (int m = 0; m < M; m += 2 * VLEN) {
            int remaining = M - m;
            int vl = remaining < 2 * VLEN ? (remaining + 1) >> 1 : VLEN;
            if (vl <= 0) break;
            
            /* Load W[:,k] as packed BF16->FP32 (contiguous!) */
            __vr wv;
            load_bf16_to_packed_fp32(wv, w_col + m, bf16mskl, vl);
            
            /* Load current y as packed */
            __vr yv;
            if (((unsigned long)(y + m)) & 0x7) {
                yv = _vel_vldu_vssl(8, (void*)(y + m + 1), vl);
                __vr ylv = _vel_vldlzx_vssl(8, (void*)(y + m), vl);
                yv = _vel_pvor_vvvl(yv, ylv, vl);
            } else {
                yv = _vel_vld_vssl(8, (void*)(y + m), vl);
            }
            
            /* Packed FMA: y += x[k] * W[:,k] */
            yv = _vel_pvfmad_vvvvl(yv, xv, wv, vl);
            
            /* Store back */
            if (((unsigned long)(y + m)) & 0x7) {
                _vel_vstunc_vssl(yv, 8, (void*)(y + m + 1), vl);
                _vel_vstlnc_vssl(yv, 8, (void*)(y + m), vl);
            } else {
                _vel_vstnc_vssl(yv, 8, (void*)(y + m), vl);
            }
        }
    }
}

/*
 * Tiled matvec for a single output column with K-tiling
 * Accumulates into y (caller must zero y first)
 * 
 * This version processes a slice of K: [k_start, k_end)
 * Caller can parallelize over row chunks and call this for each K-tile.
 */
void bf16_matvec_colmajor_tiled_intrinsics(
    float* y,           /* Output [M] - must be zeroed by caller */
    const bf16* W,      /* Weights [M x K] column-major BF16 */
    const float* x,     /* Input [K] */
    int M,
    int K,
    int k_start,        /* Start of K-tile */
    int k_end)          /* End of K-tile (exclusive) */
{
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);
    
    /* For each input element k in this tile */
    for (int k = k_start; k < k_end; k++) {
        /* Broadcast x[k] to packed format */
        uint32_t x32;
        __builtin_memcpy(&x32, &x[k], sizeof(x32));
        uint64_t x_bits = ((uint64_t)x32 << 32) | x32;
        __vr xv = _vel_vbrdl_vsl(x_bits, VLEN);
        
        /* Pointer to W[:,k] which is contiguous in column-major */
        const bf16* w_col = W + k * M;
        
        /* Process M output elements in chunks of 2*VLEN */
        for (int m = 0; m < M; m += 2 * VLEN) {
            int remaining = M - m;
            int vl = remaining < 2 * VLEN ? (remaining + 1) >> 1 : VLEN;
            if (vl <= 0) break;
            
            /* Load W[:,k] as packed BF16->FP32 (contiguous!) */
            __vr wv;
            load_bf16_to_packed_fp32(wv, w_col + m, bf16mskl, vl);
            
            /* Load current y as packed */
            __vr yv;
            if (((unsigned long)(y + m)) & 0x7) {
                yv = _vel_vldu_vssl(8, (void*)(y + m + 1), vl);
                __vr ylv = _vel_vldlzx_vssl(8, (void*)(y + m), vl);
                yv = _vel_pvor_vvvl(yv, ylv, vl);
            } else {
                yv = _vel_vld_vssl(8, (void*)(y + m), vl);
            }
            
            /* Packed FMA: y += x[k] * W[:,k] */
            yv = _vel_pvfmad_vvvvl(yv, xv, wv, vl);
            
            /* Store back */
            if (((unsigned long)(y + m)) & 0x7) {
                _vel_vstunc_vssl(yv, 8, (void*)(y + m + 1), vl);
                _vel_vstlnc_vssl(yv, 8, (void*)(y + m), vl);
            } else {
                _vel_vstnc_vssl(yv, 8, (void*)(y + m), vl);
            }
        }
    }
}

/*
 * Process a single output column with row chunking
 * 
 * This processes rows [m_start, m_end) for all K.
 * Caller can parallelize over row chunks.
 */
void bf16_matvec_colmajor_rowchunk_intrinsics(
    float* y,           /* Output [M] */
    const bf16* W,      /* Weights [M x K] column-major BF16 */
    const float* x,     /* Input [K] */
    int M,
    int K,
    int m_start,        /* Start row */
    int m_end)          /* End row (exclusive) */
{
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000UL, VLEN);
    
    int m_count = m_end - m_start;
    float* y_ptr = y + m_start;
    
    /* Zero this chunk */
    for (int i = 0; i < m_count; i++) {
        y_ptr[i] = 0.0f;
    }
    
    /* For each input element k */
    for (int k = 0; k < K; k++) {
        /* Broadcast x[k] to packed format */
        uint32_t x32;
        __builtin_memcpy(&x32, &x[k], sizeof(x32));
        uint64_t x_bits = ((uint64_t)x32 << 32) | x32;
        __vr xv = _vel_vbrdl_vsl(x_bits, VLEN);
        
        /* Pointer to W[m_start:m_end, k] */
        const bf16* w_ptr = W + k * M + m_start;
        
        /* Process m_count elements in chunks of 2*VLEN */
        for (int m = 0; m < m_count; m += 2 * VLEN) {
            int remaining = m_count - m;
            int vl = remaining < 2 * VLEN ? (remaining + 1) >> 1 : VLEN;
            if (vl <= 0) break;
            
            /* Load W[m_start+m:..., k] as packed BF16->FP32 */
            __vr wv;
            load_bf16_to_packed_fp32(wv, w_ptr + m, bf16mskl, vl);
            
            /* Load current y as packed */
            float* yp = y_ptr + m;
            __vr yv;
            if (((unsigned long)yp) & 0x7) {
                yv = _vel_vldu_vssl(8, (void*)(yp + 1), vl);
                __vr ylv = _vel_vldlzx_vssl(8, (void*)yp, vl);
                yv = _vel_pvor_vvvl(yv, ylv, vl);
            } else {
                yv = _vel_vld_vssl(8, (void*)yp, vl);
            }
            
            /* Packed FMA: y += x[k] * W[:,k] */
            yv = _vel_pvfmad_vvvvl(yv, xv, wv, vl);
            
            /* Store back */
            if (((unsigned long)yp) & 0x7) {
                _vel_vstunc_vssl(yv, 8, (void*)(yp + 1), vl);
                _vel_vstlnc_vssl(yv, 8, (void*)yp, vl);
            } else {
                _vel_vstnc_vssl(yv, 8, (void*)yp, vl);
            }
        }
    }
}
