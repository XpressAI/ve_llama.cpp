/* NCC-side dispatcher for q4k_std_intrin.c (direct-dispatch Q4_K matvec).
 *
 * Builds bp-major x_low_perm and x_high_perm ONCE per matvec, then
 * dispatches OMP-parallel rows. Each row's inner kernel uses a
 * per-thread qs scratch buffer (cn*128 bytes) to pack qs from raw
 * blocks contiguously, then issues VL=cn*32 chunked FMAs. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

extern int vedaMemPtr(void **ptr, uint64_t vptr);

extern float q4k_std_row_dot_xperm_extern(const uint8_t *blk_row,
                                            const float *x_low_perm,
                                            const float *x_high_perm,
                                            int nb);
extern void  q4k_std_8rows_xperm_extern(const uint8_t * const blk_rows[8],
                                          const float *x_low_perm,
                                          const float *x_high_perm,
                                          float *y_out[8], int nb);
extern float q4k_std_row_dot_chunked_extern(const uint8_t *blk_row,
                                              const float *x_low_perm,
                                              const float *x_high_perm,
                                              uint8_t *qs_scratch,
                                              int nb);
extern float q4k_std_row_dot_chunked_hdr_extern(const uint8_t *blk_row,
                                                  const float *hdr_decoded_row,
                                                  const float *x_low_perm,
                                                  const float *x_high_perm,
                                                  uint8_t *qs_scratch,
                                                  int nb);
extern void  q4k_std_build_x_perm_extern(const float *x,
                                          float *x_low_perm,
                                          float *x_high_perm, int K);

/* Reusable per-matvec buffers; grow monotonically. */
static float * g_xlo_perm = NULL;
static float * g_xhi_perm = NULL;
static size_t  g_xperm_cap = 0;

/* Per-thread qs scratch pool. Sized for nb*128 bytes * nthr_cap. */
static uint8_t * g_qs_pool = NULL;
static size_t    g_qs_per_thread = 0;
static int       g_qs_nthr_cap = 0;

/* Optional pre-decoded headers (per-tensor cache). If non-zero, kernel
 * loads 16 fp32 / block from this HBM buffer instead of doing scalar
 * h2f + q4k_sm per block. Layout: M * nb * 64 bytes, contiguous, row r
 * block b at offset (r*nb + b) * 64. */
uint64_t ve_q4k_matvec_std_hdr_hbm(uint64_t y_vptr, uint64_t W_vptr,
                                    uint64_t hdr_vptr, uint64_t x_vptr,
                                    uint64_t M, uint64_t K);

uint64_t ve_q4k_matvec_std_hdr_hbm(uint64_t y_vptr, uint64_t W_vptr,
                                    uint64_t hdr_vptr, uint64_t x_vptr,
                                    uint64_t M, uint64_t K);

uint64_t ve_q4k_matvec_std_hbm(uint64_t y_vptr, uint64_t W_vptr,
                                uint64_t x_vptr,
                                uint64_t M, uint64_t K) {
    return ve_q4k_matvec_std_hdr_hbm(y_vptr, W_vptr, /*hdr=*/0, x_vptr, M, K);
}

uint64_t ve_q4k_matvec_std_hdr_hbm(uint64_t y_vptr, uint64_t W_vptr,
                                    uint64_t hdr_vptr, uint64_t x_vptr,
                                    uint64_t M, uint64_t K) {
    void *p;
    if (vedaMemPtr(&p, y_vptr) != 0) return 1; float         *y = (float *)p;
    if (vedaMemPtr(&p, W_vptr) != 0) return 2; const uint8_t *W = (const uint8_t *)p;
    const float *hdr_all = NULL;
    if (hdr_vptr != 0) {
        if (vedaMemPtr(&p, hdr_vptr) != 0) return 7;
        hdr_all = (const float *)p;
    }
    if (vedaMemPtr(&p, x_vptr) != 0) return 3; const float   *x = (const float *)p;

    const int nb = (int) K / 256;
    if (nb <= 0) return 4;
    const size_t row_bytes = (size_t) nb * 144;

    int nthr = omp_get_max_threads();
    if (nthr < 1) nthr = 1;
    if (nthr > 8) nthr = 8;
    if (getenv("GGML_VE_Q4K_STD_ST")) nthr = 1;

    /* Grow permuted-x buffers if needed. */
    const size_t need_bytes = (size_t) K * sizeof(float);
    if (need_bytes > g_xperm_cap) {
        if (g_xlo_perm) free(g_xlo_perm);
        if (g_xhi_perm) free(g_xhi_perm);
        g_xlo_perm = (float *) aligned_alloc(64, need_bytes);
        g_xhi_perm = (float *) aligned_alloc(64, need_bytes);
        g_xperm_cap = need_bytes;
        if (g_xlo_perm == NULL || g_xhi_perm == NULL) return 5;
    }

    q4k_std_build_x_perm_extern(x, g_xlo_perm, g_xhi_perm, (int) K);

    /* GGML_VE_Q4K_STD_TILE=1 to try the 8-row tile (slower at the moment
     * due to register pressure -- left in for future tuning). */
    const int use_tile = (getenv("GGML_VE_Q4K_STD_TILE") != NULL);
    /* GGML_VE_Q4K_STD_CHUNK=1 to use cross-block chunking (VL up to 256). */
    const int use_chunk = (getenv("GGML_VE_Q4K_STD_CHUNK") != NULL);

    if (use_chunk) {
        /* Grow per-thread qs scratch. nb*128 bytes per thread. */
        const size_t qs_need = (size_t) nb * 128;
        if (qs_need > g_qs_per_thread || nthr > g_qs_nthr_cap) {
            if (g_qs_pool) free(g_qs_pool);
            g_qs_per_thread = qs_need;
            g_qs_nthr_cap = nthr;
            /* 64-byte aligned, total = nthr * qs_per_thread. */
            const size_t aligned_per = (qs_need + 63) & ~(size_t) 63;
            g_qs_per_thread = aligned_per;
            g_qs_pool = (uint8_t *) aligned_alloc(64,
                (size_t) nthr * aligned_per);
            if (g_qs_pool == NULL) return 6;
        }

        const size_t hdr_row_floats = (size_t) nb * 16;  /* 16 fp32 per block */
        #pragma omp parallel num_threads(nthr)
        {
            int tid = omp_get_thread_num();
            uint8_t *qs_scratch = g_qs_pool + (size_t) tid * g_qs_per_thread;
            #pragma omp for
            for (uint64_t m = 0; m < M; m++) {
                const uint8_t *blk_row = W + m * row_bytes;
                const float *hdr_row = hdr_all
                    ? hdr_all + m * hdr_row_floats
                    : NULL;
                y[m] = q4k_std_row_dot_chunked_hdr_extern(blk_row, hdr_row,
                    g_xlo_perm, g_xhi_perm, qs_scratch, nb);
            }
        }
    } else if (use_tile) {
        const uint64_t M8 = M & ~(uint64_t) 7;
        #pragma omp parallel for num_threads(nthr)
        for (uint64_t m = 0; m < M8; m += 8) {
            const uint8_t *blk_rows[8] = {
                W + (m + 0) * row_bytes, W + (m + 1) * row_bytes,
                W + (m + 2) * row_bytes, W + (m + 3) * row_bytes,
                W + (m + 4) * row_bytes, W + (m + 5) * row_bytes,
                W + (m + 6) * row_bytes, W + (m + 7) * row_bytes };
            float *y_out[8] = {
                &y[m + 0], &y[m + 1], &y[m + 2], &y[m + 3],
                &y[m + 4], &y[m + 5], &y[m + 6], &y[m + 7] };
            q4k_std_8rows_xperm_extern(blk_rows, g_xlo_perm, g_xhi_perm, y_out, nb);
        }
        #pragma omp parallel for num_threads(nthr)
        for (uint64_t m = M8; m < M; m++) {
            const uint8_t *blk_row = W + m * row_bytes;
            y[m] = q4k_std_row_dot_xperm_extern(blk_row, g_xlo_perm, g_xhi_perm, nb);
        }
    } else {
        #pragma omp parallel for num_threads(nthr)
        for (uint64_t m = 0; m < M; m++) {
            const uint8_t *blk_row = W + m * row_bytes;
            y[m] = q4k_std_row_dot_xperm_extern(blk_row, g_xlo_perm, g_xhi_perm, nb);
        }
    }
    return 0;
}
