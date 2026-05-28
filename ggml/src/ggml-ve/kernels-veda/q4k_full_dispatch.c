/* NCC-compiled dispatcher for q4k_full_intrin.c.
 *
 * The inner per-row dot function is written with LLVM-VE-RV intrinsics
 * (clang-compiled). LLVM's OpenMP runtime hangs in VEDA's thread model,
 * so we use NCC's OpenMP runtime here and call the clang inner from a
 * parallelised loop. This file is compiled with ncc -fopenmp.
 *
 * The dispatcher also pre-permutes x ONCE per matvec so the inner kernel
 * can use sequential loads (cache-friendly) instead of stride-64 strided
 * loads (each one touching 256 cache lines).
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

extern int vedaMemPtr(void **ptr, uint64_t vptr);

/* clang-compiled inner kernels: see q4k_full_intrin.c. */
extern float q4k_full_row_dot_extern(const uint8_t *qs_row,
                                      const uint8_t *hdr_row,
                                      const float *x, int nb);
extern float q4k_full_row_dot_xperm_extern(const uint8_t *qs_row,
                                            const uint8_t *hdr_row,
                                            const float *x_perm, int nb);
extern void  q4k_build_x_perm_extern(const float *x, float *x_perm, int K);
extern void  q4k_full_4rows_xperm_extern(const uint8_t *qs_r0, const uint8_t *qs_r1,
                                           const uint8_t *qs_r2, const uint8_t *qs_r3,
                                           const uint8_t *hdr_r0, const uint8_t *hdr_r1,
                                           const uint8_t *hdr_r2, const uint8_t *hdr_r3,
                                           const float *x_perm,
                                           float *y0, float *y1, float *y2, float *y3, int nb);
extern void  q4k_full_8rows_xperm_extern(const uint8_t * const qs_r[8],
                                          const uint8_t * const hdr_r[8],
                                          const float *x_perm,
                                          float *y_out[8], int nb);

uint64_t ve_q4k_matvec_full_hbm(uint64_t y_vptr, uint64_t qs_vptr,
                                 uint64_t hdr_vptr, uint64_t x_vptr,
                                 uint64_t M, uint64_t K) {
    void *p;
    if (vedaMemPtr(&p, y_vptr)   != 0) return 1; float         *y   = (float *)p;
    if (vedaMemPtr(&p, qs_vptr)  != 0) return 2; const uint8_t *qs  = (const uint8_t *)p;
    if (vedaMemPtr(&p, hdr_vptr) != 0) return 3; const uint8_t *hdr = (const uint8_t *)p;
    if (vedaMemPtr(&p, x_vptr)   != 0) return 4; const float   *x   = (const float *)p;

    const int nb = (int) K / 256;
    const int row_qs_bytes  = nb * 128;
    const int row_hdr_bytes = nb * 16;
    int nthr = omp_get_max_threads();
    if (nthr > 8) nthr = 8;

    /* Pre-permute x once per matvec for sequential loads. */
    float *x_perm = (float *) aligned_alloc(64, K * sizeof(float));
    if (x_perm == 0) return 5;
    q4k_build_x_perm_extern(x, x_perm, (int) K);

    /* Process rows in groups of 8. */
    const uint64_t M8 = M & ~(uint64_t) 7;
    #pragma omp parallel for num_threads(nthr)
    for (uint64_t m = 0; m < M8; m += 8) {
        const uint8_t *qs_r[8]  = {
            qs  + (m+0) * row_qs_bytes, qs  + (m+1) * row_qs_bytes,
            qs  + (m+2) * row_qs_bytes, qs  + (m+3) * row_qs_bytes,
            qs  + (m+4) * row_qs_bytes, qs  + (m+5) * row_qs_bytes,
            qs  + (m+6) * row_qs_bytes, qs  + (m+7) * row_qs_bytes };
        const uint8_t *hdr_r[8] = {
            hdr + (m+0) * row_hdr_bytes, hdr + (m+1) * row_hdr_bytes,
            hdr + (m+2) * row_hdr_bytes, hdr + (m+3) * row_hdr_bytes,
            hdr + (m+4) * row_hdr_bytes, hdr + (m+5) * row_hdr_bytes,
            hdr + (m+6) * row_hdr_bytes, hdr + (m+7) * row_hdr_bytes };
        float *y_out[8] = {
            &y[m+0], &y[m+1], &y[m+2], &y[m+3],
            &y[m+4], &y[m+5], &y[m+6], &y[m+7] };
        q4k_full_8rows_xperm_extern(qs_r, hdr_r, x_perm, y_out, nb);
    }
    for (uint64_t m = M8; m < M; m++) {
        y[m] = q4k_full_row_dot_xperm_extern(qs  + m * row_qs_bytes,
                                              hdr + m * row_hdr_bytes,
                                              x_perm, nb);
    }
    free(x_perm);
    return 0;
}
