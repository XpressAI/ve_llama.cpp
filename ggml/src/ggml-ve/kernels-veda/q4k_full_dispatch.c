/* NCC-compiled dispatcher for q4k_full_intrin.c.
 *
 * The inner per-row dot function is written with LLVM-VE-RV intrinsics
 * (clang-compiled). LLVM's OpenMP runtime hangs in VEDA's thread model,
 * so we use NCC's OpenMP runtime here and call the clang inner from a
 * parallelised loop. This file is compiled with ncc -fopenmp.
 */
#include <stdint.h>
#include <stddef.h>
#include <omp.h>

extern int vedaMemPtr(void **ptr, uint64_t vptr);

/* clang-compiled inner: see q4k_full_intrin.c. Same signature, but
 * takes raw pointers (no VEDA resolution). */
extern float q4k_full_row_dot_extern(const uint8_t *qs_row,
                                      const uint8_t *hdr_row,
                                      const float *x, int nb);

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

    #pragma omp parallel for num_threads(nthr)
    for (uint64_t m = 0; m < M; m++) {
        y[m] = q4k_full_row_dot_extern(qs  + m * row_qs_bytes,
                                        hdr + m * row_hdr_bytes,
                                        x, nb);
    }
    return 0;
}
