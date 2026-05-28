/* NCC-side dispatcher for q4k_std_intrin.c (direct-dispatch Q4_K matvec).
 *
 * No canon-split cache. No header pre-decode. Just: lay out the standard
 * block_q4_K bytes (144 B/block) on VE0_HBM, give us a (qs+hdr+x+y)
 * pointer set, and we dot every row.
 *
 * Compared to the canon-cache path:
 *   - HBM footprint: 1× raw weights only (no 192/144 expansion)
 *   - First-call overhead: 0 (no canon transform; no host bounce)
 *   - Per-call cost: slightly higher (header decode + 6-bit unpack inside
 *     the kernel) but the row-dot inner loop has comparable VL/Op.Ratio
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

extern int vedaMemPtr(void **ptr, uint64_t vptr);

extern float q4k_std_row_dot_extern(const uint8_t *blk_row,
                                     const float *x, int nb);

/* Direct-dispatch matvec on standard block_q4_K layout.
 *   y[M]   : F32 output
 *   W[M*K] : standard block_q4_K (M * nb * 144 bytes, nb = K/256)
 *   x[K]   : F32 input
 * All pointers are HBM (VEDAdeviceptr -> raw VE pointer via vedaMemPtr). */
uint64_t ve_q4k_matvec_std_hbm(uint64_t y_vptr, uint64_t W_vptr,
                                uint64_t x_vptr,
                                uint64_t M, uint64_t K) {
    void *p;
    if (vedaMemPtr(&p, y_vptr) != 0) return 1; float         *y = (float *)p;
    if (vedaMemPtr(&p, W_vptr) != 0) return 2; const uint8_t *W = (const uint8_t *)p;
    if (vedaMemPtr(&p, x_vptr) != 0) return 3; const float   *x = (const float *)p;

    const int nb = (int) K / 256;
    if (nb <= 0) return 4;
    const size_t row_bytes = (size_t) nb * 144;

    int nthr = omp_get_max_threads();
    if (nthr < 1) nthr = 1;
    if (nthr > 8) nthr = 8;

    #pragma omp parallel for num_threads(nthr)
    for (uint64_t m = 0; m < M; m++) {
        const uint8_t *blk_row = W + m * row_bytes;
        y[m] = q4k_std_row_dot_extern(blk_row, x, nb);
    }
    return 0;
}
