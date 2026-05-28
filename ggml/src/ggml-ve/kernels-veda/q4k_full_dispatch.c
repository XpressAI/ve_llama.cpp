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
                                            const float *x_perm,
                                            const float *sx_full, int nb);
extern void  q4k_build_x_perm_extern(const float *x, float *x_perm, int K);
extern void  q4k_build_sx_full_extern(const float *x, float *sx_full, int K);
extern float q4k_full_row_dot_tile_extern(const uint8_t *qs_row,
                                           const uint8_t *hdr_row,
                                           const float *x, int nb);
extern void  q4k_full_4rows_decoded_extern(const uint8_t * const qs_r[4],
                                            const uint8_t * const hdr_r[4],
                                            const float *x_perm,
                                            const float *sx_full,
                                            float *y_out[4], int nb);
extern void  q4k_full_8rows_decoded_extern(const uint8_t * const qs_r[8],
                                            const uint8_t * const hdr_r[8],
                                            const float *x_perm,
                                            const float *sx_full,
                                            float *y_out[8], int nb);
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

/* Reusable x_perm + sx_full buffers, grown monotonically. NCC's VEDA
 * kernel context is single-threaded across calls. */
static float * g_x_perm_buf = NULL;
static size_t  g_x_perm_cap = 0;
static float * g_sx_full_buf = NULL;
static size_t  g_sx_full_cap = 0;

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
    const int row_hdr_bytes = nb * 64;  /* pre-decoded: 8 fp32 d_sub + 8 fp32 m_sub per block */
    int nthr = omp_get_max_threads();
    if (nthr > 8) nthr = 8;

    /* Pre-permute x and pre-compute Σx per lane. Both grow monotonically. */
    const size_t need = (size_t) K * sizeof(float);
    if (need > g_x_perm_cap) {
        if (g_x_perm_buf) free(g_x_perm_buf);
        g_x_perm_buf = (float *) aligned_alloc(64, need);
        g_x_perm_cap = need;
    }
    float *x_perm = g_x_perm_buf;
    if (x_perm == 0) return 5;
    q4k_build_x_perm_extern(x, x_perm, (int) K);

    const size_t sx_need = (size_t)(K / 16) * sizeof(float);
    if (sx_need > g_sx_full_cap) {
        if (g_sx_full_buf) free(g_sx_full_buf);
        g_sx_full_buf = (float *) aligned_alloc(64, sx_need);
        g_sx_full_cap = sx_need;
    }
    float *sx_full = g_sx_full_buf;
    if (sx_full == 0) return 6;
    q4k_build_sx_full_extern(x, sx_full, (int) K);

    /* Single-row only for now. The 8-row variant allocates 8x ~2KB of
     * stack arrays per call which may overflow VE thread stacks on
     * deep models (64-layer Qwen3.6-27B crashed in node 57). */
    /* GGML_VE_Q4K_TILE=1 selects the tile-based dequant-to-F32-cache
     * variant (per-row F32 buffer stays in L1/LLC, then F32 dot). */
    const int use_tile = (getenv("GGML_VE_Q4K_TILE") != NULL);
    if (use_tile) {
        #pragma omp parallel for num_threads(nthr)
        for (uint64_t m = 0; m < M; m++) {
            y[m] = q4k_full_row_dot_tile_extern(qs + m * row_qs_bytes,
                                                 hdr + m * row_hdr_bytes,
                                                 x, nb);
        }
    } else {
        /* 8-row tile dispatch: shares one x_perm load per nibble position
         * across 8 row FMAs. Each x_perm load is a 256-element vldu --
         * sharing it cuts x-load count by 8× in the inner loop. */
        const uint64_t M8 = M & ~(uint64_t) 7;
        #pragma omp parallel for num_threads(nthr)
        for (uint64_t m = 0; m < M8; m += 8) {
            const uint8_t *qs_r[8] = {
                qs + (m+0) * row_qs_bytes, qs + (m+1) * row_qs_bytes,
                qs + (m+2) * row_qs_bytes, qs + (m+3) * row_qs_bytes,
                qs + (m+4) * row_qs_bytes, qs + (m+5) * row_qs_bytes,
                qs + (m+6) * row_qs_bytes, qs + (m+7) * row_qs_bytes };
            const uint8_t *hdr_r[8] = {
                hdr + (m+0) * row_hdr_bytes, hdr + (m+1) * row_hdr_bytes,
                hdr + (m+2) * row_hdr_bytes, hdr + (m+3) * row_hdr_bytes,
                hdr + (m+4) * row_hdr_bytes, hdr + (m+5) * row_hdr_bytes,
                hdr + (m+6) * row_hdr_bytes, hdr + (m+7) * row_hdr_bytes };
            float *y_out[8] = {
                &y[m+0], &y[m+1], &y[m+2], &y[m+3],
                &y[m+4], &y[m+5], &y[m+6], &y[m+7] };
            q4k_full_8rows_decoded_extern(qs_r, hdr_r, x_perm, sx_full, y_out, nb);
        }
        /* Tail: M%8 leftover rows go through the per-row variant. */
        #pragma omp parallel for num_threads(nthr)
        for (uint64_t m = M8; m < M; m++) {
            y[m] = q4k_full_row_dot_xperm_extern(qs  + m * row_qs_bytes,
                                                  hdr + m * row_hdr_bytes,
                                                  x_perm, sx_full, nb);
        }
    }
    /* don't free x_perm -- reused across calls */
    return 0;
}
