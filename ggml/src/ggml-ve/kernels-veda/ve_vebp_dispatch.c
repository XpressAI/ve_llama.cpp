/* NCC dispatcher for the VEBP matvec (calls the clang inner kernel).
 *
 * Receives HBM pointers for the interleaved weight planes + scales and the
 * F32 activation, quantises the activation to int8 (absmax) and bit-slices
 * it into sign + NB magnitude planes ON DEVICE, then OMP-parallelises the
 * rowblock loop over vebp_block_vpcnt_scaled.
 *
 * Interleaved weight buffers (built by the HBM cache):
 *   Ws_il, Wn_il : M/256 blocks * (K/64) words * 256 rows, uint64
 *   wscale_il    : M/256 blocks * (K/128) groups * 256 rows, f32
 *
 * Build: ncc -O4 -fopenmp -c ve_vebp_dispatch.c
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern int vedaMemPtr(void **ptr, uint64_t vptr);
extern void vebp_block_vpcnt_scaled(const uint64_t *Ws_il, const uint64_t *Wn_il,
                                    const float *wscale_il, const uint64_t *a_sign,
                                    const uint64_t *a_mag, int nbits, long wpr,
                                    float ax, float *y_block);
extern void vebp_build_act_planes(const float *x, int nbits, long wpr,
                                  uint64_t *a_sign, uint64_t *a_mag, float inv);

#define VEBP_NB 7   /* int8 magnitude bitplanes */
#define VEBP_RB 256

/* reusable activation scratch (single-threaded VEDA kernel context) */
static uint64_t *g_asign = NULL, *g_amag = NULL;
static size_t    g_acap = 0;   /* in uint64 units of wpr */

uint64_t ve_vebp_matvec_hbm(uint64_t y_vptr, uint64_t Ws_vptr, uint64_t Wn_vptr,
                            uint64_t wscale_vptr, uint64_t x_vptr,
                            uint64_t M, uint64_t K) {
    void *p;
    if (vedaMemPtr(&p, y_vptr)      != 0) return 1; float          *y   = (float *)p;
    if (vedaMemPtr(&p, Ws_vptr)     != 0) return 2; const uint64_t *Ws  = (const uint64_t *)p;
    if (vedaMemPtr(&p, Wn_vptr)     != 0) return 3; const uint64_t *Wn  = (const uint64_t *)p;
    if (vedaMemPtr(&p, wscale_vptr) != 0) return 4; const float    *Wsc = (const float *)p;
    if (vedaMemPtr(&p, x_vptr)      != 0) return 5; const float    *x   = (const float *)p;

    const long wpr = (long) K / 64;
    const long ng  = (long) K / 128;
    if (K % 256) return 6;
    const long Mfull = (long) M / VEBP_RB;     /* full 256-row blocks */
    const long Mtail = (long) M % VEBP_RB;     /* leftover rows in last block */
    /* cache provides ceil(M/256)*256 padded rows, so every block's 256-lane
     * loads are in-bounds; the partial last block writes only Mtail rows. */

    /* grow activation plane scratch */
    if ((size_t) wpr > g_acap) {
        if (g_asign) free(g_asign);
        if (g_amag)  free(g_amag);
        g_asign = (uint64_t *) aligned_alloc(64, (size_t) wpr * sizeof(uint64_t));
        g_amag  = (uint64_t *) aligned_alloc(64, (size_t) VEBP_NB * wpr * sizeof(uint64_t));
        g_acap  = wpr;
        if (!g_asign || !g_amag) return 7;
    }

    int nthr = omp_get_max_threads(); if (nthr > 8) nthr = 8; if (nthr < 1) nthr = 1;

    /* int8 quantise activation (absmax) + bit-slice into sign + 7 mag planes.
     * absmax is a vectorizable reduction; the bit-slice is parallelised over
     * words (each word w writes only its own plane entries -> independent). */
    float amax = 0.0f;
    #pragma _NEC ivdep
    for (long k = 0; k < (long) K; k++) { float a = fabsf(x[k]); if (a > amax) amax = a; }
    const float ax = amax / 127.0f + 1e-12f;
    const float inv = 1.0f / ax;
    /* vectorized bit-slice (lane = word, V.LEN = wpr) */
    vebp_build_act_planes(x, VEBP_NB, wpr, g_asign, g_amag, inv);

    #pragma omp parallel for num_threads(nthr)
    for (long rb = 0; rb < Mfull; rb++) {
        vebp_block_vpcnt_scaled(Ws  + rb * wpr * VEBP_RB,
                                Wn  + rb * wpr * VEBP_RB,
                                Wsc + rb * ng  * VEBP_RB,
                                g_asign, g_amag, VEBP_NB, wpr, ax,
                                y + rb * VEBP_RB);
    }
    if (Mtail) {
        /* partial last block: compute 256 rows into scratch, copy Mtail out */
        float ytail[VEBP_RB];
        const long rb = Mfull;
        vebp_block_vpcnt_scaled(Ws  + rb * wpr * VEBP_RB,
                                Wn  + rb * wpr * VEBP_RB,
                                Wsc + rb * ng  * VEBP_RB,
                                g_asign, g_amag, VEBP_NB, wpr, ax, ytail);
        for (long r = 0; r < Mtail; r++) y[rb * VEBP_RB + r] = ytail[r];
    }
    return 0;
}
