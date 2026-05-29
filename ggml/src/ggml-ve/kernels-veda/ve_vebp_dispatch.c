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
static float     g_ptr_ax = 0.0f;  /* shares ax single->for in ptr_inner */

/* batched (N-column) activation scratch for ve_vebp_matmul_ptr_inner */
static uint64_t *g_asignN = NULL, *g_amagN = NULL;
static float    *g_axN    = NULL;
static long      g_Ncap = 0, g_wprcapN = 0;

/* Pointer-arg variant for the graph compiler's generated kernel. Called
 * from INSIDE the kernel's `#pragma omp parallel` region (all threads enter).
 * Builds the activation planes once via `#pragma omp single` (implicit
 * barrier publishes the shared scratch + scale), then shares the rowblock
 * matmul across the team with `#pragma omp for`. Pointers are raw HBM
 * (resolved by the slot table) -- no vedaMemPtr here. */
void ve_vebp_matvec_ptr_inner(float *y, const uint64_t *ws, const uint64_t *wn,
                              const float *wsc, const float *x, int M, int K) {
    const long wpr = (long) K / 64;
    const long ng  = (long) K / 128;
    const long Mfull = (long) M / VEBP_RB;
    const long Mtail = (long) M % VEBP_RB;

    #pragma omp single
    {
        if ((size_t) wpr > g_acap) {
            if (g_asign) free(g_asign);
            if (g_amag)  free(g_amag);
            g_asign = (uint64_t *) aligned_alloc(64, (size_t) wpr * sizeof(uint64_t));
            g_amag  = (uint64_t *) aligned_alloc(64, (size_t) VEBP_NB * wpr * sizeof(uint64_t));
            g_acap  = wpr;
        }
        float amax = 0.0f;
        for (long k = 0; k < (long) K; k++) { float a = fabsf(x[k]); if (a > amax) amax = a; }
        g_ptr_ax = amax / 127.0f + 1e-12f;
        vebp_build_act_planes(x, VEBP_NB, wpr, g_asign, g_amag, 1.0f / g_ptr_ax);
    }  /* implicit barrier */

    #pragma omp for
    for (long rb = 0; rb < Mfull; rb++) {
        vebp_block_vpcnt_scaled(ws  + rb * wpr * VEBP_RB,
                                wn  + rb * wpr * VEBP_RB,
                                wsc + rb * ng  * VEBP_RB,
                                g_asign, g_amag, VEBP_NB, wpr, g_ptr_ax,
                                y + rb * VEBP_RB);
    }
    if (Mtail) {
        #pragma omp single
        {
            float ytail[VEBP_RB];
            const long rb = Mfull;
            vebp_block_vpcnt_scaled(ws  + rb * wpr * VEBP_RB,
                                    wn  + rb * wpr * VEBP_RB,
                                    wsc + rb * ng  * VEBP_RB,
                                    g_asign, g_amag, VEBP_NB, wpr, g_ptr_ax, ytail);
            for (long r = 0; r < Mtail; r++) y[rb * VEBP_RB + r] = ytail[r];
        }
    }
}

/* Batched matmul variant: y[M,N] = W[M,K] @ X[K,N], one fused call instead of
 * the graph compiler's per-column matvec loop. Called from INSIDE the kernel's
 * `#pragma omp parallel`. Quantises all N activation columns in parallel, then
 * shares ONE rowblock loop across the team — each rowblock's interleaved weight
 * is loaded from HBM once and reused (from cache) across all N columns, instead
 * of the col-loop re-traversing the whole weight from HBM N times. The vpcnt
 * work itself is inherently N x (one dot product per column), so this recovers
 * the weight-traffic + barrier + serial-quant overhead, not the compute.
 * y column j is at y + j*M (matches the col-major dst the codegen used). */
void ve_vebp_matmul_ptr_inner(float *y, const uint64_t *ws, const uint64_t *wn,
                              const float *wsc, const float *x,
                              int M, int K, int N) {
    const long wpr   = (long) K / 64;
    const long ng    = (long) K / 128;
    const long Mfull = (long) M / VEBP_RB;
    const long Mtail = (long) M % VEBP_RB;

    #pragma omp single
    {
        if ((long) N > g_Ncap || wpr > g_wprcapN) {
            if (g_asignN) free(g_asignN);
            if (g_amagN)  free(g_amagN);
            if (g_axN)    free(g_axN);
            g_asignN = (uint64_t *) aligned_alloc(64, (size_t) N * wpr * sizeof(uint64_t));
            g_amagN  = (uint64_t *) aligned_alloc(64, (size_t) N * VEBP_NB * wpr * sizeof(uint64_t));
            g_axN    = (float *)    aligned_alloc(64, (size_t) N * sizeof(float));
            g_Ncap = N; g_wprcapN = wpr;
        }
    }  /* implicit barrier: scratch published */

    /* Quantise each of the N activation columns (independent -> omp for). */
    #pragma omp for
    for (long j = 0; j < (long) N; j++) {
        const float *xj = x + j * (long) K;
        float amax = 0.0f;
        for (long k = 0; k < (long) K; k++) { float a = fabsf(xj[k]); if (a > amax) amax = a; }
        float axj = amax / 127.0f + 1e-12f;
        g_axN[j] = axj;
        vebp_build_act_planes(xj, VEBP_NB, wpr,
                              g_asignN + j * wpr,
                              g_amagN  + j * VEBP_NB * wpr,
                              1.0f / axj);
    }  /* implicit barrier: all planes ready */

    /* One rowblock loop; inner N columns reuse the cached weight block. */
    #pragma omp for
    for (long rb = 0; rb < Mfull; rb++) {
        const uint64_t *wsb  = ws  + rb * wpr * VEBP_RB;
        const uint64_t *wnb  = wn  + rb * wpr * VEBP_RB;
        const float    *wscb = wsc + rb * ng  * VEBP_RB;
        for (long j = 0; j < (long) N; j++) {
            vebp_block_vpcnt_scaled(wsb, wnb, wscb,
                                    g_asignN + j * wpr,
                                    g_amagN  + j * VEBP_NB * wpr,
                                    VEBP_NB, wpr, g_axN[j],
                                    y + j * (long) M + rb * VEBP_RB);
        }
    }
    if (Mtail) {
        #pragma omp single
        {
            float ytail[VEBP_RB];
            const long rb = Mfull;
            const uint64_t *wsb  = ws  + rb * wpr * VEBP_RB;
            const uint64_t *wnb  = wn  + rb * wpr * VEBP_RB;
            const float    *wscb = wsc + rb * ng  * VEBP_RB;
            for (long j = 0; j < (long) N; j++) {
                vebp_block_vpcnt_scaled(wsb, wnb, wscb,
                                        g_asignN + j * wpr,
                                        g_amagN  + j * VEBP_NB * wpr,
                                        VEBP_NB, wpr, g_axN[j], ytail);
                for (long r = 0; r < Mtail; r++)
                    y[j * (long) M + rb * VEBP_RB + r] = ytail[r];
            }
        }
    }
}

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
