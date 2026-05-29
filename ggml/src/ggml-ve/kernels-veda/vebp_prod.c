/* VEBP production matvec inner kernel (LLVM-VE-RV clang intrinsics).
 *
 * Row-block-interleaved, lane = output row, AVL=256, per-128-group fp16
 * weight scales. Bit-sliced int8 activations (sign + NB magnitude planes)
 * with one global activation scale applied at the end.
 *
 * Interleaved weight layout (one rowblock of RB=256 rows):
 *   Ws_il[w*RB + r] / Wn_il[w*RB + r] : uint64, word w = k/64, row r in block
 *   wscale_il[g*RB + r]               : f32,   group g = k/128
 * Activations: a_sign[wpr], a_mag[NB*wpr] uint64 ; global scale ax.
 *
 * y[r] = ax * sum_g wscale[g,r] * ( 2*pc(pos) - pc(both) over the group )
 *
 * Validated standalone (test_vebp_scaled.c) on a real Bonsai tensor:
 * relerr 2.5e-8 vs exact integer ref, 0.39% vs full fp.
 *
 * Build: clang --target=ve-linux -O3 -c vebp_prod.c
 */
#include <stdint.h>
#include <velintrin.h>

#define VEBP_RB 256
#define VEBP_GS 128
#define VEBP_GW (VEBP_GS / 64)   /* 2 words per 128-group */

void vebp_block_vpcnt_scaled(const uint64_t *Ws_il, const uint64_t *Wn_il,
                             const float *wscale_il, const uint64_t *a_sign,
                             const uint64_t *a_mag, int nbits, long wpr,
                             float ax, float *y_block);

/* Vectorized activation bit-slice: lane = word w (V.LEN = wpr, up to 256),
 * vs the scalar per-word loop (which vectorized at V.LEN=7 and dominated
 * decode). For each bit position kk in 0..63, strided-load x[64w+kk] across
 * all words, int8-quantise, and OR bit kk into the sign / magnitude planes.
 *   a_sign[w], a_mag[b*wpr + w]  built for b in 0..nbits-1.
 */
void vebp_build_act_planes(const float *x, int nbits, long wpr,
                           uint64_t *a_sign, uint64_t *a_mag, float inv);

void vebp_build_act_planes(const float *x, int nbits, long wpr,
                           uint64_t *a_sign, uint64_t *a_mag, float inv) {
    const int vl = (int) wpr;
    __vr asign = _vel_vbrdl_vsl(0UL, vl);
    __vr amag[8];
    for (int b = 0; b < nbits; b++) amag[b] = _vel_vbrdl_vsl(0UL, vl);

    for (int kk = 0; kk < 64; kk++) {
        /* x[64w + kk] for all w: stride 64 floats = 256 bytes */
        __vr xv = _vel_vldu_vssl(256, (void *)(x + kk), vl);
        __vr sc = _vel_vfmuls_vsvl(inv, xv, vl);
        __vr qi = _vel_vcvtwssx_vvl(sc, vl);              /* f32 -> i32 (round), sx to i64 */
        qi = _vel_vminswsx_vsvl(127, qi, vl);
        qi = _vel_vmaxswsx_vsvl(-127, qi, vl);
        __vr neg = _vel_vminswsx_vsvl(0, qi, vl);         /* min(qi,0) <= 0 */
        __vr pos = _vel_vmaxswsx_vsvl(0, qi, vl);         /* max(qi,0) >= 0 */
        __vr absq = _vel_vsubsl_vvvl(pos, neg, vl);       /* |qi| */
        __vr sbit = _vel_vsrl_vvsl(neg, 63, vl);          /* 1 if qi<0 else 0 */
        asign = _vel_vor_vvvl(asign, _vel_vsll_vvsl(sbit, kk, vl), vl);
        for (int b = 0; b < nbits; b++) {
            __vr bit = _vel_vand_vsvl(1UL, _vel_vsrl_vvsl(absq, b, vl), vl);
            amag[b] = _vel_vor_vvvl(amag[b], _vel_vsll_vvsl(bit, kk, vl), vl);
        }
    }
    _vel_vst_vssl(asign, 8, (void *) a_sign, vl);
    for (int b = 0; b < nbits; b++)
        _vel_vst_vssl(amag[b], 8, (void *)(a_mag + (long) b * wpr), vl);
}

void vebp_block_vpcnt_scaled(const uint64_t *Ws_il, const uint64_t *Wn_il,
                             const float *wscale_il, const uint64_t *a_sign,
                             const uint64_t *a_mag, int nbits, long wpr,
                             float ax, float *y_block) {
    const long ngrp = wpr / VEBP_GW;
    __vr yacc = _vel_vbrds_vsl(0.0f, VEBP_RB);

    for (long g = 0; g < ngrp; g++) {
        __vr acc_both = _vel_vbrdl_vsl(0UL, VEBP_RB);
        __vr acc_pos  = _vel_vbrdl_vsl(0UL, VEBP_RB);
        for (long ww = 0; ww < VEBP_GW; ww++) {
            const long w = g * VEBP_GW + ww;
            __vr vws  = _vel_vld_vssl(8, (void *)(Ws_il + w * VEBP_RB), VEBP_RB);
            __vr vwn  = _vel_vld_vssl(8, (void *)(Wn_il + w * VEBP_RB), VEBP_RB);
            __vr vdiff = _vel_vxor_vsvl(a_sign[w], vws, VEBP_RB);
            for (int b = 0; b < nbits; b++) {
                uint64_t abw = a_mag[(long) b * wpr + w];
                __vr both = _vel_vand_vsvl(abw, vwn, VEBP_RB);
                __vr pos  = _vel_vand_vvvl(both, vdiff, VEBP_RB);
                __vr pcb  = _vel_vpcnt_vvl(both, VEBP_RB);
                __vr pcp  = _vel_vpcnt_vvl(pos,  VEBP_RB);
                acc_both  = _vel_vaddsl_vvvl(acc_both, _vel_vsll_vvsl(pcb, b, VEBP_RB), VEBP_RB);
                acc_pos   = _vel_vaddsl_vvvl(acc_pos,  _vel_vsll_vvsl(pcp, b, VEBP_RB), VEBP_RB);
            }
        }
        __vr idot   = _vel_vsubsl_vvvl(_vel_vaddsl_vvvl(acc_pos, acc_pos, VEBP_RB),
                                       acc_both, VEBP_RB);
        __vr fdot   = _vel_vcvtdl_vvl(idot, VEBP_RB);
        __vr fdot32 = _vel_vcvtsd_vvl(fdot, VEBP_RB);
        __vr ws     = _vel_vldu_vssl(4, (void *)(wscale_il + g * VEBP_RB), VEBP_RB);
        yacc = _vel_vfmads_vvvvl(yacc, ws, fdot32, VEBP_RB);
    }
    yacc = _vel_vfmuls_vsvl(ax, yacc, VEBP_RB);
    _vel_vstu_vssl(yacc, 4, (void *) y_block, VEBP_RB);
}
