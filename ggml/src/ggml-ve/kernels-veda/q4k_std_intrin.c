/* Q4_K matvec: DIRECT-DISPATCH variant.
 *
 * Operates on the STANDARD ggml block_q4_K layout (144 bytes/block) in
 * HBM. NO canonical-nibble reorder, NO pre-decoded header, NO host-side
 * staging. Reads weights byte-for-byte the same way the CPU/CUDA paths
 * do, so the scheduler can place Q4_K weights on VE0_HBM without
 * paying the 192/144 storage blow-up of the canon-split cache.
 *
 * Why this exists: enabling GGML_VE_Q4K_N_GT_1 with the canon-split
 * cache OOMs Qwen3.6-27B at layer 43 (raw weights on VE0_HBM + canon
 * cache + KV cache ≈ 58 GB on a 48 GB device). With direct dispatch
 * only the raw 17 GB lives on HBM and 27B fits.
 *
 * Standard block_q4_K layout (144 B):
 *   bytes 0..1     : d        (fp16, super-block scale for scales)
 *   bytes 2..3     : dmin     (fp16, super-block scale for mins)
 *   bytes 4..15    : scales[12] (6-bit packed sub-scales + sub-mins, 8 each)
 *   bytes 16..143  : qs[128]  (4-bit nibbles, 256 elements / 2)
 *
 * qs layout (per CPU dequant code):
 *   bytes 32p..32p+31 (p=0..3): two sub-blocks of 32 elements
 *     low  nibble of each byte → sub-block 2p   element (l in 0..31)
 *     high nibble of each byte → sub-block 2p+1 element (l in 0..31)
 *
 * Per-element contribution to dot product:
 *   contrib(e) = (d_sub[sb(e)] * nib(e) - m_sub[sb(e)]) * x[e]
 *   where d_sub[s] = d * sc[s], m_sub[s] = dmin * mn[s]
 *
 * Per-block kernel: one FMA chain per (quarter, nibble-half), VL=32,
 * accumulating into a single VL=32 register. One vfsums at end.
 *
 * Build with LLVM-VE-RV clang for VE intrinsics:
 *   clang --target=ve-linux -O3 -fpic -c q4k_std_intrin.c
 *   ncc -fpic -shared -fopenmp -o libve_sgemv.so ... q4k_std_intrin.o
 */
#include <stdint.h>
#include <string.h>
#include <velintrin.h>

/* fp16 -> fp32, branchless-ish. Same impl as q4k_full_intrin.c. */
static inline float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else { while (!(m & 0x400u)) { m <<= 1; e--; } m &= 0x3FFu; e++;
               u = (s << 31) | ((e + 112u) << 23) | (m << 13); }
    } else if (e == 31) u = (s << 31) | 0x7F800000u | (m << 13);
    else u = (s << 31) | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

/* Decode the s'th 6-bit sub-scale and 6-bit sub-min from the 12-byte
 * scales[] array. Match ggml's dequantize_row_q4_K layout exactly:
 *   s in 0..3: bytes [s] and [s+4]
 *   s in 4..7: bits packed across [s-4], [s], [s+4] -- see ggml-quants.c
 */
static inline void q4k_sm(int s, const uint8_t *sc12, uint8_t *sc, uint8_t *mn) {
    if (s < 4) {
        *sc = sc12[s]     & 0x3F;
        *mn = sc12[s + 4] & 0x3F;
    } else {
        *sc = (sc12[s + 4] & 0x0F) | ((sc12[s - 4] >> 6) << 4);
        *mn = (sc12[s + 4] >>   4) | ((sc12[s]     >> 6) << 4);
    }
}

/* Inner per-row dot. blk_row points at row m's first block (M rows × nb
 * blocks × 144 bytes). x is the F32 input (K floats, contiguous). */
float q4k_std_row_dot_extern(const uint8_t *blk_row, const float *x, int nb);

float q4k_std_row_dot_extern(const uint8_t *blk_row, const float *x, int nb) {
    float acc = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = blk_row + (size_t) b * 144;
        const float   *xb  = x       + (size_t) b * 256;

        /* Header decode (scalar). 2 h2f + 8 q4k_sm + 8 scalar multiplies. */
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw,    blk + 0, 2);
        memcpy(&dmin_raw, blk + 2, 2);
        const float d_super    = h2f(d_raw);
        const float dmin_super = h2f(dmin_raw);
        const uint8_t *sc12 = blk + 4;

        float d_sub[8], m_sub[8];
        for (int s = 0; s < 8; s++) {
            uint8_t sc, mn;
            q4k_sm(s, sc12, &sc, &mn);
            d_sub[s] = d_super    * (float) sc;
            m_sub[s] = dmin_super * (float) mn;
        }

        const uint8_t *qs = blk + 16;

        /* Per-block accumulator. One vfsums at the bottom. */
        __vr acc_block = _vel_vbrds_vsl(0.0f, 8);

        /* 4 byte-quarters × 4 byte-positions within each quarter's u32-lane
         * × 2 nibble-halves = 32 sub-FMAs per block, but each FMA processes
         * 8 elements (VL=8).
         *
         * Load pattern: 32-byte quarter loaded as 8 u32 lanes (stride=4).
         * Lane i holds bytes [4i..4i+3] of the quarter.
         *   nibble at byte_pos b (b in 0..3) of lane i:
         *     low nibble  = (lane[i] >> (8b))     & 0x0F  -> element 4i+b of sub 2p
         *     high nibble = (lane[i] >> (8b + 4)) & 0x0F  -> element 4i+b of sub 2p+1
         *   x for that element: xl[4i+b]  (stride 16 bytes = 4 floats)
         * No overrun: total bytes accessed = 32 exactly.
         */
        for (int p = 0; p < 4; p++) {
            const uint8_t *qp = qs + 32 * p;

            /* 8 u32 lanes, 4 bytes each = 32 bytes total. Exactly the
             * quarter's 32 qs bytes, no overrun. */
            __vr qb     = _vel_vldlzx_vssl(4, (void *)qp, 8);
            __vr mask4  = _vel_vbrdl_vsl(0x0FUL, 8);

            const float d_lo = d_sub[2 * p];
            const float m_lo = m_sub[2 * p];
            const float d_hi = d_sub[2 * p + 1];
            const float m_hi = m_sub[2 * p + 1];
            __vr m_lo_v = _vel_vbrds_vsl(m_lo, 8);
            __vr m_hi_v = _vel_vbrds_vsl(m_hi, 8);

            /* 4 byte-positions within each u32 lane. */
            for (int bp = 0; bp < 4; bp++) {
                __vr shifted   = _vel_vsrl_vvsl(qb, 8 * bp, 8);
                __vr nib_lo    = _vel_vand_vvvl(shifted, mask4, 8);
                __vr nib_hi    = _vel_vand_vvvl(_vel_vsrl_vvsl(shifted, 4, 8),
                                                  mask4, 8);
                __vr nlf       = _vel_vcvtsw_vvl(nib_lo, 8);
                __vr nhf       = _vel_vcvtsw_vvl(nib_hi, 8);

                /* x for element {4i + bp} across i in 0..7 -> stride=16 B,
                 * VL=8 in low/high sub-blocks. */
                __vr xl = _vel_vldu_vssl(16, (void *)(xb + 64 * p      + bp), 8);
                __vr xh = _vel_vldu_vssl(16, (void *)(xb + 64 * p + 32 + bp), 8);

                __vr w_lo = _vel_vfmuls_vsvl(d_lo, nlf, 8);
                w_lo      = _vel_vfsubs_vvvl(w_lo, m_lo_v, 8);
                acc_block = _vel_vfmads_vvvvl(acc_block, w_lo, xl, 8);

                __vr w_hi = _vel_vfmuls_vsvl(d_hi, nhf, 8);
                w_hi      = _vel_vfsubs_vvvl(w_hi, m_hi_v, 8);
                acc_block = _vel_vfmads_vvvvl(acc_block, w_hi, xh, 8);
            }
        }

        /* Reduce the 8-lane accumulator and add to row total. */
        __vr red = _vel_vfsums_vvl(acc_block, 8);
        acc += _vel_lvss_svs(red, 0);
    }

    return acc;
}
