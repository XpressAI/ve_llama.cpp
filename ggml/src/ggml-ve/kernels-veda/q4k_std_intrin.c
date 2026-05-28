/* Q4_K matvec: DIRECT-DISPATCH variant, VL=32 build.
 *
 * Operates on the STANDARD ggml block_q4_K layout (144 bytes/block) in
 * HBM. NO canonical-nibble reorder, NO pre-decoded header, NO host-side
 * staging. Reads weights byte-for-byte the same way the CPU/CUDA paths
 * do, so the scheduler can place Q4_K weights on VE0_HBM without
 * paying the 192/144 storage blow-up of the canon-split cache.
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
 * Load pattern: vldlzx stride=4 VL=32 reads 32 u32s at byte positions
 * [0, 4, 8, ..., 124] of qs -> covers ALL 128 qs bytes EXACTLY, with
 * 4-byte aligned addressing (required by VE; stride-1 raises sig 7).
 *
 *   Lane i (i in 0..31) holds u32 = bytes [4i..4i+3]
 *     Quarter q = i/8 (lanes 0..7 = q0, 8..15 = q1, 16..23 = q2, 24..31 = q3)
 *     Within-quarter byte index = 4*(i%8) + bp (bp in 0..3)
 *
 *   For byte position bp in 0..3:
 *     low_nib  = (lane >> (8*bp))     & 0x0F   -> 32 low nibbles
 *     high_nib = (lane >> (8*bp + 4)) & 0x0F   -> 32 high nibbles
 *
 *   x lookup for low at (lane i, byte_pos bp):
 *     element index = 64*(i/8) + 4*(i%8) + bp           (sub 2*(i/8))
 *   for high:
 *     element index = 64*(i/8) + 32 + 4*(i%8) + bp      (sub 2*(i/8)+1)
 *
 * The x address pattern isn't a clean stride, so the dispatcher pre-
 * permutes x into x_low_perm and x_high_perm (K floats each). After
 * that, each block's x lookup is a contiguous vldu stride=4 VL=32.
 *
 * Per-block FMA count: 4 byte-positions × 2 nibble-halves = 8 FMAs at
 * VL=32 (vs 32 FMAs at VL=8 before, vs 64 FMAs at VL=4). Plus 4
 * broadcasts for the dlane/mlane vectors (4 distinct d/m values per
 * block, each repeated 8 times across lanes).
 *
 * Build with LLVM-VE-RV clang for VE intrinsics. */
#include <stdint.h>
#include <string.h>
#include <velintrin.h>

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

static inline void q4k_sm(int s, const uint8_t *sc12, uint8_t *sc, uint8_t *mn) {
    if (s < 4) {
        *sc = sc12[s]     & 0x3F;
        *mn = sc12[s + 4] & 0x3F;
    } else {
        *sc = (sc12[s + 4] & 0x0F) | ((sc12[s - 4] >> 6) << 4);
        *mn = (sc12[s + 4] >>   4) | ((sc12[s]     >> 6) << 4);
    }
}

/* Build x_low_perm[K] and x_high_perm[K] from x[K]. Called ONCE per
 * matvec, then shared across all M rows.
 *
 * CHUNK layout: per byte-position bp (0..3), nb*32 floats contiguous.
 * That way a chunk of cn blocks can vldu cn*32 consecutive floats from
 * x_low_perm[bp * nb*32 + chunk_start*32]. Total = 4*nb*32 = 128*nb = K. */
void q4k_std_build_x_perm_extern(const float *x, float *x_low_perm,
                                  float *x_high_perm, int K);

void q4k_std_build_x_perm_extern(const float *x, float *x_low_perm,
                                  float *x_high_perm, int K) {
    const int nb = K / 256;
    for (int bp = 0; bp < 4; bp++) {
        float *xlo_bp = x_low_perm  + (size_t) bp * nb * 32;
        float *xhi_bp = x_high_perm + (size_t) bp * nb * 32;
        for (int b = 0; b < nb; b++) {
            const float *xb = x + (size_t) b * 256;
            for (int i = 0; i < 32; i++) {
                const int qq = i / 8;
                const int ii = i % 8;
                xlo_bp[b * 32 + i] = xb[64 * qq      + 4 * ii + bp];
                xhi_bp[b * 32 + i] = xb[64 * qq + 32 + 4 * ii + bp];
            }
        }
    }
}

/* Inner per-row dot using pre-permuted x. blk_row points at row m's
 * first block. x_low_perm and x_high_perm are nb*128 floats each. */
float q4k_std_row_dot_xperm_extern(const uint8_t *blk_row,
                                    const float *x_low_perm,
                                    const float *x_high_perm,
                                    int nb);

/* 8-ROW tile. Shares one x_low/x_high vldu per (block, byte-position)
 * across 8 row FMAs. Per-tile dlane/mlane built per-block (cheap). */
void q4k_std_8rows_xperm_extern(const uint8_t * const blk_rows[8],
                                  const float *x_low_perm,
                                  const float *x_high_perm,
                                  float *y_out[8], int nb);

void q4k_std_8rows_xperm_extern(const uint8_t * const blk_rows[8],
                                  const float *x_low_perm,
                                  const float *x_high_perm,
                                  float *y_out[8], int nb) {
    float ay[8] = {0};

    for (int b = 0; b < nb; b++) {
        const float *xlo_b = x_low_perm  + (size_t) b * 128;
        const float *xhi_b = x_high_perm + (size_t) b * 128;

        /* Per-row qs_v loaded once per block, kept in 8 registers across
         * the 4 byte-position iterations. */
        __vr qs_v[8];
        __vr dlv[8], mlv[8], dhv[8], mhv[8];
        __vr acc[8];
        for (int r = 0; r < 8; r++) {
            const uint8_t *blk = blk_rows[r] + (size_t) b * 144;

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

            float dlane_lo[32], mlane_lo[32], dlane_hi[32], mlane_hi[32];
            for (int q = 0; q < 4; q++) {
                const float d_l = d_sub[2 * q],     m_l = m_sub[2 * q];
                const float d_h = d_sub[2 * q + 1], m_h = m_sub[2 * q + 1];
                for (int j = 0; j < 8; j++) {
                    dlane_lo[8 * q + j] = d_l;
                    mlane_lo[8 * q + j] = m_l;
                    dlane_hi[8 * q + j] = d_h;
                    mlane_hi[8 * q + j] = m_h;
                }
            }
            dlv[r] = _vel_vldu_vssl(4, (void *) dlane_lo, 32);
            mlv[r] = _vel_vldu_vssl(4, (void *) mlane_lo, 32);
            dhv[r] = _vel_vldu_vssl(4, (void *) dlane_hi, 32);
            mhv[r] = _vel_vldu_vssl(4, (void *) mlane_hi, 32);

            qs_v[r] = _vel_vldlzx_vssl(4, (void *)(blk + 16), 32);
            acc[r]  = _vel_vbrds_vsl(0.0f, 32);
        }

        __vr mask = _vel_vbrdl_vsl(0x0FUL, 32);

        for (int bp = 0; bp < 4; bp++) {
            /* x loads SHARED across all 8 rows -- 8× reuse. */
            __vr xl = _vel_vldu_vssl(4, (void *)(xlo_b + bp * 32), 32);
            __vr xh = _vel_vldu_vssl(4, (void *)(xhi_b + bp * 32), 32);

            for (int r = 0; r < 8; r++) {
                __vr shifted = _vel_vsrl_vvsl(qs_v[r], 8 * bp, 32);
                __vr nib_lo  = _vel_vand_vvvl(shifted, mask, 32);
                __vr nib_hi  = _vel_vand_vvvl(_vel_vsrl_vvsl(shifted, 4, 32),
                                                mask, 32);
                __vr nlf     = _vel_vcvtsw_vvl(nib_lo, 32);
                __vr nhf     = _vel_vcvtsw_vvl(nib_hi, 32);

                __vr w_lo = _vel_vfmuls_vvvl(dlv[r], nlf, 32);
                w_lo      = _vel_vfsubs_vvvl(w_lo, mlv[r], 32);
                acc[r]    = _vel_vfmads_vvvvl(acc[r], w_lo, xl, 32);

                __vr w_hi = _vel_vfmuls_vvvl(dhv[r], nhf, 32);
                w_hi      = _vel_vfsubs_vvvl(w_hi, mhv[r], 32);
                acc[r]    = _vel_vfmads_vvvvl(acc[r], w_hi, xh, 32);
            }
        }

        for (int r = 0; r < 8; r++) {
            __vr red = _vel_vfsums_vvl(acc[r], 32);
            ay[r] += _vel_lvss_svs(red, 0);
        }
    }

    for (int r = 0; r < 8; r++) *y_out[r] = ay[r];
}

float q4k_std_row_dot_xperm_extern(const uint8_t *blk_row,
                                    const float *x_low_perm,
                                    const float *x_high_perm,
                                    int nb) {
    float acc = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = blk_row + (size_t) b * 144;

        /* Header decode. */
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

        /* Build dlane_lo[32] / mlane_lo[32] / dlane_hi[32] / mlane_hi[32]
         * on the stack. Per-block cost ≈ 128 stores (very cheap). */
        float dlane_lo[32], mlane_lo[32], dlane_hi[32], mlane_hi[32];
        for (int q = 0; q < 4; q++) {
            const float d_lo = d_sub[2 * q],     m_lo = m_sub[2 * q];
            const float d_hi = d_sub[2 * q + 1], m_hi = m_sub[2 * q + 1];
            for (int j = 0; j < 8; j++) {
                dlane_lo[8 * q + j] = d_lo;
                mlane_lo[8 * q + j] = m_lo;
                dlane_hi[8 * q + j] = d_hi;
                mlane_hi[8 * q + j] = m_hi;
            }
        }
        __vr dlv = _vel_vldu_vssl(4, (void *) dlane_lo, 32);
        __vr mlv = _vel_vldu_vssl(4, (void *) mlane_lo, 32);
        __vr dhv = _vel_vldu_vssl(4, (void *) dlane_hi, 32);
        __vr mhv = _vel_vldu_vssl(4, (void *) mlane_hi, 32);

        /* Load whole block of qs as 32 u32 lanes (stride=4, aligned). */
        const uint8_t *qs = blk + 16;
        __vr qs_v = _vel_vldlzx_vssl(4, (void *) qs, 32);
        __vr mask = _vel_vbrdl_vsl(0x0FUL, 32);

        __vr acc_block = _vel_vbrds_vsl(0.0f, 32);

        for (int bp = 0; bp < 4; bp++) {
            __vr shifted = _vel_vsrl_vvsl(qs_v, 8 * bp, 32);
            __vr nib_lo  = _vel_vand_vvvl(shifted, mask, 32);
            __vr nib_hi  = _vel_vand_vvvl(_vel_vsrl_vvsl(shifted, 4, 32),
                                            mask, 32);
            __vr nlf     = _vel_vcvtsw_vvl(nib_lo, 32);
            __vr nhf     = _vel_vcvtsw_vvl(nib_hi, 32);

            /* x layout (bp-major): per bp, nb*32 contiguous floats.
             * For this row, block b at byte position bp: 32 floats at
             * x_low_perm[bp * nb*32 + b * 32]. */
            __vr xl = _vel_vldu_vssl(4,
                (void *)(x_low_perm  + (size_t) bp * nb * 32 + (size_t) b * 32), 32);
            __vr xh = _vel_vldu_vssl(4,
                (void *)(x_high_perm + (size_t) bp * nb * 32 + (size_t) b * 32), 32);

            /* w_lo = dlv * nlf - mlv ; acc += w_lo * xl */
            __vr w_lo = _vel_vfmuls_vvvl(dlv, nlf, 32);
            w_lo      = _vel_vfsubs_vvvl(w_lo, mlv, 32);
            acc_block = _vel_vfmads_vvvvl(acc_block, w_lo, xl, 32);

            __vr w_hi = _vel_vfmuls_vvvl(dhv, nhf, 32);
            w_hi      = _vel_vfsubs_vvvl(w_hi, mhv, 32);
            acc_block = _vel_vfmads_vvvvl(acc_block, w_hi, xh, 32);
        }

        __vr red = _vel_vfsums_vvl(acc_block, 32);
        acc += _vel_lvss_svs(red, 0);
    }

    return acc;
}

/* -------- CHUNKED variant: process multiple blocks per FMA chain. -------
 *
 * Per row, pack qs from the raw row into a per-thread scratch buffer (so
 * cn*128 qs bytes are contiguous and stride-4 vldlzx can load them at
 * VL=cn*32). Then VL=cn*32 FMAs amortise dispatch / pipeline cost across
 * cn blocks.
 *
 *   cn = 8  => VL = 256  (MVL)
 *   cn = 4  => VL = 128
 *
 * Caller-provided qs_scratch must be at least nb*128 bytes. The x_*_perm
 * inputs are in bp-major layout (see q4k_std_build_x_perm_extern):
 *   x_*_perm[bp * nb*32 + b*32 + lane]   for bp 0..3, b 0..nb-1, lane 0..31
 *
 * Per chunk: 1 qs vldlzx + 4 dlane builds + 4 byte-pos × 2 FMA chains.
 * Total FMAs per row = (nb/cn) * 8  vs the single-block kernel's nb * 8. */
float q4k_std_row_dot_chunked_extern(const uint8_t *blk_row,
                                      const float *x_low_perm,
                                      const float *x_high_perm,
                                      uint8_t *qs_scratch,
                                      int nb);

#define Q4K_STD_CHUNK 8                       /* 8 blocks per chunk => VL=256 */

/* Same kernel with OPTIONAL pre-decoded header source.
 *   hdr_decoded != NULL : skip per-block scalar decode, read 16 fp32 per
 *                          block from hdr_decoded (8 d_sub + 8 m_sub).
 *                          hdr_decoded layout: row r, block b at
 *                          hdr_decoded + (r*nb + b) * 64 -- BUT this
 *                          inner func takes the ROW POINTER so caller
 *                          passes hdr_decoded + r*nb*64. Then per block:
 *                          16 floats at offset b*64.
 *   hdr_decoded == NULL : decode on the fly (same as the
 *                          q4k_std_row_dot_chunked_extern shim below). */
float q4k_std_row_dot_chunked_hdr_extern(const uint8_t *blk_row,
                                          const float *hdr_decoded_row,
                                          const float *x_low_perm,
                                          const float *x_high_perm,
                                          uint8_t *qs_scratch,
                                          int nb);

float q4k_std_row_dot_chunked_hdr_extern(const uint8_t *blk_row,
                                          const float *hdr_decoded_row,
                                          const float *x_low_perm,
                                          const float *x_high_perm,
                                          uint8_t *qs_scratch,
                                          int nb);

float q4k_std_row_dot_chunked_extern(const uint8_t *blk_row,
                                      const float *x_low_perm,
                                      const float *x_high_perm,
                                      uint8_t *qs_scratch,
                                      int nb) {
    return q4k_std_row_dot_chunked_hdr_extern(blk_row, /*hdr_decoded=*/NULL,
                                              x_low_perm, x_high_perm,
                                              qs_scratch, nb);
}

float q4k_std_row_dot_chunked_hdr_extern(const uint8_t *blk_row,
                                          const float *hdr_decoded_row,
                                          const float *x_low_perm,
                                          const float *x_high_perm,
                                          uint8_t *qs_scratch,
                                          int nb) {
    /* Pack qs into scratch via 16-lane u64 vector copy (1 vld + 1 vst per
     * block, ~2 cycles). Total: nb such copies. */
    for (int b = 0; b < nb; b++) {
        __vr v = _vel_vld_vssl(8, (void *)(blk_row + (size_t) b * 144 + 16), 16);
        _vel_vst_vssl(v, 8, qs_scratch + (size_t) b * 128, 16);
    }

    float acc = 0.0f;

    /* Chunk loop: decode headers for the CHUNK and process it. Per-chunk
     * d/m arrays sized for Q4K_STD_CHUNK blocks max — bounded regardless
     * of nb, so it handles real-model shapes like Qwen FFN-down K=17408
     * (nb=68) without overflow. */
    for (int chunk_start = 0; chunk_start < nb; chunk_start += Q4K_STD_CHUNK) {
        int cn = (nb - chunk_start) < Q4K_STD_CHUNK ? (nb - chunk_start) : Q4K_STD_CHUNK;
        const int VL = cn * 32;

        /* Chunk's 8 d_sub + 8 m_sub per block: either load pre-decoded from
         * cache or fall back to per-block scalar decode. */
        float d_sub_chunk[Q4K_STD_CHUNK * 8];
        float m_sub_chunk[Q4K_STD_CHUNK * 8];
        if (hdr_decoded_row != NULL) {
            /* hdr layout: per block 16 fp32 = [d0..d7, m0..m7]. */
            const float *hdr_chunk = hdr_decoded_row + (size_t) chunk_start * 16;
            for (int cb = 0; cb < cn; cb++) {
                const float *blk_hdr = hdr_chunk + (size_t) cb * 16;
                for (int s = 0; s < 8; s++) {
                    d_sub_chunk[cb * 8 + s] = blk_hdr[s    ];
                    m_sub_chunk[cb * 8 + s] = blk_hdr[8 + s];
                }
            }
        } else {
            for (int cb = 0; cb < cn; cb++) {
                const uint8_t *blk = blk_row + (size_t)(chunk_start + cb) * 144;
                uint16_t d_raw, dmin_raw;
                memcpy(&d_raw,    blk + 0, 2);
                memcpy(&dmin_raw, blk + 2, 2);
                const float d_super    = h2f(d_raw);
                const float dmin_super = h2f(dmin_raw);
                const uint8_t *sc12 = blk + 4;
                for (int s = 0; s < 8; s++) {
                    uint8_t sc, mn;
                    q4k_sm(s, sc12, &sc, &mn);
                    d_sub_chunk[cb * 8 + s] = d_super    * (float) sc;
                    m_sub_chunk[cb * 8 + s] = dmin_super * (float) mn;
                }
            }
        }

        /* Load qs chunk: cn*32 u32 lanes from packed scratch (stride=4). */
        __vr qs_chunk = _vel_vldlzx_vssl(4,
            (void *)(qs_scratch + (size_t) chunk_start * 128), VL);
        __vr mask = _vel_vbrdl_vsl(0x0FUL, VL);

        /* Build dlane/mlane for low + high at VL.
         * Lane i in 0..VL-1: block_in_chunk = i/32, quarter q = (i%32)/8.
         *   dlane_lo[i] = d_sub_chunk[(i/32)*8 + 2*q]   etc. */
        float dlane_lo[256], mlane_lo[256], dlane_hi[256], mlane_hi[256];
        for (int cb = 0; cb < cn; cb++) {
            const float *d_blk = d_sub_chunk + (size_t) cb * 8;
            const float *m_blk = m_sub_chunk + (size_t) cb * 8;
            for (int q = 0; q < 4; q++) {
                const float d_l = d_blk[2 * q],     m_l = m_blk[2 * q];
                const float d_h = d_blk[2 * q + 1], m_h = m_blk[2 * q + 1];
                for (int j = 0; j < 8; j++) {
                    const int lane = cb * 32 + q * 8 + j;
                    dlane_lo[lane] = d_l;
                    mlane_lo[lane] = m_l;
                    dlane_hi[lane] = d_h;
                    mlane_hi[lane] = m_h;
                }
            }
        }
        __vr dlv = _vel_vldu_vssl(4, (void *) dlane_lo, VL);
        __vr mlv = _vel_vldu_vssl(4, (void *) mlane_lo, VL);
        __vr dhv = _vel_vldu_vssl(4, (void *) dlane_hi, VL);
        __vr mhv = _vel_vldu_vssl(4, (void *) mlane_hi, VL);

        __vr acc_v = _vel_vbrds_vsl(0.0f, VL);

        for (int bp = 0; bp < 4; bp++) {
            __vr shifted = _vel_vsrl_vvsl(qs_chunk, 8 * bp, VL);
            __vr nib_lo  = _vel_vand_vvvl(shifted, mask, VL);
            __vr nib_hi  = _vel_vand_vvvl(_vel_vsrl_vvsl(shifted, 4, VL),
                                            mask, VL);
            __vr nlf     = _vel_vcvtsw_vvl(nib_lo, VL);
            __vr nhf     = _vel_vcvtsw_vvl(nib_hi, VL);

            /* x_*_perm bp-major: per bp, nb*32 floats; this chunk wants
             * cn*32 floats starting at chunk_start*32. */
            __vr xl = _vel_vldu_vssl(4,
                (void *)(x_low_perm  + (size_t) bp * nb * 32 + (size_t) chunk_start * 32), VL);
            __vr xh = _vel_vldu_vssl(4,
                (void *)(x_high_perm + (size_t) bp * nb * 32 + (size_t) chunk_start * 32), VL);

            __vr w_lo = _vel_vfmuls_vvvl(dlv, nlf, VL);
            w_lo      = _vel_vfsubs_vvvl(w_lo, mlv, VL);
            acc_v     = _vel_vfmads_vvvvl(acc_v, w_lo, xl, VL);

            __vr w_hi = _vel_vfmuls_vvvl(dhv, nhf, VL);
            w_hi      = _vel_vfsubs_vvvl(w_hi, mhv, VL);
            acc_v     = _vel_vfmads_vvvvl(acc_v, w_hi, xh, VL);
        }

        __vr red = _vel_vfsums_vvl(acc_v, VL);
        acc += _vel_lvss_svs(red, 0);
    }

    return acc;
}
