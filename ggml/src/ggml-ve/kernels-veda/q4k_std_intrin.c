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

/* PACKED variant of the x permute: per bp, nb*32 u64 where each u64 packs
 * (x_low, x_high) as low|high<<32. Used by the pvfmad kernel below. */
void q4k_std_build_x_perm_packed_extern(const float *x, uint64_t *x_pk_perm, int K);

void q4k_std_build_x_perm_packed_extern(const float *x, uint64_t *x_pk_perm, int K) {
    const int nb = K / 256;
    for (int bp = 0; bp < 4; bp++) {
        uint64_t *xpk_bp = x_pk_perm + (size_t) bp * nb * 32;
        for (int b = 0; b < nb; b++) {
            const float *xb = x + (size_t) b * 256;
            for (int i = 0; i < 32; i++) {
                const int qq = i / 8;
                const int ii = i % 8;
                const float x_lo = xb[64 * qq      + 4 * ii + bp];
                const float x_hi = xb[64 * qq + 32 + 4 * ii + bp];
                uint32_t lo_bits, hi_bits;
                memcpy(&lo_bits, &x_lo, 4);
                memcpy(&hi_bits, &x_hi, 4);
                xpk_bp[b * 32 + i] = ((uint64_t) hi_bits << 32) | lo_bits;
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

/* GATHER variant: skips the scratch pack and uses vgtlzx + vsfa to load
 * cn*32 u32 lanes directly from raw HBM. Address pattern per chunk:
 *   addr[i] = chunk_base + (i/32)*144 + 16 + (i%32)*4
 * with chunk_base = blk_row + chunk_start * 144.
 *
 * The offset vector (i/32)*144 + 16 + (i%32)*4 doesn't depend on
 * chunk_start, so it's precomputed once into g_qs_gather_offsets (a
 * static MAX_VL=256 u64 array) and loaded per-chunk. */
float q4k_std_row_dot_chunked_gather_hdr_extern(const uint8_t *blk_row,
                                                  const float *hdr_decoded_row,
                                                  const float *x_low_perm,
                                                  const float *x_high_perm,
                                                  int nb);

float q4k_std_row_dot_chunked_gather_hdr_extern(const uint8_t *blk_row,
                                                  const float *hdr_decoded_row,
                                                  const float *x_low_perm,
                                                  const float *x_high_perm,
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

/* ---- Gather variant ---- */

/* Static offset vector. Initialised lazily on first call. Pattern:
 *   off[i] = (i/32)*144 + 16 + (i%32)*4   for i in 0..MAX-1
 * MAX = Q4K_STD_CHUNK*32 = 256 (matches MVL). */
#define Q4K_STD_GATHER_VL (Q4K_STD_CHUNK * 32)
static uint64_t g_qs_gather_offsets[Q4K_STD_GATHER_VL] __attribute__((aligned(64)));
static int      g_qs_gather_init = 0;

static void q4k_std_init_gather_offsets(void) {
    for (int i = 0; i < Q4K_STD_GATHER_VL; i++) {
        const int cb = i / 32;
        const int ii = i % 32;
        g_qs_gather_offsets[i] = (uint64_t)(cb * 144 + 16 + ii * 4);
    }
    g_qs_gather_init = 1;
}

float q4k_std_row_dot_chunked_gather_hdr_extern(const uint8_t *blk_row,
                                                  const float *hdr_decoded_row,
                                                  const float *x_low_perm,
                                                  const float *x_high_perm,
                                                  int nb) {
    if (!g_qs_gather_init) q4k_std_init_gather_offsets();

    /* Preload the offset vector at MAX VL. */
    __vr off_v = _vel_vld_vssl(8, (void *)g_qs_gather_offsets, Q4K_STD_GATHER_VL);

    float acc = 0.0f;

    for (int chunk_start = 0; chunk_start < nb; chunk_start += Q4K_STD_CHUNK) {
        int cn = (nb - chunk_start) < Q4K_STD_CHUNK ? (nb - chunk_start) : Q4K_STD_CHUNK;
        const int VL = cn * 32;

        /* Build chunk-relative absolute addresses: chunk_base + offsets. */
        const uint64_t chunk_base = (uint64_t)(uintptr_t)(blk_row + (size_t) chunk_start * 144);
        __vr addrs = _vel_vsfa_vvssl(off_v, /*shift=*/0, chunk_base, VL);

        /* Gather load: each lane reads u32 at addrs[lane]. */
        __vr qs_chunk = _vel_vgtlzx_vvssl(addrs, /*sw=*/0, /*sz=*/0, VL);
        __vr mask     = _vel_vbrdl_vsl(0x0FUL, VL);

        /* Headers: same logic as the scratch-pack variant. */
        float d_sub_chunk[Q4K_STD_CHUNK * 8];
        float m_sub_chunk[Q4K_STD_CHUNK * 8];
        if (hdr_decoded_row != NULL) {
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
            __vr nib_hi  = _vel_vand_vvvl(_vel_vsrl_vvsl(shifted, 4, VL), mask, VL);
            __vr nlf     = _vel_vcvtsw_vvl(nib_lo, VL);
            __vr nhf     = _vel_vcvtsw_vvl(nib_hi, VL);

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

/* ---- Packed pvfmad variant ---- *
 *
 * Packs low+high nibble FMAs into _vel_pvfmad_vvvvl (packed FP32 -- 2
 * elements per 64-bit lane). Per chunk:
 *   - 4 byte-positions × 1 packed FMA chain = 4 packed FMAs (vs 8 in
 *     the non-packed chunked kernel: 4 bp × 2 halves).
 *   - dlane_pk[i] = pack(d_low, d_high), mlane_pk[i] = pack(-m_lo, -m_hi)
 *   - x_pk[i] = pack(x_low, x_high) -- preloaded once per matvec
 *   - nib_pk[i] = pack(low_nib_i, high_nib_i)
 *   - w_pk = pvfmad(neg_m_pk, d_pk, nib_f_pk)   # = d*nib - m  (packed)
 *   - acc_pk = pvfmad(acc_pk, w_pk, x_pk)
 *
 * Reduction: extract low and high halves of each lane, sum to scalar.
 *
 * Inputs:
 *   x_pk_perm: packed x permute (low|high<<32 per bp×nb×32 layout)
 */
float q4k_std_row_dot_chunked_packed_hdr_extern(const uint8_t *blk_row,
                                                  const float *hdr_decoded_row,
                                                  const uint64_t *x_pk_perm,
                                                  int nb);

float q4k_std_row_dot_chunked_packed_hdr_extern(const uint8_t *blk_row,
                                                  const float *hdr_decoded_row,
                                                  const uint64_t *x_pk_perm,
                                                  int nb) {
    if (!g_qs_gather_init) q4k_std_init_gather_offsets();

    /* Preload the offset vector at MAX VL (only used by gather path). */
    __vr off_v = _vel_vld_vssl(8, (void *) g_qs_gather_offsets, Q4K_STD_GATHER_VL);

    float acc = 0.0f;

    for (int chunk_start = 0; chunk_start < nb; chunk_start += Q4K_STD_CHUNK) {
        int cn = (nb - chunk_start) < Q4K_STD_CHUNK ? (nb - chunk_start) : Q4K_STD_CHUNK;
        const int VL = cn * 32;

        /* qs gather (same as gather variant). */
        const uint64_t chunk_base = (uint64_t)(uintptr_t)(blk_row + (size_t) chunk_start * 144);
        __vr addrs    = _vel_vsfa_vvssl(off_v, 0, chunk_base, VL);
        __vr qs_chunk = _vel_vgtlzx_vvssl(addrs, 0, 0, VL);

        /* Header source: cached pre-decoded, else live decode. */
        float d_sub_chunk[Q4K_STD_CHUNK * 8];
        float m_sub_chunk[Q4K_STD_CHUNK * 8];
        if (hdr_decoded_row != NULL) {
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

        /* Build packed dlane/mlane (negated m for pvfmad).
         *   dlane_pk[lane] = pack(d_low, d_high) = hi<<32 | lo bits
         *   mlane_pk[lane] = pack(-m_low, -m_high) */
        uint64_t dlane_pk[256], mlane_pk[256];
        for (int cb = 0; cb < cn; cb++) {
            const float *d_blk = d_sub_chunk + (size_t) cb * 8;
            const float *m_blk = m_sub_chunk + (size_t) cb * 8;
            for (int q = 0; q < 4; q++) {
                const float d_l =  d_blk[2 * q],     d_h =  d_blk[2 * q + 1];
                const float m_l = -m_blk[2 * q],     m_h = -m_blk[2 * q + 1];
                uint32_t dl_b, dh_b, ml_b, mh_b;
                memcpy(&dl_b, &d_l, 4); memcpy(&dh_b, &d_h, 4);
                memcpy(&ml_b, &m_l, 4); memcpy(&mh_b, &m_h, 4);
                const uint64_t d_pk = ((uint64_t) dh_b << 32) | dl_b;
                const uint64_t m_pk = ((uint64_t) mh_b << 32) | ml_b;
                for (int j = 0; j < 8; j++) {
                    const int lane = cb * 32 + q * 8 + j;
                    dlane_pk[lane] = d_pk;
                    mlane_pk[lane] = m_pk;
                }
            }
        }
        __vr d_pk_v = _vel_vld_vssl(8, (void *) dlane_pk, VL);
        __vr m_pk_v = _vel_vld_vssl(8, (void *) mlane_pk, VL);

        __vr acc_pk = _vel_vbrdl_vsl(0UL, VL);
        __vr lo_mask = _vel_vbrdl_vsl(0x000000000000000FUL, VL);

        for (int bp = 0; bp < 4; bp++) {
            /* nib_pk lane i = low_nib_i | (high_nib_i << 32).
             *   low_nib_i  = (qs >> 8bp)      & 0x0F
             *   high_nib_i = (qs >> (8bp+4))  & 0x0F  -> shift left 32 */
            __vr shifted_lo = _vel_vsrl_vvsl(qs_chunk, 8 * bp,     VL);
            __vr shifted_hi = _vel_vsrl_vvsl(qs_chunk, 8 * bp + 4, VL);
            __vr low_nib    = _vel_vand_vvvl(shifted_lo, lo_mask,  VL);
            __vr high_nib   = _vel_vand_vvvl(shifted_hi, lo_mask,  VL);
            __vr high_upper = _vel_vsll_vvsl(high_nib, 32,         VL);
            __vr nib_pk     = _vel_vor_vvvl (low_nib,  high_upper, VL);

            /* Packed int32 -> packed fp32. */
            __vr nib_f_pk = _vel_pvcvtsw_vvl(nib_pk, VL);

            /* w_pk = -m + d*nib  (packed FMA). */
            __vr w_pk = _vel_pvfmad_vvvvl(m_pk_v, d_pk_v, nib_f_pk, VL);

            /* Load packed x. */
            __vr x_pk = _vel_vld_vssl(8,
                (void *)(x_pk_perm + (size_t) bp * nb * 32 + (size_t) chunk_start * 32), VL);

            /* acc_pk += w_pk * x_pk  (packed FMA). */
            acc_pk = _vel_pvfmad_vvvvl(acc_pk, w_pk, x_pk, VL);
        }

        /* Reduce packed accumulator. Pattern mirrors canon's packed
         * matvec at q4k_full_intrin.c:698-705. */
        __vr lo32_mask = _vel_vbrdl_vsl(0x00000000FFFFFFFFUL, VL);
        __vr acc_lo32 = _vel_vand_vvvl(acc_pk, lo32_mask, VL);
        __vr acc_hi32 = _vel_vsrl_vvsl(acc_pk, 32, VL);
        acc_lo32 = _vel_vsll_vvsl(acc_lo32, 32, VL);
        __vr acc_hi32_up = _vel_vsll_vvsl(acc_hi32, 32, VL);
        __vr acc_sum = _vel_vfadds_vvvl(acc_lo32, acc_hi32_up, VL);
        acc_sum = _vel_vfsums_vvl(acc_sum, VL);
        acc += _vel_lvss_svs(acc_sum, 0);
    }

    return acc;
}
