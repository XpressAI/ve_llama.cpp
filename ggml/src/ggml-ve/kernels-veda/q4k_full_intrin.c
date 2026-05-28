/* Q4_K matvec: FULL high-VL implementation.
 *
 * The user's spec:
 *   "256 FP64 is the vector register. It can handle 512 32-bit and 1024
 *    BF16. So what we really want to do is load 4096 Q4 blocks and mask
 *    it into 16 registers using bit masks and shifts."
 *
 * Translation:
 *   - VE vector register: 256 × 64-bit lanes
 *   - At 4 bits per Q4 value: 256 × 16 = 4096 Q4 values per register
 *   - Load all 4096 nibbles of one row (K=4096) into ONE register
 *   - Extract 16 nibble positions (k=0..15) into 16 separate registers
 *     via shift + mask -- each register has 256 lanes × 1 nibble value
 *   - That's all 4096 element values in 16 registers, ready for FMA
 *
 * To make this load possible we use TWO restructurings of standard Q4_K:
 *
 * (1) CANONICAL NIBBLE ORDER (same byte count):
 *     byte k holds (element 2k+1) << 4 | (element 2k)
 *     So nibble position k of byte k_b corresponds to element 2*k_b + (k%2),
 *     and within a vector lane (8 bytes), nibble position k corresponds
 *     to element offset k.
 *
 * (2) QS/HDR SPLIT (same total bytes):
 *     Headers (d, dmin, scales) stored in a separate per-row array; qs
 *     bytes stored contiguous per row. Lets us load the entire row's qs
 *     in one vector load.
 *
 * Result: row's qs = nb * 128 bytes contiguous. For K=4096, that's 2048
 * bytes = vl=256 stride=8 in one register. The user's "4096 Q4 values
 * in one register" is literally what we load.
 *
 * Build: clang --target=ve-linux -O3 -fpic -c q4k_full_intrin.c
 *        ncc -fpic -shared -fopenmp -o libq4k_full.so q4k_full_intrin.o
 *
 * NOTE: We use NCC's OpenMP runtime by linking with ncc. Single-core if
 * OMP isn't available.
 */

#include <stdint.h>
#include <string.h>
#include <velintrin.h>
/* OpenMP commented out — VEDA's kernel runtime doesn't play with clang's
 * libomp (hangs forever in pthread spawn). The host side can parallelise
 * the call by chunking M and launching multiple kernels. For now this
 * kernel is single-core; perf numbers below are per-core. */

extern int vedaMemPtr(void **ptr, uint64_t vptr);

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

static inline void q4k_sm(int idx, const uint8_t *sc12, uint8_t *s, uint8_t *m) {
    if (idx < 4) {
        *s = sc12[idx]     & 0x3F;
        *m = sc12[idx + 4] & 0x3F;
    } else {
        *s = (sc12[idx + 4] & 0x0F) | ((sc12[idx - 4] >> 6) << 4);
        *m = (sc12[idx + 4] >>   4) | ((sc12[idx]     >> 6) << 4);
    }
}

/* Reorder a full M×K Q4_K weight tensor into split (qs_arr, hdr_arr) with
 * canonical nibble ordering. Called once at upload. Total bytes unchanged.
 *
 *   qs_arr:  M × nb × 128 bytes  (contiguous canonical-ordered nibbles)
 *   hdr_arr: M × nb × 16 bytes   (d, dmin, scales -- unchanged from standard)
 */
uint64_t ve_q4k_full_reorder_hmem(void *src, void *qs_out, void *hdr_out,
                                   uint64_t M, uint64_t K) {
    int nb = (int) K / 256;
    const uint8_t *S = (const uint8_t *) src;
    uint8_t *QS  = (uint8_t *) qs_out;
    uint8_t *HDR = (uint8_t *) hdr_out;

    for (uint64_t m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = S + (m * nb + b) * 144;
            uint8_t *qd = QS  + (m * nb + b) * 128;
            uint8_t *hd = HDR + (m * nb + b) * 16;

            memcpy(hd, blk, 16);  /* d, dmin, scales */

            /* Standard → canonical for the 128 qs bytes. */
            uint8_t elem[256];
            const uint8_t *src_qs = blk + 16;
            for (int p = 0; p < 4; p++) {
                for (int l = 0; l < 32; l++) {
                    elem[64*p + l]      = src_qs[32*p + l] & 0x0F;
                    elem[64*p + 32 + l] = src_qs[32*p + l] >> 4;
                }
            }
            for (int k = 0; k < 128; k++) {
                qd[k] = (uint8_t) ((elem[2*k + 1] << 4) | elem[2*k]);
            }
        }
    }
    return 0;
}

/* Process up to 16 blocks (K=4096 elements) per chunk at vl<=256. */
#define MAX_CHUNK_BLOCKS 16
#define MAX_CHUNK_VL    (MAX_CHUNK_BLOCKS * 16)

/* Per-row matvec on split, canonical-ordered storage.
 * For K<=4096: one chunk at vl=nb*16. For K>4096: multiple chunks.
 * Exported as q4k_full_row_dot_extern for the NCC-compiled OMP dispatcher. */
float q4k_full_row_dot_extern(const uint8_t *qs_row, const uint8_t *hdr_row,
                              const float *x, int nb);

float q4k_full_row_dot_extern(const uint8_t *qs_row, const uint8_t *hdr_row,
                              const float *x, int nb) {
    float acc_scalar = 0.0f;
    int blk_offset = 0;
    while (blk_offset < nb) {
        int chunk_nb = nb - blk_offset;
        if (chunk_nb > MAX_CHUNK_BLOCKS) chunk_nb = MAX_CHUNK_BLOCKS;
        const int VL = chunk_nb * 16;

        const uint8_t *qs_chunk  = qs_row  + (size_t) blk_offset * 128;
        const uint8_t *hdr_chunk = hdr_row + (size_t) blk_offset * 16;
        const float   *x_chunk   = x       + (size_t) blk_offset * 256;

        /* Build per-lane d and m vectors ONCE per chunk. Each lane i covers
         * 16 consecutive elements (16i .. 16i+15). All 16 elements in a lane
         * belong to the same sub-block (sub-block index = i/2). So:
         *   dlane[i] = d_super[i/16/8] * scale[(i/16)*8 + (i/2)%8]
         *            = d_full[i/2]
         *   mlane[i] = m_full[i/2]   (analogous) */
        float dlane[MAX_CHUNK_VL];
        float mlane[MAX_CHUNK_VL];
        for (int b = 0; b < chunk_nb; b++) {
            const uint8_t *hdr = hdr_chunk + (size_t) b * 16;
            uint16_t d_raw, dmin_raw;
            memcpy(&d_raw,    hdr,     2);
            memcpy(&dmin_raw, hdr + 2, 2);
            const float d_super    = h2f(d_raw);
            const float dmin_super = h2f(dmin_raw);
            const uint8_t *sc12 = hdr + 4;
            for (int s = 0; s < 8; s++) {
                uint8_t sc, mn;
                q4k_sm(s, sc12, &sc, &mn);
                const float d = d_super    * (float) sc;
                const float m = dmin_super * (float) mn;
                /* Sub-block s of block b covers 32 elements at offset
                 *   (b * 256 + s * 32)
                 * which span lanes [(b * 16 + s * 2) .. (b * 16 + s * 2 + 1)]
                 * since each lane covers 16 elements. */
                int lane0 = b * 16 + s * 2;
                dlane[lane0    ] = d;
                dlane[lane0 + 1] = d;
                mlane[lane0    ] = m;
                mlane[lane0 + 1] = m;
            }
        }

        __vr qs_v   = _vel_vld_vssl(8, (void *)qs_chunk, VL);
        __vr mask_f = _vel_vbrdl_vsl(0x0FUL, VL);
        __vr acc_v  = _vel_vbrds_vsl(0.0f, VL);
        __vr dlane_v = _vel_vldu_vssl(4, (void *) dlane, VL);
        __vr mlane_v = _vel_vldu_vssl(4, (void *) mlane, VL);

        /* Σ (d * q - m) * x  per element across all 16 nibble positions.
         * Per extract: w = d*q - m, acc += w * x. */
        for (int k = 0; k < 16; k++) {
            __vr nib = _vel_vand_vvvl(_vel_vsrl_vvsl(qs_v, 4 * k, VL),
                                      mask_f, VL);
            __vr nib_f = _vel_vcvtsw_vvl(nib, VL);
            __vr xv = _vel_vldu_vssl(64, (void *)(x_chunk + k), VL);
            /* w = d * nib - m -- vfnmsbs is the wrong sign, use vfmuls + vfsubs. */
            __vr w  = _vel_vfmuls_vvvl(dlane_v, nib_f, VL);
            w       = _vel_vfsubs_vvvl(w, mlane_v, VL);
            acc_v = _vel_vfmads_vvvvl(acc_v, w, xv, VL);
        }
        acc_v = _vel_vfsums_vvl(acc_v, VL);
        acc_scalar += _vel_lvss_svs(acc_v, 0);
        blk_offset += chunk_nb;
    }
    return acc_scalar;
}

/* Entry point lives in q4k_full_dispatch.c (compiled with NCC for OMP).
 * It calls q4k_full_row_dot_extern per row in parallel. */
