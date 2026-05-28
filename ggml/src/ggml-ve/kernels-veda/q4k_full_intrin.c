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

/* Permuted-x variant: x_perm[k * n_lanes + i] = x[16*i + k] where
 * n_lanes = nb * 16. Caller builds once per matvec to make x loads
 * sequential inside the inner extract loop. sx_full[i] = Σ_k x[16i+k]
 * precomputed so the inner can apply the -m*Σx correction once per
 * chunk instead of per nibble extract. */
float q4k_full_row_dot_xperm_extern(const uint8_t *qs_row, const uint8_t *hdr_row,
                                     const float *x_perm, const float *sx_full,
                                     int nb);

/* Build x_perm AND sx_full (Σ x per lane) in a single pass.
 * sx_full[lane] = Σ_{k=0..15} x[16*lane + k]
 * Used by the optimised single-FMA inner kernel to apply the -m * Σx
 * correction outside the per-element loop. */
void q4k_build_x_perm_extern(const float *x, float *x_perm, int K) {
    const int n_lanes = K / 16;
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < n_lanes; i++) {
            x_perm[k * n_lanes + i] = x[16 * i + k];
        }
    }
}

void q4k_build_sx_full_extern(const float *x, float *sx_full, int K) {
    const int n_lanes = K / 16;
    for (int i = 0; i < n_lanes; i++) {
        float s = 0.0f;
        #pragma _NEC ivdep
        for (int k = 0; k < 16; k++) {
            s += x[16 * i + k];
        }
        sx_full[i] = s;
    }
}

/* Packed-mode x permute: for pair (k_even, k_odd) build per-lane uint64
 * with x[16i+k_even] in lower 32 bits, x[16i+k_odd] in upper 32 bits.
 * Layout: x_pk[p * n_lanes + i] for p in 0..7, i in 0..n_lanes-1. */
void q4k_build_x_pk_extern(const float *x, uint64_t *x_pk, int K) {
    const int n_lanes = K / 16;
    for (int p = 0; p < 8; p++) {
        for (int i = 0; i < n_lanes; i++) {
            uint32_t lo_bits, hi_bits;
            float fl = x[16 * i + 2*p];
            float fh = x[16 * i + 2*p + 1];
            memcpy(&lo_bits, &fl, 4);
            memcpy(&hi_bits, &fh, 4);
            x_pk[p * n_lanes + i] = ((uint64_t) hi_bits << 32) | lo_bits;
        }
    }
}

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

/* xperm variant: x already permuted so x_perm[k * n_lanes + i] = x[16i + k].
 * Loads are now SEQUENTIAL (stride=4). Also fuses vfmuls + vfsubs into a
 * single vfmads by pre-negating m: w = d*nib - m  →  vfmads(neg_m, d, nib). */
float q4k_full_row_dot_xperm_extern(const uint8_t *qs_row, const uint8_t *hdr_row,
                                     const float *x_perm, const float *sx_full,
                                     int nb) {
    float acc_scalar = 0.0f;
    const int n_lanes_total = nb * 16;
    int blk_offset = 0;
    while (blk_offset < nb) {
        int chunk_nb = nb - blk_offset;
        if (chunk_nb > MAX_CHUNK_BLOCKS) chunk_nb = MAX_CHUNK_BLOCKS;
        const int VL = chunk_nb * 16;

        const uint8_t *qs_chunk  = qs_row  + (size_t) blk_offset * 128;
        /* Pre-decoded header layout: 64 bytes/block = 8 fp32 d_sub + 8 fp32 m_sub.
         * Decoded at upload by hbm_cache::get_or_upload_q4k_canon, so no per-row
         * SPU work needed (h2f, q4k_sm, dlane build all GONE). */
        const float   *hdr_chunk = (const float *)(hdr_row + (size_t) blk_offset * 64);
        const int      lane_off  = blk_offset * 16;

        /* Build dlane[VL]/neg_mlane[VL] from the pre-decoded 8-fp32 vectors.
         * Each lane i covers 16 elements = half a sub-block. Sub-block index
         * for lane i is i/2 (within the chunk). The pre-decoded array has
         * d_sub at slot s and m_sub at slot 8+s per block. */
        float dlane[MAX_CHUNK_VL];
        float neg_mlane[MAX_CHUNK_VL];
        for (int b = 0; b < chunk_nb; b++) {
            const float *blk_hdr = hdr_chunk + b * 16;  /* 8 d + 8 m = 16 floats */
            for (int s = 0; s < 8; s++) {
                const float d = blk_hdr[s];
                const float nm = -blk_hdr[8 + s];
                int lane0 = b * 16 + s * 2;
                dlane[lane0    ] = d;
                dlane[lane0 + 1] = d;
                neg_mlane[lane0    ] = nm;
                neg_mlane[lane0 + 1] = nm;
            }
        }

        __vr qs_v   = _vel_vld_vssl(8, (void *)qs_chunk, VL);
        __vr mask_f = _vel_vbrdl_vsl(0x0FUL, VL);

        /* Decompose: y = Σ (d*nib - m)*x = d * Σ(nib*x) - m * Σx.
         * Σx per lane is precomputed below ONCE per call (cheap; doesn't
         * depend on row). The inner loop then needs only ONE FMA per
         * nibble extract (sum_nibx += nib * x) instead of two (w = d*nib
         * - m; acc += w * x). The d/m multiply happens once per chunk
         * outside the loop.
         *
         * 4-way ILP on sum_nibx accumulators (8-way and 16-way tested,
         * no gain -- inner is now too short to benefit from more chains). */
        __vr s0 = _vel_vbrds_vsl(0.0f, VL);
        __vr s1 = _vel_vbrds_vsl(0.0f, VL);
        __vr s2 = _vel_vbrds_vsl(0.0f, VL);
        __vr s3 = _vel_vbrds_vsl(0.0f, VL);
        __vr * acs[4] = {&s0, &s1, &s2, &s3};

        for (int k = 0; k < 16; k++) {
            __vr nib   = _vel_vand_vvvl(_vel_vsrl_vvsl(qs_v, 4 * k, VL),
                                        mask_f, VL);
            __vr nib_f = _vel_vcvtsw_vvl(nib, VL);
            __vr xv = _vel_vldu_vssl(4,
                (void *)(x_perm + k * n_lanes_total + lane_off), VL);
            *acs[k & 3] = _vel_vfmads_vvvvl(*acs[k & 3], nib_f, xv, VL);
        }
        s0 = _vel_vfadds_vvvl(s0, s1, VL);
        s2 = _vel_vfadds_vvvl(s2, s3, VL);
        s0 = _vel_vfadds_vvvl(s0, s2, VL);

        /* sx_full was precomputed by dispatcher (build_sx_full_extern):
         * sx_full[i] = Σ_k x[16i + k]. Per chunk we need lanes
         * [lane_off..lane_off+VL-1] which is just a sequential slice. */
        __vr dlane_v   = _vel_vldu_vssl(4, (void *) dlane, VL);
        __vr nmlane_v  = _vel_vldu_vssl(4, (void *) neg_mlane, VL);
        __vr sx_lane_v = _vel_vldu_vssl(4, (void *)(sx_full + lane_off), VL);

        /* y_lane = dlane * sum_nibx + (-mlane) * sum_x_lane */
        __vr y_lane = _vel_vfmads_vvvvl(
            _vel_vfmuls_vvvl(nmlane_v, sx_lane_v, VL),
            dlane_v, s0, VL);
        y_lane = _vel_vfsums_vvl(y_lane, VL);
        acc_scalar += _vel_lvss_svs(y_lane, 0);
        blk_offset += chunk_nb;
    }
    return acc_scalar;
}

/* 4-ROW tile, pre-decoded header. Processes 4 rows in parallel
 * sharing one x_perm load per nibble position. Inner-loop x_loads
 * drop from 16 (×row) to 16 (shared across 4 rows) -- 4× fewer
 * x-load issues per row. With pre-decoded headers the per-row stack
 * cost is just dlane[VL]+neg_mlane[VL] = 2 KB/row, so 8 KB/tile --
 * easily fits in any thread's stack (previously the 16-byte packed
 * header + scalar setup approach was already 2 KB/row of stack
 * scratch; this is no worse). */
void q4k_full_4rows_decoded_extern(const uint8_t * const qs_r[4],
                                    const uint8_t * const hdr_r[4],
                                    const float *x_perm,
                                    const float *sx_full,
                                    float *y_out[4], int nb);

void q4k_full_4rows_decoded_extern(const uint8_t * const qs_r[4],
                                    const uint8_t * const hdr_r[4],
                                    const float *x_perm,
                                    const float *sx_full,
                                    float *y_out[4], int nb) {
    float ay[4] = {0};
    const int n_lanes_total = nb * 16;
    int blk_offset = 0;
    while (blk_offset < nb) {
        int chunk_nb = nb - blk_offset;
        if (chunk_nb > MAX_CHUNK_BLOCKS) chunk_nb = MAX_CHUNK_BLOCKS;
        const int VL = chunk_nb * 16;
        const int lane_off = blk_offset * 16;

        /* Per-row: build dlane/neg_mlane from pre-decoded fp32 hdr. */
        float dlane[4][MAX_CHUNK_VL];
        float neg_mlane[4][MAX_CHUNK_VL];
        for (int r = 0; r < 4; r++) {
            const float *hdr_chunk =
                (const float *)(hdr_r[r] + (size_t) blk_offset * 64);
            for (int b = 0; b < chunk_nb; b++) {
                const float *blk_hdr = hdr_chunk + b * 16;
                for (int s = 0; s < 8; s++) {
                    const float d  =  blk_hdr[s];
                    const float nm = -blk_hdr[8 + s];
                    int lane0 = b * 16 + s * 2;
                    dlane[r][lane0    ]   = d;
                    dlane[r][lane0 + 1]   = d;
                    neg_mlane[r][lane0  ] = nm;
                    neg_mlane[r][lane0+1] = nm;
                }
            }
        }

        /* Per-row qs load. */
        __vr qs0_v = _vel_vld_vssl(8, (void *)(qs_r[0] + blk_offset * 128), VL);
        __vr qs1_v = _vel_vld_vssl(8, (void *)(qs_r[1] + blk_offset * 128), VL);
        __vr qs2_v = _vel_vld_vssl(8, (void *)(qs_r[2] + blk_offset * 128), VL);
        __vr qs3_v = _vel_vld_vssl(8, (void *)(qs_r[3] + blk_offset * 128), VL);
        __vr mask_f = _vel_vbrdl_vsl(0x0FUL, VL);

        /* Per-row sum_nibx accumulator (one shared chain per row -- the
         * x_perm load is the shared resource). */
        __vr sum0 = _vel_vbrds_vsl(0.0f, VL);
        __vr sum1 = _vel_vbrds_vsl(0.0f, VL);
        __vr sum2 = _vel_vbrds_vsl(0.0f, VL);
        __vr sum3 = _vel_vbrds_vsl(0.0f, VL);

        for (int k = 0; k < 16; k++) {
            /* ONE x load shared by 4 rows. */
            __vr xv = _vel_vldu_vssl(4,
                (void *)(x_perm + k * n_lanes_total + lane_off), VL);

            __vr nib0 = _vel_vand_vvvl(_vel_vsrl_vvsl(qs0_v, 4 * k, VL), mask_f, VL);
            __vr nib1 = _vel_vand_vvvl(_vel_vsrl_vvsl(qs1_v, 4 * k, VL), mask_f, VL);
            __vr nib2 = _vel_vand_vvvl(_vel_vsrl_vvsl(qs2_v, 4 * k, VL), mask_f, VL);
            __vr nib3 = _vel_vand_vvvl(_vel_vsrl_vvsl(qs3_v, 4 * k, VL), mask_f, VL);
            __vr nf0  = _vel_vcvtsw_vvl(nib0, VL);
            __vr nf1  = _vel_vcvtsw_vvl(nib1, VL);
            __vr nf2  = _vel_vcvtsw_vvl(nib2, VL);
            __vr nf3  = _vel_vcvtsw_vvl(nib3, VL);
            sum0 = _vel_vfmads_vvvvl(sum0, nf0, xv, VL);
            sum1 = _vel_vfmads_vvvvl(sum1, nf1, xv, VL);
            sum2 = _vel_vfmads_vvvvl(sum2, nf2, xv, VL);
            sum3 = _vel_vfmads_vvvvl(sum3, nf3, xv, VL);
        }

        __vr sx_lane_v = _vel_vldu_vssl(4, (void *)(sx_full + lane_off), VL);
        for (int r = 0; r < 4; r++) {
            __vr dlane_v   = _vel_vldu_vssl(4, (void *) dlane[r], VL);
            __vr nmlane_v  = _vel_vldu_vssl(4, (void *) neg_mlane[r], VL);
            __vr s = (r == 0) ? sum0 : (r == 1) ? sum1 : (r == 2) ? sum2 : sum3;
            __vr y_lane = _vel_vfmads_vvvvl(
                _vel_vfmuls_vvvl(nmlane_v, sx_lane_v, VL), dlane_v, s, VL);
            y_lane = _vel_vfsums_vvl(y_lane, VL);
            ay[r] += _vel_lvss_svs(y_lane, 0);
        }
        blk_offset += chunk_nb;
    }
    *y_out[0] = ay[0]; *y_out[1] = ay[1]; *y_out[2] = ay[2]; *y_out[3] = ay[3];
}

/* 8-ROW tile, pre-decoded header. 8 rows share one x_perm load per
 * nibble position -- 8× fewer x-loads per row vs single-row. Per-tile
 * stack: 8 × (dlane + neg_mlane) = 8 × 2 KB = 16 KB. Safely fits in any
 * thread's stack. */
void q4k_full_8rows_decoded_extern(const uint8_t * const qs_r[8],
                                    const uint8_t * const hdr_r[8],
                                    const float *x_perm,
                                    const float *sx_full,
                                    float *y_out[8], int nb);

void q4k_full_8rows_decoded_extern(const uint8_t * const qs_r[8],
                                    const uint8_t * const hdr_r[8],
                                    const float *x_perm,
                                    const float *sx_full,
                                    float *y_out[8], int nb) {
    float ay[8] = {0};
    const int n_lanes_total = nb * 16;
    int blk_offset = 0;
    while (blk_offset < nb) {
        int chunk_nb = nb - blk_offset;
        if (chunk_nb > MAX_CHUNK_BLOCKS) chunk_nb = MAX_CHUNK_BLOCKS;
        const int VL = chunk_nb * 16;
        const int lane_off = blk_offset * 16;

        float dlane[8][MAX_CHUNK_VL];
        float neg_mlane[8][MAX_CHUNK_VL];
        for (int r = 0; r < 8; r++) {
            const float *hdr_chunk =
                (const float *)(hdr_r[r] + (size_t) blk_offset * 64);
            for (int b = 0; b < chunk_nb; b++) {
                const float *blk_hdr = hdr_chunk + b * 16;
                for (int s = 0; s < 8; s++) {
                    const float d  =  blk_hdr[s];
                    const float nm = -blk_hdr[8 + s];
                    int lane0 = b * 16 + s * 2;
                    dlane[r][lane0    ]   = d;
                    dlane[r][lane0 + 1]   = d;
                    neg_mlane[r][lane0  ] = nm;
                    neg_mlane[r][lane0+1] = nm;
                }
            }
        }

        __vr qs_v[8];
        for (int r = 0; r < 8; r++)
            qs_v[r] = _vel_vld_vssl(8, (void *)(qs_r[r] + blk_offset * 128), VL);
        __vr mask_f = _vel_vbrdl_vsl(0x0FUL, VL);

        __vr sum[8];
        for (int r = 0; r < 8; r++) sum[r] = _vel_vbrds_vsl(0.0f, VL);

        for (int k = 0; k < 16; k++) {
            __vr xv = _vel_vldu_vssl(4,
                (void *)(x_perm + k * n_lanes_total + lane_off), VL);
            for (int r = 0; r < 8; r++) {
                __vr nib = _vel_vand_vvvl(_vel_vsrl_vvsl(qs_v[r], 4 * k, VL),
                                          mask_f, VL);
                __vr nf  = _vel_vcvtsw_vvl(nib, VL);
                sum[r] = _vel_vfmads_vvvvl(sum[r], nf, xv, VL);
            }
        }

        __vr sx_lane_v = _vel_vldu_vssl(4, (void *)(sx_full + lane_off), VL);
        for (int r = 0; r < 8; r++) {
            __vr dlane_v  = _vel_vldu_vssl(4, (void *) dlane[r], VL);
            __vr nmlane_v = _vel_vldu_vssl(4, (void *) neg_mlane[r], VL);
            __vr y_lane = _vel_vfmads_vvvvl(
                _vel_vfmuls_vvvl(nmlane_v, sx_lane_v, VL), dlane_v, sum[r], VL);
            y_lane = _vel_vfsums_vvl(y_lane, VL);
            ay[r] += _vel_lvss_svs(y_lane, 0);
        }
        blk_offset += chunk_nb;
    }
    for (int r = 0; r < 8; r++) *y_out[r] = ay[r];
}

/* PACKED-FP32 variant. Processes 2 nibble positions per FMA via VE's
 * packed-fp32 ops (pvfmad). Halves the inner-loop count from 16 to 8.
 *
 *   For pair p in 0..7, nibble positions 2p and 2p+1:
 *     - lower 32 bits of packed lane holds the value for k = 2p
 *     - upper 32 bits holds the value for k = 2p+1
 *
 *   Same dlane/mlane apply (since 16i + 2p and 16i + 2p+1 are in the same
 *   /32 sub-block bucket). dlane_pk[i] = pack(d, d) per lane.
 *
 * Expects x_pk built by q4k_build_x_pk_extern. */
float q4k_full_row_dot_packed_extern(const uint8_t *qs_row, const uint8_t *hdr_row,
                                      const uint64_t *x_pk, int nb);

float q4k_full_row_dot_packed_extern(const uint8_t *qs_row, const uint8_t *hdr_row,
                                      const uint64_t *x_pk, int nb) {
    float acc_scalar = 0.0f;
    const int n_lanes_total = nb * 16;
    int blk_offset = 0;
    while (blk_offset < nb) {
        int chunk_nb = nb - blk_offset;
        if (chunk_nb > MAX_CHUNK_BLOCKS) chunk_nb = MAX_CHUNK_BLOCKS;
        const int VL = chunk_nb * 16;

        const uint8_t *qs_chunk  = qs_row  + (size_t) blk_offset * 128;
        const uint8_t *hdr_chunk = hdr_row + (size_t) blk_offset * 16;
        const int      lane_off  = blk_offset * 16;

        /* Build packed dlane / neg_mlane: each lane holds (d, d) and (-m, -m). */
        uint64_t dlane_pk[MAX_CHUNK_VL];
        uint64_t neg_mlane_pk[MAX_CHUNK_VL];
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
                const float d  = d_super    * (float) sc;
                const float nm = -(dmin_super * (float) mn);
                uint32_t d_bits, nm_bits;
                memcpy(&d_bits,  &d,  4);
                memcpy(&nm_bits, &nm, 4);
                const uint64_t d_pk  = ((uint64_t) d_bits  << 32) | d_bits;
                const uint64_t nm_pk = ((uint64_t) nm_bits << 32) | nm_bits;
                int lane0 = b * 16 + s * 2;
                dlane_pk[lane0    ] = d_pk;
                dlane_pk[lane0 + 1] = d_pk;
                neg_mlane_pk[lane0    ] = nm_pk;
                neg_mlane_pk[lane0 + 1] = nm_pk;
            }
        }

        __vr qs_v        = _vel_vld_vssl(8, (void *)qs_chunk, VL);
        __vr dlane_pk_v  = _vel_vld_vssl(8, (void *) dlane_pk, VL);
        __vr neg_mlane_pk_v = _vel_vld_vssl(8, (void *) neg_mlane_pk, VL);
        __vr mask_lo_pk  = _vel_vbrdl_vsl(0x0000000F0000000FUL, VL); /* low nibble in each fp32 half */

        /* 4 parallel packed accumulators for ILP. */
        __vr acc0 = _vel_vbrdl_vsl(0, VL);
        __vr acc1 = _vel_vbrdl_vsl(0, VL);
        __vr acc2 = _vel_vbrdl_vsl(0, VL);
        __vr acc3 = _vel_vbrdl_vsl(0, VL);
        __vr * accs[4] = {&acc0, &acc1, &acc2, &acc3};

        /* 8 packed iterations covering 16 nibble positions. */
        for (int p = 0; p < 8; p++) {
            /* Extract nibbles at positions 2p (lower 32) and 2p+1 (upper 32).
             * For each lane: bits [8p..8p+3] go to low fp32 position; bits
             * [8p+4..8p+7] go to high fp32 position. We can compute both
             * by shifting qs_v right by 8p, then ANDing with mask
             * 0x0000000F0000000F (selects bits 0..3 in lower 32 AND bits
             * 0..3 in upper 32 simultaneously). For the upper 32 we want
             * bits 4..7 of the original byte, so we need a different shift
             * for the upper half.
             *
             * Simpler: build the packed nibble register from two unpacked
             * extracts. low = vand(vsrl(qs, 8p), 0xF); high = vand(vsrl(qs,
             * 8p + 4), 0xF); packed = low | (high << 32). */
            __vr nib_lo = _vel_vand_vsvl(0x0FUL,
                              _vel_vsrl_vvsl(qs_v, 8 * p,     VL), VL);
            __vr nib_hi = _vel_vand_vsvl(0x0FUL,
                              _vel_vsrl_vvsl(qs_v, 8 * p + 4, VL), VL);
            __vr nib_pk = _vel_vor_vvvl(nib_lo,
                              _vel_vsll_vvsl(nib_hi, 32, VL), VL);

            /* Packed int32 → packed fp32. */
            __vr nib_f_pk = _vel_pvcvtsw_vvl(nib_pk, VL);

            /* w = neg_m + d * nib (packed FMA -- 2 elements per lane). */
            __vr w_pk = _vel_pvfmad_vvvvl(neg_mlane_pk_v, dlane_pk_v, nib_f_pk, VL);

            /* Load packed x for this pair. */
            __vr xv_pk = _vel_vld_vssl(8,
                (void *)(x_pk + p * n_lanes_total + lane_off), VL);

            /* acc += w * x  (packed). 4-way ILP via 4 accumulators. */
            *accs[p & 3] = _vel_pvfmad_vvvvl(*accs[p & 3], w_pk, xv_pk, VL);
        }

        /* Combine 4 packed accumulators, then reduce upper+lower halves
         * and finally lane-wise sum. */
        acc0 = _vel_pvfadd_vvvl(acc0, acc1, VL);
        acc2 = _vel_pvfadd_vvvl(acc2, acc3, VL);
        acc0 = _vel_pvfadd_vvvl(acc0, acc2, VL);
        /* Sum upper + lower halves of each packed lane into low half. */
        __vr acc_lo32 = _vel_vand_vvvl(acc0,
                            _vel_vbrdl_vsl(0x00000000FFFFFFFFUL, VL), VL);
        __vr acc_hi32 = _vel_vsrl_vvsl(acc0, 32, VL);
        acc_lo32 = _vel_vsll_vvsl(acc_lo32, 32, VL);  /* low → upper position for vfadds */
        __vr acc_sum = _vel_vfadds_vvvl(acc_lo32, _vel_vsll_vvsl(acc_hi32, 32, VL), VL);
        acc_sum = _vel_vfsums_vvl(acc_sum, VL);
        acc_scalar += _vel_lvss_svs(acc_sum, 0);
        blk_offset += chunk_nb;
    }
    return acc_scalar;
}

/* TILE variant: per row, vectorised dequant Q4_K → F32 into a stack
 * buffer (16 KB for K=4096 — fits in L1 per core), then compute the dot
 * with x using a vectorised F32 inner loop ON CACHED DATA.
 *
 * The hypothesis: BF16 matvec on cache-resident weights runs at multiples
 * of HBM bandwidth (LLC is ~3 TB/s aggregate vs HBM 1.2 TB/s). The
 * dequant cost is fixed but the cached matvec is much faster than direct
 * Q4_K-from-HBM. Net: per-row total time may drop below direct
 * Q4_K_FULL's 2.4 ms even with the extra dequant write.
 *
 * Output buffer layout: standard row-major (element index = canonical
 * order from dequant), so a subsequent ve_f32_matvec inner could be
 * called -- but we just inline the dot here for simplicity.
 *
 * Stack usage: K * 4 bytes. For K=17408 (Qwen FFN), that's 70 KB --
 * exceeds default L1 (32 KB) but well within LLC (2 MB per core). */
float q4k_full_row_dot_tile_extern(const uint8_t *qs_row, const uint8_t *hdr_row,
                                    const float *x, int nb);

float q4k_full_row_dot_tile_extern(const uint8_t *qs_row, const uint8_t *hdr_row,
                                    const float *x, int nb) {
    /* Stack alloc one row's worth of F32 dequant. Max nb=68 (K=17408 = Qwen FFN). */
    float row_f32[68 * 256];
    if (nb > 68) return 0.0f;  /* unsupported, shouldn't happen */

    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = qs_row + (size_t) b * 128;     /* canonical qs */
        const uint8_t *hdr = hdr_row + (size_t) b * 16;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw,    hdr,     2);
        memcpy(&dmin_raw, hdr + 2, 2);
        const float d_super    = h2f(d_raw);
        const float dmin_super = h2f(dmin_raw);
        const uint8_t *sc12 = hdr + 4;

        /* Per-lane scale + min (lane=16 elements; 2 lanes per sub-block). */
        float dlane[16], neg_mlane[16];
        for (int s = 0; s < 8; s++) {
            uint8_t sc, mn;
            q4k_sm(s, sc12, &sc, &mn);
            const float d  = d_super    * (float) sc;
            const float nm = -(dmin_super * (float) mn);
            dlane[s*2]     = d;  dlane[s*2 + 1] = d;
            neg_mlane[s*2] = nm; neg_mlane[s*2 + 1] = nm;
        }

        /* Load this block's canonical qs (128 bytes = 16 lanes of 8 bytes). */
        __vr qs_v       = _vel_vld_vssl(8, (void *) blk, 16);
        __vr mask_f     = _vel_vbrdl_vsl(0x0FUL, 16);
        __vr dlane_v    = _vel_vldu_vssl(4, (void *) dlane, 16);
        __vr neg_mlane_v = _vel_vldu_vssl(4, (void *) neg_mlane, 16);

        /* Per nibble position k: extract, convert, scale, store strided.
         * Strided store: lane i contributes element [16i + k] of the row.
         * For block b, output offset = b * 256 + 16i + k → store at
         * stride 64 bytes (16 floats = stride between consecutive lanes
         * in same nibble position) starting at offset 4*(b*256 + k). */
        float *blk_out = row_f32 + (size_t) b * 256;
        for (int k = 0; k < 16; k++) {
            __vr nib   = _vel_vand_vvvl(_vel_vsrl_vvsl(qs_v, 4 * k, 16), mask_f, 16);
            __vr nib_f = _vel_vcvtsw_vvl(nib, 16);
            __vr fp32_out = _vel_vfmads_vvvvl(neg_mlane_v, dlane_v, nib_f, 16);
            /* vstu: store upper 32 bits (the fp32) at stride 64 bytes. */
            _vel_vstu_vssl(fp32_out, 64, (void *)(blk_out + k), 16);
        }
    }

    /* F32 dot product on the cache-resident row buffer. */
    const int K_total = nb * 256;
    __vr acc = _vel_vbrds_vsl(0.0f, 256);
    int kk = 0;
    while (kk + 256 <= K_total) {
        __vr w  = _vel_vldu_vssl(4, (void *)(row_f32 + kk), 256);
        __vr xv = _vel_vldu_vssl(4, (void *)(x       + kk), 256);
        acc = _vel_vfmads_vvvvl(acc, w, xv, 256);
        kk += 256;
    }
    if (kk < K_total) {
        int rem = K_total - kk;
        __vr w  = _vel_vldu_vssl(4, (void *)(row_f32 + kk), rem);
        __vr xv = _vel_vldu_vssl(4, (void *)(x       + kk), rem);
        acc = _vel_vfmads_vvvvl(acc, w, xv, rem);
    }
    acc = _vel_vfsums_vvl(acc, 256);
    return _vel_lvss_svs(acc, 0);
}

/* 4-row unrolled variant. Processes 4 rows in parallel, sharing one x_perm
 * load across all 4 rows for each nibble position. The dlane/mlane builds
 * are still per-row but the inner FMA chains share x loads, which halves
 * the x-load issue rate and exposes 4x more independent FMA chains. */
void q4k_full_4rows_xperm_extern(const uint8_t *qs_r0, const uint8_t *qs_r1,
                                  const uint8_t *qs_r2, const uint8_t *qs_r3,
                                  const uint8_t *hdr_r0, const uint8_t *hdr_r1,
                                  const uint8_t *hdr_r2, const uint8_t *hdr_r3,
                                  const float *x_perm,
                                  float *y0_out, float *y1_out,
                                  float *y2_out, float *y3_out, int nb);

void q4k_full_4rows_xperm_extern(const uint8_t *qs_r0, const uint8_t *qs_r1,
                                  const uint8_t *qs_r2, const uint8_t *qs_r3,
                                  const uint8_t *hdr_r0, const uint8_t *hdr_r1,
                                  const uint8_t *hdr_r2, const uint8_t *hdr_r3,
                                  const float *x_perm,
                                  float *y0_out, float *y1_out,
                                  float *y2_out, float *y3_out, int nb) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const int n_lanes_total = nb * 16;
    int blk_offset = 0;
    while (blk_offset < nb) {
        int chunk_nb = nb - blk_offset;
        if (chunk_nb > MAX_CHUNK_BLOCKS) chunk_nb = MAX_CHUNK_BLOCKS;
        const int VL = chunk_nb * 16;

        const uint8_t *qs_chunks[4] = {
            qs_r0 + (size_t) blk_offset * 128,
            qs_r1 + (size_t) blk_offset * 128,
            qs_r2 + (size_t) blk_offset * 128,
            qs_r3 + (size_t) blk_offset * 128,
        };
        const uint8_t *hdr_chunks[4] = {
            hdr_r0 + (size_t) blk_offset * 16,
            hdr_r1 + (size_t) blk_offset * 16,
            hdr_r2 + (size_t) blk_offset * 16,
            hdr_r3 + (size_t) blk_offset * 16,
        };
        const int lane_off = blk_offset * 16;

        float dlane[4][MAX_CHUNK_VL];
        float neg_mlane[4][MAX_CHUNK_VL];
        for (int r = 0; r < 4; r++) {
            for (int b = 0; b < chunk_nb; b++) {
                const uint8_t *hdr = hdr_chunks[r] + (size_t) b * 16;
                uint16_t d_raw, dmin_raw;
                memcpy(&d_raw,    hdr,     2);
                memcpy(&dmin_raw, hdr + 2, 2);
                const float d_super    = h2f(d_raw);
                const float dmin_super = h2f(dmin_raw);
                const uint8_t *sc12 = hdr + 4;
                for (int s = 0; s < 8; s++) {
                    uint8_t sc, mn;
                    q4k_sm(s, sc12, &sc, &mn);
                    const float d  = d_super    * (float) sc;
                    const float nm = -(dmin_super * (float) mn);
                    int lane0 = b * 16 + s * 2;
                    dlane[r][lane0    ] = d;
                    dlane[r][lane0 + 1] = d;
                    neg_mlane[r][lane0    ] = nm;
                    neg_mlane[r][lane0 + 1] = nm;
                }
            }
        }

        __vr qs0_v = _vel_vld_vssl(8, (void *) qs_chunks[0], VL);
        __vr qs1_v = _vel_vld_vssl(8, (void *) qs_chunks[1], VL);
        __vr qs2_v = _vel_vld_vssl(8, (void *) qs_chunks[2], VL);
        __vr qs3_v = _vel_vld_vssl(8, (void *) qs_chunks[3], VL);
        __vr mask_f = _vel_vbrdl_vsl(0x0FUL, VL);
        __vr d0_v   = _vel_vldu_vssl(4, (void *) dlane[0], VL);
        __vr d1_v   = _vel_vldu_vssl(4, (void *) dlane[1], VL);
        __vr d2_v   = _vel_vldu_vssl(4, (void *) dlane[2], VL);
        __vr d3_v   = _vel_vldu_vssl(4, (void *) dlane[3], VL);
        __vr nm0_v  = _vel_vldu_vssl(4, (void *) neg_mlane[0], VL);
        __vr nm1_v  = _vel_vldu_vssl(4, (void *) neg_mlane[1], VL);
        __vr nm2_v  = _vel_vldu_vssl(4, (void *) neg_mlane[2], VL);
        __vr nm3_v  = _vel_vldu_vssl(4, (void *) neg_mlane[3], VL);
        __vr acc0 = _vel_vbrds_vsl(0.0f, VL);
        __vr acc1 = _vel_vbrds_vsl(0.0f, VL);
        __vr acc2 = _vel_vbrds_vsl(0.0f, VL);
        __vr acc3 = _vel_vbrds_vsl(0.0f, VL);

        for (int k = 0; k < 16; k++) {
            /* SHARED x load (4 rows share). */
            __vr xv = _vel_vldu_vssl(4,
                (void *)(x_perm + k * n_lanes_total + lane_off), VL);

            /* 4 independent extract + FMA chains. */
            __vr nib0 = _vel_vand_vvvl(_vel_vsrl_vvsl(qs0_v, 4 * k, VL), mask_f, VL);
            __vr nib1 = _vel_vand_vvvl(_vel_vsrl_vvsl(qs1_v, 4 * k, VL), mask_f, VL);
            __vr nib2 = _vel_vand_vvvl(_vel_vsrl_vvsl(qs2_v, 4 * k, VL), mask_f, VL);
            __vr nib3 = _vel_vand_vvvl(_vel_vsrl_vvsl(qs3_v, 4 * k, VL), mask_f, VL);
            __vr nf0  = _vel_vcvtsw_vvl(nib0, VL);
            __vr nf1  = _vel_vcvtsw_vvl(nib1, VL);
            __vr nf2  = _vel_vcvtsw_vvl(nib2, VL);
            __vr nf3  = _vel_vcvtsw_vvl(nib3, VL);
            __vr w0   = _vel_vfmads_vvvvl(nm0_v, d0_v, nf0, VL);
            __vr w1   = _vel_vfmads_vvvvl(nm1_v, d1_v, nf1, VL);
            __vr w2   = _vel_vfmads_vvvvl(nm2_v, d2_v, nf2, VL);
            __vr w3   = _vel_vfmads_vvvvl(nm3_v, d3_v, nf3, VL);
            acc0 = _vel_vfmads_vvvvl(acc0, w0, xv, VL);
            acc1 = _vel_vfmads_vvvvl(acc1, w1, xv, VL);
            acc2 = _vel_vfmads_vvvvl(acc2, w2, xv, VL);
            acc3 = _vel_vfmads_vvvvl(acc3, w3, xv, VL);
        }

        acc0 = _vel_vfsums_vvl(acc0, VL); a0 += _vel_lvss_svs(acc0, 0);
        acc1 = _vel_vfsums_vvl(acc1, VL); a1 += _vel_lvss_svs(acc1, 0);
        acc2 = _vel_vfsums_vvl(acc2, VL); a2 += _vel_lvss_svs(acc2, 0);
        acc3 = _vel_vfsums_vvl(acc3, VL); a3 += _vel_lvss_svs(acc3, 0);
        blk_offset += chunk_nb;
    }
    *y0_out = a0; *y1_out = a1; *y2_out = a2; *y3_out = a3;
}

/* 8-row variant: same idea, 2x more shared-x sharing. */
void q4k_full_8rows_xperm_extern(const uint8_t * const qs_r[8],
                                  const uint8_t * const hdr_r[8],
                                  const float *x_perm,
                                  float *y_out[8], int nb);

void q4k_full_8rows_xperm_extern(const uint8_t * const qs_r[8],
                                  const uint8_t * const hdr_r[8],
                                  const float *x_perm,
                                  float *y_out[8], int nb) {
    float a[8] = {0};
    const int n_lanes_total = nb * 16;
    int blk_offset = 0;
    while (blk_offset < nb) {
        int chunk_nb = nb - blk_offset;
        if (chunk_nb > MAX_CHUNK_BLOCKS) chunk_nb = MAX_CHUNK_BLOCKS;
        const int VL = chunk_nb * 16;
        const int lane_off = blk_offset * 16;

        float dlane[8][MAX_CHUNK_VL];
        float neg_mlane[8][MAX_CHUNK_VL];
        for (int r = 0; r < 8; r++) {
            const uint8_t *hdr_chunk = hdr_r[r] + (size_t) blk_offset * 16;
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
                    const float d  = d_super    * (float) sc;
                    const float nm = -(dmin_super * (float) mn);
                    int lane0 = b * 16 + s * 2;
                    dlane[r][lane0]     = d;
                    dlane[r][lane0 + 1] = d;
                    neg_mlane[r][lane0]     = nm;
                    neg_mlane[r][lane0 + 1] = nm;
                }
            }
        }

        __vr qs_v[8];
        __vr d_v[8], nm_v[8], acc_v[8];
        for (int r = 0; r < 8; r++) {
            qs_v[r] = _vel_vld_vssl(8, (void *)(qs_r[r] + blk_offset * 128), VL);
            d_v[r]  = _vel_vldu_vssl(4, (void *) dlane[r], VL);
            nm_v[r] = _vel_vldu_vssl(4, (void *) neg_mlane[r], VL);
            acc_v[r] = _vel_vbrds_vsl(0.0f, VL);
        }
        __vr mask_f = _vel_vbrdl_vsl(0x0FUL, VL);

        for (int k = 0; k < 16; k++) {
            /* Single shared x load. */
            __vr xv = _vel_vldu_vssl(4,
                (void *)(x_perm + k * n_lanes_total + lane_off), VL);
            /* 8 independent extract + FMA chains. */
            for (int r = 0; r < 8; r++) {
                __vr nib = _vel_vand_vvvl(_vel_vsrl_vvsl(qs_v[r], 4 * k, VL),
                                          mask_f, VL);
                __vr nib_f = _vel_vcvtsw_vvl(nib, VL);
                __vr w = _vel_vfmads_vvvvl(nm_v[r], d_v[r], nib_f, VL);
                acc_v[r] = _vel_vfmads_vvvvl(acc_v[r], w, xv, VL);
            }
        }

        for (int r = 0; r < 8; r++) {
            __vr s = _vel_vfsums_vvl(acc_v[r], VL);
            a[r] += _vel_lvss_svs(s, 0);
        }
        blk_offset += chunk_nb;
    }
    for (int r = 0; r < 8; r++) *y_out[r] = a[r];
}

/* Entry point lives in q4k_full_dispatch.c (compiled with NCC for OMP).
 * It calls q4k_full_row_dot_extern per row in parallel. */
