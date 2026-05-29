#ifndef GGML_VE_HBM_CACHE_HPP
#define GGML_VE_HBM_CACHE_HPP

// HBM weight cache. Upload weights to VE device memory once, reuse them at
// 1.2 TB/s. Keyed by host pointer (primary) and tensor name (for the graph
// compiler, which can identify the same tensor across different host
// addresses after view/reshape). The dual-key lookup is exactly the fix for
// the December-2025 cache-miss bug (kb/performance/current-investigation.md).

#include "common.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace ggml_ve {

// ---- Q4_K header decode helpers (used by get_or_upload_q4k_canon) ----
//
// q4k_h2f       : fp16 → fp32 (manual, no compiler builtin dependency)
// q4k_unpack_sm : extract the 6-bit (scale, min) pair for sub-block idx (0..7)
//                 from the 12-byte scales array; same packing rule as
//                 ggml's CPU Q4_K decode.
static inline float q4k_h2f(uint16_t h) {
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
    float f; std::memcpy(&f, &u, 4); return f;
}

static inline void q4k_unpack_sm(int idx, const uint8_t * sc12,
                                  uint8_t * scale, uint8_t * mn) {
    if (idx < 4) {
        *scale = sc12[idx]     & 0x3F;
        *mn    = sc12[idx + 4] & 0x3F;
    } else {
        *scale = (sc12[idx + 4] & 0x0F) | ((sc12[idx - 4] >> 6) << 4);
        *mn    = (sc12[idx + 4] >>   4) | ((sc12[idx]     >> 6) << 4);
    }
}

struct hbm_cache_entry {
    VEDAdeviceptr        vptr      = 0;
    size_t               size      = 0;
    const void *         host_data = nullptr;
    std::string          name;
    ggml_ve_hbm_format   format    = GGML_VE_HBM_FP32;
};

class hbm_weight_cache {
public:
    void set_context(VEDAcontext ctx) { ctx_ = ctx; }

    // Standard row-major lookup-or-upload by host pointer. Skips column-major
    // entries (they have the same key but a different layout).
    VEDAdeviceptr get_or_upload(const void * host_data, size_t size) {
        for (auto & e : entries_) {
            if (e.host_data == host_data && e.size == size &&
                e.format != GGML_VE_HBM_FP32_COLMAJOR &&
                e.format != GGML_VE_HBM_BF16_COLMAJOR) {
                hits_++;
                return e.vptr;
            }
        }
        VEDAdeviceptr v = upload(host_data, size);
        if (v == 0) return 0;
        record(v, size, host_data, /*name=*/nullptr, GGML_VE_HBM_FP32);
        return v;
    }

    // Pointer-or-name lookup. Used by the graph compiler where tensor identity
    // is more reliably tracked by name than by pointer.
    VEDAdeviceptr get_or_upload_by_name(const char * tensor_name,
                                        const void * host_data,
                                        size_t size) {
        if (tensor_name != nullptr) {
            for (auto & e : entries_) {
                if (e.name == tensor_name && e.size == size) {
                    hits_++;
                    return e.vptr;
                }
            }
        }
        for (auto & e : entries_) {
            if (e.host_data == host_data && e.size == size) {
                if (e.name.empty() && tensor_name) e.name = tensor_name;
                hits_++;
                return e.vptr;
            }
        }
        VEDAdeviceptr v = upload(host_data, size);
        if (v == 0) return 0;
        record(v, size, host_data, tensor_name, GGML_VE_HBM_FP32);
        return v;
    }

    // Q4_K canonical-split upload WITH PRE-DECODED HEADERS.
    //
    // Pre-decoding moves the per-block scale work (h2f conversion of d/dmin,
    // 6-bit unpack of 8 sub-block scales/mins, fp32 multiplication for d_sub
    // and m_sub) from per-MUL_MAT VE-side scalar code to a ONE-TIME CPU pass
    // at upload. Per VE_PROGINF on Q4_K_FULL at 14336x4096: the prior packed
    // header forced ~1ms of SPU scalar work per matvec call (h2f + q4k_sm +
    // dlane/mlane build) -- pre-decoded headers eliminate all of it. The VE
    // kernel just vector-loads d_sub[8] and m_sub[8] directly.
    //
    // Layout:
    //   qs_vptr  : M × nb × 128 bytes  (canonical nibbles, unchanged)
    //   hdr_vptr : M × nb × 64 bytes   (8 fp32 d_sub + 8 fp32 m_sub per block)
    //
    // 64-byte decoded header vs the original 16-byte packed header: +48 bytes
    // per block = ~4× header expansion. For Qwen3.6-27B Q4_K_M (110M blocks):
    // +5.3 GB total cache, taking weights from 16 GB to ~21 GB (still fits
    // in the 48 GB HBM with KV cache + activations).
    bool get_or_upload_q4k_canon(const char * tensor_name,
                                  const void * src_blocks,
                                  uint64_t M, uint64_t K,
                                  VEDAdeviceptr * qs_vptr,
                                  VEDAdeviceptr * hdr_vptr) {
        const int    nb        = (int) K / 256;
        const size_t qs_total  = (size_t) M * nb * 128;
        const size_t hdr_total = (size_t) M * nb * 64;  // PRE-DECODED 64-byte hdr

        // Lookup by name+suffix.
        std::string qs_key  = std::string(tensor_name ? tensor_name : "?") + "/q4k_qs";
        std::string hdr_key = std::string(tensor_name ? tensor_name : "?") + "/q4k_dhdr64";
        VEDAdeviceptr qs_v = 0, hdr_v = 0;
        for (auto & e : entries_) {
            if (e.name == qs_key  && e.size == qs_total)  qs_v  = e.vptr;
            if (e.name == hdr_key && e.size == hdr_total) hdr_v = e.vptr;
        }
        if (qs_v && hdr_v) {
            hits_ += 2;
            *qs_vptr = qs_v; *hdr_vptr = hdr_v;
            return true;
        }

        // Need to build the split arrays. Allocate host-side staging.
        uint8_t * qs_host  = (uint8_t *) std::aligned_alloc(64, qs_total);
        uint8_t * hdr_host = (uint8_t *) std::aligned_alloc(64, hdr_total);
        if (!qs_host || !hdr_host) {
            if (qs_host)  std::free(qs_host);
            if (hdr_host) std::free(hdr_host);
            return false;
        }
        const uint8_t * S = (const uint8_t *) src_blocks;
        for (uint64_t m = 0; m < M; m++) {
            for (int b = 0; b < nb; b++) {
                const uint8_t * blk = S + (m * nb + b) * 144;
                uint8_t * qd  = qs_host  + (m * nb + b) * 128;
                float   * hd  = (float *)(hdr_host + (m * nb + b) * 64);

                // ---- Decode header into 8 fp32 d_sub + 8 fp32 m_sub ----
                uint16_t d_raw, dmin_raw;
                std::memcpy(&d_raw,    blk + 0, 2);
                std::memcpy(&dmin_raw, blk + 2, 2);
                const float d_super    = q4k_h2f(d_raw);
                const float dmin_super = q4k_h2f(dmin_raw);
                const uint8_t * sc12 = blk + 4;
                for (int s = 0; s < 8; s++) {
                    uint8_t sc, mn;
                    q4k_unpack_sm(s, sc12, &sc, &mn);
                    hd[s    ] = d_super    * (float) sc;   // d_sub[s]
                    hd[8 + s] = dmin_super * (float) mn;   // m_sub[s]
                }

                // ---- Canonical-pack qs (byte k = elem[2k+1]<<4 | elem[2k]) ----
                uint8_t elem[256];
                const uint8_t * src_qs = blk + 16;
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

        if (qs_v == 0) {
            qs_v = upload(qs_host, qs_total);
            // Record host_data as src_blocks (the original Q4_K weights)
            // so that pointer-based lookups still work; name is primary key.
            if (qs_v) record(qs_v, qs_total, src_blocks, qs_key.c_str(),
                             GGML_VE_HBM_Q4K_CANON_QS);
        }
        if (hdr_v == 0) {
            hdr_v = upload(hdr_host, hdr_total);
            if (hdr_v) record(hdr_v, hdr_total, src_blocks, hdr_key.c_str(),
                              GGML_VE_HBM_Q4K_CANON_HDR);
        }
        // Temp buffers are no longer needed -- upload() did the HtoD copy.
        std::free(qs_host);
        std::free(hdr_host);
        if (!qs_v || !hdr_v) return false;
        *qs_vptr = qs_v; *hdr_vptr = hdr_v;
        return true;
    }

    // ---- VEBP upload: repack on-disk block_vebp -> interleaved planes ----
    //
    // On-disk block_vebp (row-major [M][nblk][68]): per 256-elem block,
    //   d[2] fp16 (2 group-128 scales), nz[32] (256-bit), sign[32] (256-bit).
    // Interleaved (lane = row, block of 256 rows) for the VE kernel:
    //   Ws_il/Wn_il: [rowblk][word=k/64][256 rows] uint64
    //   wscale_il  : [rowblk][group=k/128][256 rows] f32
    // Three HBM buffers, cached by name. ~2 GB for an 8B model.
    // Hit-only lookup (no upload). Lets the caller skip an expensive
    // device->host bounce of the weight when it's already cached.
    bool vebp_lookup(const char * tensor_name, uint64_t M, uint64_t K,
                     VEDAdeviceptr * ws_vptr, VEDAdeviceptr * wn_vptr,
                     VEDAdeviceptr * wsc_vptr) {
        const long   wpr  = (long) K / 64;
        const long   ng   = (long) K / 128;
        const long   Mpad = ((long) M + 255) / 256 * 256;
        const size_t ws_total  = (size_t) Mpad * wpr * sizeof(uint64_t);
        const size_t wsc_total = (size_t) Mpad * ng  * sizeof(float);
        std::string ws_key  = std::string(tensor_name ? tensor_name : "?") + "/vebp_ws";
        std::string wn_key  = std::string(tensor_name ? tensor_name : "?") + "/vebp_wn";
        std::string wsc_key = std::string(tensor_name ? tensor_name : "?") + "/vebp_wsc";
        VEDAdeviceptr ws = 0, wn = 0, wsc = 0;
        for (auto & e : entries_) {
            if (e.name == ws_key  && e.size == ws_total)  ws  = e.vptr;
            if (e.name == wn_key  && e.size == ws_total)  wn  = e.vptr;
            if (e.name == wsc_key && e.size == wsc_total) wsc = e.vptr;
        }
        if (ws && wn && wsc) { hits_ += 3; *ws_vptr=ws; *wn_vptr=wn; *wsc_vptr=wsc; return true; }
        return false;
    }

    bool get_or_upload_vebp(const char * tensor_name, const void * src_blocks,
                            uint64_t M, uint64_t K,
                            VEDAdeviceptr * ws_vptr, VEDAdeviceptr * wn_vptr,
                            VEDAdeviceptr * wsc_vptr) {
        const int    nblk = (int) K / 256;
        const long   wpr  = (long) K / 64;
        const long   ng   = (long) K / 128;
        // Pad rows up to a multiple of 256 (the kernel's rowblock width).
        // Pad rows are zero (calloc) -> all-zero ternary -> y=0, harmless;
        // they only exist so the last block's 256-lane loads stay in-bounds.
        const long   Mpad = ((long) M + 255) / 256 * 256;
        const size_t ws_total  = (size_t) Mpad * wpr * sizeof(uint64_t);
        const size_t wsc_total = (size_t) Mpad * ng  * sizeof(float);

        std::string ws_key  = std::string(tensor_name ? tensor_name : "?") + "/vebp_ws";
        std::string wn_key  = std::string(tensor_name ? tensor_name : "?") + "/vebp_wn";
        std::string wsc_key = std::string(tensor_name ? tensor_name : "?") + "/vebp_wsc";
        VEDAdeviceptr ws = 0, wn = 0, wsc = 0;
        for (auto & e : entries_) {
            if (e.name == ws_key  && e.size == ws_total)  ws  = e.vptr;
            if (e.name == wn_key  && e.size == ws_total)  wn  = e.vptr;
            if (e.name == wsc_key && e.size == wsc_total) wsc = e.vptr;
        }
        if (ws && wn && wsc) { hits_ += 3; *ws_vptr=ws; *wn_vptr=wn; *wsc_vptr=wsc; return true; }

        // calloc so the padded rows (M..Mpad-1) are zero.
        uint64_t * ws_h  = (uint64_t *) std::calloc(ws_total,  1);
        uint64_t * wn_h  = (uint64_t *) std::calloc(ws_total,  1);
        float    * wsc_h = (float *)    std::calloc(wsc_total, 1);
        if (!ws_h || !wn_h || !wsc_h) { std::free(ws_h); std::free(wn_h); std::free(wsc_h); return false; }

        const uint8_t * S = (const uint8_t *) src_blocks;  // M*nblk*68, row-major
        for (long m = 0; m < (long) M; m++) {
            const long rb = m / 256, r = m % 256;
            const uint8_t * row = S + (size_t) m * nblk * 68;
            for (int b = 0; b < nblk; b++) {
                const uint8_t * blk = row + (size_t) b * 68;
                for (int gg = 0; gg < 2; gg++) {
                    uint16_t h; std::memcpy(&h, blk + gg*2, 2);
                    const long g = 2*b + gg;
                    wsc_h[(rb*ng + g)*256 + r] = q4k_h2f(h);
                }
                for (int ww = 0; ww < 4; ww++) {
                    uint64_t nzw, sgw;
                    std::memcpy(&nzw, blk + 4  + ww*8, 8);
                    std::memcpy(&sgw, blk + 36 + ww*8, 8);
                    const long w = 4*b + ww;
                    wn_h[(rb*wpr + w)*256 + r] = nzw;
                    ws_h[(rb*wpr + w)*256 + r] = sgw;
                }
            }
        }

        ws  = upload(ws_h,  ws_total);  if (ws)  record(ws,  ws_total,  src_blocks, ws_key.c_str(),  GGML_VE_HBM_VEBP_WS);
        wn  = upload(wn_h,  ws_total);  if (wn)  record(wn,  ws_total,  src_blocks, wn_key.c_str(),  GGML_VE_HBM_VEBP_WN);
        wsc = upload(wsc_h, wsc_total); if (wsc) record(wsc, wsc_total, src_blocks, wsc_key.c_str(), GGML_VE_HBM_VEBP_WSCALE);
        std::free(ws_h); std::free(wn_h); std::free(wsc_h);
        if (!ws || !wn || !wsc) return false;
        *ws_vptr = ws; *wn_vptr = wn; *wsc_vptr = wsc;
        return true;
    }

    // ---- Decoded-header ONLY upload (for direct-dispatch Q4_K kernel) ----
    //
    // Same per-block decoded header layout as get_or_upload_q4k_canon (8 fp32
    // d_sub + 8 fp32 m_sub = 64 B/blk) but with NO canon-qs side table. The
    // direct kernel reads qs from raw HBM weights and only needs the decoded
    // scales here. Costs +5.3 GB on 27B Q4_K_M (vs +21 GB for full canon).
    bool get_or_upload_q4k_hdr_decoded(const char * tensor_name,
                                        const void * src_blocks,
                                        uint64_t M, uint64_t K,
                                        VEDAdeviceptr * hdr_vptr) {
        const int    nb        = (int) K / 256;
        const size_t hdr_total = (size_t) M * nb * 64;  // 8 d_sub + 8 m_sub fp32

        std::string hdr_key = std::string(tensor_name ? tensor_name : "?") + "/q4k_dhdr64_std";
        for (auto & e : entries_) {
            if (e.name == hdr_key && e.size == hdr_total) {
                ++hits_;
                *hdr_vptr = e.vptr;
                return true;
            }
        }

        uint8_t * hdr_host = (uint8_t *) std::aligned_alloc(64, hdr_total);
        if (!hdr_host) return false;

        const uint8_t * S = (const uint8_t *) src_blocks;
        for (uint64_t m = 0; m < M; m++) {
            for (int b = 0; b < nb; b++) {
                const uint8_t * blk = S + (m * nb + b) * 144;
                float         * hd  = (float *)(hdr_host + (m * nb + b) * 64);
                uint16_t d_raw, dmin_raw;
                std::memcpy(&d_raw,    blk + 0, 2);
                std::memcpy(&dmin_raw, blk + 2, 2);
                const float d_super    = q4k_h2f(d_raw);
                const float dmin_super = q4k_h2f(dmin_raw);
                const uint8_t * sc12 = blk + 4;
                for (int s = 0; s < 8; s++) {
                    uint8_t sc, mn;
                    q4k_unpack_sm(s, sc12, &sc, &mn);
                    hd[s    ] = d_super    * (float) sc;
                    hd[8 + s] = dmin_super * (float) mn;
                }
            }
        }

        VEDAdeviceptr hdr_v = upload(hdr_host, hdr_total);
        if (hdr_v) record(hdr_v, hdr_total, src_blocks, hdr_key.c_str(),
                          GGML_VE_HBM_Q4K_CANON_HDR);
        std::free(hdr_host);
        if (!hdr_v) return false;
        *hdr_vptr = hdr_v;
        return true;
    }

    void clear() {
        for (auto & e : entries_) {
            if (e.vptr) vedaMemFreeAsync(e.vptr, 0);
        }
        entries_.clear();
        total_allocated_ = 0;
    }

    void stats(size_t * allocated, int64_t * hits, int64_t * misses) const {
        if (allocated) *allocated = total_allocated_;
        if (hits)      *hits      = hits_;
        if (misses)    *misses    = misses_;
    }

private:
    VEDAdeviceptr upload(const void * host_data, size_t size) {
        if (ctx_ == nullptr) {
            fprintf(stderr, "hbm_weight_cache: no VEDA context set\n");
            return 0;
        }
        VEDAResult_push();

        VEDAdeviceptr vptr = 0;
        VEDAresult err = vedaMemAllocAsync(&vptr, size, 0);
        if (err != VEDA_SUCCESS) {
            fprintf(stderr, "hbm_weight_cache: vedaMemAllocAsync (size=%zu) failed: %s\n",
                    size, ggml_ve_err_str(err));
            VEDAResult_pop();
            return 0;
        }
        err = vedaMemcpyHtoD(vptr, host_data, size);
        if (err != VEDA_SUCCESS) {
            fprintf(stderr, "hbm_weight_cache: vedaMemcpyHtoD (size=%zu) failed: %s\n",
                    size, ggml_ve_err_str(err));
            vedaMemFreeAsync(vptr, 0);
            VEDAResult_pop();
            return 0;
        }
        vedaCtxSynchronize();
        VEDAResult_pop();
        return vptr;
    }

    void record(VEDAdeviceptr vptr, size_t size, const void * host_data,
                const char * name, ggml_ve_hbm_format fmt) {
        hbm_cache_entry e;
        e.vptr      = vptr;
        e.size      = size;
        e.host_data = host_data;
        if (name) e.name = name;
        e.format    = fmt;
        entries_.push_back(std::move(e));
        total_allocated_ += size;
        misses_++;
    }

    // Push/pop helpers — sync before realloc to avoid use-after-free of
    // in-flight buffers (matches the old implementation's safety net).
    void VEDAResult_push() {
        VEDAresult err = vedaCtxPushCurrent(ctx_);
        if (err != VEDA_SUCCESS) {
            fprintf(stderr, "hbm_weight_cache: vedaCtxPushCurrent failed: %s\n",
                    ggml_ve_err_str(err));
        }
        vedaCtxSynchronize();
    }
    void VEDAResult_pop() {
        VEDAcontext prev = nullptr;
        vedaCtxPopCurrent(&prev);
    }

    std::vector<hbm_cache_entry> entries_;
    size_t      total_allocated_ = 0;
    int64_t     hits_            = 0;
    int64_t     misses_          = 0;
    VEDAcontext ctx_             = nullptr;
};

} // namespace ggml_ve

#endif // GGML_VE_HBM_CACHE_HPP
