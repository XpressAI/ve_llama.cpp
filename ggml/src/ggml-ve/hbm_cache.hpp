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

    // Q4_K canonical-split upload. Given a tensor of standard-layout Q4_K
    // blocks (144 B / 256 elem, M rows × nb blocks/row), produce two HBM
    // regions:
    //   qs_vptr  : M × nb × 128 bytes  -- nibble bytes in canonical order
    //              (byte k = (element 2k+1) << 4 | element 2k)
    //   hdr_vptr : M × nb × 16 bytes   -- d, dmin, scales (verbatim copy)
    // Same total bytes as standard Q4_K (no expansion). Keyed by name+suffix
    // so each is cacheable across MUL_MAT calls touching the same weight.
    // Returns true if both uploads succeed; sets out params.
    bool get_or_upload_q4k_canon(const char * tensor_name,
                                  const void * src_blocks,
                                  uint64_t M, uint64_t K,
                                  VEDAdeviceptr * qs_vptr,
                                  VEDAdeviceptr * hdr_vptr) {
        const int    nb        = (int) K / 256;
        const size_t qs_total  = (size_t) M * nb * 128;
        const size_t hdr_total = (size_t) M * nb * 16;

        // Lookup by name+suffix.
        std::string qs_key  = std::string(tensor_name ? tensor_name : "?") + "/q4k_qs";
        std::string hdr_key = std::string(tensor_name ? tensor_name : "?") + "/q4k_hdr";
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
                uint8_t * hd  = hdr_host + (m * nb + b) * 16;
                std::memcpy(hd, blk, 16);
                // Reconstruct element[256] from standard layout, repack canonical.
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
