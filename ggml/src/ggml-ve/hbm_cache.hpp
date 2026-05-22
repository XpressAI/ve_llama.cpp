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
