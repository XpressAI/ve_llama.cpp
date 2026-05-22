#ifndef GGML_VE_BACKEND_CTX_HPP
#define GGML_VE_BACKEND_CTX_HPP

// Per-backend-instance state. One per `ggml_backend_t`. Holds the deferred-sync
// queue plus pointers to the per-device kernel handles and weight cache.
//
// The deferred-sync queue is the key host-side performance optimisation: each
// kernel that wants its output copied back enqueues a pending_output instead
// of calling vedaCtxSynchronize immediately, and we flush once per graph
// compute. See kb/bugs-lessons/deferred-sync.md (3.0x win on Llama-3-8B).

#include "common.hpp"
#include "device.hpp"
#include "hbm_cache.hpp"
#include "hmem_pool.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <ctime>
#include <vector>

namespace ggml_ve {

// True iff `t` lives in a VE_HBM buffer — meaning tensor->data is a
// VEDAdeviceptr cast to void*, not a host pointer.
inline bool tensor_is_hbm(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr) return false;
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    const char * name = buft ? ggml_backend_buft_name(buft) : nullptr;
    if (name == nullptr) return false;
    return std::strncmp(name, "VE", 2) == 0 && std::strstr(name, "_HBM") != nullptr;
}

inline VEDAdeviceptr tensor_hbm_ptr(const ggml_tensor * t) {
    return reinterpret_cast<VEDAdeviceptr>(reinterpret_cast<uintptr_t>(t->data));
}

struct pending_output {
    VEDAhmemptr hmem     = 0;
    void *      host_dst = nullptr;
    size_t      size     = 0;
};

class backend_context {
public:
    explicit backend_context(device * dev) : dev_(dev) {
        if (dev_) cache_.set_context(dev_->context);
    }

    ~backend_context() {
        pool_.clear();
        // Don't call cache_.clear() here — the cached weights may still be
        // referenced by other backend instances on the same device.
    }

    device *           dev()      { return dev_; }
    hmem_pool &        pool()     { return pool_; }
    hbm_weight_cache & cache()    { return cache_; }
    ggml_cgraph *      cgraph()   { return cgraph_; }
    void               set_cgraph(ggml_cgraph * g) { cgraph_ = g; }

    VEDAfunction fn(kernel_id id) const {
        return dev_ ? dev_->fn(id) : 0;
    }

    // ---- Deferred-sync API ----
    void enqueue_output(VEDAhmemptr hmem, void * dst, size_t size) {
        pending_outputs_.push_back({hmem, dst, size});
        needs_sync_ = true;
    }
    void enqueue_input(VEDAhmemptr hmem) {
        pending_inputs_.push_back(hmem);
        needs_sync_ = true;
    }
    void enqueue_hbm_free(VEDAdeviceptr hbm) {
        pending_hbm_frees_.push_back(hbm);
        needs_sync_ = true;
    }
    void mark_sync_pending() { needs_sync_ = true; }
    bool has_pending() const { return needs_sync_; }

    // Single sync for the whole batch of queued ops. After the sync, copy
    // every queued HMEM output back to host, recycle HMEM buffers, free
    // pending HBM allocs.
    void flush(const char * caller = "unknown") {
        if (!needs_sync_) return;

        if (std::getenv("GGML_VE_DEBUG_SYNC")) {
            fprintf(stderr, "[SYNC-DEBUG] flush by %s (out=%zu in=%zu hbm_free=%zu)\n",
                    caller, pending_outputs_.size(),
                    pending_inputs_.size(), pending_hbm_frees_.size());
        }

        struct timespec t0{}, t1{}, t2{};
        clock_gettime(CLOCK_MONOTONIC, &t0);

        sync_count_++;
        vedaCtxSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &t1);

        for (auto & o : pending_outputs_) {
            vedaHMemcpy(o.host_dst, reinterpret_cast<void *>(o.hmem), o.size);
            pool_.release(o.hmem);
        }
        for (auto & h : pending_inputs_) {
            pool_.release(h, nullptr, 0, /*clear_cache=*/true);
        }
        for (auto & v : pending_hbm_frees_) {
            vedaMemFreeAsync(v, 0);
        }

        clock_gettime(CLOCK_MONOTONIC, &t2);
        sync_time_us_    += (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
        hmemcpy_time_us_ += (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_nsec - t1.tv_nsec) / 1e3;

        pending_outputs_.clear();
        pending_inputs_.clear();
        pending_hbm_frees_.clear();
        needs_sync_ = false;
    }

    // Drop the queue without syncing. Used on error paths.
    void abort_pending() {
        for (auto & o : pending_outputs_) pool_.release(o.hmem);
        for (auto & h : pending_inputs_)  pool_.release(h);
        if (!pending_hbm_frees_.empty()) {
            vedaCtxSynchronize();
            for (auto & v : pending_hbm_frees_) vedaMemFreeAsync(v, 0);
        }
        pending_outputs_.clear();
        pending_inputs_.clear();
        pending_hbm_frees_.clear();
        needs_sync_ = false;
    }

    // ---- Per-op statistics, useful for telemetry on free() ----
    int64_t & ops_total()      { return ops_total_; }
    int64_t & ops_mul_mat()    { return ops_mul_mat_; }
    int64_t & ops_flash_attn() { return ops_flash_attn_; }
    int64_t & ops_hbm()        { return ops_hbm_; }
    int64_t   sync_count() const     { return sync_count_; }
    double    sync_time_us() const   { return sync_time_us_; }
    double    hmemcpy_time_us() const{ return hmemcpy_time_us_; }

private:
    device *         dev_     = nullptr;
    ggml_cgraph *    cgraph_  = nullptr;
    hmem_pool        pool_;
    hbm_weight_cache cache_;

    std::vector<pending_output> pending_outputs_;
    std::vector<VEDAhmemptr>    pending_inputs_;
    std::vector<VEDAdeviceptr>  pending_hbm_frees_;
    bool                        needs_sync_ = false;

    int64_t sync_count_      = 0;
    double  sync_time_us_    = 0.0;
    double  hmemcpy_time_us_ = 0.0;

    int64_t ops_total_       = 0;
    int64_t ops_mul_mat_     = 0;
    int64_t ops_flash_attn_  = 0;
    int64_t ops_hbm_         = 0;
};

} // namespace ggml_ve

#endif // GGML_VE_BACKEND_CTX_HPP
