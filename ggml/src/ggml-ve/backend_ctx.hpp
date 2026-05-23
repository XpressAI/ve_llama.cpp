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

// True if `t` lives in a buffer ggml flagged as WEIGHTS (stable for the
// model's lifetime). Used to decide between cache-forever vs free-after-
// kernel uploads.
inline bool tensor_is_weight(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr) return false;
    return ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
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

    // ---- Tensor → HBM resolution ----
    //
    // Returns a VEDAdeviceptr addressing `t`'s contents on the VE, uploading
    // from CPU memory if needed. There are three cases:
    //
    //   1. tensor_is_hbm(t)  → already on the VE; return tensor_hbm_ptr(t).
    //   2. tensor_is_weight(t) on CPU → the data is stable for the model's
    //      lifetime; cache via hbm_weight_cache (keyed by host pointer)
    //      and return the cached HBM ptr. Re-used across every op that
    //      touches this weight.
    //   3. otherwise (CPU activation) → allocate a temp HBM, copy host →
    //      HBM, enqueue the temp for free-after-sync. The caller never
    //      sees the temp; it just gets a valid HBM ptr.
    //
    // Returns 0 on failure (HBM OOM, ctx not set, etc.). Note: case (3)
    // marks the context dirty (`needs_sync_=true`), so an immediate
    // `vedaCtxSynchronize` is the wrong thing to do after — let the
    // normal flush() at end-of-graph handle it.
    // Hot path: split the all-HBM short-circuit out so gcc always inlines
    // it. The slow path (CPU upload) is in resolve_in_slow.
    __attribute__((always_inline))
    VEDAdeviceptr resolve_in(const ggml_tensor * t) {
        if (tensor_is_hbm(t)) return tensor_hbm_ptr(t);
        return resolve_in_slow(t);
    }

    VEDAdeviceptr resolve_in_slow(const ggml_tensor * t) {
        if (t == nullptr || t->data == nullptr) return 0;
        const size_t size = ggml_nbytes(t);
        if (tensor_is_weight(t)) {
            // Key the weight cache by tensor NAME (stable across calls),
            // not by host pointer: weights sometimes arrive as views into
            // a larger mmap'd buffer, and the view's `t->data` includes
            // the offset — different view objects for the same logical
            // weight can have different `t->data` even though the bytes
            // they point at are the same. tensor->name (e.g.,
            // "blk.0.attn_q.weight") is set by the loader once and never
            // changes. get_or_upload_by_name tries name first, falls back
            // to host pointer.
            const char * name = (t->name && t->name[0]) ? t->name : nullptr;
            return cache_.get_or_upload_by_name(name, t->data, size);
        }
        // Transient activation: temp HBM alloc.
        VEDAdeviceptr tmp = 0;
        if (vedaMemAllocAsync(&tmp, size, 0) != VEDA_SUCCESS || tmp == 0) {
            return 0;
        }
        if (vedaMemcpyHtoDAsync(tmp, t->data, size, 0) != VEDA_SUCCESS) {
            vedaMemFreeAsync(tmp, 0);
            return 0;
        }
        enqueue_hbm_free(tmp);
        return tmp;
    }

    // Output-side resolver. If `dst` is HBM, returns its ptr directly.
    // Otherwise allocates a temp HBM that the kernel will write into,
    // and enqueues a copy-back from temp → dst->data after the next
    // sync (using a pooled HMEM staging buffer because vedaMemcpyDtoH
    // can't write straight into arbitrary host memory).
    //
    // Returns 0 on failure.
    __attribute__((always_inline))
    VEDAdeviceptr resolve_out(ggml_tensor * dst) {
        if (tensor_is_hbm(dst)) return tensor_hbm_ptr(dst);
        return resolve_out_slow(dst);
    }

    VEDAdeviceptr resolve_out_slow(ggml_tensor * dst) {
        if (dst == nullptr || dst->data == nullptr) return 0;
        const size_t size = ggml_nbytes(dst);
        VEDAdeviceptr tmp = 0;
        if (vedaMemAllocAsync(&tmp, size, 0) != VEDA_SUCCESS || tmp == 0) {
            return 0;
        }
        // Stage HMEM for the deferred D→H bounce.
        VEDAhmemptr hmem = pool_.acquire(size);
        if (hmem == 0) {
            vedaMemFreeAsync(tmp, 0);
            return 0;
        }
        // After the kernel runs, copy tmp → hmem (D→H from VE's side),
        // then flush() copies hmem → dst->data (H→H), then releases both.
        deferred_dtoh_.push_back({tmp, hmem, dst->data, size});
        needs_sync_ = true;
        return tmp;
    }

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
        // CPU-dst ops (from resolve_out): stage temp HBM → HMEM, then memcpy
        // to the host destination. We do this BEFORE freeing the temp HBMs.
        for (auto & d : deferred_dtoh_) {
            if (vedaMemcpyDtoH(reinterpret_cast<void *>(d.hmem), d.temp_hbm, d.size) == VEDA_SUCCESS) {
                std::memcpy(d.host_dst, reinterpret_cast<void *>(d.hmem), d.size);
            }
            pool_.release(d.hmem);
            vedaMemFreeAsync(d.temp_hbm, 0);
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
        deferred_dtoh_.clear();
        needs_sync_ = false;
    }

    // Drop the queue without syncing. Used on error paths.
    void abort_pending() {
        for (auto & o : pending_outputs_) pool_.release(o.hmem);
        for (auto & h : pending_inputs_)  pool_.release(h);
        for (auto & d : deferred_dtoh_)   { pool_.release(d.hmem); }
        if (!pending_hbm_frees_.empty() || !deferred_dtoh_.empty()) {
            vedaCtxSynchronize();
            for (auto & v : pending_hbm_frees_) vedaMemFreeAsync(v, 0);
            for (auto & d : deferred_dtoh_)     vedaMemFreeAsync(d.temp_hbm, 0);
        }
        pending_outputs_.clear();
        pending_inputs_.clear();
        pending_hbm_frees_.clear();
        deferred_dtoh_.clear();
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

    // Deferred D→H copies for ops whose `dst` lives in a CPU buffer.
    // After the next sync, we copy temp_hbm → hmem (D→H), then
    // hmem → host_dst (memcpy), then free temp_hbm and release hmem.
    struct deferred_dtoh {
        VEDAdeviceptr temp_hbm = 0;
        VEDAhmemptr   hmem     = 0;
        void *        host_dst = nullptr;
        size_t        size     = 0;
    };
    std::vector<deferred_dtoh> deferred_dtoh_;

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
