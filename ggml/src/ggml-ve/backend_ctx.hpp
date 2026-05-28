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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    void mark_sync_pending() {
        needs_sync_ = true;
        // In canary mode, force a per-op flush so that any kernel that
        // tramples its dst's trailing canary is caught immediately --
        // we log the op name, then continue and likely crash on the
        // next op. The crash trace then pinpoints the offender.
        if (canary_enabled()) flush("canary-per-op");
    }
    bool has_pending() const { return needs_sync_; }

    // ---- Tensor → HBM resolution ----
    //
    // Returns a VEDAdeviceptr addressing `t`'s contents on the VE,
    // uploading from CPU memory if needed.
    //
    //   1. tensor_is_hbm(t)            → return tensor_hbm_ptr(t).
    //   2. CPU tensor, WEIGHTS buffer  → cache via hbm_weight_cache
    //      (keyed by tensor name) and return the cached HBM ptr.
    //      Re-used across every op that touches the weight.
    //   3. CPU tensor, transient       → allocate a temp HBM, copy host
    //      → HBM, enqueue the temp for free-after-sync.
    //
    // resolve_out is the write-side equivalent.
    //
    // Why these are two-function (inline wrapper + out-of-line slow):
    // an earlier version kept the slow body inline in the class. Even
    // with __attribute__((always_inline)) on the wrapper, gcc saw the
    // body could mutate `this` (cache_, deferred_dtoh_, needs_sync_)
    // and refused to elide work at call sites — measured -28% decode
    // (8.8 → 6.3 t/s on Llama-3.2-3B BF16) even when the slow path was
    // unreachable. Hiding the body behind a forward declaration leaves
    // gcc with a pure-function fast path plus an unconditional call on
    // the miss branch, which is properly inlined back to the original.
    inline VEDAdeviceptr resolve_in(const ggml_tensor * t) {
        if (tensor_is_hbm(t)) return tensor_hbm_ptr(t);
        return resolve_in_slow(t);
    }
    inline VEDAdeviceptr resolve_out(ggml_tensor * dst) {
        if (tensor_is_hbm(dst)) return tensor_hbm_ptr(dst);
        return resolve_out_slow(dst);
    }

    // Out-of-line: defined in backend_ctx.cpp. The bodies must NOT be
    // visible to callers of resolve_in / resolve_out.
    VEDAdeviceptr resolve_in_slow (const ggml_tensor * t);
    VEDAdeviceptr resolve_out_slow(ggml_tensor *       dst);

    // Constants for the canary guard mode.
    static constexpr size_t  CANARY_BYTES   = 64;
    static constexpr uint8_t CANARY_PATTERN = 0xCD;
    static bool canary_enabled() {
        static bool on = (std::getenv("GGML_VE_DEBUG_CANARY") != nullptr);
        return on;
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

        // Check guard bands -- any byte != 0xCD means a kernel wrote
        // past the dst it was given.
        if (canary_enabled() && !canaries_.empty()) {
            uint8_t guard[CANARY_BYTES];
            for (auto & c : canaries_) {
                VEDAdeviceptr trailer = (VEDAdeviceptr)((uintptr_t)c.base + c.logical_sz);
                if (vedaMemcpyDtoH(guard, trailer, CANARY_BYTES) != VEDA_SUCCESS) continue;
                for (size_t i = 0; i < CANARY_BYTES; ++i) {
                    if (guard[i] != CANARY_PATTERN) {
                        fprintf(stderr,
                                "[CANARY] OVERFLOW in '%s' (logical_sz=%zu): byte +%zu past end "
                                "= 0x%02x (expected 0x%02x). First 16 trailer bytes: ",
                                c.label ? c.label : "?", c.logical_sz, i,
                                guard[i], CANARY_PATTERN);
                        for (int j = 0; j < 16; ++j) fprintf(stderr, "%02x ", guard[j]);
                        fprintf(stderr, "\n");
                        break;
                    }
                }
            }
            canaries_.clear();
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);

        for (auto & o : pending_outputs_) {
            vedaHMemcpy(o.host_dst, reinterpret_cast<void *>(o.hmem), o.size);
            pool_.release(o.hmem);
        }
        // CPU-dst ops (from resolve_out): direct DtoH from temp HBM
        // into the original host destination. Issue all as async, then
        // sync once at the end -- much faster than N synchronous DtoHs
        // when many small intermediates pile up (typical of safe-mode
        // post-flush).
        if (!deferred_dtoh_.empty()) {
            for (auto & d : deferred_dtoh_) {
                vedaMemcpyDtoHAsync(d.host_dst, d.temp_hbm, d.size, 0);
            }
            vedaCtxSynchronize();  // single sync for all queued DtoHs
            for (auto & d : deferred_dtoh_) {
                vedaMemFreeAsync(d.temp_hbm, 0);
            }
        }
        for (auto & h : pending_inputs_) {
            pool_.release(h, nullptr, 0, /*clear_cache=*/true);
        }
        for (auto & v : pending_hbm_frees_) {
            vedaMemFreeAsync(v, 0);
        }
        // Free bounce buffers used to align misaligned host sources for
        // HtoD. AVEO's DMA path requires >= 8-byte aligned source; view
        // tensors whose data pointer carries a sub-8-byte offset (e.g.
        // Qwen3.5 conv_state_last at parent+36) must be staged through a
        // properly-aligned host buffer before async HtoD.
        for (auto & b : pending_host_frees_) {
            std::free(b);
        }

        clock_gettime(CLOCK_MONOTONIC, &t2);
        sync_time_us_    += (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
        hmemcpy_time_us_ += (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_nsec - t1.tv_nsec) / 1e3;

        pending_outputs_.clear();
        pending_inputs_.clear();
        pending_hbm_frees_.clear();
        pending_host_frees_.clear();
        deferred_dtoh_.clear();
        needs_sync_ = false;
    }

    // Drop the queue without syncing. Used on error paths.
    void abort_pending() {
        for (auto & o : pending_outputs_) pool_.release(o.hmem);
        for (auto & h : pending_inputs_)  pool_.release(h);
        if (!pending_hbm_frees_.empty() || !deferred_dtoh_.empty()) {
            vedaCtxSynchronize();
            for (auto & v : pending_hbm_frees_) vedaMemFreeAsync(v, 0);
            for (auto & d : deferred_dtoh_)     vedaMemFreeAsync(d.temp_hbm, 0);
        }
        for (auto & b : pending_host_frees_) std::free(b);
        pending_outputs_.clear();
        pending_inputs_.clear();
        pending_hbm_frees_.clear();
        pending_host_frees_.clear();
        canaries_.clear();
        deferred_dtoh_.clear();
        needs_sync_ = false;
    }

    // Bounce-buffer hook for resolve_in: track an aligned host buffer
    // that holds a copy of a misaligned tensor's bytes. Must outlive
    // the async HtoD that reads from it, so we free at flush() time.
    void enqueue_host_free(void * p) { pending_host_frees_.push_back(p); }

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
    std::vector<void *>         pending_host_frees_;

    // HBM guard-band canaries (GGML_VE_DEBUG_CANARY=1). Each temp HBM
    // allocation that resolve_*_slow makes carries an extra 64-byte
    // trailer filled with 0xCD. flush() checks each trailer after
    // sync; any byte that's not 0xCD points to a kernel that wrote
    // past its declared dst size.
public:
    struct canary_record {
        VEDAdeviceptr base       = 0;     // start of the allocation
        size_t        logical_sz = 0;     // bytes the kernel was told it has
        const char *  label      = nullptr;
    };
    std::vector<canary_record> canaries_;

    // Deferred D→H copies for ops whose `dst` lives in a CPU buffer.
    // After the next sync, we copy temp_hbm → host_dst directly with
    // vedaMemcpyDtoH and free temp_hbm. Previous design staged through
    // an HMEM intermediate which added a redundant memcpy hop and
    // introduced HMEM-pool lifetime bugs in long queues (the recvBuff
    // segfault that bit Qwen3.5 op #38).
    struct deferred_dtoh {
        VEDAdeviceptr temp_hbm = 0;
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
