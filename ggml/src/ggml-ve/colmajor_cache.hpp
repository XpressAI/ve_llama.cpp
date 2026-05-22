#ifndef GGML_VE_COLMAJOR_CACHE_HPP
#define GGML_VE_COLMAJOR_CACHE_HPP

// Per-device cache of F32 column-major weight copies.
//
// The original ve_llama.cpp port keeps weight tensors in the ggml-allocated
// VE_HBM buffer (BF16, row-major). The fastest BF16 matvec / matmul kernel
// on the VE is `ve_f32_sgemm_batched_cblas_hbm_notrans`, which wants F32
// column-major weights. The legacy llama.cpp VE backend got "10-30x faster"
// prompt eval and the bulk of its decode win by pre-transposing each weight
// to F32 column-major *once* and caching the result in HBM.
//
// This cache wraps that pattern:
//
//   - keyed by the source BF16 weight's HBM address (stable for the model's
//     lifetime, ggml allocator doesn't shuffle weights);
//   - the transpose itself runs on the VE (the in-tree
//     ve_bf16_to_f32_colmajor_hbm kernel), so no PCIe round-trip;
//   - cached entries live until the device shuts down.
//
// Memory cost: 2 * sizeof(F32) / sizeof(BF16) = 4x the BF16 weight size for
// the column-major copy. For Llama-3.2-3B BF16 (~3 GB of weights), the F32
// col-major versions add ~12 GB — well within a 48 GB HBM budget.

#include "common.hpp"
#include "device.hpp"

#include <cstdio>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace ggml_ve {

class colmajor_weight_cache {
public:
    void set_context(VEDAcontext ctx) { ctx_ = ctx; }

    // Return the F32 column-major HBM pointer for the BF16 weight at
    // bf16_hbm (M x K row-major), creating it on first call. Returns 0 on
    // failure.
    VEDAdeviceptr get_or_create(VEDAdeviceptr bf16_hbm, int64_t M, int64_t K,
                                VEDAfunction transpose_fn) {
        if (bf16_hbm == 0 || M <= 0 || K <= 0) return 0;

        std::lock_guard<std::mutex> lk(mu_);
        auto it = entries_.find((uint64_t) bf16_hbm);
        if (it != entries_.end()) {
            ++hits_;
            return it->second.f32_colmajor_hbm;
        }

        if (transpose_fn == 0) {
            fprintf(stderr, "[ve-colmajor-cache] no transpose fn\n");
            return 0;
        }

        // Caller is responsible for having the device's VEDA context pushed
        // (backend_graph_compute does it via VEDAContextGuard). Pushing it
        // again here deadlocks on later iterations once VEDA's per-thread
        // context stack saturates.

        // Drain any in-flight ops before reallocating HBM — the source BF16
        // tensor may have just been written; we read it.
        vedaCtxSynchronize();

        VEDAresult err;
        const size_t f32_bytes = (size_t) M * (size_t) K * sizeof(float);
        static const bool dbg = std::getenv("GGML_VE_DEBUG_COLMAJOR") != nullptr;
        if (dbg) {
            fprintf(stderr, "[ve-colmajor-cache] create entry %d: M=%ld K=%ld bytes=%zu (total cached: %zu MB)\n",
                    (int) entries_.size(), (long) M, (long) K, f32_bytes,
                    (total_bytes_ + f32_bytes) / (1024 * 1024));
        }
        VEDAdeviceptr f32_colmajor = 0;
        err = vedaMemAllocAsync(&f32_colmajor, f32_bytes, 0);
        if (err != VEDA_SUCCESS) {
            fprintf(stderr, "[ve-colmajor-cache] vedaMemAllocAsync(%zu) failed: %s\n",
                    f32_bytes, ggml_ve_err_str(err));
            return 0;
        }

        VEDAargs args = nullptr;
        if (vedaArgsCreate(&args) != VEDA_SUCCESS) {
            vedaMemFreeAsync(f32_colmajor, 0);
            return 0;
        }
        vedaArgsSetVPtr(args, 0, bf16_hbm);
        vedaArgsSetVPtr(args, 1, f32_colmajor);
        vedaArgsSetU64 (args, 2, (uint64_t) M);
        vedaArgsSetU64 (args, 3, (uint64_t) K);

        uint64_t rc = 0;
        err = vedaLaunchKernelEx(transpose_fn, 0, args, /*destroyArgs=*/1, &rc);
        if (err != VEDA_SUCCESS) {
            fprintf(stderr, "[ve-colmajor-cache] transpose launch failed: %s\n",
                    ggml_ve_err_str(err));
            vedaMemFreeAsync(f32_colmajor, 0);
            return 0;
        }
        vedaCtxSynchronize();

        entry e;
        e.f32_colmajor_hbm = f32_colmajor;
        e.bytes            = f32_bytes;
        e.M                = M;
        e.K                = K;
        entries_[(uint64_t) bf16_hbm] = e;
        total_bytes_ += f32_bytes;
        ++misses_;

        return f32_colmajor;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        if (ctx_) vedaCtxPushCurrent(ctx_);
        for (auto & kv : entries_) {
            if (kv.second.f32_colmajor_hbm) vedaMemFreeAsync(kv.second.f32_colmajor_hbm, 0);
        }
        if (ctx_) {
            vedaCtxSynchronize();
            VEDAcontext prev = nullptr; vedaCtxPopCurrent(&prev);
        }
        entries_.clear();
        total_bytes_ = 0;
    }

    void stats(size_t * bytes, int64_t * hits, int64_t * misses) const {
        if (bytes)  *bytes  = total_bytes_;
        if (hits)   *hits   = hits_;
        if (misses) *misses = misses_;
    }

private:
    struct entry {
        VEDAdeviceptr f32_colmajor_hbm = 0;
        size_t        bytes            = 0;
        int64_t       M = 0, K = 0;
    };

    std::unordered_map<uint64_t, entry> entries_;
    std::mutex   mu_;
    VEDAcontext  ctx_         = nullptr;
    size_t       total_bytes_ = 0;
    int64_t      hits_        = 0;
    int64_t      misses_      = 0;
};

} // namespace ggml_ve

#endif // GGML_VE_COLMAJOR_CACHE_HPP
