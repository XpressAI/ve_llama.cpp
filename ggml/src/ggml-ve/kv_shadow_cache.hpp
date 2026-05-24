#ifndef GGML_VE_KV_SHADOW_CACHE_HPP
#define GGML_VE_KV_SHADOW_CACHE_HPP

// Per-device shadow KV cache in column-major (seq-as-unit-stride) layout.
//
// Each entry is a parallel BF16 buffer in HBM, sized to match the
// original KV cache tensor (so they're interchangeable from a "how much
// of the seq dim has valid data" standpoint), but transposed so the FA
// kernel can stream-load along the seq dim without strided cache-line
// waste.
//
// Why this exists: the row-major K cache stores [head_dim, seq, n_kv_heads]
// with nb_k1 (stride between seq positions) = head_dim*n_kv_heads*2 bytes.
// For Llama-3.2-3B that's 2048 bytes between adjacent seq positions per
// kv_head — FA's per-seq loop becomes 8x cache-line wasteful, and per-
// token cost grows linearly with KV occupancy. The col-major shadow
// turns those strided loads into unit-stride; the new attention kernel
// (ve_flash_attn_ext_f32q_bf16kv_colmajor_hbm) runs ~50x faster at
// long seq_len.
//
// Validity model (per codex review):
//   - `generation` is bumped any time the shadow is invalidated wholesale
//     (e.g. cache reset between requests, prompt-eval bulk write).
//   - `watermark` tracks how many rows from row 0 have been mirrored
//     since the last invalidation. FA may read shadow rows [0, watermark).
//   - For non-monotonic writes (slot reuse, cache shift, multi-row prompt
//     prefill) we invalidate and fall back to row-major FA until the
//     shadow is rebuilt.
//
// Memory cost: BF16 shadow = same bytes as the BF16 original KV cache.
// One shadow per cache tensor. For Llama-3.2-3B (28 layers × 2 caches
// each × ~16 MB per cache at ctx=4096) ≈ 0.9 GB. Comfortable in 48 GB.

#include "common.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace ggml_ve {

struct kv_shadow {
    VEDAdeviceptr shadow_hbm = 0;     // BF16 col-major buffer
    int64_t       channels   = 0;     // head_dim * n_kv_heads
    int64_t       seq_max    = 0;     // allocated capacity (stride for seq dim)
    int64_t       watermark  = 0;     // rows [0, watermark) mirrored since last invalidation
    int64_t       generation = 0;     // bumped on invalidate
};

class kv_shadow_cache {
public:
    void set_context(VEDAcontext ctx) { ctx_ = ctx; }

    /* Get or lazily create a shadow for the KV cache tensor whose
     * canonical HBM data ptr is source_hbm. Same source -> same shadow
     * for the model's lifetime. Returns nullptr if the alloc fails. */
    kv_shadow * get_or_create(VEDAdeviceptr source_hbm,
                              int64_t       channels,
                              int64_t       seq_max) {
        if (source_hbm == 0 || channels <= 0 || seq_max <= 0) return nullptr;

        std::lock_guard<std::mutex> lk(mu_);
        auto it = entries_.find((uint64_t) source_hbm);
        if (it != entries_.end()) {
            kv_shadow & s = it->second;
            if (s.channels != channels || s.seq_max != seq_max) {
                if (s.shadow_hbm) vedaMemFreeAsync(s.shadow_hbm, 0);
                entries_.erase(it);
            } else {
                return &it->second;
            }
        }

        const size_t bytes = (size_t) channels * seq_max * sizeof(uint16_t);
        VEDAdeviceptr buf = 0;
        VEDAresult err = vedaMemAllocAsync(&buf, bytes, 0);
        if (err != VEDA_SUCCESS) {
            fprintf(stderr, "[kv-shadow] vedaMemAllocAsync(%zu) failed: %s\n",
                    bytes, ggml_ve_err_str(err));
            return nullptr;
        }
        /* Zero-init the shadow: unmirrored positions in the shadow must
         * match the zeros that fresh-allocated HBM gives the row-major
         * K cache, otherwise FA reads garbage at masked-out positions
         * and the mask doesn't fully suppress them (e.g. softmax
         * normalisation amplifies small noise into wrong output). */
        VEDAresult zerr = vedaMemsetD8Async(buf, 0, bytes, 0);
        if (zerr != VEDA_SUCCESS) {
            fprintf(stderr, "[kv-shadow] vedaMemsetD8Async failed: %s\n",
                    ggml_ve_err_str(zerr));
            vedaMemFreeAsync(buf, 0);
            return nullptr;
        }

        kv_shadow s;
        s.shadow_hbm = buf;
        s.channels   = channels;
        s.seq_max    = seq_max;
        s.watermark  = 0;
        s.generation = 1;
        entries_[(uint64_t) source_hbm] = s;
        return &entries_[(uint64_t) source_hbm];
    }

    /* Look up an existing shadow (no create). Returns nullptr if absent. */
    kv_shadow * find(VEDAdeviceptr source_hbm) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = entries_.find((uint64_t) source_hbm);
        return (it == entries_.end()) ? nullptr : &it->second;
    }

    /* Mark rows [row_start, row_start+nrows) as mirrored. Caller has
     * launched the mirror kernel; this just bookkeeps the validity range. */
    void note_rows_mirrored(kv_shadow * s, int64_t row_start, int64_t nrows) {
        if (s == nullptr || nrows <= 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (row_start == 0) {
            // Fresh start (warmup, new request, cache reset). Whatever
            // the previous watermark was, the cache contents from row 0
            // are now fresh, so the shadow's row 0..nrows is current.
            s->watermark = nrows;
            s->generation++;
        } else if (row_start == s->watermark) {
            // Monotonic continuation.
            s->watermark = row_start + nrows;
        } else if (row_start > s->watermark) {
            // Forward gap — caller wrote rows we haven't mirrored. Can't
            // claim the shadow is valid through the new high-water mark.
            s->watermark = 0;
            s->generation++;
        } else {
            // Backward write into mid-range. Slot reuse / cache shift.
            s->watermark = 0;
            s->generation++;
        }
    }

    /* Wholesale invalidate (e.g. before a multi-row prefill we don't
     * mirror through, or on session reset). */
    void invalidate(kv_shadow * s) {
        if (s == nullptr) return;
        std::lock_guard<std::mutex> lk(mu_);
        s->watermark = 0;
        s->generation++;
    }

    /* Can FA safely read shadow rows [0, kv_len)?
     *
     * Note: the shadow is zero-initialised, so unmirrored positions
     * contain zeros — same as the row-major K cache, where positions
     * never written by SET_ROWS are also zero from fresh HBM
     * allocation. So as long as the watermark covers all
     * SET_ROWS-touched positions (which the per-write mirror enforces
     * by construction), FA produces the same result on shadow as on the
     * row-major cache, regardless of the K view's ne[1] (which may be
     * allocation-rounded above the actual valid kv_len).
     *
     * We still want a nontrivial watermark so we don't trigger the
     * colmajor path before any real data has been mirrored (e.g. before
     * prompt prefill). */
    bool valid_for_range(kv_shadow * s, int64_t /*kv_len*/) const {
        if (s == nullptr) return false;
        return s->watermark > 0;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        if (ctx_) vedaCtxPushCurrent(ctx_);
        for (auto & kv : entries_) {
            if (kv.second.shadow_hbm) vedaMemFreeAsync(kv.second.shadow_hbm, 0);
        }
        if (ctx_) {
            vedaCtxSynchronize();
            VEDAcontext prev = nullptr; vedaCtxPopCurrent(&prev);
        }
        entries_.clear();
    }

private:
    std::unordered_map<uint64_t, kv_shadow> entries_;
    std::mutex                              mu_;
    VEDAcontext                             ctx_ = nullptr;
};

} // namespace ggml_ve

#endif // GGML_VE_KV_SHADOW_CACHE_HPP
