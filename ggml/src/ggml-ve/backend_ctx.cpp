// Out-of-line slow paths for backend_context::resolve_in/out. Kept in a
// separate translation unit on purpose: when these bodies were inline in
// backend_ctx.hpp, gcc refused to elide work at the call sites (see
// resolve_in declaration). Moving them here costs one function call per
// CPU-tensor resolution (rare) and recovers the inlined fast-path
// behaviour for every all-HBM call site.

#include "backend_ctx.hpp"

#include <cstdlib>
#include <cstring>

namespace ggml_ve {

// VEDA's AVEO HtoD path uses an internal DMA engine that requires the
// source host pointer to be at least 8-byte aligned (matches the
// "vectorizable alignment conditions" the NCC compiler assumes -- see
// NCC User Guide ch.5). View tensors whose data pointer carries an
// offset smaller than 8 bytes (e.g. Qwen3.5 conv_state_last lives at
// conv_input + 36 -- only 4-byte aligned) trigger a host-side memmove
// segfault deep inside libveo's recvBuffAsync when the deferred sync
// finally progresses the transfer. We stage them through an aligned
// bounce buffer.
// 64 bytes covers VE vector register alignment + cache lines. Worth
// staying above 8 because AVEO's DMA path and NCC's vector loads both
// like larger alignment; some kernels assume cache-line alignment.
static constexpr size_t HTOD_ALIGNMENT = 64;

static inline bool is_aligned(const void * p, size_t n) {
    return (reinterpret_cast<uintptr_t>(p) & (n - 1)) == 0;
}

VEDAdeviceptr backend_context::resolve_in_slow(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) return 0;
    const size_t size = ggml_nbytes(t);
    if (size == 0) return 0;
    if (tensor_is_weight(t)) {
        // Key the cache by tensor NAME — weights sometimes arrive as
        // views into a packed mmap'd buffer; the view's `t->data`
        // includes the offset and changes per view object, but the
        // name ("blk.0.attn_q.weight") is stable.
        const char * name = (t->name && t->name[0]) ? t->name : nullptr;
        return cache_.get_or_upload_by_name(name, t->data, size);
    }
    // PRODUCER-CONSUMER FRESH-DATA LOOKUP:
    // Within a single VE graph split, if an earlier VE op wrote to this
    // same CPU-backed tensor and queued the DtoH copy, the fresh data
    // lives in temp_hbm. The CPU bytes are stale until flush() runs.
    //
    // KNOWN-BUG: this returns the WRONG entry on Q-quant models where
    // CPU MUL_MATs and VE ops alternate frequently. The deferred queue
    // ends up holding stale entries whose host_dst matches a freshly
    // CPU-written tensor (because the cgraph allocator reuses the same
    // CPU memory address for tensors at different points in the graph,
    // and resolve_out's same-address purge only fires when the new
    // writer is also VE — a CPU writer between two VE ops leaves the
    // stale entry intact). GGML_VE_QUANT_SAFE_MODE=1 (or
    // GGML_VE_NO_FRESH_HIT=1) skips the lookup; combined with always-on
    // per-op sync (also flipped by QUANT_SAFE_MODE) this gives correct
    // output on Q-quant models at ~9% perf cost on BF16 and 58× speedup
    // + correctness on Q4_K_M (vs the broken default). The minimal
    // correct fix is open in task #60.
    static const bool no_fresh_hit =
        std::getenv("GGML_VE_NO_FRESH_HIT") != nullptr;
    if (!no_fresh_hit) {
        // Only return entries that have been MARKED INITIALIZED by a
        // prior op's commit_pending_initialized() call. Skipping
        // uninitialized entries handles the in-place op case correctly:
        // resolve_out adds an entry for dst BEFORE the kernel runs (so
        // the temp_hbm is empty); if src aliases dst the naive fresh-hit
        // would return that empty buffer. The initialized flag ensures
        // we walk back to the LATEST genuinely-written entry, or fall
        // through to a clean CPU upload if none exists.
        for (auto it = deferred_dtoh_.rbegin(); it != deferred_dtoh_.rend(); ++it) {
            if (it->host_dst == t->data && it->size == size && it->initialized) {
                return it->temp_hbm;
            }
        }
    }
    // Transient activation: temp HBM, freed after sync. With canary
    // mode on, allocate an extra trailer and seed it with 0xCD so we
    // can detect kernels that read past `size` (irrelevant here -- this
    // buffer is a source, but if a kernel writes past it accidentally
    // we'll catch it too).
    const size_t alloc_sz = backend_context::canary_enabled()
        ? size + backend_context::CANARY_BYTES
        : size;
    VEDAdeviceptr tmp = 0;
    if (vedaMemAllocAsync(&tmp, alloc_sz, 0) != VEDA_SUCCESS || tmp == 0) {
        return 0;
    }
    if (backend_context::canary_enabled()) {
        vedaMemsetD8Async(
            (VEDAdeviceptr)((uintptr_t)tmp + size),
            backend_context::CANARY_PATTERN,
            backend_context::CANARY_BYTES, 0);
        canaries_.push_back({tmp, size,
            (t->name && t->name[0]) ? t->name : "?in?"});
    }

    const void * src = t->data;
    // If the host source is sub-8-byte-aligned (common for view tensors
    // whose offset isn't a multiple of 8), copy into an aligned bounce
    // buffer first. The bounce outlives this call -- freed at flush().
    if (!is_aligned(src, HTOD_ALIGNMENT)) {
        const size_t b_sz = (size + HTOD_ALIGNMENT - 1) & ~(HTOD_ALIGNMENT - 1);
        void * bounce = std::aligned_alloc(HTOD_ALIGNMENT, b_sz);
        if (bounce == nullptr) {
            vedaMemFreeAsync(tmp, 0);
            return 0;
        }
        std::memcpy(bounce, t->data, size);
        enqueue_host_free(bounce);
        src = bounce;
    }

    if (vedaMemcpyHtoDAsync(tmp, src, size, 0) != VEDA_SUCCESS) {
        vedaMemFreeAsync(tmp, 0);
        return 0;
    }
    enqueue_hbm_free(tmp);
    return tmp;
}

VEDAdeviceptr backend_context::resolve_out_slow(ggml_tensor * dst) {
    if (dst == nullptr || dst->data == nullptr) return 0;
    const size_t size     = ggml_nbytes(dst);
    if (size == 0) return 0;
    const size_t alloc_sz = backend_context::canary_enabled()
        ? size + backend_context::CANARY_BYTES
        : size;
    VEDAdeviceptr tmp = 0;
    if (vedaMemAllocAsync(&tmp, alloc_sz, 0) != VEDA_SUCCESS || tmp == 0) {
        return 0;
    }
    if (backend_context::canary_enabled()) {
        vedaMemsetD8Async(
            (VEDAdeviceptr)((uintptr_t)tmp + size),
            backend_context::CANARY_PATTERN,
            backend_context::CANARY_BYTES, 0);
        canaries_.push_back({tmp, size,
            (dst->name && dst->name[0]) ? dst->name : "?out?"});
    }
    // DELIBERATELY NO PURGE here. Keeping older entries with the same
    // host_dst preserves the "prior data" for in-place ops:
    //   resolve_out(dst) adds a fresh uninitialized entry; resolve_in(src)
    //   with src->data == dst->data walks back, skips this uninitialized
    //   entry, and returns the PRIOR initialized entry's temp_hbm.
    // flush() copies all entries with the same host_dst in insertion
    // order, so the LATEST write wins in CPU memory. The cgraph-reuse
    // case (X dies, Z reuses addr P) is safe because the scheduler
    // already calls backend_synchronize between graph splits, which
    // runs flush() and clears the queue.
    // After the kernel runs, flush() copies tmp → dst->data directly
    // via vedaMemcpyDtoH and frees tmp. No HMEM intermediate. Mark
    // initialized=false so resolve_in_slow's fresh-hit won't return this
    // entry until the producing op's compute_forward completes (the
    // backend_context::commit_pending_initialized() call at the end of
    // each op walks deferred_dtoh_ and flips initialized=true).
    deferred_dtoh_.push_back({tmp, dst->data, size, /*initialized=*/false});
    needs_sync_ = true;
    return tmp;
}

} // namespace ggml_ve
