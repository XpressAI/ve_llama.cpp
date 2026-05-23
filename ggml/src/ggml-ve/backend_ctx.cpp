// Out-of-line slow paths for backend_context::resolve_in/out. Kept in a
// separate translation unit on purpose: when these bodies were inline in
// backend_ctx.hpp, gcc refused to elide work at the call sites (see
// resolve_in declaration). Moving them here costs one function call per
// CPU-tensor resolution (rare) and recovers the inlined fast-path
// behaviour for every all-HBM call site.

#include "backend_ctx.hpp"

namespace ggml_ve {

VEDAdeviceptr backend_context::resolve_in_slow(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) return 0;
    const size_t size = ggml_nbytes(t);
    if (tensor_is_weight(t)) {
        // Key the cache by tensor NAME — weights sometimes arrive as
        // views into a packed mmap'd buffer; the view's `t->data`
        // includes the offset and changes per view object, but the
        // name ("blk.0.attn_q.weight") is stable.
        const char * name = (t->name && t->name[0]) ? t->name : nullptr;
        return cache_.get_or_upload_by_name(name, t->data, size);
    }
    // Transient activation: temp HBM, freed after sync.
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

VEDAdeviceptr backend_context::resolve_out_slow(ggml_tensor * dst) {
    if (dst == nullptr || dst->data == nullptr) return 0;
    const size_t size = ggml_nbytes(dst);
    VEDAdeviceptr tmp = 0;
    if (vedaMemAllocAsync(&tmp, size, 0) != VEDA_SUCCESS || tmp == 0) {
        return 0;
    }
    VEDAhmemptr hmem = pool_.acquire(size);
    if (hmem == 0) {
        vedaMemFreeAsync(tmp, 0);
        return 0;
    }
    // After the kernel runs, flush() copies tmp → hmem (D→H), then
    // memcpy hmem → dst->data, then frees both.
    deferred_dtoh_.push_back({tmp, hmem, dst->data, size});
    needs_sync_ = true;
    return tmp;
}

} // namespace ggml_ve
