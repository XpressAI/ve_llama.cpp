// Phase-5 GGML_OP_GET_ROWS — embedding-table lookup. The src0 (embeddings)
// is BF16 or F32 in HBM; src1 (indices) is i32 in HBM. We stage the indices
// to HMEM since the kernel reads them as a host pointer.
//
// The HBM→HBM kernel variant means we don't need an HMEM staging for the
// output — it writes directly to dst's HBM buffer.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool get_rows_supports(const ggml_tensor * op) {

    if (op->op != GGML_OP_GET_ROWS) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * src = op->src[0];  // embeddings
    const ggml_tensor * idx = op->src[1];  // i32 indices
    if (src == nullptr || idx == nullptr) return false;
    if (src->type != GGML_TYPE_BF16 && src->type != GGML_TYPE_F32) return false;
    if (idx->type != GGML_TYPE_I32) return false;
    // Need contiguous indices and output.
    if (!ggml_is_contiguous(idx) || !ggml_is_contiguous(op)) return false;
    // 2D outputs only for now (cols=ne[0], rows=ne[1]).
    if (op->ne[2] != 1 || op->ne[3] != 1) return false;
    return true;
}

bool get_rows(backend_context * ctx, ggml_tensor * dst) {
    if (!get_rows_supports(dst)) return false;
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * idx = dst->src[1];
    if (!tensor_is_hbm(src) || !tensor_is_hbm(idx) || !tensor_is_hbm(dst)) return false;

    const uint64_t nc      = (uint64_t) dst->ne[0];      // cols / embedding dim
    const uint64_t nr      = (uint64_t) ggml_nelements(idx);  // n indices
    const uint64_t nb_src  = (uint64_t) src->nb[1];
    const uint64_t nb_dst  = (uint64_t) dst->nb[1];

    // Stage indices HBM → HMEM (small buffer, but kernel wants HMEM).
    const size_t idx_bytes = nr * sizeof(int32_t);
    VEDAhmemptr idx_hmem = ctx->pool().acquire(idx_bytes);
    if (idx_hmem == 0) return false;

    if (!ggml_ve_ok(vedaHMemcpyDtoX(reinterpret_cast<void *>(idx_hmem),
                                     tensor_hbm_ptr(idx), idx_bytes),
                    "vedaHMemcpyDtoX (get_rows idx: HBM->HMEM)")) {
        ctx->pool().release(idx_hmem);
        return false;
    }

    kernel_id kid = (src->type == GGML_TYPE_BF16)
        ? K_GET_ROWS_BF16_F32_HBM_HBM
        : K_GET_ROWS_F32_F32_HBM_HBM;
    VEDAfunction fn = ctx->fn(kid);
    if (fn == 0) {
        ctx->pool().release(idx_hmem);
        return false;
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(get_rows)")) {
        ctx->pool().release(idx_hmem);
        return false;
    }
    vedaArgsSetVPtr(args, 0, tensor_hbm_ptr(dst));
    vedaArgsSetVPtr(args, 1, tensor_hbm_ptr(src));
    vedaArgsSetHMEM(args, 2, idx_hmem);
    vedaArgsSetU64 (args, 3, nc);
    vedaArgsSetU64 (args, 4, nr);
    vedaArgsSetU64 (args, 5, nb_src);
    vedaArgsSetU64 (args, 6, nb_dst);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(get_rows_hbm_hbm)")) {
        ctx->pool().release(idx_hmem);
        return false;
    }

    // The kernel reads the index HMEM buffer in-flight, so defer release
    // until after the next flush.
    ctx->enqueue_input(idx_hmem);
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
