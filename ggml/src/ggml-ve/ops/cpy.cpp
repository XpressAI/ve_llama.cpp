// Phase-5 GGML_OP_CPY / GGML_OP_CONT / GGML_OP_DUP — contiguous, same dtype,
// all HBM. The kernel is a thin device-side memcpy: ve_copy_hbm_full.
//
// Non-contiguous strided / mixed-precision copies happen via the dedicated
// ve_cpy_strided_*_hbm and ve_cpy_*_*_hmem kernels — those need HMEM
// staging for the precision-conversion variants, which is blocked on the
// K-quant HMEM transfer fix.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool cpy_supports(const ggml_tensor * op) {

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            break;
        default:
            return false;
    }
    const ggml_tensor * x = op->src[0];
    if (x == nullptr) return false;
    // Same dtype, both contiguous, same total nbytes — i.e. a straight memcpy.
    if (x->type != op->type) return false;
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(op)) return false;
    if (ggml_nbytes(x) != ggml_nbytes(op)) return false;
    return true;
}

bool cpy_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!cpy_supports(dst)) return false;
    if (!tensor_is_hbm(dst->src[0]) || !tensor_is_hbm(dst)) return false;

    VEDAfunction fn = ctx->fn(K_COPY_HBM_FULL);
    if (fn == 0) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(cpy)")) return false;
    vedaArgsSetVPtr(args, 0, tensor_hbm_ptr(dst));
    vedaArgsSetVPtr(args, 1, tensor_hbm_ptr(dst->src[0]));
    vedaArgsSetU64 (args, 2, (uint64_t) ggml_nbytes(dst));

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(copy_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
