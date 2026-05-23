// GGML_OP_ADD on F32.
//
// Phase-1 scope: contiguous tensors of equal element count, both srcs and
// dst in VE HBM. Broadcast / mixed-precision / HMEM-resident paths land in
// Phase 5 (elementwise).

#include "../ops.hpp"
#include "../backend_ctx.hpp"
#include "../device.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool add_supports(const ggml_tensor * op) {

    if (op->op != GGML_OP_ADD || op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (a == nullptr || b == nullptr ||
        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b)) {
        return false;
    }
    return ggml_nelements(op) == ggml_nelements(a) &&
           ggml_nelements(op) == ggml_nelements(b);
}

bool add_f32(backend_context * ctx, ggml_tensor * dst) {
    VEDAfunction fn = ctx->fn(K_ADD_HBM_FULL);
    if (fn == 0 || !add_supports(dst)) {
        return false;
    }

    VEDAdeviceptr y = ctx->resolve_out(dst);
    VEDAdeviceptr a = ctx->resolve_in(dst->src[0]);
    VEDAdeviceptr b = ctx->resolve_in(dst->src[1]);
    if (y == 0 || a == 0 || b == 0) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(add)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, a);
    vedaArgsSetVPtr(args, 2, b);
    vedaArgsSetU64 (args, 3, (uint64_t) ggml_nelements(dst));

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(ve_add_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    ctx->ops_hbm()++;
    return true;
}

} // namespace ops
} // namespace ggml_ve
