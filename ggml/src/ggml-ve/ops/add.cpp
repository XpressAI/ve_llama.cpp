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
    const int64_t na = ggml_nelements(a);
    const int64_t nb = ggml_nelements(b);
    const int64_t nd = ggml_nelements(op);
    if (nd != na) return false;                  // output matches src0 shape
    if (nb == na) return true;                   // same-shape path
    // Broadcast path: src1 has fewer elements; the broadcast kernel
    // distinguishes "inner dim match" (nb == ne00) vs "outer broadcast"
    // (nb divides na and nb < na). Both require na divisible by nb.
    if (nb == 0 || (na % nb) != 0) return false;
    return true;
}

bool add_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!add_supports(dst)) return false;

    VEDAdeviceptr y = ctx->resolve_out(dst);
    VEDAdeviceptr a = ctx->resolve_in(dst->src[0]);
    VEDAdeviceptr b = ctx->resolve_in(dst->src[1]);
    if (y == 0 || a == 0 || b == 0) return false;

    const uint64_t na = (uint64_t) ggml_nelements(dst->src[0]);
    const uint64_t nb = (uint64_t) ggml_nelements(dst->src[1]);
    const uint64_t ne00 = (uint64_t) dst->src[0]->ne[0];

    VEDAfunction fn;
    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(add)")) return false;

    if (na == nb) {
        fn = ctx->fn(K_ADD_HBM_FULL);
        vedaArgsSetVPtr(args, 0, y);
        vedaArgsSetVPtr(args, 1, a);
        vedaArgsSetVPtr(args, 2, b);
        vedaArgsSetU64 (args, 3, na);
    } else {
        fn = ctx->fn(K_ADD_HBM_FULL_BROADCAST);
        vedaArgsSetVPtr(args, 0, y);
        vedaArgsSetVPtr(args, 1, a);
        vedaArgsSetVPtr(args, 2, b);
        vedaArgsSetU64 (args, 3, na);
        vedaArgsSetU64 (args, 4, nb);
        vedaArgsSetU64 (args, 5, ne00);
    }
    if (fn == 0) { vedaArgsDestroy(args); return false; }

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(ve_add)")) {
        return false;
    }
    ctx->mark_sync_pending();
    ctx->ops_hbm()++;
    return true;
}

} // namespace ops
} // namespace ggml_ve
