// GGML_OP_CONCAT: dst = concat(src0, src1) along axis = op_params[0].
// Used by Qwen3.5 (conv_input = concat conv_states + qkv_mixed) and many
// other archs.
//
// F32 only, both sources + dst contiguous. dim ∈ [0, 3].

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool concat_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_CONCAT) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (!a || !b) return false;
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(op)) {
        return false;
    }
    const int32_t dim = ggml_get_op_params_i32(op, 0);
    if (dim < 0 || dim >= GGML_MAX_DIMS) return false;
    // dst dims = a dims, with a[dim] + b[dim] along the concat axis.
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const int64_t want = (i == dim) ? a->ne[i] + b->ne[i] : a->ne[i];
        if (op->ne[i] != want) return false;
        if (i != dim && a->ne[i] != b->ne[i]) return false;
    }
    return true;
}

bool concat_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!concat_supports(dst)) return false;
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];

    VEDAfunction fn = ctx->fn(K_CONCAT_F32_HBM);
    if (fn == 0) return false;

    const VEDAdeviceptr y_v = ctx->resolve_out(dst);
    const VEDAdeviceptr a_v = ctx->resolve_in(a);
    const VEDAdeviceptr b_v = ctx->resolve_in(b);
    if (!y_v || !a_v || !b_v) return false;

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(concat)")) return false;
    vedaArgsSetVPtr(args, 0, y_v);
    vedaArgsSetVPtr(args, 1, a_v);
    vedaArgsSetVPtr(args, 2, b_v);
    vedaArgsSetU64 (args, 3, (uint64_t) dim);
    vedaArgsSetU64 (args, 4, (uint64_t) a->ne[0]);
    vedaArgsSetU64 (args, 5, (uint64_t) a->ne[1]);
    vedaArgsSetU64 (args, 6, (uint64_t) a->ne[2]);
    vedaArgsSetU64 (args, 7, (uint64_t) a->ne[3]);
    vedaArgsSetU64 (args, 8, (uint64_t) b->ne[0]);
    vedaArgsSetU64 (args, 9, (uint64_t) b->ne[1]);
    vedaArgsSetU64 (args, 10, (uint64_t) b->ne[2]);
    vedaArgsSetU64 (args, 11, (uint64_t) b->ne[3]);

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(concat)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
