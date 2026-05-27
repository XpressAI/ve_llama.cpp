// GGML_OP_SUM_ROWS: dst[0, i1, i2, i3] = sum over i0 of src[i0, i1, i2, i3]
// (innermost-axis reduction). Used by Qwen3.5 GDN.
//
// All F32, src can be a strided view, dst is contiguous along ne[0]=1.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool sum_rows_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_SUM_ROWS) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * src = op->src[0];
    if (!src || src->type != GGML_TYPE_F32) return false;
    if (src->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) return false;
    if (op->ne[0] != 1) return false;
    if (op->ne[1] != src->ne[1] || op->ne[2] != src->ne[2] || op->ne[3] != src->ne[3]) return false;
    return true;
}

bool sum_rows_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!sum_rows_supports(dst)) return false;
    const ggml_tensor * src = dst->src[0];

    VEDAfunction fn = ctx->fn(K_SUM_ROWS_F32_HBM);
    if (fn == 0) return false;

    const VEDAdeviceptr y = ctx->resolve_out(dst);
    const VEDAdeviceptr x = ctx->resolve_in(src);
    if (!y || !x) return false;

    auto f = [](size_t b) { return (uint64_t)(b / sizeof(float)); };

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(sum_rows)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, x);
    vedaArgsSetU64 (args, 2, (uint64_t) src->ne[0]);
    vedaArgsSetU64 (args, 3, (uint64_t) src->ne[1]);
    vedaArgsSetU64 (args, 4, (uint64_t) src->ne[2]);
    vedaArgsSetU64 (args, 5, (uint64_t) src->ne[3]);
    vedaArgsSetU64 (args, 6, f(src->nb[1]));
    vedaArgsSetU64 (args, 7, f(src->nb[2]));
    vedaArgsSetU64 (args, 8, f(src->nb[3]));
    vedaArgsSetU64 (args, 9, f(dst->nb[1]));
    vedaArgsSetU64 (args, 10, f(dst->nb[2]));
    vedaArgsSetU64 (args, 11, f(dst->nb[3]));

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(sum_rows)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
