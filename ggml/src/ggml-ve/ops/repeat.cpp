// GGML_OP_REPEAT: tile src so dst->ne = nr * src->ne (per dim integer ratio).
// Used by Qwen3.5 GDN to broadcast k_in / v_in across the head_v dim.
//
// F32 only, both contiguous (src has nb[0] == sizeof(float)).

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool repeat_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_REPEAT) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * src = op->src[0];
    if (!src || src->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
    // Each dst dim must be an integer multiple of the corresponding src dim.
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] == 0 || (op->ne[i] % src->ne[i]) != 0) return false;
    }
    return true;
}

bool repeat_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!repeat_supports(dst)) return false;
    const ggml_tensor * src = dst->src[0];

    VEDAfunction fn = ctx->fn(K_REPEAT_F32_HBM);
    if (fn == 0) return false;

    const VEDAdeviceptr y = ctx->resolve_out(dst);
    const VEDAdeviceptr x = ctx->resolve_in(src);
    if (!y || !x) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(repeat)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, x);
    vedaArgsSetU64 (args, 2, (uint64_t) src->ne[0]);
    vedaArgsSetU64 (args, 3, (uint64_t) src->ne[1]);
    vedaArgsSetU64 (args, 4, (uint64_t) src->ne[2]);
    vedaArgsSetU64 (args, 5, (uint64_t) src->ne[3]);
    vedaArgsSetU64 (args, 6, (uint64_t) (dst->ne[0] / src->ne[0]));
    vedaArgsSetU64 (args, 7, (uint64_t) (dst->ne[1] / src->ne[1]));
    vedaArgsSetU64 (args, 8, (uint64_t) (dst->ne[2] / src->ne[2]));
    vedaArgsSetU64 (args, 9, (uint64_t) (dst->ne[3] / src->ne[3]));

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(repeat)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
