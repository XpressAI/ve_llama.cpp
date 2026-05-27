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
    static const bool dbg = std::getenv("GGML_VE_DEBUG_CPY") != nullptr;
    auto rej = [&](const char * why) {
        if (dbg) fprintf(stderr, "[VE-CPY-rej] %s : %s %s -> %s src=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                         op->name[0]?op->name:"?", why,
                         op->src[0]?ggml_type_name(op->src[0]->type):"?",
                         ggml_type_name(op->type),
                         op->src[0]?(long long)op->src[0]->ne[0]:0, op->src[0]?(long long)op->src[0]->ne[1]:0,
                         op->src[0]?(long long)op->src[0]->ne[2]:0, op->src[0]?(long long)op->src[0]->ne[3]:0,
                         (long long)op->ne[0], (long long)op->ne[1], (long long)op->ne[2], (long long)op->ne[3]);
        return false;
    };

    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            break;
        default:
            return false;
    }
    const ggml_tensor * x = op->src[0];
    if (x == nullptr) return rej("missing src");
    if (ggml_nelements(x) != ggml_nelements(op)) return rej("nelements mismatch");

    if (x->type != op->type) {
        // F32 -> F16 conversion path (used for attention KQ mask).
        if (x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F16
            && ggml_is_contiguous(x) && ggml_is_contiguous(op)) {
            return true;
        }
        return rej("dtype mismatch");
    }

    if (x->type != GGML_TYPE_F32) {
        if (!ggml_is_contiguous(x)) return rej("non-f32 x not contig");
        if (!ggml_is_contiguous(op)) return rej("non-f32 dst not contig");
        if (ggml_nbytes(x) != ggml_nbytes(op)) return rej("non-f32 nbytes mismatch");
        return true;
    }

    if (x->nb[0] != sizeof(float)) return rej("x nb[0] != f32");
    if (op->nb[0] != sizeof(float)) return rej("dst nb[0] != f32");
    return true;
}

bool cpy_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!cpy_supports(dst)) return false;

    const ggml_tensor * src = dst->src[0];

    const VEDAdeviceptr y   = ctx->resolve_out(dst);
    const VEDAdeviceptr x   = ctx->resolve_in(src);
    if (y == 0 || x == 0) return false;

    const bool both_contig = ggml_is_contiguous(src) && ggml_is_contiguous(dst);
    const bool same_bytes  = (ggml_nbytes(src) == ggml_nbytes(dst));
    bool same_ne = true;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] != dst->ne[i]) { same_ne = false; break; }
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(cpy)")) return false;

    VEDAfunction fn = 0;
    if (src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        // F32 -> F16 conversion (KQ mask).
        fn = ctx->fn(K_CPY_F32_TO_F16_HBM);
        if (fn == 0) { vedaArgsDestroy(args); return false; }
        vedaArgsSetVPtr(args, 0, y);
        vedaArgsSetVPtr(args, 1, x);
        vedaArgsSetU64 (args, 2, (uint64_t) ggml_nelements(dst));
    } else if (both_contig && same_bytes && same_ne) {
        fn = ctx->fn(K_COPY_HBM_FULL);
        if (fn == 0) { vedaArgsDestroy(args); return false; }
        vedaArgsSetVPtr(args, 0, y);
        vedaArgsSetVPtr(args, 1, x);
        vedaArgsSetU64 (args, 2, (uint64_t) ggml_nbytes(dst));
    } else if (same_ne) {
        // Strided same-shape, F32 (view-of-view, conv_state mid-update).
        fn = ctx->fn(K_COPY_STRIDED_F32_HBM);
        if (fn == 0) { vedaArgsDestroy(args); return false; }
        vedaArgsSetVPtr(args, 0,  y);
        vedaArgsSetVPtr(args, 1,  x);
        vedaArgsSetU64 (args, 2,  (uint64_t) dst->ne[0]);
        vedaArgsSetU64 (args, 3,  (uint64_t) dst->ne[1]);
        vedaArgsSetU64 (args, 4,  (uint64_t) dst->ne[2]);
        vedaArgsSetU64 (args, 5,  (uint64_t) dst->ne[3]);
        vedaArgsSetU64 (args, 6,  (uint64_t) (src->nb[1] / sizeof(float)));
        vedaArgsSetU64 (args, 7,  (uint64_t) (src->nb[2] / sizeof(float)));
        vedaArgsSetU64 (args, 8,  (uint64_t) (src->nb[3] / sizeof(float)));
        vedaArgsSetU64 (args, 9,  (uint64_t) (dst->nb[1] / sizeof(float)));
        vedaArgsSetU64 (args, 10, (uint64_t) (dst->nb[2] / sizeof(float)));
        vedaArgsSetU64 (args, 11, (uint64_t) (dst->nb[3] / sizeof(float)));
    } else {
        // General reshape-on-copy (conv_state slide back into recurrent cache).
        // Walks both ne[]'s independently and remaps each linear index.
        fn = ctx->fn(K_COPY_BYTES_F32_HBM);
        if (fn == 0) { vedaArgsDestroy(args); return false; }
        vedaArgsSetVPtr(args, 0,  y);
        vedaArgsSetVPtr(args, 1,  x);
        vedaArgsSetU64 (args, 2,  (uint64_t) src->ne[0]);
        vedaArgsSetU64 (args, 3,  (uint64_t) src->ne[1]);
        vedaArgsSetU64 (args, 4,  (uint64_t) src->ne[2]);
        vedaArgsSetU64 (args, 5,  (uint64_t) src->ne[3]);
        vedaArgsSetU64 (args, 6,  (uint64_t) (src->nb[1] / sizeof(float)));
        vedaArgsSetU64 (args, 7,  (uint64_t) (src->nb[2] / sizeof(float)));
        vedaArgsSetU64 (args, 8,  (uint64_t) (src->nb[3] / sizeof(float)));
        vedaArgsSetU64 (args, 9,  (uint64_t) dst->ne[0]);
        vedaArgsSetU64 (args, 10, (uint64_t) dst->ne[1]);
        vedaArgsSetU64 (args, 11, (uint64_t) dst->ne[2]);
        vedaArgsSetU64 (args, 12, (uint64_t) dst->ne[3]);
        vedaArgsSetU64 (args, 13, (uint64_t) (dst->nb[1] / sizeof(float)));
        vedaArgsSetU64 (args, 14, (uint64_t) (dst->nb[2] / sizeof(float)));
        vedaArgsSetU64 (args, 15, (uint64_t) (dst->nb[3] / sizeof(float)));
    }

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(cpy)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
