// Phase-5 GGML_OP_RMS_NORM on F32 tensors, all-HBM resident.
//
// The HBM kernels in libve_sgemv.so:
//   ve_rms_norm_hbm_simple(y, x, n, eps)       single row
//   ve_rms_norm_hbm_omp   (y, x, ne00, ne01, eps)   multi-row (ne01 batches)
//
// ggml's RMS_NORM does:  y[i] = x[i] / sqrt(mean(x*x) + eps), per row of ne00.
// The multiplication with the learned weight is a *separate* GGML_OP_MUL,
// already covered by ops/elementwise.cpp.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

#include <cstring>

namespace ggml_ve {
namespace ops {

bool rms_norm_supports(const ggml_tensor * op) {
    static const bool dbg = std::getenv("GGML_VE_DEBUG_RMS_NORM") != nullptr;
    auto rej = [&](const char * why) {
        if (dbg) fprintf(stderr, "[VE-RMS-rej] %s : %s x=%s[%lld,%lld,%lld,%lld] op=%s[%lld,%lld,%lld,%lld]\n",
                         op->name[0]?op->name:"?", why,
                         op->src[0]?ggml_type_name(op->src[0]->type):"?",
                         op->src[0]?(long long)op->src[0]->ne[0]:0, op->src[0]?(long long)op->src[0]->ne[1]:0,
                         op->src[0]?(long long)op->src[0]->ne[2]:0, op->src[0]?(long long)op->src[0]->ne[3]:0,
                         ggml_type_name(op->type),
                         (long long)op->ne[0], (long long)op->ne[1], (long long)op->ne[2], (long long)op->ne[3]);
        return false;
    };

    if (op->op != GGML_OP_RMS_NORM || op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * x = op->src[0];
    if (x == nullptr || x->type != GGML_TYPE_F32) return rej("x type");
    if (!ggml_is_contiguous(x)) return rej("x not contig");
    if (!ggml_is_contiguous(op)) return rej("dst not contig");
    if (x->ne[0] != op->ne[0] || ggml_nelements(x) != ggml_nelements(op)) return rej("shape mismatch");
    return true;
}

bool rms_norm_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!rms_norm_supports(dst)) return false;
    const ggml_tensor * x = dst->src[0];

    VEDAdeviceptr y_vptr = ctx->resolve_out(dst);
    VEDAdeviceptr x_vptr = ctx->resolve_in(x);
    if (y_vptr == 0 || x_vptr == 0) return false;

    const int64_t ne00   = x->ne[0];                    // cols to normalise over
    const int64_t n_rows = ggml_nelements(x) / ne00;    // total rows

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(float));
    uint64_t eps_bits = 0;
    std::memcpy(&eps_bits, &eps, sizeof(float));

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(rms_norm)")) return false;

    VEDAfunction fn = 0;
    if (n_rows == 1) {
        fn = ctx->fn(K_RMS_NORM_HBM_SIMPLE);
        if (fn == 0) {
            vedaArgsDestroy(args);
            return false;
        }
        vedaArgsSetVPtr(args, 0, y_vptr);
        vedaArgsSetVPtr(args, 1, x_vptr);
        vedaArgsSetU64 (args, 2, (uint64_t) ne00);
        vedaArgsSetU64 (args, 3, eps_bits);
    } else {
        fn = ctx->fn(K_RMS_NORM_HBM_OMP);
        if (fn == 0) {
            vedaArgsDestroy(args);
            return false;
        }
        vedaArgsSetVPtr(args, 0, y_vptr);
        vedaArgsSetVPtr(args, 1, x_vptr);
        vedaArgsSetU64 (args, 2, (uint64_t) ne00);
        vedaArgsSetU64 (args, 3, (uint64_t) n_rows);
        vedaArgsSetU64 (args, 4, eps_bits);
    }

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(rms_norm_hbm)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
