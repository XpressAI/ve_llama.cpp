// Phase-5 elementwise ops on F32 tensors, all-HBM resident.
//   GGML_OP_MUL    -> ve_mul_hbm_full      (no broadcast yet)
//   GGML_OP_SCALE  -> ve_scale_hbm_full    (s*x; bias must be 0)
//   GGML_OP_UNARY  -> ve_silu_hbm_full     (GGML_UNARY_OP_SILU only)
//
// Broadcast / mixed-precision / non-HBM paths land alongside the K-quant
// HMEM-transfer fix.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

#include <cstring>

namespace ggml_ve {
namespace ops {

namespace {

// Shape / dtype gating. The HBM residency check happens at compute time;
// `supports_op` is called before tensors are allocated so the buffer is null.
bool same_elemcount_f32(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * dst) {
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(dst)) return false;
    if (ggml_nelements(a) != ggml_nelements(b)) return false;
    if (ggml_nelements(a) != ggml_nelements(dst)) return false;
    return true;
}

bool single_f32(const ggml_tensor * x, const ggml_tensor * dst) {
    if (x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(dst)) return false;
    if (ggml_nelements(x) != ggml_nelements(dst)) return false;
    return true;
}

}  // namespace

// ---------------- MUL ----------------
bool mul_supports(const ggml_tensor * op) {
    static const bool dbg = std::getenv("GGML_VE_DEBUG_MUL") != nullptr;
    auto rej = [&](const char * why) {
        if (dbg) fprintf(stderr, "[VE-MUL-rej] %s : %s a=%s[%lld,%lld,%lld,%lld] b=%s[%lld,%lld,%lld,%lld]\n",
                         op->name[0]?op->name:"?", why,
                         op->src[0]?ggml_type_name(op->src[0]->type):"?",
                         op->src[0]?(long long)op->src[0]->ne[0]:0, op->src[0]?(long long)op->src[0]->ne[1]:0,
                         op->src[0]?(long long)op->src[0]->ne[2]:0, op->src[0]?(long long)op->src[0]->ne[3]:0,
                         op->src[1]?ggml_type_name(op->src[1]->type):"?",
                         op->src[1]?(long long)op->src[1]->ne[0]:0, op->src[1]?(long long)op->src[1]->ne[1]:0,
                         op->src[1]?(long long)op->src[1]->ne[2]:0, op->src[1]?(long long)op->src[1]->ne[3]:0);
        return false;
    };

    if (op->op != GGML_OP_MUL) return false;
    if (op->type != GGML_TYPE_F32) return rej("dst type");
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (!a || !b) return rej("missing src");
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) return rej("src type");
    if (!ggml_is_contiguous(a)) return rej("a not contig");
    if (!ggml_is_contiguous(b)) return rej("b not contig");
    if (!ggml_is_contiguous(op)) return rej("dst not contig");
    const int64_t na = ggml_nelements(a);
    const int64_t nb = ggml_nelements(b);
    const int64_t nd = ggml_nelements(op);
    if (nd != na) return rej("nd!=na");
    if (nb != na) {
        if (nb == 0 || (na % nb) != 0) return rej("nb doesn't divide na");
    }
    return true;
}

bool mul_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!mul_supports(dst)) return false;
    const VEDAdeviceptr ya = ctx->resolve_out(dst);
    const VEDAdeviceptr a0 = ctx->resolve_in(dst->src[0]);
    const VEDAdeviceptr a1 = ctx->resolve_in(dst->src[1]);
    if (ya == 0 || a0 == 0 || a1 == 0) return false;

    const uint64_t na = (uint64_t) ggml_nelements(dst->src[0]);
    const uint64_t nb = (uint64_t) ggml_nelements(dst->src[1]);
    const uint64_t ne00 = (uint64_t) dst->src[0]->ne[0];

    VEDAfunction fn;
    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(mul)")) return false;
    if (na == nb) {
        fn = ctx->fn(K_MUL_HBM_FULL);
        vedaArgsSetVPtr(args, 0, ya);
        vedaArgsSetVPtr(args, 1, a0);
        vedaArgsSetVPtr(args, 2, a1);
        vedaArgsSetU64 (args, 3, na);
    } else {
        fn = ctx->fn(K_MUL_HBM_FULL_BCAST);
        vedaArgsSetVPtr(args, 0, ya);
        vedaArgsSetVPtr(args, 1, a0);
        vedaArgsSetVPtr(args, 2, a1);
        vedaArgsSetU64 (args, 3, na);
        vedaArgsSetU64 (args, 4, nb);
        vedaArgsSetU64 (args, 5, ne00);
    }
    if (fn == 0) { vedaArgsDestroy(args); return false; }

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(mul)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

// ---------------- SCALE ----------------
bool scale_supports(const ggml_tensor * op) {

    if (op->op != GGML_OP_SCALE) return false;
    const ggml_tensor * x = op->src[0];
    if (x == nullptr) return false;

    // ggml SCALE is y = s*x + b; the HBM kernel only does y = s*x, so we
    // only claim cases where bias == 0.
    float params[2] = {0.0f, 0.0f};
    std::memcpy(params, op->op_params, sizeof(params));
    if (params[1] != 0.0f) return false;

    return single_f32(x, op);
}

bool scale_f32(backend_context * ctx, ggml_tensor * dst) {
    VEDAfunction fn = ctx->fn(K_SCALE_HBM_FULL);
    if (fn == 0 || !scale_supports(dst)) return false;
    const VEDAdeviceptr y = ctx->resolve_out(dst);
    const VEDAdeviceptr x = ctx->resolve_in(dst->src[0]);
    if (y == 0 || x == 0) return false;

    float params[2] = {0.0f, 0.0f};
    std::memcpy(params, dst->op_params, sizeof(params));

    uint64_t scale_bits = 0;
    std::memcpy(&scale_bits, &params[0], sizeof(float));   // pass as bits (low 32)

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(scale)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, x);
    vedaArgsSetU64 (args, 2, scale_bits);
    vedaArgsSetU64 (args, 3, (uint64_t) ggml_nelements(dst));

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(scale_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

// ---------------- UNARY family ----------------
// All claim the op only when src+dst are HBM-resident F32 with matching
// element count. The supported unary kinds map to one kernel each:
//
//   SILU      -> ve_silu_hbm_full
//   SIGMOID   -> ve_sigmoid_hbm_full
//   EXP       -> ve_exp_hbm_full
//   NEG       -> ve_neg_hbm_full
//   SQR       -> ve_sqr_hbm_full
//
// SOFTPLUS lives in GGML_OP_SOFTPLUS (not UNARY); see softplus_* below.

namespace {
kernel_id unary_kernel_id(const ggml_tensor * op) {
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_SILU:     return K_SILU_HBM_FULL;
        case GGML_UNARY_OP_SIGMOID:  return K_SIGMOID_HBM_FULL;
        case GGML_UNARY_OP_EXP:      return K_EXP_HBM_FULL;
        case GGML_UNARY_OP_NEG:      return K_NEG_HBM_FULL;
        case GGML_UNARY_OP_SOFTPLUS: return K_SOFTPLUS_HBM_FULL;
        default:                     return K_COUNT;  // unsupported
    }
}
}  // namespace

bool silu_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_UNARY) return false;
    const ggml_tensor * x = op->src[0];
    if (x == nullptr) return false;
    if (unary_kernel_id(op) == K_COUNT) return false;
    return single_f32(x, op);
}

bool silu_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!silu_supports(dst)) return false;
    VEDAfunction fn = ctx->fn(unary_kernel_id(dst));
    if (fn == 0) return false;
    const VEDAdeviceptr y = ctx->resolve_out(dst);
    const VEDAdeviceptr x = ctx->resolve_in(dst->src[0]);
    if (y == 0 || x == 0) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(unary)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, x);
    vedaArgsSetU64 (args, 2, (uint64_t) ggml_nelements(dst));

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(unary_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

// ---------------- SQR (GGML_OP_SQR) ----------------
bool sqr_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_SQR) return false;
    const ggml_tensor * x = op->src[0];
    if (!x) return false;
    return single_f32(x, op);
}

bool sqr_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!sqr_supports(dst)) return false;
    VEDAfunction fn = ctx->fn(K_SQR_HBM_FULL);
    if (fn == 0) return false;
    const VEDAdeviceptr y = ctx->resolve_out(dst);
    const VEDAdeviceptr x = ctx->resolve_in(dst->src[0]);
    if (!y || !x) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(sqr)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, x);
    vedaArgsSetU64 (args, 2, (uint64_t) ggml_nelements(dst));
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(sqr_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

// ---------------- SUB (element-wise, same-shape F32) ----------------
bool sub_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_SUB) return false;
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (!a || !b || !same_elemcount_f32(a, b, op)) return false;
    return true;
}

bool sub_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!sub_supports(dst)) return false;
    VEDAfunction fn = ctx->fn(K_SUB_HBM_FULL);
    if (fn == 0) return false;
    const VEDAdeviceptr y = ctx->resolve_out(dst);
    const VEDAdeviceptr a = ctx->resolve_in(dst->src[0]);
    const VEDAdeviceptr b = ctx->resolve_in(dst->src[1]);
    if (!y || !a || !b) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(sub)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, a);
    vedaArgsSetVPtr(args, 2, b);
    vedaArgsSetU64 (args, 3, (uint64_t) ggml_nelements(dst));
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(sub_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

// ---------------- L2_NORM (per-row normalisation) ----------------
bool l2_norm_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_L2_NORM) return false;
    const ggml_tensor * x = op->src[0];
    if (!x) return false;
    if (x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(op)) return false;
    if (ggml_nelements(x) != ggml_nelements(op)) return false;
    return true;
}

bool l2_norm_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!l2_norm_supports(dst)) return false;
    VEDAfunction fn = ctx->fn(K_L2_NORM_HBM_FULL);
    if (fn == 0) return false;
    const VEDAdeviceptr y = ctx->resolve_out(dst);
    const VEDAdeviceptr x = ctx->resolve_in(dst->src[0]);
    if (!y || !x) return false;

    /* op_params[0] = eps (float). */
    float eps = 1e-6f;
    std::memcpy(&eps, dst->op_params, sizeof(float));
    uint32_t eps_bits;
    std::memcpy(&eps_bits, &eps, sizeof(uint32_t));

    const uint64_t nc = (uint64_t) dst->ne[0];
    const uint64_t nr = (uint64_t) (ggml_nelements(dst) / dst->ne[0]);

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(l2_norm)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, x);
    vedaArgsSetU64 (args, 2, nc);
    vedaArgsSetU64 (args, 3, nr);
    vedaArgsSetU64 (args, 4, (uint64_t) eps_bits);
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(l2_norm_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

// ---------------- GGML_OP_GLU ----------------
// Fused gate activation -- SWIGLU is `out = silu(gate) * up`.
//
// Llama-3 / Llama-3.2 GGUFs emit GGML_OP_GLU instead of separate
// MUL_MAT/SILU/MUL nodes; without claiming it the scheduler splits the
// FFN cgraph at every GLU node (codex finding #3). We currently only
// handle the split-mode SWIGLU variant (gate and up in different
// tensors, both F32, all HBM) -- that's the shape this model uses.
bool glu_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_GLU) return false;
    const ggml_glu_op glu = ggml_get_glu_op(op);
    if (glu != GGML_GLU_OP_SWIGLU) return false;            // others not wired yet
    const ggml_tensor * gate = op->src[0];
    const ggml_tensor * up   = op->src[1];
    if (gate == nullptr || up == nullptr) return false;     // single-source not handled
    if (gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32) return false;
    if (op->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(gate) || !ggml_is_contiguous(up) || !ggml_is_contiguous(op)) return false;
    if (gate->ne[0] != up->ne[0] || gate->ne[1] != up->ne[1]) return false;
    return true;
}

bool glu_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!glu_supports(dst)) return false;
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up   = dst->src[1];

    VEDAfunction fn = ctx->fn(K_SWIGLU_HBM_FULL_OMP);
    if (fn == 0) fn = ctx->fn(K_SWIGLU_HBM_FULL);
    if (fn == 0) return false;

    const VEDAdeviceptr y_v  = ctx->resolve_out(dst);
    const VEDAdeviceptr g_v  = ctx->resolve_in(gate);
    const VEDAdeviceptr u_v  = ctx->resolve_in(up);
    if (y_v == 0 || g_v == 0 || u_v == 0) return false;

    // Kernel signature: ve_swiglu_hbm_full_omp(y, gate, up, ne0, ne1)
    //   nc = elements per row = ne0
    //   nr = number of rows   = nrows(gate)
    const uint64_t nc = (uint64_t) gate->ne[0];
    const uint64_t nr = (uint64_t) ggml_nrows(gate);

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(glu)")) return false;
    vedaArgsSetVPtr(args, 0, y_v);
    vedaArgsSetVPtr(args, 1, g_v);
    vedaArgsSetVPtr(args, 2, u_v);
    vedaArgsSetU64 (args, 3, nc);
    vedaArgsSetU64 (args, 4, nr);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(swiglu_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
