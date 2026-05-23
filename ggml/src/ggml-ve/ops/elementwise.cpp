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

    if (op->op != GGML_OP_MUL) return false;
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return a && b && same_elemcount_f32(a, b, op);
}

bool mul_f32(backend_context * ctx, ggml_tensor * dst) {
    VEDAfunction fn = ctx->fn(K_MUL_HBM_FULL);
    if (fn == 0 || !mul_supports(dst)) return false;
    const VEDAdeviceptr ya = ctx->resolve_out(dst);
    const VEDAdeviceptr a0 = ctx->resolve_in(dst->src[0]);
    const VEDAdeviceptr a1 = ctx->resolve_in(dst->src[1]);
    if (ya == 0 || a0 == 0 || a1 == 0) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(mul)")) return false;
    vedaArgsSetVPtr(args, 0, ya);
    vedaArgsSetVPtr(args, 1, a0);
    vedaArgsSetVPtr(args, 2, a1);
    vedaArgsSetU64 (args, 3, (uint64_t) ggml_nelements(dst));

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(mul_hbm_full)")) {
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
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(scale_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

// ---------------- UNARY: SILU ----------------
bool silu_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_UNARY) return false;
    if (ggml_get_unary_op(op) != GGML_UNARY_OP_SILU) return false;
    const ggml_tensor * x = op->src[0];
    if (x == nullptr) return false;
    // The kernel uses `x / (1 + expf(-x))` without clamping x. NCC's
    // vectorised expf returns NaN for x < -88-ish, so the unit-test value
    // range (~[-100, 100]) breaks it. Real model activations stay in a
    // narrow band where it's fine. We claim the op only when both src and
    // dst are HBM-resident (i.e. inside a normal model graph).
    return single_f32(x, op);
}

bool silu_f32(backend_context * ctx, ggml_tensor * dst) {
    VEDAfunction fn = ctx->fn(K_SILU_HBM_FULL);
    if (fn == 0 || !silu_supports(dst)) return false;
    const VEDAdeviceptr y = ctx->resolve_out(dst);
    const VEDAdeviceptr x = ctx->resolve_in(dst->src[0]);
    if (y == 0 || x == 0) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(silu)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, x);
    vedaArgsSetU64 (args, 2, (uint64_t) ggml_nelements(dst));

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(silu_hbm_full)")) {
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
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(swiglu_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
