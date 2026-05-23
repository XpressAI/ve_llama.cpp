// GGML_OP_MUL_MAT on quantized weight tensors.
//
// Phase-3 scope: Q8_0 (full-HBM fused kernel) and Q2_K (HBM weights, HMEM I/O
// staged through the pool). N=1 only. F32 input, F32 output.
//
// Q4_K / Q5_K / Q6_K / MXFP4 land in a follow-up — libve_sgemv.so doesn't
// ship genuinely all-HBM kernels for them today, and the
// dequantise-to-FP32-into-HBM strategy belongs alongside the HBM weight
// cache work (Phase 5 / 7 territory).

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

namespace {

bool is_supported_quant_type(ggml_type t) {
    // Q8_0 only: it has a genuine all-HBM fused kernel.
    // Q2_K's `ve_q2k_bf16_matvec_hbm` actually expects pre-dequantised BF16
    // weights despite its name; running it on raw Q2_K bytes yields NaN.
    // Re-enable via the dequantise-to-HBM strategy (the 147× legacy win).
    return t == GGML_TYPE_Q8_0;
}

// Block size for a given quant — used to validate K is aligned.
int blk_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q2_K: return 256;
        default:             return 0;
    }
}

bool launch_matvec_q8_0_full_hbm(VEDAfunction fn,
                                 VEDAdeviceptr y, VEDAdeviceptr W, VEDAdeviceptr x,
                                 uint64_t M, uint64_t K) {
    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(q8_0)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, W);
    vedaArgsSetVPtr(args, 2, x);
    vedaArgsSetU64 (args, 3, M);
    vedaArgsSetU64 (args, 4, K);

    uint64_t result = 0;
    return ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                      "vedaLaunchKernelEx(q8_0_fused_matvec_hbm_full)");
}

// HBM-weights, HMEM I/O for K-quants. Stages x and y through the HMEM pool.
// We currently sync between kernel-launch and the y→HBM copy because there's
// no HMEM→HBM transfer in the deferred-sync queue yet (TODO Phase 7-ish).
bool launch_kquant_hbm_weights_hmem_io(backend_context * ctx,
                                       VEDAfunction fn,
                                       VEDAdeviceptr y_hbm,
                                       VEDAdeviceptr W_hbm,
                                       VEDAdeviceptr x_hbm,
                                       uint64_t M, uint64_t K) {
    const size_t x_bytes = K * sizeof(float);
    const size_t y_bytes = M * sizeof(float);

    VEDAhmemptr x_hmem = ctx->pool().acquire(x_bytes);
    VEDAhmemptr y_hmem = ctx->pool().acquire(y_bytes);
    if (x_hmem == 0 || y_hmem == 0) {
        if (x_hmem) ctx->pool().release(x_hmem);
        if (y_hmem) ctx->pool().release(y_hmem);
        return false;
    }

    // vedaHMemcpy{X,D}toX takes the *tagged* VEDAhmemptr cast to void* — NOT
    // the host-converted address from vedaHMemPtr. See veda/tests/FT/
    // FT_VEDA_mem_HMem_01.cpp for the canonical usage.
    if (!ggml_ve_ok(vedaHMemcpyDtoX(reinterpret_cast<void *>(x_hmem), x_hbm, x_bytes),
                    "vedaHMemcpyDtoX (kquant x: HBM->HMEM)")) {
        ctx->pool().release(x_hmem);
        ctx->pool().release(y_hmem);
        return false;
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(kquant)")) {
        ctx->pool().release(x_hmem);
        ctx->pool().release(y_hmem);
        return false;
    }
    vedaArgsSetHMEM(args, 0, y_hmem);
    vedaArgsSetVPtr(args, 1, W_hbm);
    vedaArgsSetHMEM(args, 2, x_hmem);
    vedaArgsSetU64 (args, 3, M);
    vedaArgsSetU64 (args, 4, K);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(kquant matvec)")) {
        ctx->pool().release(x_hmem);
        ctx->pool().release(y_hmem);
        return false;
    }

    // Need the kernel finished before we read y_hmem. This is a local sync
    // (kills the deferred-sync win for this op); the proper fix is to extend
    // the deferred-sync queue with HMEM→HBM copies. Tracked as follow-up.
    vedaCtxSynchronize();

    if (!ggml_ve_ok(vedaHMemcpyXtoD(y_hbm, reinterpret_cast<void *>(y_hmem), y_bytes),
                    "vedaHMemcpyXtoD (kquant y: HMEM->HBM)")) {
        ctx->pool().release(x_hmem);
        ctx->pool().release(y_hmem);
        return false;
    }
    vedaCtxSynchronize();

    ctx->pool().release(x_hmem);
    ctx->pool().release(y_hmem);
    return true;
}

}  // namespace

bool mul_mat_q_supports(const ggml_tensor * op) {

    if (op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w == nullptr || x == nullptr) return false;
    if (x->type != GGML_TYPE_F32) return false;
    if (!is_supported_quant_type(w->type)) return false;

    const int64_t K = w->ne[0];
    const int64_t M = w->ne[1];
    const int64_t N = x->ne[1];

    if (N != 1) return false;                                 // matvec only
    if (op->ne[0] != M || op->ne[1] != N) return false;
    if (w->ne[0]  != x->ne[0]) return false;                  // K must match
    if (op->ne[2] != 1 || op->ne[3] != 1) return false;
    if (w->ne[2]  != 1 || w->ne[3]  != 1) return false;
    if (x->ne[2]  != 1 || x->ne[3]  != 1) return false;
    if (!ggml_is_contiguous(w))  return false;
    if (!ggml_is_contiguous(x))  return false;
    if (!ggml_is_contiguous(op)) return false;

    const int b = blk_size(w->type);
    if (b == 0 || (K % b) != 0) return false;
    if (M <= 0 || K <= 0) return false;
    return true;
}

bool mul_mat_q(backend_context * ctx, ggml_tensor * dst) {
    if (!mul_mat_q_supports(dst)) return false;

    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    const VEDAdeviceptr y_hbm = ctx->resolve_out(dst);
    const VEDAdeviceptr w_hbm = ctx->resolve_in(w);
    const VEDAdeviceptr x_hbm = ctx->resolve_in(x);
    if (y_hbm == 0 || w_hbm == 0 || x_hbm == 0) return false;

    const uint64_t M = (uint64_t) w->ne[1];
    const uint64_t K = (uint64_t) w->ne[0];

    bool ok = false;
    if (w->type == GGML_TYPE_Q8_0) {
        VEDAfunction fn = ctx->fn(K_Q8_0_FUSED_MATVEC_HBM_FULL);
        if (fn == 0) return false;
        ok = launch_matvec_q8_0_full_hbm(fn, y_hbm, w_hbm, x_hbm, M, K);
    } else if (w->type == GGML_TYPE_Q2_K) {
        VEDAfunction fn = ctx->fn(K_Q2K_BF16_MATVEC_HBM);
        if (fn == 0) return false;
        ok = launch_kquant_hbm_weights_hmem_io(ctx, fn, y_hbm, w_hbm, x_hbm, M, K);
    }

    if (ok) {
        ctx->mark_sync_pending();
        ctx->ops_mul_mat()++;
        ctx->ops_hbm()++;
    }
    return ok;
}

}  // namespace ops
}  // namespace ggml_ve
