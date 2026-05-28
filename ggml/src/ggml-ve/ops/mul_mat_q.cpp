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
    // Q8_0: all-HBM fused kernel.
    // Q4_K: direct matvec exists (ve_q4k_matvec_f32_hbm) and passes all
    //       standalone correctness shapes bit-exact. BUT enabling it on
    //       real Q-quant models triggers a separate VE↔CPU intermediate
    //       transfer bug that produces garbage regardless of whether the
    //       Q4_K kernel runs (verified: a dummy kernel body that just
    //       writes y[i] = constant produces the same broken output as
    //       my real kernel; even disabling the kernel entirely and
    //       routing all Q4_K to CPU produces the same garbage).
    //       Until that boundary bug is found, leave Q4_K opt-in.
    if (std::getenv("GGML_VE_Q4K") == nullptr) return t == GGML_TYPE_Q8_0;
    return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_K;
}

// Block size for a given quant — used to validate K is aligned.
int blk_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q2_K: return 256;
        case GGML_TYPE_Q4_K: return 256;
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
    return ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                      "vedaLaunchKernelEx(q8_0_fused_matvec_hbm_full)");
}

// HBM-weights, temp-HBM staging for K-quant I/O (HMEM is for inter-VE / MPI
// only). The kernel reads x from HBM and writes y to HBM; we then D->D copy
// y to the caller's HBM output.
bool launch_kquant_hbm_weights_hmem_io(backend_context * ctx,
                                       VEDAfunction fn,
                                       VEDAdeviceptr y_hbm,
                                       VEDAdeviceptr W_hbm,
                                       VEDAdeviceptr x_hbm,
                                       uint64_t M, uint64_t K) {
    const size_t x_bytes = K * sizeof(float);
    const size_t y_bytes = M * sizeof(float);
    if (x_bytes == 0 || y_bytes == 0) return true;

    VEDAdeviceptr x_tmp = 0, y_tmp = 0;
    if (vedaMemAllocAsync(&x_tmp, x_bytes, 0) != VEDA_SUCCESS) return false;
    if (vedaMemAllocAsync(&y_tmp, y_bytes, 0) != VEDA_SUCCESS) {
        vedaMemFreeAsync(x_tmp, 0);
        return false;
    }
    ctx->enqueue_hbm_free(x_tmp);
    ctx->enqueue_hbm_free(y_tmp);

    if (!ggml_ve_ok(vedaMemcpyDtoDAsync(x_tmp, x_hbm, x_bytes, 0),
                    "vedaMemcpyDtoDAsync (kquant x: HBM->HBM)")) {
        return false;
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(kquant)")) return false;
    vedaArgsSetVPtr(args, 0, y_tmp);
    vedaArgsSetVPtr(args, 1, W_hbm);
    vedaArgsSetVPtr(args, 2, x_tmp);
    vedaArgsSetU64 (args, 3, M);
    vedaArgsSetU64 (args, 4, K);

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(kquant matvec)")) {
        return false;
    }

    if (!ggml_ve_ok(vedaMemcpyDtoDAsync(y_hbm, y_tmp, y_bytes, 0),
                    "vedaMemcpyDtoDAsync (kquant y: HBM->HBM)")) {
        return false;
    }
    ctx->mark_sync_pending();
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

    // Reject CPU_REPACK weights. llama.cpp's CPU backend repacks K-quant
    // weights into SIMD-friendly interleaved layouts (e.g. block_q4_Kx8 —
    // 8 blocks interleaved for AVX). Our kernels expect the canonical
    // block_q4_K layout; running them on repacked bytes gives garbage.
    // Falling back lets the CPU backend handle these (which understands
    // its own repacked layout). To use VE, either disable repack at build
    // (-DGGML_CPU_REPACK=OFF) or convert the GGUF.
    if (w->buffer) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(w->buffer);
        if (buft) {
            const char * bn = ggml_backend_buft_name(buft);
            if (bn && std::strcmp(bn, "CPU_REPACK") == 0) return false;
        }
    }

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
    } else if (w->type == GGML_TYPE_Q4_K) {
        VEDAfunction fn = ctx->fn(K_Q4K_MATVEC_F32_HBM);
        if (fn == 0) return false;
        // Same all-HBM launch signature as Q8_0 (y, W, x, M, K).
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
