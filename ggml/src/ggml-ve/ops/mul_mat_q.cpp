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
    // Q4_K: canonical-nibble + qs/hdr-split HBM kernel. Microbench: 94x
    //       speedup over the old NCC scalar kernel, ~10x off BF16 speed.
    //       Requires GGML_VE_QUANT_SAFE_MODE=1 for correct output on
    //       real models (see resolve_in_slow KNOWN-BUG in backend_ctx.cpp,
    //       and commit 7ddc2fa26). Opt-out via GGML_VE_NO_Q4K=1.
    if (t == GGML_TYPE_Q8_0) return true;
    if (t == GGML_TYPE_Q4_K) return std::getenv("GGML_VE_NO_Q4K") == nullptr;
    return false;
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

    auto reject = [&](const char *why) {
        if (std::getenv("GGML_VE_Q4K_DEBUG")) {
            fprintf(stderr, "[Q4K-REJECT] %s op=%s w=%s w_ne=[%ld,%ld,%ld,%ld] x_ne=[%ld,%ld,%ld,%ld] dst_ne=[%ld,%ld,%ld,%ld]\n",
                    why, op->name, w->name,
                    (long)w->ne[0], (long)w->ne[1], (long)w->ne[2], (long)w->ne[3],
                    (long)x->ne[0], (long)x->ne[1], (long)x->ne[2], (long)x->ne[3],
                    (long)op->ne[0], (long)op->ne[1], (long)op->ne[2], (long)op->ne[3]);
        }
        return false;
    };

    if (N != 1) return reject("N!=1");
    if (op->ne[0] != M || op->ne[1] != N) return reject("op_ne mismatch");
    if (w->ne[0]  != x->ne[0]) return reject("K mismatch");
    if (op->ne[2] != 1 || op->ne[3] != 1) return reject("op_ne23");
    if (w->ne[2]  != 1 || w->ne[3]  != 1) return reject("w_ne23");
    if (x->ne[2]  != 1 || x->ne[3]  != 1) return reject("x_ne23");
    if (!ggml_is_contiguous(w))  return reject("w not contig");
    if (!ggml_is_contiguous(x))  return reject("x not contig");
    if (!ggml_is_contiguous(op)) return reject("op not contig");

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
    if (std::getenv("GGML_VE_Q4K_DEBUG")) {
        fprintf(stderr, "[mul_mat_q ENTRY] dst=%s w=%s/%s\n",
            dst->name, dst->src[0]?dst->src[0]->name:"?",
            dst->src[0]?ggml_type_name(dst->src[0]->type):"?");
        fflush(stderr);
    }
    if (!mul_mat_q_supports(dst)) {
        if (std::getenv("GGML_VE_Q4K_DEBUG")) {
            fprintf(stderr, "[mul_mat_q REJECTED]\n");
            fflush(stderr);
        }
        return false;
    }

    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    const VEDAdeviceptr y_hbm = ctx->resolve_out(dst);
    const VEDAdeviceptr x_hbm = ctx->resolve_in(x);
    // For Q4_K we use the canonical-split cache below (qs+hdr) instead of
    // the raw-weight cache, so skip resolve_in(w) -- otherwise the same
    // weight gets uploaded TWICE (raw + canonical-split), doubling the
    // HBM footprint and OOMing 27B-class models around layer 42.
    const VEDAdeviceptr w_hbm = (w->type == GGML_TYPE_Q4_K) ? 0 : ctx->resolve_in(w);
    if (y_hbm == 0 || x_hbm == 0) return false;
    if (w->type != GGML_TYPE_Q4_K && w_hbm == 0) return false;

    const uint64_t M = (uint64_t) w->ne[1];
    const uint64_t K = (uint64_t) w->ne[0];

    bool ok = false;
    if (w->type == GGML_TYPE_Q8_0) {
        VEDAfunction fn = ctx->fn(K_Q8_0_FUSED_MATVEC_HBM_FULL);
        if (fn == 0) return false;
        ok = launch_matvec_q8_0_full_hbm(fn, y_hbm, w_hbm, x_hbm, M, K);
    } else if (w->type == GGML_TYPE_Q4_K) {
        // Canonical-split kernel: takes y, qs_split, hdr_split, x, M, K.
        VEDAfunction fn = ctx->fn(K_Q4K_MATVEC_FULL_HBM);
        if (fn == 0) return false;
        VEDAdeviceptr qs_v = 0, hdr_v = 0;
        const char * name = (w->name && w->name[0]) ? w->name : nullptr;
        if (std::getenv("GGML_VE_Q4K_DEBUG")) {
            fprintf(stderr, "[Q4K-DEBUG] name=%s M=%ld K=%ld y_hbm=%lx x_hbm=%lx ...",
                    name ? name : "?", (long)M, (long)K,
                    (long)y_hbm, (long)x_hbm);
            fflush(stderr);
        }
        if (!ctx->cache().get_or_upload_q4k_canon(
                name, w->data, (uint64_t) M, (uint64_t) K, &qs_v, &hdr_v)) {
            if (std::getenv("GGML_VE_Q4K_DEBUG")) fprintf(stderr, " UPLOAD FAIL\n");
            return false;
        }
        if (std::getenv("GGML_VE_Q4K_DEBUG")) {
            fprintf(stderr, " qs_v=%lx hdr_v=%lx ", (long)qs_v, (long)hdr_v);
            fflush(stderr);
        }
        VEDAargs args = nullptr;
        if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(q4k_full)")) return false;
        vedaArgsSetVPtr(args, 0, y_hbm);
        vedaArgsSetVPtr(args, 1, qs_v);
        vedaArgsSetVPtr(args, 2, hdr_v);
        vedaArgsSetVPtr(args, 3, x_hbm);
        vedaArgsSetU64 (args, 4, (uint64_t) M);
        vedaArgsSetU64 (args, 5, (uint64_t) K);
        ok = ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, 1, nullptr),
                        "vedaLaunchKernelEx(q4k_matvec_full_hbm)");
        if (ok && std::getenv("GGML_VE_Q4K_SYNC")) vedaCtxSynchronize();
        if (std::getenv("GGML_VE_Q4K_DEBUG")) fprintf(stderr, " OK\n");
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
