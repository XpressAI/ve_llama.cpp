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

#include <memory>

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

    /* N>1 currently CRASHES the kernel chain (likely cache reentry or
     * VEDA arg lifetime issue across queued launches). Reverting to
     * N=1 only until investigated. The dispatch code that loops N times
     * is left in place for when the underlying issue is fixed.
     * Opt-in for testing: GGML_VE_Q4K_N_GT_1=1 to try N>1 path. */
    if (w->type == GGML_TYPE_Q4_K && std::getenv("GGML_VE_Q4K_N_GT_1") != nullptr) {
        if (N < 1) return reject("N<1");
    } else {
        if (N != 1) return reject("N!=1");
    }
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
        const int64_t N = dst->ne[1];
        if (std::getenv("GGML_VE_Q4K_DEBUG")) {
            fprintf(stderr, "[Q4K-DEBUG] name=%s M=%ld K=%ld N=%ld\n",
                    name ? name : "?", (long)M, (long)K, (long)N);
        }
        // The host-side canonical-pack reads from src_blocks via memcpy. If
        // the weight buffer is device-resident (VE_HBM), w->data is an HBM
        // pointer and host memcpy SEGVs. Download to a host bounce buffer
        // first.
        //
        // This happens with GGML_VE_Q4K_N_GT_1: once our backend accepts
        // Q4_K MUL_MATs in batched form, the scheduler decides Q4_K weights
        // are best placed on VE_HBM (the buffer they're consumed on). The
        // cache then needs to read them BACK to host to do the canonical
        // pack + pre-decode. One-time cost per weight at first lookup.
        const void * src_for_cache = w->data;
        std::unique_ptr<uint8_t[]> bounce;
        const int64_t weight_bytes = (int64_t) M * (K / 256) * 144;
        if (w->buffer && !ggml_backend_buffer_is_host(w->buffer)) {
            bounce.reset(new uint8_t[weight_bytes]);
            if (vedaMemcpyDtoH(bounce.get(),
                               (VEDAdeviceptr)(uintptr_t) w->data,
                               weight_bytes) != VEDA_SUCCESS) {
                if (std::getenv("GGML_VE_Q4K_DEBUG")) {
                    fprintf(stderr, "[Q4K-FAIL] DtoH bounce for %s failed\n",
                            name ? name : "?");
                }
                return false;
            }
            src_for_cache = bounce.get();
            if (std::getenv("GGML_VE_Q4K_DEBUG")) {
                fprintf(stderr, "[Q4K-DEVICE-BOUNCE] %s: %ld bytes downloaded for canon pack\n",
                        name ? name : "?", (long) weight_bytes);
            }
        }
        if (!ctx->cache().get_or_upload_q4k_canon(
                name, src_for_cache, (uint64_t) M, (uint64_t) K, &qs_v, &hdr_v)) {
            return false;
        }
        /* N>1 path: loop the matvec column-by-column. y[m,n] is stored at
         * y_hbm + (n*M + m)*4 (ggml's column-major). Each column matvec is
         * independent so we queue them all without sync (VEDA stream
         * serialises). For N=1 this is just one call.
         *
         * KNOWN LIMITATION: enabling GGML_VE_Q4K_N_GT_1 on models that
         * already saturate HBM (e.g. Qwen3.6-27B) WILL crash with OOM.
         * When N>1 is accepted, the scheduler places Q4_K weights on
         * VE_HBM directly (so consumers can read them at HBM bandwidth);
         * our canonical-split cache then DOUBLES the storage (raw + canon)
         * because we still need to canon-pack for the optimised kernel.
         * Fix path: write a direct-dispatch kernel that operates on the
         * standard Q4_K layout (task #58) so the canon cache can be
         * deleted entirely. */
        ok = true;
        for (int64_t n = 0; n < N && ok; n++) {
            VEDAargs args = nullptr;
            if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(q4k_full)")) {
                ok = false; break;
            }
            VEDAdeviceptr y_col = (VEDAdeviceptr)((uintptr_t) y_hbm + n * M * sizeof(float));
            VEDAdeviceptr x_col = (VEDAdeviceptr)((uintptr_t) x_hbm + n * K * sizeof(float));
            vedaArgsSetVPtr(args, 0, y_col);
            vedaArgsSetVPtr(args, 1, qs_v);
            vedaArgsSetVPtr(args, 2, hdr_v);
            vedaArgsSetVPtr(args, 3, x_col);
            vedaArgsSetU64 (args, 4, (uint64_t) M);
            vedaArgsSetU64 (args, 5, (uint64_t) K);
            ok = ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, 1, nullptr),
                            "vedaLaunchKernelEx(q4k_matvec_full_hbm)");
            /* Sync each iteration so VEDA's static x_perm staging buffer in
             * the kernel doesn't get clobbered by the next column. */
            if (ok && N > 1) vedaCtxSynchronize();
        }
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
