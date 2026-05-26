// Phase-6 MoE ops:
//   GGML_OP_MUL_MAT_ID  -> ve_mul_mat_id_bf16_f32_hbm_full
//   GGML_OP_ADD_ID      -> ve_add_id_f32_hbm_full
//   GGML_OP_ARGSORT     -> ve_argsort_f32_omp_hmem  (HMEM-staged)
//
// MUL_MAT_ID semantics (ggml):
//   as   = src0 [K, M, n_experts]    BF16 weight matrices
//   b    = src1 [K, n_expert_used, n_tokens]  F32 input
//   ids  = src2 [n_expert_used, n_tokens]     I32 expert IDs
//   dst  =      [M, n_expert_used, n_tokens]  F32
//
// ADD_ID semantics:
//   src0 [ne0, n_expert_used, n_tokens]  F32 input
//   src1 [ne0, n_experts]                F32 bias table
//   ids  [n_expert_used, n_tokens]       I32 expert IDs
//   dst  = src0 + src1[ids[i1,i2]]
//
// ARGSORT: dst[I32, ne0, n_rows] = argsort_descending(src[F32, ne0, n_rows]).

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

#include <cstring>
#include <vector>

namespace ggml_ve {
namespace ops {

// ---------------- MUL_MAT_ID ----------------
bool mul_mat_id_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT_ID || op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * as  = op->src[0];
    const ggml_tensor * b   = op->src[1];
    const ggml_tensor * ids = op->src[2];
    if (!as || !b || !ids) return false;
    if (as->type != GGML_TYPE_BF16) return false;        // only BF16 weights kernel today
    if (b->type   != GGML_TYPE_F32) return false;
    if (ids->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(as) || !ggml_is_contiguous(b) ||
        !ggml_is_contiguous(ids) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (as->ne[3] != 1 || b->ne[3] != 1) return false;
    // Kernel assumes a single input column per token broadcast to all experts
    // (b->ne[1] == 1). The general case (b->ne[1] == n_expert_used) needs a
    // different x_col selection inside the kernel.
    if (b->ne[1] != 1) return false;
    return true;
}

bool mul_mat_id(backend_context * ctx, ggml_tensor * dst) {
    if (!mul_mat_id_supports(dst)) return false;
    const ggml_tensor * as  = dst->src[0];
    const ggml_tensor * b   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];

    const VEDAdeviceptr dst_hbm = ctx->resolve_out(dst);
    const VEDAdeviceptr as_hbm  = ctx->resolve_in(as);
    const VEDAdeviceptr b_hbm   = ctx->resolve_in(b);
    const VEDAdeviceptr ids_hbm = ctx->resolve_in(ids);
    if (dst_hbm == 0 || as_hbm == 0 || b_hbm == 0 || ids_hbm == 0) return false;

    VEDAfunction fn = ctx->fn(K_MUL_MAT_ID_BF16_F32_HBM_FULL);
    if (fn == 0) return false;

    const uint64_t K              = (uint64_t) as->ne[0];
    const uint64_t M              = (uint64_t) as->ne[1];
    const uint64_t n_experts      = (uint64_t) as->ne[2];
    const uint64_t n_expert_used  = (uint64_t) ids->ne[0];
    const uint64_t n_tokens       = (uint64_t) b->ne[2];

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(mul_mat_id)")) return false;
    vedaArgsSetVPtr(args, 0, dst_hbm);
    vedaArgsSetVPtr(args, 1, as_hbm);
    vedaArgsSetVPtr(args, 2, b_hbm);
    vedaArgsSetVPtr(args, 3, ids_hbm);
    vedaArgsSetU64 (args, 4, M);
    vedaArgsSetU64 (args, 5, K);
    vedaArgsSetU64 (args, 6, n_experts);
    vedaArgsSetU64 (args, 7, n_expert_used);
    vedaArgsSetU64 (args, 8, n_tokens);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(mul_mat_id_bf16_f32_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    ctx->ops_mul_mat()++;
    ctx->ops_hbm()++;
    return true;
}

// ---------------- ADD_ID ----------------
bool add_id_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_ADD_ID || op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * ids  = op->src[2];
    if (!src0 || !src1 || !ids) return false;
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) return false;
    if (ids->type  != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(ids)  || !ggml_is_contiguous(op)) {
        return false;
    }
    return true;
}

bool add_id(backend_context * ctx, ggml_tensor * dst) {
    if (!add_id_supports(dst)) return false;
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const VEDAdeviceptr dst_hbm  = ctx->resolve_out(dst);
    const VEDAdeviceptr src0_hbm = ctx->resolve_in(src0);
    const VEDAdeviceptr src1_hbm = ctx->resolve_in(src1);
    const VEDAdeviceptr ids_hbm  = ctx->resolve_in(ids);
    if (dst_hbm == 0 || src0_hbm == 0 || src1_hbm == 0 || ids_hbm == 0) return false;

    VEDAfunction fn = ctx->fn(K_ADD_ID_F32_HBM_FULL);
    if (fn == 0) return false;

    const uint64_t ne0 = (uint64_t) dst->ne[0];
    const uint64_t ne1 = (uint64_t) dst->ne[1];
    const uint64_t ne2 = (uint64_t) dst->ne[2];

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(add_id)")) return false;
    vedaArgsSetVPtr(args, 0, dst_hbm);
    vedaArgsSetVPtr(args, 1, src0_hbm);
    vedaArgsSetVPtr(args, 2, src1_hbm);
    vedaArgsSetVPtr(args, 3, ids_hbm);
    vedaArgsSetU64 (args, 4, ne0);
    vedaArgsSetU64 (args, 5, ne1);
    vedaArgsSetU64 (args, 6, ne2);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(add_id_f32_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

// ---------------- ARGSORT ----------------
bool argsort_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_ARGSORT) return false;
    if (op->type != GGML_TYPE_I32) return false;
    const ggml_tensor * src = op->src[0];
    if (!src || src->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
    // The kernel uses O(N^2) insertion sort. Cap row width so it doesn't hang
    // on large arrays — MoE expert selection (the intended use case) has
    // ne[0] == n_experts which is small (8..256 typical).
    if (src->ne[0] > 256) return false;
    return true;
}

bool argsort(backend_context * ctx, ggml_tensor * dst) {
    if (!argsort_supports(dst)) return false;
    const ggml_tensor * src = dst->src[0];

    VEDAfunction fn = ctx->fn(K_ARGSORT_F32_OMP_HMEM);
    if (fn == 0) return false;

    const uint64_t ne0    = (uint64_t) dst->ne[0];
    const uint64_t nrows  = (uint64_t) ggml_nelements(dst) / ne0;
    const size_t src_bytes = nrows * ne0 * sizeof(float);
    const size_t dst_bytes = nrows * ne0 * sizeof(int32_t);

    // Stage src + dst through temp HBM (HMEM is for multi-VE / MPI only).
    VEDAdeviceptr src_tmp = 0, dst_tmp = 0;
    if (vedaMemAllocAsync(&src_tmp, src_bytes, 0) != VEDA_SUCCESS) return false;
    if (vedaMemAllocAsync(&dst_tmp, dst_bytes, 0) != VEDA_SUCCESS) {
        vedaMemFreeAsync(src_tmp, 0);
        return false;
    }
    ctx->enqueue_hbm_free(src_tmp);
    ctx->enqueue_hbm_free(dst_tmp);

    VEDAresult src_err;
    if (tensor_is_hbm(src)) {
        src_err = vedaMemcpyDtoDAsync(src_tmp, tensor_hbm_ptr(src), src_bytes, 0);
    } else if (src->data != nullptr) {
        src_err = vedaMemcpyHtoDAsync(src_tmp, src->data, src_bytes, 0);
    } else {
        return false;
    }
    if (!ggml_ve_ok(src_err, "vedaMemcpy* (argsort src)")) return false;

    const int32_t * params = (const int32_t *) dst->op_params;
    const uint64_t order = (uint64_t) params[0];   // 0=ASC, 1=DESC

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(argsort)")) return false;
    vedaArgsSetVPtr(args, 0, dst_tmp);
    vedaArgsSetVPtr(args, 1, src_tmp);
    vedaArgsSetU64 (args, 2, ne0);
    vedaArgsSetU64 (args, 3, nrows);
    vedaArgsSetU64 (args, 4, order);

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(argsort_f32_omp_hmem)")) {
        return false;
    }

    // Output unstage: HBM dst = D->D, CPU dst = sync + D->H.
    bool dst_ok = false;
    if (tensor_is_hbm(dst)) {
        dst_ok = ggml_ve_ok(vedaMemcpyDtoDAsync(tensor_hbm_ptr(dst), dst_tmp, dst_bytes, 0),
                            "vedaMemcpyDtoDAsync (argsort dst)");
    } else if (dst->data != nullptr) {
        vedaCtxSynchronize();
        dst_ok = (vedaMemcpyDtoH(dst->data, dst_tmp, dst_bytes) == VEDA_SUCCESS);
    }
    if (!dst_ok) return false;
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
