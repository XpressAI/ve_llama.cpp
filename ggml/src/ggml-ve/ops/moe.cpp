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
    if (!tensor_is_hbm(as) || !tensor_is_hbm(b) ||
        !tensor_is_hbm(ids) || !tensor_is_hbm(dst)) {
        return false;
    }

    VEDAfunction fn = ctx->fn(K_MUL_MAT_ID_BF16_F32_HBM_FULL);
    if (fn == 0) return false;

    const uint64_t K              = (uint64_t) as->ne[0];
    const uint64_t M              = (uint64_t) as->ne[1];
    const uint64_t n_experts      = (uint64_t) as->ne[2];
    const uint64_t n_expert_used  = (uint64_t) ids->ne[0];
    const uint64_t n_tokens       = (uint64_t) b->ne[2];

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(mul_mat_id)")) return false;
    vedaArgsSetVPtr(args, 0, tensor_hbm_ptr(dst));
    vedaArgsSetVPtr(args, 1, tensor_hbm_ptr(as));
    vedaArgsSetVPtr(args, 2, tensor_hbm_ptr(b));
    vedaArgsSetVPtr(args, 3, tensor_hbm_ptr(ids));
    vedaArgsSetU64 (args, 4, M);
    vedaArgsSetU64 (args, 5, K);
    vedaArgsSetU64 (args, 6, n_experts);
    vedaArgsSetU64 (args, 7, n_expert_used);
    vedaArgsSetU64 (args, 8, n_tokens);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
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
    if (!tensor_is_hbm(src0) || !tensor_is_hbm(src1) ||
        !tensor_is_hbm(ids)  || !tensor_is_hbm(dst)) {
        return false;
    }

    VEDAfunction fn = ctx->fn(K_ADD_ID_F32_HBM_FULL);
    if (fn == 0) return false;

    const uint64_t ne0 = (uint64_t) dst->ne[0];
    const uint64_t ne1 = (uint64_t) dst->ne[1];
    const uint64_t ne2 = (uint64_t) dst->ne[2];

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(add_id)")) return false;
    vedaArgsSetVPtr(args, 0, tensor_hbm_ptr(dst));
    vedaArgsSetVPtr(args, 1, tensor_hbm_ptr(src0));
    vedaArgsSetVPtr(args, 2, tensor_hbm_ptr(src1));
    vedaArgsSetVPtr(args, 3, tensor_hbm_ptr(ids));
    vedaArgsSetU64 (args, 4, ne0);
    vedaArgsSetU64 (args, 5, ne1);
    vedaArgsSetU64 (args, 6, ne2);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
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
    if (!tensor_is_hbm(src) || !tensor_is_hbm(dst)) return false;

    VEDAfunction fn = ctx->fn(K_ARGSORT_F32_OMP_HMEM);
    if (fn == 0) return false;

    const uint64_t ne0    = (uint64_t) dst->ne[0];
    const uint64_t nrows  = (uint64_t) ggml_nelements(dst) / ne0;
    const size_t src_bytes = nrows * ne0 * sizeof(float);
    const size_t dst_bytes = nrows * ne0 * sizeof(int32_t);

    // The argsort kernel is HMEM-IO; stage both src and dst through the pool.
    VEDAhmemptr src_hmem = ctx->pool().acquire(src_bytes);
    VEDAhmemptr dst_hmem = ctx->pool().acquire(dst_bytes);
    if (src_hmem == 0 || dst_hmem == 0) {
        if (src_hmem) ctx->pool().release(src_hmem);
        if (dst_hmem) ctx->pool().release(dst_hmem);
        return false;
    }

    if (!ggml_ve_ok(vedaHMemcpyDtoX(reinterpret_cast<void *>(src_hmem),
                                     tensor_hbm_ptr(src), src_bytes),
                    "vedaHMemcpyDtoX (argsort src)")) {
        ctx->pool().release(src_hmem);
        ctx->pool().release(dst_hmem);
        return false;
    }

    const int32_t * params = (const int32_t *) dst->op_params;
    const uint64_t order = (uint64_t) params[0];   // 0=ASC, 1=DESC

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(argsort)")) {
        ctx->pool().release(src_hmem);
        ctx->pool().release(dst_hmem);
        return false;
    }
    vedaArgsSetHMEM(args, 0, dst_hmem);
    vedaArgsSetHMEM(args, 1, src_hmem);
    vedaArgsSetU64 (args, 2, ne0);
    vedaArgsSetU64 (args, 3, nrows);
    vedaArgsSetU64 (args, 4, order);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(argsort_f32_omp_hmem)")) {
        ctx->pool().release(src_hmem);
        ctx->pool().release(dst_hmem);
        return false;
    }

    // Need kernel done before staging output back. (HMEM->HBM deferred-sync
    // wiring is the same TODO as the K-quant path.)
    vedaCtxSynchronize();

    if (!ggml_ve_ok(vedaHMemcpyXtoD(tensor_hbm_ptr(dst),
                                     reinterpret_cast<void *>(dst_hmem), dst_bytes),
                    "vedaHMemcpyXtoD (argsort dst)")) {
        ctx->pool().release(src_hmem);
        ctx->pool().release(dst_hmem);
        return false;
    }
    vedaCtxSynchronize();

    ctx->pool().release(src_hmem);
    ctx->pool().release(dst_hmem);
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
