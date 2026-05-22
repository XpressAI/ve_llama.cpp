// GGML_OP_MUL_MAT: dst = src0 @ src1^T (ggml semantics).
//
// src0 [K, M]  weights   (row-major, BF16 or F32)
// src1 [K, N]  input     (row-major, F32, contiguous)
// dst  [M, N]  output    (F32, contiguous)
//
// Phase-2 scope: all three tensors resident in VE HBM. The full-HBM path is
// the fast path (1.2 TB/s); CPU-resident weights with HBM-cache upload land
// in a follow-up.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool mul_mat_supports(const ggml_tensor * op) {

    if (op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32) return false;

    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w == nullptr || x == nullptr) return false;
    if (x->type != GGML_TYPE_F32) return false;
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_BF16) return false;

    // Shapes: dst = src0 @ src1^T  ->  K matches, M = src0 row dim, N = src1 row dim.
    const int64_t K = w->ne[0];
    const int64_t M = w->ne[1];
    const int64_t N = x->ne[1];
    if (K != x->ne[0])   return false;        // K must match
    if (op->ne[0] != M)  return false;        // dst rows = M
    if (op->ne[1] != N)  return false;        // dst cols = N

    // No broadcasting yet.
    if (op->ne[2] != 1 || op->ne[3] != 1) return false;
    if (w->ne[2]  != 1 || w->ne[3]  != 1) return false;
    if (x->ne[2]  != 1 || x->ne[3]  != 1) return false;

    if (!ggml_is_contiguous(w))  return false;
    if (!ggml_is_contiguous(x))  return false;
    if (!ggml_is_contiguous(op)) return false;

    // We need an all-HBM fast path; we don't yet have an HMEM-round-trip
    // fallback for the cases libve_sgemv.so only ships HMEM-IO kernels for.
    // The kernels we actually have full-HBM variants of:
    //   BF16  N=1  : ve_bf16_matvec_hbm_full
    //   BF16  N>1  : ve_bf16_matmul_hbm_full     (sgemv_packed_bf16_unr per batch)
    //    F32  N=1  : ve_f32_matvec_hbm_full
    //    F32  N>1  : (no all-HBM kernel)  -> reject for now
    if (w->type == GGML_TYPE_F32 && N > 1) {
        return false;
    }
    // The packed BF16 sgemv kernel uses 16-way unrolling on K, so K must be
    // a multiple of 16 to stay inside the row.
    if (w->type == GGML_TYPE_BF16 && (K % 16) != 0) {
        return false;
    }
    // FP32 matvec kernel inner loop assumes K >= 1 (trivially true) but we
    // still need M > 0 and K > 0.
    if (M <= 0 || K <= 0 || N <= 0) return false;

    return true;
}

namespace {

// Launch a kernel with arg layout (y_vptr, W_vptr, x_vptr, ...dims...).
// 5 args for matvec (N=1), 6 for matmul (N>1). Returns true on success.
bool launch_matmul(VEDAfunction fn,
                   VEDAdeviceptr y, VEDAdeviceptr W, VEDAdeviceptr x,
                   uint64_t M, uint64_t K, uint64_t N, bool include_N) {
    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(mul_mat)")) return false;
    vedaArgsSetVPtr(args, 0, y);
    vedaArgsSetVPtr(args, 1, W);
    vedaArgsSetVPtr(args, 2, x);
    vedaArgsSetU64 (args, 3, M);
    vedaArgsSetU64 (args, 4, K);
    if (include_N) vedaArgsSetU64(args, 5, N);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(mul_mat)")) {
        return false;
    }
    return true;
}

}  // namespace

bool mul_mat(backend_context * ctx, ggml_tensor * dst) {
    if (!mul_mat_supports(dst)) return false;

    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    const uint64_t M = (uint64_t) w->ne[1];
    const uint64_t K = (uint64_t) w->ne[0];
    const uint64_t N = (uint64_t) x->ne[1];

    const bool full_hbm = tensor_is_hbm(w) && tensor_is_hbm(x) && tensor_is_hbm(dst);
    if (!full_hbm) {
        // Mixed CPU-weight / HMEM-IO paths come in a follow-up.
        return false;
    }

    const VEDAdeviceptr y_vptr = tensor_hbm_ptr(dst);
    const VEDAdeviceptr w_vptr = tensor_hbm_ptr(w);
    const VEDAdeviceptr x_vptr = tensor_hbm_ptr(x);

    // Pick the right kernel.  BF16 vs F32 weights, N==1 (matvec) vs N>1 (matmul).
    VEDAfunction fn = 0;
    bool include_N = false;

    if (w->type == GGML_TYPE_BF16) {
        if (N == 1) {
            fn = ctx->fn(K_BF16_MATVEC_HBM_FULL);
        } else {
            // For N>1, prefer batched CBLAS (10-30x faster on prompt eval);
            // ve_bf16_matmul_hbm_full is the row-major intrinsics fallback.
            fn        = ctx->fn(K_BF16_MATMUL_HBM_FULL);
            include_N = true;
        }
    } else {  // F32
        if (N == 1) {
            fn = ctx->fn(K_F32_MATVEC_HBM_FULL);
        } else {
            fn        = ctx->fn(K_F32_MATMUL_HBM_OMP);
            include_N = true;
        }
    }

    if (fn == 0) {
        return false;
    }

    if (!launch_matmul(fn, y_vptr, w_vptr, x_vptr, M, K, N, include_N)) {
        return false;
    }
    ctx->mark_sync_pending();
    ctx->ops_mul_mat()++;
    ctx->ops_hbm()++;
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
