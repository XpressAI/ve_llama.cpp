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
#include "../colmajor_cache.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool mul_mat_supports(const ggml_tensor * op) {
    static const bool dbg = std::getenv("GGML_VE_DEBUG_MUL_MAT") != nullptr;
    auto rej = [&](const char * why) {
        if (dbg) fprintf(stderr, "[VE-MM-rej] %s : %s w=%s[%lld,%lld,%lld,%lld] x=%s[%lld,%lld,%lld,%lld]\n",
                         op->name[0]?op->name:"?", why,
                         op->src[0]?ggml_type_name(op->src[0]->type):"?",
                         op->src[0]?(long long)op->src[0]->ne[0]:0, op->src[0]?(long long)op->src[0]->ne[1]:0,
                         op->src[0]?(long long)op->src[0]->ne[2]:0, op->src[0]?(long long)op->src[0]->ne[3]:0,
                         op->src[1]?ggml_type_name(op->src[1]->type):"?",
                         op->src[1]?(long long)op->src[1]->ne[0]:0, op->src[1]?(long long)op->src[1]->ne[1]:0,
                         op->src[1]?(long long)op->src[1]->ne[2]:0, op->src[1]?(long long)op->src[1]->ne[3]:0);
        return false;
    };

    if (op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32) return false;

    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w == nullptr || x == nullptr) return rej("missing srcs");
    if (x->type != GGML_TYPE_F32) return rej("x not f32");
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_BF16) return rej("w type");

    // Shapes: dst = src0 @ src1^T  ->  K matches, M = src0 row dim, N = src1 row dim.
    const int64_t K = w->ne[0];
    const int64_t M = w->ne[1];
    const int64_t N = x->ne[1];
    if (K != x->ne[0])   return rej("K mismatch");
    if (op->ne[0] != M)  return rej("dst rows");
    if (op->ne[1] != N)  return rej("dst cols");

    // No broadcasting yet.
    if (op->ne[2] != 1 || op->ne[3] != 1) return rej("dst ne2/ne3");
    if (w->ne[2]  != 1 || w->ne[3]  != 1) return rej("w ne2/ne3");
    if (x->ne[2]  != 1 || x->ne[3]  != 1) return rej("x ne2/ne3");

    if (!ggml_is_contiguous(w))  return rej("w not contig");
    if (!ggml_is_contiguous(x))  return rej("x not contig");
    if (!ggml_is_contiguous(op)) return rej("dst not contig");

    // Kernels we have:
    //   BF16  N=1  : ve_bf16_matvec_hbm_full
    //   BF16  N>1  : ve_bf16_matmul_hbm_full     (sgemv_packed_bf16_unr per batch)
    //    F32  N=1  : ve_f32_matvec_hbm_full
    //    F32  N>1  : ve_f32_sgemm_batched_cblas_hbm  (NLC cblas_sgemm)
    if (w->type == GGML_TYPE_BF16 && (K % 16) != 0) {
        // The packed BF16 sgemv unrolls 16 across K, so K must be a multiple
        // of 16 to stay inside the row.
        return rej("BF16 K%16 != 0");
    }
    if (M <= 0 || K <= 0 || N <= 0) return rej("zero dim");

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
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
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

    const VEDAdeviceptr w_vptr = ctx->resolve_in(w);
    const VEDAdeviceptr x_vptr = ctx->resolve_in(x);
    const VEDAdeviceptr y_vptr = ctx->resolve_out(dst);
    if (w_vptr == 0 || x_vptr == 0 || y_vptr == 0) return false;

    // BF16 weight × F32 input precision-matching: handled inline in
    // sgemv_packed_bf16_unr (it ANDs the loaded F32 with 0xFFFF0000 to
    // truncate to BF16 precision before the packed FMA). That makes
    // every BF16 matvec bit-exact against CPU's ggml_vec_dot_bf16 —
    // necessary for Qwen3.5 (precision-sensitive GDN block) and
    // harmless for Llama-3.x. No host-side staging needed.

    // Opt-in until validated end-to-end.
    static const bool colmajor_enabled =
        (std::getenv("GGML_VE_NO_COLMAJOR") == nullptr);

    // FAST PATH: BF16 weights via cached F32 column-major + CBLAS NoTrans.
    //
    // The legacy backend's primary perf lever: transpose each BF16 weight
    // to F32 column-major once (on the VE, no PCIe), cache, then dispatch
    // through NEC's CBLAS SGEMM with CblasNoTrans. Wins big:
    //   - prompt eval (N>1): +77% pp32, +89% pp128 measured on Llama-3.2-3B
    //   - decode (N=1):      profiling showed the packed-BF16 N=1 matvec
    //                        scalarises and burns ~80ms per call on the
    //                        FFN gate/up projections; the CBLAS path
    //                        vectorises with V.OP=99% and demolishes that.
    //
    // CBLAS sync caveat: NLC CBLAS hangs the VE if we stack async
    // launches, so we sync inside the path. The sync adds ~100us per
    // call but the kernel saves milliseconds, so it's still a huge net
    // win. The first MUL_MAT for each weight pays a one-time transpose
    // cost (~25ms for a 3072x3072) — make sure llama-bench warms up so
    // the cache fills before the timing window opens.
    // GGML_VE_COLMAJOR_N1=1 forces N=1 to also use the colmajor F32 CBLAS path
    // — more precise (true F32 accumulation vs. packed-FP32 matvec) at the cost
    // of a per-call sync. Default off because Llama-3.x-class models tolerate
    // the matvec's precision and the synchronization cost is non-trivial.
    static const bool n1_colmajor =
        (std::getenv("GGML_VE_COLMAJOR_N1") != nullptr);
    if (colmajor_enabled
        && (N > 1 || n1_colmajor)
        && w->type == GGML_TYPE_BF16
        && ctx->dev() && ctx->dev()->colmajor
        && ctx->fn(K_BF16_TO_F32_COLMAJOR_HBM) != 0
        && ctx->fn(K_F32_SGEMM_BATCHED_CBLAS_HBM_NOTRANS) != 0) {

        // Why colmajor + CblasNoTrans (not rowmajor + CblasTrans):
        // NEC CBLAS's Trans path is measurably slower (~3-5x on pp) than
        // its NoTrans path — empirically confirmed by trying the row-
        // major variant and watching pp128 drop from 143 t/s to 21 t/s.
        // So we stick with column-major weights. The colmajor dequant
        // kernel itself doesn't vectorise (strided store kills it; see
        // aveorun-ftrace data — 76% of VE time in the transpose at
        // V.OP=0 / MFLOPS=0 when ftrace overhead is included) but it's
        // a one-time-per-weight cost amortised across all subsequent
        // matmuls of that weight, so net it's still a huge win.
        VEDAdeviceptr w_f32_colmajor = ctx->dev()->colmajor->get_or_create(
            w_vptr, (int64_t) M, (int64_t) K,
            ctx->fn(K_BF16_TO_F32_COLMAJOR_HBM));

        if (w_f32_colmajor != 0) {
            VEDAfunction fn = ctx->fn(K_F32_SGEMM_BATCHED_CBLAS_HBM_NOTRANS);
            VEDAargs args = nullptr;
            if (ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(cblas_notrans)")) {
                vedaArgsSetVPtr(args, 0, y_vptr);
                vedaArgsSetVPtr(args, 1, w_f32_colmajor);
                vedaArgsSetVPtr(args, 2, x_vptr);
                vedaArgsSetU64 (args, 3, M);
                vedaArgsSetU64 (args, 4, K);
                vedaArgsSetU64 (args, 5, N);
                uint64_t rc = 0;
                if (ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                               "vedaLaunchKernelEx(f32_sgemm_cblas_notrans)")) {
                    vedaCtxSynchronize();
                    ctx->ops_mul_mat()++;
                    ctx->ops_hbm()++;
                    return true;
                }
            }
        }
    }

    // Original BF16 / F32 kernels as a fallback (also still the path for F32
    // weights, which we don't need to col-major-cache since they're already
    // F32 in HBM).
    VEDAfunction fn = 0;
    bool include_N = false;

    if (w->type == GGML_TYPE_BF16) {
        if (N == 1) {
            fn = ctx->fn(K_BF16_MATVEC_HBM_FULL);
        } else {
            fn        = ctx->fn(K_BF16_MATMUL_HBM_FULL);
            include_N = true;
        }
    } else {  // F32
        if (N == 1) {
            fn = ctx->fn(K_F32_MATVEC_HBM_FULL);
        } else {
            // NLC's cblas_sgemm is comprehensively tuned for VE and beats
            // anything we'd hand-roll. ve_f32_sgemm_batched_cblas_hbm just
            // reinterprets W[M,K] / x[K,N] as col-major and calls
            // cblas_sgemm(ColMajor, Trans, NoTrans, ...) -- that produces the
            // exact GGML MUL_MAT semantics dst[m,n] = sum_k W[m,k] * x[n,k].
            // CBLAS hangs the VE if launches stack async, so we sync inline.
            VEDAfunction fn_cblas = ctx->fn(K_F32_SGEMM_BATCHED_CBLAS_HBM);
            if (fn_cblas != 0) {
                VEDAargs args = nullptr;
                if (ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(f32_sgemm_cblas)")) {
                    vedaArgsSetVPtr(args, 0, y_vptr);
                    vedaArgsSetVPtr(args, 1, w_vptr);
                    vedaArgsSetVPtr(args, 2, x_vptr);
                    vedaArgsSetU64 (args, 3, M);
                    vedaArgsSetU64 (args, 4, K);
                    vedaArgsSetU64 (args, 5, N);
                    if (ggml_ve_ok(vedaLaunchKernelEx(fn_cblas, 0, args, /*destroyArgs=*/1, nullptr),
                                   "vedaLaunchKernelEx(f32_sgemm_cblas)")) {
                        vedaCtxSynchronize();
                        ctx->ops_mul_mat()++;
                        ctx->ops_hbm()++;
                        return true;
                    }
                }
            }
            // Fallback: hand-rolled F32 matmul. Slower but doesn't depend on
            // NLC being loadable.
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
