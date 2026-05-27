// GGML_OP_FLASH_ATTN_EXT: fused multi-head attention.
//
// We support three configurations, all with Q/K/V/dst in HBM and the mask
// staged to HMEM through the pool:
//   1) Q F32, K F32, V F32       -> ve_flash_attn_f32_hbm
//   2) Q F32, K BF16, V BF16     -> ve_flash_attn_ext_f32q_bf16kv_hbm
//   3) Q BF16, K BF16, V BF16    -> ve_flash_attn_bf16_intrinsics_hbm (LLVM-VE intrinsics)
//
// Constraints (Phase-4 first cut):
//   - sinks (op->src[4]) must be null. Attention sinks is gpt-oss territory.
//   - permute must be [0,1,2,3] (no transpose). Anything else is rare in
//     standard inference and would need stride remap logic.
//   - logit_softcap == 0, max_bias == 0 — these enable specialised paths in
//     the kernel but we keep the dispatch simple by requiring the common case.
//
// Mask handling: the kernel takes the mask via HMEM; we acquire a buffer,
// stage HBM->HMEM with vedaHMemcpyDtoX, and release it via the deferred
// input queue after the next sync.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../kv_shadow_cache.hpp"
#include "../ops.hpp"

#include "ggml.h"

#include <cstdlib>
#include <cstring>

namespace ggml_ve {
namespace ops {

/* ------------------------------------------------------------------ *
 * Stage 3: column-major shadow KV cache fast path.
 *
 * Pre-conditions checked here (above and beyond flash_attn_supports):
 *   - Q F32, K/V BF16 (col-major shadow is only built for BF16 caches).
 *   - Single-token decode (q->ne[1] == 1). Multi-token prefill writes
 *     don't go through the per-row mirror hook, so the shadow watermark
 *     can't catch up in one step — those keep using row-major FA.
 *   - kv_len >= GGML_VE_COLMAJOR_FA_MIN (default 96 — Stage 1 crossover).
 *   - Mask F16, no head/batch broadcast (ne[2] == ne[3] == 1).
 *   - Q and dst are contiguous in (head_dim, head)-major order so the
 *     kernel's `q + h*head_dim` indexing matches reality.
 *   - The shadow exists for both K and V canonical tensors, and the
 *     watermark covers [0, kv_len). If not, fall back to the row-major
 *     path (it'll get rebuilt on the next SET_ROWS). */
static bool try_flash_attn_colmajor(backend_context * ctx, ggml_tensor * dst) {
    static const bool enabled = std::getenv("GGML_VE_NO_KV_SHADOW") == nullptr;
    if (!enabled) return false;

    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_BF16 || v->type != GGML_TYPE_BF16) {
        return false;
    }
    if (q->ne[1] != 1) return false;
    if (mask == nullptr || mask->type != GGML_TYPE_F16) return false;
    if (mask->ne[2] != 1 || mask->ne[3] != 1)            return false;

    static const int colmajor_min = []() {
        const char * e = std::getenv("GGML_VE_COLMAJOR_FA_MIN");
        return e ? std::atoi(e) : 96;
    }();
    const int64_t kv_len = k->ne[1];
    if (kv_len < (int64_t) colmajor_min) return false;

    VEDAfunction fn = ctx->fn(K_FLASH_ATTN_EXT_F32Q_BF16KV_COLMAJOR_HBM);
    if (fn == 0) return false;

    if (!ctx->dev() || !ctx->dev()->kv_shadow) return false;
    auto * shadows = ctx->dev()->kv_shadow;

    /* Walk view_src chain to the bottom-most tensor — for permuted KV
     * views ("cache_k_l0 (view) (permuted)") the immediate view_src is
     * the permutation; the actual KV cache canonical is one or more
     * levels down. */
    const ggml_tensor * k_canon = k;
    while (k_canon->view_src) k_canon = k_canon->view_src;
    const ggml_tensor * v_canon = v;
    while (v_canon->view_src) v_canon = v_canon->view_src;
    if (!tensor_is_hbm(k_canon) || !tensor_is_hbm(v_canon)) return false;

    kv_shadow * k_sh = shadows->find(tensor_hbm_ptr(k_canon));
    kv_shadow * v_sh = shadows->find(tensor_hbm_ptr(v_canon));
    if (!shadows->valid_for_range(k_sh, kv_len)) return false;
    if (!shadows->valid_for_range(v_sh, kv_len)) return false;

    const uint64_t head_dim   = (uint64_t) q->ne[0];
    const uint64_t n_q_heads  = (uint64_t) q->ne[2];
    const uint64_t n_kv_heads = (uint64_t) k->ne[2];

    /* The shadow's channels axis must agree with [head_dim * n_kv_heads]
     * to be safely reinterpretable as [n_kv_heads][head_dim][seq_max]. */
    if (k_sh->channels != (int64_t) (head_dim * n_kv_heads)) return false;
    if (v_sh->channels != (int64_t) (head_dim * n_kv_heads)) return false;

    /* Q is [head_dim, N=1, n_q_heads, batch] so per-head stride = nb[2].
     * dst is [head_dim, n_q_heads, N=1, batch] (note dims 1 and 2 are
     * swapped vs Q) so per-head stride = nb[1]. The kernel needs both
     * head strides as bytes so it can index q + h*q_hstride and
     * out + h*out_hstride correctly. */
    if (dst->ne[0] != q->ne[0])  return false;
    if (dst->ne[1] != q->ne[2])  return false;
    if (q->nb[0]   != sizeof(float) || dst->nb[0] != sizeof(float)) return false;
    if (q->nb[2]   % sizeof(float) != 0)                            return false;
    if (dst->nb[1] % sizeof(float) != 0)                            return false;

    const VEDAdeviceptr q_hbm   = ctx->resolve_in(q);
    const VEDAdeviceptr dst_hbm = ctx->resolve_out(dst);
    if (q_hbm == 0 || dst_hbm == 0) return false;

    /* Stage mask into temp HBM (HMEM is for multi-VE / MPI only). */
    const size_t mask_bytes = ggml_nbytes(mask);
    if (mask_bytes == 0) return false;  /* fall back, mask required */
    VEDAdeviceptr mask_tmp = 0;
    if (vedaMemAllocAsync(&mask_tmp, mask_bytes, 0) != VEDA_SUCCESS) return false;
    VEDAresult mask_err;
    if (tensor_is_hbm(mask)) {
        mask_err = vedaMemcpyDtoDAsync(mask_tmp, tensor_hbm_ptr(mask), mask_bytes, 0);
    } else if (mask->data != nullptr) {
        mask_err = vedaMemcpyHtoDAsync(mask_tmp, mask->data, mask_bytes, 0);
    } else {
        vedaMemFreeAsync(mask_tmp, 0);
        return false;
    }
    if (!ggml_ve_ok(mask_err, "vedaMemcpy* (flash_attn_colmajor mask)")) {
        vedaMemFreeAsync(mask_tmp, 0);
        return false;
    }
    ctx->enqueue_hbm_free(mask_tmp);

    float scale = 0.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    uint64_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(float));
    /* max_bias / softcap are already verified zero by flash_attn_supports,
     * so the kernel's slope param is just the identity. */
    const float    slope_f = 1.0f;
    uint64_t slope_bits = 0;
    std::memcpy(&slope_bits, &slope_f, sizeof(float));

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(flash_attn_colmajor)")) return false;
    vedaArgsSetVPtr(args,  0, dst_hbm);
    vedaArgsSetVPtr(args,  1, q_hbm);
    vedaArgsSetVPtr(args,  2, k_sh->shadow_hbm);
    vedaArgsSetVPtr(args,  3, v_sh->shadow_hbm);
    vedaArgsSetVPtr(args,  4, mask_tmp);
    vedaArgsSetU64 (args,  5, head_dim);
    vedaArgsSetU64 (args,  6, n_q_heads);
    vedaArgsSetU64 (args,  7, n_kv_heads);
    vedaArgsSetU64 (args,  8, (uint64_t) kv_len);
    vedaArgsSetU64 (args,  9, (uint64_t) k_sh->seq_max);
    vedaArgsSetU64 (args, 10, (uint64_t) q->nb[2]);    /* Q per-head stride: heads are dim 2 */
    vedaArgsSetU64 (args, 11, (uint64_t) dst->nb[1]);  /* dst per-head stride: heads are dim 1 */
    vedaArgsSetU64 (args, 12, scale_bits);
    vedaArgsSetU64 (args, 13, slope_bits);

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(flash_attn_colmajor)")) {
        return false;
    }
    ctx->mark_sync_pending();
    ctx->ops_flash_attn()++;
    ctx->ops_hbm()++;
    return true;
}

bool flash_attn_supports(const ggml_tensor * op) {
    static const bool dbg = std::getenv("GGML_VE_DEBUG_FA") != nullptr;
    auto reject = [&](const char * why) {
        if (dbg) fprintf(stderr, "[VE-FA-reject] %s : %s\n",
                         op->name[0] ? op->name : "(noname)", why);
        return false;
    };
    if (op->op != GGML_OP_FLASH_ATTN_EXT || op->type != GGML_TYPE_F32) return reject("op/type");
    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * k    = op->src[1];
    const ggml_tensor * v    = op->src[2];
    const ggml_tensor * mask = op->src[3];
    if (!q || !k || !v) return reject("missing q/k/v");

    if (op->src[4] != nullptr) return reject("attention-sinks");
    // The wrapped kernels assume mask != NULL with strides on src[3];
    // passing null would dereference garbage. Until we add an explicit
    // no-mask kernel path the safe gate is mask-must-exist-and-be-F16.
    if (mask == nullptr) return reject("no mask");
    if (mask->type != GGML_TYPE_F16) return reject("mask type != F16");
    if (dbg) {
        fprintf(stderr, "[VE-FA-shape] %s q=%s[%ld,%ld,%ld,%ld] k=%s[%ld,%ld,%ld,%ld] v=%s[%ld,%ld,%ld,%ld] mask=%s\n",
                op->name[0] ? op->name : "(noname)",
                ggml_type_name(q->type), q->ne[0], q->ne[1], q->ne[2], q->ne[3],
                ggml_type_name(k->type), k->ne[0], k->ne[1], k->ne[2], k->ne[3],
                ggml_type_name(v->type), v->ne[0], v->ne[1], v->ne[2], v->ne[3],
                ggml_type_name(mask->type));
    }

    // Type combinations we have a kernel for.
    const bool combo_f32  = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32);
    const bool combo_bf16kv = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_BF16 && v->type == GGML_TYPE_BF16);
    const bool combo_bf16 = (q->type == GGML_TYPE_BF16 && k->type == GGML_TYPE_BF16 && v->type == GGML_TYPE_BF16);
    if (!(combo_f32 || combo_bf16kv || combo_bf16)) return reject("qkv dtype combo");

    // Soft-cap and max-bias must be zero for the basic dispatch.
    float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
    std::memcpy(&scale,    (const float *) op->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
    std::memcpy(&softcap,  (const float *) op->op_params + 2, sizeof(float));
    if (max_bias != 0.0f) return reject("max_bias != 0");
    if (softcap  != 0.0f) return reject("softcap != 0");

    return true;
}

bool flash_attn(backend_context * ctx, ggml_tensor * dst) {
    if (!flash_attn_supports(dst)) return false;

    /* Try the column-major shadow path first. Falls through if any gate
     * fails (e.g. shadow not yet warm, multi-token decode, kv_len small). */
    if (try_flash_attn_colmajor(ctx, dst)) return true;

    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const VEDAdeviceptr q_hbm   = ctx->resolve_in(q);
    const VEDAdeviceptr k_hbm   = ctx->resolve_in(k);
    const VEDAdeviceptr v_hbm   = ctx->resolve_in(v);
    const VEDAdeviceptr dst_hbm = ctx->resolve_out(dst);
    if (q_hbm == 0 || k_hbm == 0 || v_hbm == 0 || dst_hbm == 0) return false;

    // Pick kernel by Q/K/V dtypes. For (F32 Q, BF16 K/V) prefill (N>1) the
    // tile-batched variant reuses each BF16 K/V load + decode across NQ_TILE
    // queries, cutting HBM traffic ~Nq× — that's the dominant cost in the
    // row-major prefill path. Decode (N=1) and the colmajor shadow path
    // remain untouched.
    VEDAfunction fn = 0;
    const bool can_tile = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_BF16 &&
                           q->ne[1] > 1 &&
                           std::getenv("GGML_VE_NO_FA_TILE") == nullptr);
    if (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32) {
        fn = ctx->fn(K_FLASH_ATTN_F32_HBM);
    } else if (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_BF16) {
        if (can_tile) {
            fn = ctx->fn(K_FLASH_ATTN_EXT_F32Q_BF16KV_TILE_HBM);
        }
        if (fn == 0) {
            fn = ctx->fn(K_FLASH_ATTN_EXT_F32Q_BF16KV_HBM);
        }
    } else if (q->type == GGML_TYPE_BF16) {
        fn = ctx->fn(K_FLASH_ATTN_BF16_INTRINSICS_HBM);
    }
    if (fn == 0) return false;

    // Dimensions.
    const uint64_t D  = (uint64_t) q->ne[0];
    const uint64_t N  = (uint64_t) q->ne[1];
    const uint64_t H  = (uint64_t) q->ne[2];
    const uint64_t B  = (uint64_t) q->ne[3];
    const uint64_t S  = (uint64_t) k->ne[1];
    const uint64_t Hk = (uint64_t) k->ne[2];
    const uint64_t Dv = (uint64_t) v->ne[0];

    // op_params: float[0]=scale, [1]=max_bias, [2]=logit_softcap (already verified 0).
    float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
    std::memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    uint64_t scale_bits = 0, max_bias_bits = 0, softcap_bits = 0;
    std::memcpy(&scale_bits,    &scale,    sizeof(float));
    std::memcpy(&max_bias_bits, &max_bias, sizeof(float));
    std::memcpy(&softcap_bits,  &softcap,  sizeof(float));

    // Stage mask into temp HBM (HMEM is for multi-VE / MPI only).
    const size_t mask_bytes = ggml_nbytes(mask);
    if (mask_bytes == 0) return false;  // fall back, mask required
    VEDAdeviceptr mask_tmp = 0;
    if (vedaMemAllocAsync(&mask_tmp, mask_bytes, 0) != VEDA_SUCCESS) return false;
    VEDAresult mask_err;
    if (tensor_is_hbm(mask)) {
        mask_err = vedaMemcpyDtoDAsync(mask_tmp, tensor_hbm_ptr(mask), mask_bytes, 0);
    } else if (mask->data != nullptr) {
        mask_err = vedaMemcpyHtoDAsync(mask_tmp, mask->data, mask_bytes, 0);
    } else {
        vedaMemFreeAsync(mask_tmp, 0);
        return false;
    }
    if (!ggml_ve_ok(mask_err, "vedaMemcpy* (flash_attn mask)")) {
        vedaMemFreeAsync(mask_tmp, 0);
        return false;
    }
    ctx->enqueue_hbm_free(mask_tmp);

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(flash_attn)")) return false;

    vedaArgsSetVPtr(args,  0, dst_hbm);
    vedaArgsSetVPtr(args,  1, q_hbm);
    vedaArgsSetVPtr(args,  2, k_hbm);
    vedaArgsSetVPtr(args,  3, v_hbm);
    vedaArgsSetVPtr(args,  4, mask_tmp);

    vedaArgsSetU64 (args,  5, D);
    vedaArgsSetU64 (args,  6, Dv);
    vedaArgsSetU64 (args,  7, N);
    vedaArgsSetU64 (args,  8, S);
    vedaArgsSetU64 (args,  9, H);
    vedaArgsSetU64 (args, 10, Hk);
    vedaArgsSetU64 (args, 11, B);

    // Q strides
    vedaArgsSetU64 (args, 12, (uint64_t) q->nb[1]);
    vedaArgsSetU64 (args, 13, (uint64_t) q->nb[2]);
    vedaArgsSetU64 (args, 14, (uint64_t) q->nb[3]);
    // K strides
    vedaArgsSetU64 (args, 15, (uint64_t) k->nb[1]);
    vedaArgsSetU64 (args, 16, (uint64_t) k->nb[2]);
    vedaArgsSetU64 (args, 17, (uint64_t) k->nb[3]);
    // V strides
    vedaArgsSetU64 (args, 18, (uint64_t) v->nb[1]);
    vedaArgsSetU64 (args, 19, (uint64_t) v->nb[2]);
    vedaArgsSetU64 (args, 20, (uint64_t) v->nb[3]);
    // Mask strides
    vedaArgsSetU64 (args, 21, (uint64_t) mask->nb[1]);
    vedaArgsSetU64 (args, 22, (uint64_t) mask->nb[2]);
    vedaArgsSetU64 (args, 23, (uint64_t) mask->nb[3]);
    // Output strides
    vedaArgsSetU64 (args, 24, (uint64_t) dst->nb[1]);
    vedaArgsSetU64 (args, 25, (uint64_t) dst->nb[2]);
    vedaArgsSetU64 (args, 26, (uint64_t) dst->nb[3]);

    // Float params as bits
    vedaArgsSetU64 (args, 27, scale_bits);
    vedaArgsSetU64 (args, 28, max_bias_bits);
    vedaArgsSetU64 (args, 29, softcap_bits);

    // Mask broadcast dims
    vedaArgsSetU64 (args, 30, (uint64_t) mask->ne[2]);
    vedaArgsSetU64 (args, 31, (uint64_t) mask->ne[3]);

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(flash_attn)")) {
        return false;
    }
    ctx->mark_sync_pending();
    ctx->ops_flash_attn()++;
    ctx->ops_hbm()++;
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
