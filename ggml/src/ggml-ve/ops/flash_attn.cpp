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
#include "../ops.hpp"

#include "ggml.h"

#include <cstring>

namespace ggml_ve {
namespace ops {

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
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!tensor_is_hbm(q) || !tensor_is_hbm(k) || !tensor_is_hbm(v) ||
        !tensor_is_hbm(dst) || !tensor_is_hbm(mask)) {
        return false;
    }

    // Pick kernel by Q/K/V dtypes.
    VEDAfunction fn = 0;
    if (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32) {
        fn = ctx->fn(K_FLASH_ATTN_F32_HBM);
    } else if (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_BF16) {
        fn = ctx->fn(K_FLASH_ATTN_EXT_F32Q_BF16KV_HBM);
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

    // Stage mask HBM -> HMEM.
    const size_t mask_bytes = ggml_nbytes(mask);
    VEDAhmemptr mask_hmem = ctx->pool().acquire(mask_bytes);
    if (mask_hmem == 0) return false;
    if (!ggml_ve_ok(vedaHMemcpyDtoX(reinterpret_cast<void *>(mask_hmem),
                                     tensor_hbm_ptr(mask), mask_bytes),
                    "vedaHMemcpyDtoX (flash_attn mask: HBM->HMEM)")) {
        ctx->pool().release(mask_hmem);
        return false;
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(flash_attn)")) {
        ctx->pool().release(mask_hmem);
        return false;
    }

    vedaArgsSetVPtr(args,  0, tensor_hbm_ptr(dst));
    vedaArgsSetVPtr(args,  1, tensor_hbm_ptr(q));
    vedaArgsSetVPtr(args,  2, tensor_hbm_ptr(k));
    vedaArgsSetVPtr(args,  3, tensor_hbm_ptr(v));
    vedaArgsSetHMEM(args,  4, mask_hmem);

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

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(flash_attn)")) {
        ctx->pool().release(mask_hmem);
        return false;
    }

    ctx->enqueue_input(mask_hmem);
    ctx->mark_sync_pending();
    ctx->ops_flash_attn()++;
    ctx->ops_hbm()++;
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
