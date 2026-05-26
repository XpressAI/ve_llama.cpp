// GGML_OP_SOFT_MAX on F32 tensors. Two variants:
//   - unmasked / no attention sinks -> ve_soft_max_f32_omp_hmem
//   - per-row F16 mask              -> ve_soft_max_f32_masked_attn_hmem
//
// The kernels are HMEM-IO. Source/dest staged through the pool, same pattern
// as ARGSORT. Only one SOFT_MAX runs per token in standard decode (the
// lm_head logits sampling), so the HMEM round-trip is a constant cost.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

#include <cstring>

namespace ggml_ve {
namespace ops {

namespace {

void unpack_softmax_params(const ggml_tensor * op, float * scale, float * max_bias) {
    *scale = 1.0f;
    *max_bias = 0.0f;
    if (op->op_params) {
        std::memcpy(scale,    (const float *) op->op_params + 0, sizeof(float));
        std::memcpy(max_bias, (const float *) op->op_params + 1, sizeof(float));
    }
}

}  // namespace

bool soft_max_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_SOFT_MAX) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * src  = op->src[0];
    const ggml_tensor * mask = op->src[1];
    if (!src || src->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;

    // The masked-attn kernel mishandles ggml's broadcast dims in our test
    // framework (~0.4 ERR across many shapes); ship unmasked only for now.
    // Real attention paths use FLASH_ATTN_EXT which has its own mask logic;
    // the bare-SOFT_MAX-with-mask case is rarely on the hot path.
    if (mask != nullptr) return false;

    // Attention-sinks variant (gpt-oss) needs extra src[2]; skip for now.
    if (op->src[2] != nullptr) return false;

    float scale = 0.0f, max_bias = 0.0f;
    unpack_softmax_params(op, &scale, &max_bias);
    if (max_bias != 0.0f) return false;   // ALiBi not in these kernels

    return true;
}

bool soft_max_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!soft_max_supports(dst)) return false;
    const ggml_tensor * src  = dst->src[0];
    const ggml_tensor * mask = dst->src[1];
    // SOFT_MAX uses HMEM-only kernels — every input/output bounces
    // through pooled HMEM regardless of where ggml put the tensors.
    // We just vary the *source* of those staging copies (vedaHMemcpyDtoX
    // from HBM, vedaHMemcpy from host memory).
    auto stage_in = [&](const ggml_tensor * t, VEDAhmemptr hmem, size_t bytes) -> bool {
        if (tensor_is_hbm(t)) {
            return ggml_ve_ok(vedaHMemcpyDtoX(reinterpret_cast<void *>(hmem),
                                               tensor_hbm_ptr(t), bytes),
                              "vedaHMemcpyDtoX (softmax stage)");
        }
        if (t->data == nullptr) return false;
        return ggml_ve_ok(vedaHMemcpy(reinterpret_cast<void *>(hmem),
                                       t->data, bytes),
                          "vedaHMemcpy (softmax stage)");
    };
    auto unstage_out = [&](ggml_tensor * t, VEDAhmemptr hmem, size_t bytes) -> bool {
        if (tensor_is_hbm(t)) {
            return ggml_ve_ok(vedaHMemcpyXtoD(tensor_hbm_ptr(t),
                                               reinterpret_cast<void *>(hmem), bytes),
                              "vedaHMemcpyXtoD (softmax dst)");
        }
        if (t->data == nullptr) return false;
        std::memcpy(t->data, reinterpret_cast<void *>(hmem), bytes);
        return true;
    };

    float scale = 0.0f, max_bias = 0.0f;
    unpack_softmax_params(dst, &scale, &max_bias);
    uint64_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(float));

    const uint64_t ne00 = (uint64_t) src->ne[0];                       // softmax row width
    const uint64_t nrows = (uint64_t) ggml_nelements(src) / ne00;

    const size_t src_bytes = ggml_nbytes(src);
    const size_t dst_bytes = ggml_nbytes(dst);

    VEDAhmemptr src_hmem = ctx->pool().acquire(src_bytes);
    VEDAhmemptr dst_hmem = ctx->pool().acquire(dst_bytes);
    if (src_hmem == 0 || dst_hmem == 0) {
        if (src_hmem) ctx->pool().release(src_hmem);
        if (dst_hmem) ctx->pool().release(dst_hmem);
        return false;
    }
    if (!stage_in(src, src_hmem, src_bytes)) {
        ctx->pool().release(src_hmem);
        ctx->pool().release(dst_hmem);
        return false;
    }

    VEDAhmemptr mask_hmem = 0;
    if (mask) {
        const size_t mask_bytes = ggml_nbytes(mask);
        mask_hmem = ctx->pool().acquire(mask_bytes);
        if (mask_hmem == 0) {
            ctx->pool().release(src_hmem);
            ctx->pool().release(dst_hmem);
            return false;
        }
        if (!stage_in(mask, mask_hmem, mask_bytes)) {
            ctx->pool().release(src_hmem);
            ctx->pool().release(dst_hmem);
            ctx->pool().release(mask_hmem);
            return false;
        }
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(softmax)")) {
        ctx->pool().release(src_hmem);
        ctx->pool().release(dst_hmem);
        if (mask_hmem) ctx->pool().release(mask_hmem);
        return false;
    }

    VEDAfunction fn = 0;
    if (mask) {
        fn = ctx->fn(K_SOFT_MAX_F32_MASKED_ATTN_HMEM);
        if (fn == 0) {
            vedaArgsDestroy(args);
            ctx->pool().release(src_hmem);
            ctx->pool().release(dst_hmem);
            ctx->pool().release(mask_hmem);
            return false;
        }
        const uint64_t ne1 = (uint64_t) src->ne[1];
        vedaArgsSetHMEM(args, 0, dst_hmem);
        vedaArgsSetHMEM(args, 1, src_hmem);
        vedaArgsSetHMEM(args, 2, mask_hmem);
        vedaArgsSetU64 (args, 3, ne00);
        vedaArgsSetU64 (args, 4, nrows);
        vedaArgsSetU64 (args, 5, ne1);
        vedaArgsSetU64 (args, 6, scale_bits);
    } else {
        fn = ctx->fn(K_SOFT_MAX_F32_OMP_HMEM);
        if (fn == 0) {
            vedaArgsDestroy(args);
            ctx->pool().release(src_hmem);
            ctx->pool().release(dst_hmem);
            return false;
        }
        vedaArgsSetHMEM(args, 0, dst_hmem);
        vedaArgsSetHMEM(args, 1, src_hmem);
        vedaArgsSetU64 (args, 2, ne00);
        vedaArgsSetU64 (args, 3, nrows);
        vedaArgsSetU64 (args, 4, scale_bits);
    }

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(softmax)")) {
        ctx->pool().release(src_hmem);
        ctx->pool().release(dst_hmem);
        if (mask_hmem) ctx->pool().release(mask_hmem);
        return false;
    }

    // sync, copy result back to HBM (or host memory), release pool entries.
    vedaCtxSynchronize();
    if (!unstage_out(dst, dst_hmem, dst_bytes)) {
        ctx->pool().release(src_hmem);
        ctx->pool().release(dst_hmem);
        if (mask_hmem) ctx->pool().release(mask_hmem);
        return false;
    }
    vedaCtxSynchronize();

    ctx->pool().release(src_hmem);
    ctx->pool().release(dst_hmem);
    if (mask_hmem) ctx->pool().release(mask_hmem);
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
