// GGML_OP_SOFT_MAX on F32 tensors. Two variants:
//   - unmasked / no attention sinks -> ve_soft_max_f32_omp_hbm
//   - per-row F16 mask              -> ve_soft_max_f32_masked_attn_hbm
//
// Source/dest stage through temp HBM (HMEM is for multi-VE / MPI only).
// Only one SOFT_MAX runs per token in standard decode (the lm_head logits
// sampling), so the round-trip is a constant cost.

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

    // Stage input/output tensors into temp HBM. For HBM-resident
    // tensors we'd ideally use resolve_in/resolve_out's existing
    // path, but the SOFT_MAX kernel was built assuming raw
    // VE-visible pointers so we keep a dedicated staging path.
    auto stage_in = [&](const ggml_tensor * t, VEDAdeviceptr tmp, size_t bytes) -> bool {
        if (tensor_is_hbm(t)) {
            return ggml_ve_ok(vedaMemcpyDtoDAsync(tmp, tensor_hbm_ptr(t), bytes, 0),
                              "vedaMemcpyDtoDAsync (softmax stage)");
        }
        if (t->data == nullptr) return false;
        return ggml_ve_ok(vedaMemcpyHtoDAsync(tmp, t->data, bytes, 0),
                          "vedaMemcpyHtoDAsync (softmax stage)");
    };

    float scale = 0.0f, max_bias = 0.0f;
    unpack_softmax_params(dst, &scale, &max_bias);
    uint64_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(float));

    const uint64_t ne00 = (uint64_t) src->ne[0];                       // softmax row width
    const uint64_t nrows = (uint64_t) ggml_nelements(src) / ne00;

    const size_t src_bytes = ggml_nbytes(src);
    const size_t dst_bytes = ggml_nbytes(dst);
    if (src_bytes == 0 || dst_bytes == 0) return true;  // no-op

    VEDAdeviceptr src_tmp = 0, dst_tmp = 0;
    if (vedaMemAllocAsync(&src_tmp, src_bytes, 0) != VEDA_SUCCESS) return false;
    if (vedaMemAllocAsync(&dst_tmp, dst_bytes, 0) != VEDA_SUCCESS) {
        vedaMemFreeAsync(src_tmp, 0);
        return false;
    }
    ctx->enqueue_hbm_free(src_tmp);
    ctx->enqueue_hbm_free(dst_tmp);

    if (!stage_in(src, src_tmp, src_bytes)) return false;

    VEDAdeviceptr mask_tmp = 0;
    if (mask) {
        const size_t mask_bytes = ggml_nbytes(mask);
        if (vedaMemAllocAsync(&mask_tmp, mask_bytes, 0) != VEDA_SUCCESS) return false;
        ctx->enqueue_hbm_free(mask_tmp);
        if (!stage_in(mask, mask_tmp, mask_bytes)) return false;
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(softmax)")) return false;

    VEDAfunction fn = 0;
    if (mask) {
        fn = ctx->fn(K_SOFT_MAX_F32_MASKED_ATTN_HMEM);
        if (fn == 0) { vedaArgsDestroy(args); return false; }
        const uint64_t ne1 = (uint64_t) src->ne[1];
        vedaArgsSetVPtr(args, 0, dst_tmp);
        vedaArgsSetVPtr(args, 1, src_tmp);
        vedaArgsSetVPtr(args, 2, mask_tmp);
        vedaArgsSetU64 (args, 3, ne00);
        vedaArgsSetU64 (args, 4, nrows);
        vedaArgsSetU64 (args, 5, ne1);
        vedaArgsSetU64 (args, 6, scale_bits);
    } else {
        fn = ctx->fn(K_SOFT_MAX_F32_OMP_HMEM);
        if (fn == 0) { vedaArgsDestroy(args); return false; }
        vedaArgsSetVPtr(args, 0, dst_tmp);
        vedaArgsSetVPtr(args, 1, src_tmp);
        vedaArgsSetU64 (args, 2, ne00);
        vedaArgsSetU64 (args, 3, nrows);
        vedaArgsSetU64 (args, 4, scale_bits);
    }

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(softmax)")) {
        return false;
    }

    // Copy result back to dst's home buffer.
    if (tensor_is_hbm(dst)) {
        if (!ggml_ve_ok(vedaMemcpyDtoDAsync(tensor_hbm_ptr(dst), dst_tmp, dst_bytes, 0),
                        "vedaMemcpyDtoDAsync (softmax unstage)")) {
            return false;
        }
    } else if (dst->data != nullptr) {
        // CPU-resident dst -- must sync first since DtoH is what eventually
        // delivers the value to host. Keep it on the existing deferred-dtoh
        // model by registering a record; resolve_out_slow's flush handles it.
        // Here we just sync-then-copy because the per-op pattern was already
        // sync-heavy.
        vedaCtxSynchronize();
        vedaMemcpyDtoH(dst->data, dst_tmp, dst_bytes);
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
