// Phase-5 GGML_OP_SET_ROWS — KV-cache writes. dst is the cache tensor in HBM
// (BF16, F16, or F32), src is the freshly computed value to write (F32).
// Indices come via an HBM tensor, staged through HMEM by the pool.
//
// ggml's SET_ROWS semantics:
//   dst[idx[i]] = src[i]
// The all-HBM kernels also do the precision conversion F32 -> {F16, BF16, F32}.
//
// IMPORTANT: ggml's SET_ROWS uses int64 indices, but the VE kernel reads
// indices as int32*. We narrow i64 -> i32 in the staging copy. Indices for
// KV caches fit comfortably in 32 bits (context length << 2^31). For i32
// inputs we pass through directly.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace ggml_ve {
namespace ops {

bool set_rows_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_SET_ROWS) return false;
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * idx = op->src[1];
    if (src == nullptr || idx == nullptr) return false;
    if (src->type != GGML_TYPE_F32) return false;
    if (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16 && op->type != GGML_TYPE_F32) {
        return false;
    }
    if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) return false;
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(idx)) return false;
    if (op->ne[2] != 1 || op->ne[3] != 1) return false;
    return true;
}

bool set_rows(backend_context * ctx, ggml_tensor * dst) {
    if (!set_rows_supports(dst)) return false;
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * idx = dst->src[1];
    if (!tensor_is_hbm(src) || !tensor_is_hbm(idx) || !tensor_is_hbm(dst)) {
        return false;
    }

    const uint64_t nc     = (uint64_t) src->ne[0];
    const uint64_t nr     = (uint64_t) ggml_nelements(idx);
    const uint64_t nb_dst = (uint64_t) dst->nb[1];
    const uint64_t nb_src = (uint64_t) src->nb[1];

    // The VE kernel reads indices as int32 with a 4-byte stride.
    // Stage them into HMEM as int32 — narrowing i64 if needed.
    const size_t idx32_bytes = nr * sizeof(int32_t);
    VEDAhmemptr idx_hmem = ctx->pool().acquire(idx32_bytes);
    if (idx_hmem == 0) return false;

    if (idx->type == GGML_TYPE_I32) {
        // Direct HBM -> HMEM copy of int32 indices.
        if (!ggml_ve_ok(vedaHMemcpyDtoX(reinterpret_cast<void *>(idx_hmem),
                                         tensor_hbm_ptr(idx), idx32_bytes),
                        "vedaHMemcpyDtoX (set_rows i32 idx)")) {
            ctx->pool().release(idx_hmem);
            return false;
        }
    } else {  // GGML_TYPE_I64
        // Read i64 indices from HBM into a host buffer, narrow to i32,
        // upload to HMEM. nr is small (number of new tokens in this graph).
        std::vector<int64_t> host_i64(nr);
        if (!ggml_ve_ok(vedaMemcpyDtoH(host_i64.data(), tensor_hbm_ptr(idx),
                                        nr * sizeof(int64_t)),
                        "vedaMemcpyDtoH (set_rows i64 idx)")) {
            ctx->pool().release(idx_hmem);
            return false;
        }
        std::vector<int32_t> host_i32(nr);
        for (uint64_t i = 0; i < nr; ++i) {
            host_i32[i] = (int32_t) host_i64[i];
        }
        if (!ggml_ve_ok(vedaHMemcpy(reinterpret_cast<void *>(idx_hmem),
                                     host_i32.data(), idx32_bytes),
                        "vedaHMemcpy (set_rows i32 narrowed idx)")) {
            ctx->pool().release(idx_hmem);
            return false;
        }
    }

    kernel_id kid;
    switch (dst->type) {
        case GGML_TYPE_F16:  kid = K_SET_ROWS_F16_HBM_FULL;  break;
        case GGML_TYPE_BF16: kid = K_SET_ROWS_BF16_HBM_FULL; break;
        case GGML_TYPE_F32:  kid = K_SET_ROWS_F32_HBM_FULL;  break;
        default:
            ctx->pool().release(idx_hmem);
            return false;
    }
    VEDAfunction fn = ctx->fn(kid);
    if (fn == 0) {
        ctx->pool().release(idx_hmem);
        return false;
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(set_rows)")) {
        ctx->pool().release(idx_hmem);
        return false;
    }
    vedaArgsSetVPtr(args, 0, tensor_hbm_ptr(dst));
    vedaArgsSetVPtr(args, 1, tensor_hbm_ptr(src));
    vedaArgsSetHMEM(args, 2, idx_hmem);
    vedaArgsSetU64 (args, 3, nc);
    vedaArgsSetU64 (args, 4, nr);
    vedaArgsSetU64 (args, 5, nb_dst);
    vedaArgsSetU64 (args, 6, nb_src);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(set_rows_hbm_full)")) {
        ctx->pool().release(idx_hmem);
        return false;
    }

    ctx->enqueue_input(idx_hmem);
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
