// GGML_OP_ROPE — Rotary Position Embedding.
//
// We use the *_hbm_omp_nocache kernels: they take the positions array via
// HMEM (we stage HBM->HMEM, no sync needed) and compute cos/sin on the VE
// side. Works for both single-token decode and multi-token prompt eval.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

#include <cstring>
#include <cstdint>

namespace ggml_ve {
namespace ops {

namespace {
constexpr int ROPE_MODE_NEOX  = 2;
constexpr int ROPE_MODE_MROPE = 8;
}

bool rope_supports(const ggml_tensor * op) {

    if (op->op != GGML_OP_ROPE) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * x   = op->src[0];
    const ggml_tensor * pos = op->src[1];
    if (x == nullptr || pos == nullptr) return false;
    if (x->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(x)) return false;

    const int32_t * params = (const int32_t *) op->op_params;
    const int mode = params[2];
    if (mode & ROPE_MODE_MROPE) return false;  // not yet

    float ext_factor = 0.0f;
    std::memcpy(&ext_factor, params + 7, sizeof(float));
    if (ext_factor != 0.0f) return false;  // YaRN-style not in the nocache kernel
    return true;
}

bool rope(backend_context * ctx, ggml_tensor * dst) {
    if (!rope_supports(dst)) return false;
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * pos = dst->src[1];
    if (!tensor_is_hbm(x) || !tensor_is_hbm(pos) || !tensor_is_hbm(dst)) {
        return false;
    }

    const int32_t * params = (const int32_t *) dst->op_params;
    const int n_dims   = params[1];
    const int mode     = params[2];
    float freq_base  = 0.0f, freq_scale = 0.0f, attn_factor = 0.0f;
    std::memcpy(&freq_base,   params + 5, sizeof(float));
    std::memcpy(&freq_scale,  params + 6, sizeof(float));
    std::memcpy(&attn_factor, params + 8, sizeof(float));

    const bool neox = (mode & ROPE_MODE_NEOX) != 0;
    VEDAfunction fn = ctx->fn(neox ? K_ROPE_NEOX_HBM_OMP_NOCACHE
                                   : K_ROPE_NORMAL_HBM_OMP_NOCACHE);
    if (fn == 0) return false;

    // x shape: [D, H, N, B] — D=head_dim, H=heads, N=tokens (= ne[2] in ggml RoPE)
    // For RoPE the n_ctx parameter is "number of token positions".
    const uint64_t ne0     = (uint64_t) x->ne[0];
    const uint64_t n_heads = (uint64_t) x->ne[1];
    const uint64_t n_ctx   = (uint64_t) x->ne[2];
    const uint64_t n_batch = (uint64_t) x->ne[3];

    // Stage positions: HBM -> HMEM. Tiny copy (4 * n_tokens bytes).
    const size_t pos_bytes = ggml_nbytes(pos);
    VEDAhmemptr pos_hmem = ctx->pool().acquire(pos_bytes);
    if (pos_hmem == 0) return false;
    if (!ggml_ve_ok(vedaHMemcpyDtoX(reinterpret_cast<void *>(pos_hmem),
                                     tensor_hbm_ptr(pos), pos_bytes),
                    "vedaHMemcpyDtoX (rope positions: HBM->HMEM)")) {
        ctx->pool().release(pos_hmem);
        return false;
    }

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(rope)")) {
        ctx->pool().release(pos_hmem);
        return false;
    }
    vedaArgsSetVPtr(args,  0, tensor_hbm_ptr(dst));
    vedaArgsSetVPtr(args,  1, tensor_hbm_ptr(x));
    vedaArgsSetHMEM(args,  2, pos_hmem);
    vedaArgsSetU64 (args,  3, ne0);
    vedaArgsSetU64 (args,  4, (uint64_t) n_dims);
    vedaArgsSetU64 (args,  5, n_heads);
    vedaArgsSetU64 (args,  6, n_ctx);
    vedaArgsSetU64 (args,  7, n_batch);
    vedaArgsSetU64 (args,  8, (uint64_t) x->nb[1]);
    vedaArgsSetU64 (args,  9, (uint64_t) x->nb[2]);
    vedaArgsSetU64 (args, 10, (uint64_t) x->nb[3]);
    vedaArgsSetF32 (args, 11, freq_base);
    vedaArgsSetF32 (args, 12, freq_scale);
    vedaArgsSetF32 (args, 13, attn_factor);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, &result),
                    "vedaLaunchKernelEx(rope_hbm_omp_nocache)")) {
        ctx->pool().release(pos_hmem);
        return false;
    }
    ctx->enqueue_input(pos_hmem);
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
