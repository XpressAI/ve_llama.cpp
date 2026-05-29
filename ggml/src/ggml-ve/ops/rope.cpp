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
constexpr int ROPE_MODE_NEOX   = 2;
constexpr int ROPE_MODE_MROPE  = 8;
constexpr int ROPE_MODE_VISION = 24;  // bit 8 + bit 16; vision = MROPE variant
constexpr int ROPE_MODE_IMROPE = 40;  // bit 8 + bit 32; Qwen3-VL interleaved
}

bool rope_supports(const ggml_tensor * op) {
    static const bool dbg = std::getenv("GGML_VE_DEBUG_ROPE") != nullptr;
    auto rej = [&](const char * why) {
        if (dbg) fprintf(stderr, "[VE-ROPE-rej] %s : %s\n",
                         op->name[0]?op->name:"?", why);
        return false;
    };

    if (op->op != GGML_OP_ROPE) return false;
    if (op->type != GGML_TYPE_F32) return rej("type");
    const ggml_tensor * x   = op->src[0];
    const ggml_tensor * pos = op->src[1];
    if (x == nullptr || pos == nullptr) return rej("missing srcs");
    if (x->type != GGML_TYPE_F32) return rej("x not f32");
    if (pos->type != GGML_TYPE_I32) return rej("pos not i32");
    if (!ggml_is_contiguous(x)) return rej("x not contiguous");

    const int32_t * params = (const int32_t *) op->op_params;
    const int mode = params[2];

    // YaRN (ext_factor != 0) is implemented only in the NeoX HBM kernels
    // (ve_rope_neox_hbm_omp_nocache). Accept it there; for normal/mrope YaRN
    // (rare) fall back to CPU rather than apply un-corrected angles.
    float ext_factor = 0.0f;
    std::memcpy(&ext_factor, params + 7, sizeof(float));
    const bool neox  = (mode & ROPE_MODE_NEOX)  != 0;
    const bool mrope = (mode & ROPE_MODE_MROPE) != 0;
    if (ext_factor != 0.0f && (!neox || mrope)) return rej("yarn only on neox");

    // VISION mode uses a different pair layout; not yet supported.
    if (mode == ROPE_MODE_VISION) return rej("vision mode");
    return true;
}

bool rope(backend_context * ctx, ggml_tensor * dst) {
    if (!rope_supports(dst)) return false;
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * pos = dst->src[1];

    const VEDAdeviceptr x_hbm   = ctx->resolve_in(x);
    const VEDAdeviceptr dst_hbm = ctx->resolve_out(dst);
    if (x_hbm == 0 || dst_hbm == 0) return false;

    const int32_t * params = (const int32_t *) dst->op_params;
    const int n_dims     = params[1];
    const int mode       = params[2];
    const uint64_t n_ctx_orig = (uint64_t) params[4];
    float freq_base = 0.0f, freq_scale = 0.0f, ext_factor = 0.0f,
          attn_factor = 0.0f, beta_fast = 0.0f, beta_slow = 0.0f;
    std::memcpy(&freq_base,   params + 5, sizeof(float));
    std::memcpy(&freq_scale,  params + 6, sizeof(float));
    std::memcpy(&ext_factor,  params + 7, sizeof(float));
    std::memcpy(&attn_factor, params + 8, sizeof(float));
    std::memcpy(&beta_fast,   params + 9, sizeof(float));
    std::memcpy(&beta_slow,   params + 10, sizeof(float));

    const bool mrope = (mode & ROPE_MODE_MROPE) != 0;
    const bool neox  = (mode & ROPE_MODE_NEOX)  != 0;

    const uint64_t ne0     = (uint64_t) x->ne[0];
    const uint64_t n_heads = (uint64_t) x->ne[1];
    const uint64_t n_ctx   = (uint64_t) x->ne[2];
    const uint64_t n_batch = (uint64_t) x->ne[3];

    // Stage positions into temp HBM (HMEM is for multi-VE / MPI only).
    const size_t pos_bytes = ggml_nbytes(pos);
    if (pos_bytes == 0) return true;
    VEDAdeviceptr pos_tmp = 0;
    if (vedaMemAllocAsync(&pos_tmp, pos_bytes, 0) != VEDA_SUCCESS || pos_tmp == 0) return false;
    VEDAresult pos_err;
    if (tensor_is_hbm(pos)) {
        pos_err = vedaMemcpyDtoDAsync(pos_tmp, tensor_hbm_ptr(pos), pos_bytes, 0);
    } else if (pos->data != nullptr) {
        pos_err = vedaMemcpyHtoDAsync(pos_tmp, pos->data, pos_bytes, 0);
    } else {
        vedaMemFreeAsync(pos_tmp, 0);
        return false;
    }
    if (!ggml_ve_ok(pos_err, "vedaMemcpy* (rope positions)")) {
        vedaMemFreeAsync(pos_tmp, 0);
        return false;
    }
    ctx->enqueue_hbm_free(pos_tmp);

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(rope)")) return false;

    if (mrope) {
        VEDAfunction fn = ctx->fn(K_ROPE_MROPE_F32_HBM);
        if (fn == 0) { vedaArgsDestroy(args); return false; }

        // sections live in params[11..14].
        int sections[4] = { params[11], params[12], params[13], params[14] };

        uint32_t fb_bits, fs_bits, af_bits;
        std::memcpy(&fb_bits, &freq_base,   sizeof(uint32_t));
        std::memcpy(&fs_bits, &freq_scale,  sizeof(uint32_t));
        std::memcpy(&af_bits, &attn_factor, sizeof(uint32_t));
        if (freq_scale == 0.0f) {
            // Llama defaults sometimes leave freq_scale at 0 in op_params when
            // it should be 1.0 (linear, no scaling). Match CPU semantics.
            float one = 1.0f;
            std::memcpy(&fs_bits, &one, sizeof(uint32_t));
        }
        if (attn_factor == 0.0f) {
            float one = 1.0f;
            std::memcpy(&af_bits, &one, sizeof(uint32_t));
        }

        const uint64_t is_imrope = (mode == ROPE_MODE_IMROPE) ? 1 : 0;

        int idx = 0;
        vedaArgsSetVPtr(args, idx++, dst_hbm);
        vedaArgsSetVPtr(args, idx++, x_hbm);
        vedaArgsSetVPtr(args, idx++, pos_tmp);
        vedaArgsSetU64 (args, idx++, ne0);
        vedaArgsSetU64 (args, idx++, n_heads);
        vedaArgsSetU64 (args, idx++, n_ctx);
        vedaArgsSetU64 (args, idx++, n_batch);
        vedaArgsSetU64 (args, idx++, (uint64_t) x->nb[1]);
        vedaArgsSetU64 (args, idx++, (uint64_t) x->nb[2]);
        vedaArgsSetU64 (args, idx++, (uint64_t) x->nb[3]);
        vedaArgsSetU64 (args, idx++, (uint64_t) n_dims);
        vedaArgsSetU64 (args, idx++, (uint64_t) sections[0]);
        vedaArgsSetU64 (args, idx++, (uint64_t) sections[1]);
        vedaArgsSetU64 (args, idx++, (uint64_t) sections[2]);
        vedaArgsSetU64 (args, idx++, (uint64_t) sections[3]);
        vedaArgsSetU64 (args, idx++, (uint64_t) fb_bits);
        vedaArgsSetU64 (args, idx++, (uint64_t) fs_bits);
        vedaArgsSetU64 (args, idx++, (uint64_t) af_bits);
        vedaArgsSetU64 (args, idx++, is_imrope);

        if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                        "vedaLaunchKernelEx(rope_mrope_f32_hbm)")) {
            return false;
        }
        ctx->mark_sync_pending();
        return true;
    }

    VEDAfunction fn = ctx->fn(neox ? K_ROPE_NEOX_HBM_OMP_NOCACHE
                                   : K_ROPE_NORMAL_HBM_OMP_NOCACHE);
    if (fn == 0) { vedaArgsDestroy(args); return false; }

    vedaArgsSetVPtr(args,  0, dst_hbm);
    vedaArgsSetVPtr(args,  1, x_hbm);
    vedaArgsSetVPtr(args,  2, pos_tmp);
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
    vedaArgsSetF32 (args, 14, ext_factor);
    vedaArgsSetU64 (args, 15, n_ctx_orig);
    vedaArgsSetF32 (args, 16, beta_fast);
    vedaArgsSetF32 (args, 17, beta_slow);

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(rope_hbm_omp_nocache)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
