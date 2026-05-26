// GGML_OP_SSM_CONV — 1D causal depthwise conv used by Qwen3.5 / Qwen3.6
// in the recurrent (gated delta net) layers. The CPU reference is
// ggml_compute_forward_ssm_conv_f32 in ggml-cpu/ops.cpp.
//
// Inputs (per the CPU spec):
//   src0 : F32 [ncs=d_conv-1+n_t, d_inner, n_seqs]   sliding window
//   src1 : F32 [d_conv, d_inner]                     conv weight
//   dst  : F32 [d_inner, n_t, n_seqs]                output
// with src0->nb[0]=src1->nb[0]=sizeof(float) and src0->nb[1]=ncs*4 (the
// inner d_conv dim contiguous within a row, and channel-row stride
// equals row length). For typical Qwen3.5 decode: n_t=n_seqs=1,
// d_conv=4, d_inner≈4096 -- the call is launch-overhead-bound.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool ssm_conv_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_SSM_CONV || op->type != GGML_TYPE_F32) return false;

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) return false;
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) return false;

    // CPU-side preconditions we rely on in the kernel:
    //   - both input strides float-aligned along their innermost dim
    //   - src0's channel-row stride equals row length (ncs floats)
    //   - src1's channel-row stride equals d_conv floats
    if (src0->nb[0] != sizeof(float)) return false;
    if (src1->nb[0] != sizeof(float)) return false;
    if (src0->nb[1] != (size_t) src0->ne[0] * sizeof(float)) return false;
    if (src1->nb[1] != (size_t) src1->ne[0] * sizeof(float)) return false;
    if (op->nb[0]   != sizeof(float))                        return false;

    // dst geometry sanity per CPU code.
    if (op->ne[0] != src0->ne[1]) return false;
    return true;
}

bool ssm_conv_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!ssm_conv_supports(dst)) return false;

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    VEDAfunction fn = ctx->fn(K_SSM_CONV_F32_HBM);
    if (fn == 0) return false;

    const VEDAdeviceptr y_vptr = ctx->resolve_out(dst);
    const VEDAdeviceptr x_vptr = ctx->resolve_in(src0);
    const VEDAdeviceptr w_vptr = ctx->resolve_in(src1);
    if (y_vptr == 0 || x_vptr == 0 || w_vptr == 0) return false;

    const uint64_t d_conv  = (uint64_t) src1->ne[0];
    const uint64_t d_inner = (uint64_t) src0->ne[1];
    const uint64_t n_t     = (uint64_t) dst->ne[1];
    const uint64_t n_seqs  = (uint64_t) dst->ne[2];

    // Pass strides in float units so the kernel doesn't have to redo
    // the byte->float math.
    const uint64_t src0_nb1_f = (uint64_t) (src0->nb[1] / sizeof(float));
    const uint64_t src0_nb2_f = (uint64_t) (src0->nb[2] / sizeof(float));
    const uint64_t dst_nb1_f  = (uint64_t) (dst->nb[1]  / sizeof(float));
    const uint64_t dst_nb2_f  = (uint64_t) (dst->nb[2]  / sizeof(float));

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(ssm_conv)")) return false;
    vedaArgsSetVPtr(args, 0,  y_vptr);
    vedaArgsSetVPtr(args, 1,  x_vptr);
    vedaArgsSetVPtr(args, 2,  w_vptr);
    vedaArgsSetU64 (args, 3,  d_conv);
    vedaArgsSetU64 (args, 4,  d_inner);
    vedaArgsSetU64 (args, 5,  n_t);
    vedaArgsSetU64 (args, 6,  n_seqs);
    vedaArgsSetU64 (args, 7,  src0_nb1_f);
    vedaArgsSetU64 (args, 8,  src0_nb2_f);
    vedaArgsSetU64 (args, 9,  dst_nb1_f);
    vedaArgsSetU64 (args, 10, dst_nb2_f);

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(ssm_conv_f32_hbm)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
