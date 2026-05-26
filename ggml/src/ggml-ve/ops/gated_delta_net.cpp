// GGML_OP_GATED_DELTA_NET — fused linear-attention + state-update kernel
// used by Qwen3.5 / Qwen3.6 in every recurrent layer. Mirrors the CPU
// reference in ggml-cpu/ops.cpp (ggml_compute_forward_gated_delta_net_*).
//
// Inputs:
//   src[0] q     : F32 [S_v, neq1, n_tokens, neq3]
//   src[1] k     : F32 [S_v, nek1, n_tokens, nek3]
//   src[2] v     : F32 [S_v, H,    n_tokens, n_seqs]   (defines S_v, H, T, B)
//   src[3] g     : F32 [neg0, neg1, n_tokens, n_seqs]   neg0 in {1, S_v}
//   src[4] beta  : F32 [1,    neb1, n_tokens, n_seqs]
//   src[5] state : F32 [S_v*S_v*H, K, n_seqs]
// Output dst: F32 4D [S_v*H, n_tokens*n_seqs + state_rows, 1, 1]
//   (logically: attn region followed by snapshot states region)

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool gated_delta_net_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_GATED_DELTA_NET || op->type != GGML_TYPE_F32) return false;

    const ggml_tensor * q     = op->src[0];
    const ggml_tensor * k     = op->src[1];
    const ggml_tensor * v     = op->src[2];
    const ggml_tensor * g     = op->src[3];
    const ggml_tensor * beta  = op->src[4];
    const ggml_tensor * state = op->src[5];
    if (!q || !k || !v || !g || !beta || !state) return false;
    for (int i = 0; i < 6; ++i) {
        if (op->src[i]->type != GGML_TYPE_F32) return false;
    }

    // The CPU constructor enforces these — re-check defensively because
    // a passing supports_op MUST mean compute_forward won't blow up.
    const int64_t S_v = v->ne[0];
    if (S_v <= 0 || S_v > 256) return false;          // GDN_MAX_SV in kernel
    if (g->ne[0] != 1 && g->ne[0] != S_v) return false;
    if (beta->ne[0] != 1) return false;
    if (state->ne[0] != S_v * S_v * v->ne[1])  return false;
    if (state->ne[2] != v->ne[3])              return false;
    if (state->ne[3] != 1)                      return false;

    // Stride-1 innermost on every input.
    if (q->nb[0]    != sizeof(float)) return false;
    if (k->nb[0]    != sizeof(float)) return false;
    if (v->nb[0]    != sizeof(float)) return false;
    if (g->nb[0]    != sizeof(float)) return false;
    if (beta->nb[0] != sizeof(float)) return false;
    if (state->nb[0]!= sizeof(float)) return false;
    return true;
}

bool gated_delta_net_f32(backend_context * ctx, ggml_tensor * dst) {
    if (!gated_delta_net_supports(dst)) return false;

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    VEDAfunction fn = ctx->fn(K_GATED_DELTA_NET_F32_HBM);
    if (fn == 0) return false;

    const VEDAdeviceptr dst_v = ctx->resolve_out(dst);
    const VEDAdeviceptr q_v   = ctx->resolve_in(q);
    const VEDAdeviceptr k_v   = ctx->resolve_in(k);
    const VEDAdeviceptr v_v   = ctx->resolve_in(v);
    const VEDAdeviceptr g_v   = ctx->resolve_in(g);
    const VEDAdeviceptr b_v   = ctx->resolve_in(beta);
    const VEDAdeviceptr s_v   = ctx->resolve_in(state);
    if (!dst_v || !q_v || !k_v || !v_v || !g_v || !b_v || !s_v) return false;

    const uint64_t S_v      = (uint64_t) v->ne[0];
    const uint64_t H        = (uint64_t) v->ne[1];
    const uint64_t n_tokens = (uint64_t) v->ne[2];
    const uint64_t n_seqs   = (uint64_t) v->ne[3];
    const uint64_t Kslots   = (uint64_t) state->ne[1];
    const uint64_t kda_flag = (uint64_t) (g->ne[0] == (int64_t) S_v ? 1 : 0);

    auto f = [](size_t b) { return (uint64_t)(b / sizeof(float)); };

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(gdn)")) return false;
    int idx = 0;
    vedaArgsSetVPtr(args, idx++, dst_v);
    vedaArgsSetVPtr(args, idx++, q_v);
    vedaArgsSetVPtr(args, idx++, k_v);
    vedaArgsSetVPtr(args, idx++, v_v);
    vedaArgsSetVPtr(args, idx++, g_v);
    vedaArgsSetVPtr(args, idx++, b_v);
    vedaArgsSetVPtr(args, idx++, s_v);
    vedaArgsSetU64 (args, idx++, S_v);
    vedaArgsSetU64 (args, idx++, H);
    vedaArgsSetU64 (args, idx++, n_tokens);
    vedaArgsSetU64 (args, idx++, n_seqs);
    vedaArgsSetU64 (args, idx++, Kslots);
    vedaArgsSetU64 (args, idx++, kda_flag);
    /* Q */
    vedaArgsSetU64 (args, idx++, (uint64_t) q->ne[1]);
    vedaArgsSetU64 (args, idx++, (uint64_t) q->ne[3]);
    vedaArgsSetU64 (args, idx++, f(q->nb[1]));
    vedaArgsSetU64 (args, idx++, f(q->nb[2]));
    vedaArgsSetU64 (args, idx++, f(q->nb[3]));
    /* K */
    vedaArgsSetU64 (args, idx++, (uint64_t) k->ne[1]);
    vedaArgsSetU64 (args, idx++, (uint64_t) k->ne[3]);
    vedaArgsSetU64 (args, idx++, f(k->nb[1]));
    vedaArgsSetU64 (args, idx++, f(k->nb[2]));
    vedaArgsSetU64 (args, idx++, f(k->nb[3]));
    /* V */
    vedaArgsSetU64 (args, idx++, f(v->nb[1]));
    vedaArgsSetU64 (args, idx++, f(v->nb[2]));
    vedaArgsSetU64 (args, idx++, f(v->nb[3]));
    /* g */
    vedaArgsSetU64 (args, idx++, (uint64_t) g->ne[0]);
    vedaArgsSetU64 (args, idx++, f(g->nb[1]));
    vedaArgsSetU64 (args, idx++, f(g->nb[2]));
    vedaArgsSetU64 (args, idx++, f(g->nb[3]));
    /* beta */
    vedaArgsSetU64 (args, idx++, f(beta->nb[1]));
    vedaArgsSetU64 (args, idx++, f(beta->nb[2]));
    vedaArgsSetU64 (args, idx++, f(beta->nb[3]));
    /* state */
    vedaArgsSetU64 (args, idx++, f(state->nb[2]));

    uint64_t result = 0;
    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(gated_delta_net_f32_hbm)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
