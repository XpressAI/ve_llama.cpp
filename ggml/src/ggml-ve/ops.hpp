#ifndef GGML_VE_OPS_HPP
#define GGML_VE_OPS_HPP

// Op dispatch. `supports_op` is what the scheduler queries; `compute_forward`
// runs a single node. The dispatcher is small on purpose — each per-op
// handler lives in its own ops/<name>.cpp file.

#include "backend_ctx.hpp"

#include "ggml.h"

namespace ggml_ve {

bool supports_op(const device * dev, const ggml_tensor * op);

bool compute_forward(backend_context * ctx, ggml_tensor * node);

namespace ops {

// Each phase adds one declaration here; the implementation lives in
// ops/<name>.cpp. Returns true on success, false to signal failure (the
// backend then aborts the whole graph). Returning false for "I don't handle
// this shape" lets the dispatcher fall through to the default-fail path —
// but the scheduler already filtered via supports_op, so by the time we get
// here we expect to handle the op.

bool add_f32(backend_context * ctx, ggml_tensor * dst);
bool add_supports(const ggml_tensor * op);

bool mul_mat(backend_context * ctx, ggml_tensor * dst);
bool mul_mat_supports(const ggml_tensor * op);

bool mul_mat_q(backend_context * ctx, ggml_tensor * dst);
bool mul_mat_q_supports(const ggml_tensor * op);

bool mul_f32(backend_context * ctx, ggml_tensor * dst);
bool mul_supports(const ggml_tensor * op);

bool scale_f32(backend_context * ctx, ggml_tensor * dst);
bool scale_supports(const ggml_tensor * op);

bool silu_f32(backend_context * ctx, ggml_tensor * dst);
bool silu_supports(const ggml_tensor * op);

// Fused gated linear units. Currently handles GGML_GLU_OP_SWIGLU (split
// gate/up F32 in HBM) — the FFN pattern in Llama-3 / Llama-3.2. Without
// this the scheduler splits cgraphs at every FFN GLU node (codex
// finding #3) and the per-token graph fragments grow.
bool glu_f32(backend_context * ctx, ggml_tensor * dst);
bool glu_supports(const ggml_tensor * op);

bool rms_norm_f32(backend_context * ctx, ggml_tensor * dst);
bool rms_norm_supports(const ggml_tensor * op);

bool cpy_f32(backend_context * ctx, ggml_tensor * dst);
bool cpy_supports(const ggml_tensor * op);

bool get_rows(backend_context * ctx, ggml_tensor * dst);
bool get_rows_supports(const ggml_tensor * op);

bool flash_attn(backend_context * ctx, ggml_tensor * dst);
bool flash_attn_supports(const ggml_tensor * op);

bool set_rows(backend_context * ctx, ggml_tensor * dst);
bool set_rows_supports(const ggml_tensor * op);

bool rope(backend_context * ctx, ggml_tensor * dst);
bool rope_supports(const ggml_tensor * op);

bool mul_mat_id(backend_context * ctx, ggml_tensor * dst);
bool mul_mat_id_supports(const ggml_tensor * op);

bool add_id(backend_context * ctx, ggml_tensor * dst);
bool add_id_supports(const ggml_tensor * op);

bool argsort(backend_context * ctx, ggml_tensor * dst);
bool argsort_supports(const ggml_tensor * op);

bool soft_max_f32(backend_context * ctx, ggml_tensor * dst);
bool soft_max_supports(const ggml_tensor * op);

}  // namespace ops

}  // namespace ggml_ve

#endif // GGML_VE_OPS_HPP
