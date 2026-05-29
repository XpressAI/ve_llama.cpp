// Phase-7 graph compiler implementation. See graph_compiler.hpp for design.

#include "graph_compiler.hpp"

#include "backend_ctx.hpp"
#include "colmajor_cache.hpp"
#include "common.hpp"
#include "device.hpp"
#include "hbm_cache.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace ggml_ve {
namespace gcomp {

namespace {

const char * op_type_name(OpType t) {
    switch (t) {
        case OpType::GET_ROWS:      return "GET_ROWS";
        case OpType::RMS_NORM:      return "RMS_NORM";
        case OpType::MUL:           return "MUL";
        case OpType::ADD:           return "ADD";
        case OpType::MUL_MAT_F32:   return "MUL_MAT_F32";
        case OpType::MUL_MAT_BF16:  return "MUL_MAT_BF16";
        case OpType::MUL_MAT_Q4K:   return "MUL_MAT_Q4K";
        case OpType::ROPE:          return "ROPE";
        case OpType::SET_ROWS:      return "SET_ROWS";
        case OpType::FLASH_ATTN:    return "FLASH_ATTN";
        case OpType::GLU_SWIGLU:    return "GLU_SWIGLU";
        case OpType::CPY:           return "CPY";
        case OpType::SOFT_MAX:      return "SOFT_MAX";
        default:                    return "UNKNOWN";
    }
}

const ggml_tensor * canonical(const ggml_tensor * t) {
    while (t && t->view_src) t = t->view_src;
    return t;
}

bool name_contains(const std::string & name, const char * needle) {
    return name.find(needle) != std::string::npos;
}

bool debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char * env = std::getenv("GGML_VE_COMPILE_DEBUG");
        v = (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0) ? 0 : 1;
    }
    return v != 0;
}

}  // namespace

// -------------------------------------------------------------------------
// Enable / disable
// -------------------------------------------------------------------------
bool GraphCompiler::enabled() {
    static int v = -1;
    if (v < 0) {
        // Accept `=0`, empty, or unset as off; any other value as on.
        // The earlier check (and the legacy backend's, at
        // llama.cpp/ggml/src/ggml-ve/ggml-ve.cpp:9527) treated bare
        // presence as on, so `GGML_VE_COMPILE_GRAPH=0` actually enabled
        // the compiler — surprising and not user-expected.
        const char * env = std::getenv("GGML_VE_COMPILE_GRAPH");
        v = (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0) ? 0 : 1;
        if (v) {
            fprintf(stderr, "[VE-GC] graph compilation ENABLED\n");
        }
    }
    return v != 0;
}

GraphCompiler::GraphCompiler() = default;
GraphCompiler::~GraphCompiler() = default;

GraphCompiler & get_compiler() {
    static GraphCompiler instance;
    return instance;
}

// -------------------------------------------------------------------------
// Tensor classification
// -------------------------------------------------------------------------
bool GraphCompiler::is_weight(const ggml_tensor * t) const {
    const ggml_tensor * c = canonical(t);
    if (output_tensors_.count(t) || output_tensors_.count(c)) return false;

    std::string name = c->name ? c->name : "";

    if (name_contains(name, "cache_k") || name_contains(name, "cache_v")) return false;
    if (name_contains(name, ".weight") && (c->buffer || c->data)) return true;
    if (c->type == GGML_TYPE_BF16 && (c->buffer || c->data)) return true;

    if (name_contains(name, "leaf_")) {
        return ggml_nbytes(c) > 100 && (c->buffer || c->data);
    }

    // Intermediate-name heuristics matching the legacy backend's rules.
    static const char * intermediate_patterns[] = {
        "Qcur", "Kcur", "Vcur",
        "attn_out", "attn_norm",
        "ffn_inp", "ffn_gate", "ffn_up", "ffn_out", "ffn_norm", "ffn_swiglu",
        "l_out", "inp_embd", "result_",
        "(view)", "(copy)", "node_", "norm-",
        "kqv_out", "__fattn__",
        nullptr,
    };
    for (int i = 0; intermediate_patterns[i]; ++i) {
        if (name_contains(name, intermediate_patterns[i])) return false;
    }

    if (ggml_nbytes(c) < 100) return false;
    if (!c->buffer && !c->data) return false;
    return name_contains(name, ".weight");
}

int GraphCompiler::assign_buffer(const ggml_tensor * t, BufferKind & kind_out) {
    const ggml_tensor * c = canonical(t);

    auto it = tensor_buf_idx_.find(c);
    if (it != tensor_buf_idx_.end()) {
        kind_out = tensor_buf_kind_[c];
        if (t != c) {
            tensor_buf_idx_[t]  = it->second;
            tensor_buf_kind_[t] = kind_out;
        }
        return it->second;
    }

    // Single global slot counter — every tensor gets a unique p[N] slot.
    // BufferKind is metadata for the host-side resolver (so we can verify
    // the right kind of tensor at the right slot) but doesn't change the
    // index space.
    std::string name = c->name ? c->name : "";
    if (name_contains(name, "cache_k") || name_contains(name, "cache_v")) {
        kind_out = BufferKind::KV_CACHE;
        kv_cache_tensors_.push_back(c);
    } else if (is_weight(c)) {
        kind_out = BufferKind::WEIGHT;
        weight_tensors_.push_back(c);
    } else {
        kind_out = BufferKind::INTERMEDIATE;
        intermediate_bufs_.push_back({c, ggml_nbytes(c)});
    }
    int idx = (int) tensor_slot_order_.size();
    tensor_slot_order_.push_back({c, kind_out});

    tensor_buf_idx_[c]  = idx;
    tensor_buf_kind_[c] = kind_out;
    if (t != c) {
        tensor_buf_idx_[t]  = idx;
        tensor_buf_kind_[t] = kind_out;
    }
    return idx;
}

// -------------------------------------------------------------------------
// Trace
// -------------------------------------------------------------------------
bool GraphCompiler::trace_one(ggml_tensor * node) {
    // Up front: NONE / RESHAPE / VIEW / PERMUTE / TRANSPOSE never run a
    // kernel; they're just metadata. We let them slide without further
    // checks (they ARE views by construction).
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            break;
    }

    // DEFENSIVE: views are OK *as long as* we pass each view's own
    // tensor->data (which already includes view_offs) to the kernel, NOT
    // the canonical's data. That's what try_push() in execute() does now.
    // Refuse only the cases the rest of the generator definitely can't
    // handle: SET_ROWS / CPY whose dst is a sliced (non-contiguous in
    // logical layout) view of something larger — the current SET_ROWS
    // codegen writes `base + idx*nb[1]`, which only matches reality
    // when nb[1] equals the parent's nb[1] (i.e. the view shares row
    // stride with its canonical). 99% of KV-cache views satisfy that,
    // but we double-check.
    if (node->view_src != nullptr) {
        if (node->nb[1] != node->view_src->nb[1]) {
            if (debug_enabled()) {
                fprintf(stderr, "[VE-GC] refuse: dst '%s' is a view of '%s' with non-matching nb[1] (%zu vs %zu)\n",
                        node->name ? node->name : "?",
                        node->view_src->name ? node->view_src->name : "?",
                        (size_t) node->nb[1], (size_t) node->view_src->nb[1]);
            }
            return false;
        }
    }

    TracedOp op;
    op.name = node->name ? node->name : "";
    for (int i = 0; i < 4; ++i) {
        op.ne[i] = node->ne[i];
        op.nb[i] = node->nb[i];
    }
    op.dst_type = node->type;

    output_tensors_.insert(node);
    output_tensors_.insert(canonical(node));

    op.dst_idx = assign_buffer(node, op.dst_kind);

    auto fill_src = [&](int slot, ggml_tensor * src,
                        int & out_idx, BufferKind & out_kind, int64_t * out_ne) {
        if (!src) {
            out_idx = -1;
            return;
        }
        out_idx = assign_buffer(src, out_kind);
        for (int i = 0; i < 4; ++i) out_ne[i] = src->ne[i];
        if (slot == 0) op.src_type = src->type;
    };
    fill_src(0, node->src[0], op.src0_idx, op.src0_kind, op.src0_ne);

    // For ROPE / SET_ROWS / GET_ROWS, src[1] is a position/index leaf tensor
    // (i32 / i64) — not a weight, not really an intermediate either.
    // Classifying it pulls it into the wrong slot at execute time, so just
    // leave src1 unassigned: the codegen for those ops reads the index from
    // the `pos` kernel arg instead of dereferencing src1.
    bool src1_is_position_index =
           (node->op == GGML_OP_ROPE
         || node->op == GGML_OP_SET_ROWS
         || node->op == GGML_OP_GET_ROWS);
    if (src1_is_position_index) {
        op.src1_idx = -1;
        if (node->src[1]) {
            for (int i = 0; i < 4; ++i) op.src1_ne[i] = node->src[1]->ne[i];
        }
    } else {
        fill_src(1, node->src[1], op.src1_idx, op.src1_kind, op.src1_ne);
    }

    if (node->src[2]) {
        BufferKind k;
        op.src2_idx = assign_buffer(node->src[2], k);
        op.src2_kind = k;
    }

    // GET_ROWS codegen below (OpType::GET_ROWS) branches per-op on
    // whether src0 is the embedding WEIGHT (real token-id lookup from
    // input[0]) or an intermediate (memcpy first row). Multi-GET_ROWS
    // graphs work as long as at most one is WEIGHT-backed — same
    // constraint the legacy backend operates under.

    // Reject multi-token (prompt-eval) shapes — the codegen bakes a
    // per-token element count into the .so and would run off the end
    // on the subsequent decode call. Shape semantics differ by op:
    //   - Hidden-state ops carry n_tokens in ne[2].
    //   - FLASH_ATTN_EXT Q is permuted to [D, N, H, B] — n_tokens is ne[1],
    //     ne[2] holds heads. K/V grow seq-wise but that's allowed.
    //   - SET_ROWS dst is the KV-cache view; ne[1..2] are layout, not tokens.
    //   - ne[3] > 1 is always a real batch dim.
    auto is_multi_token = [&](const ggml_tensor * t,
                              ggml_op for_op,
                              bool is_dst,
                              int src_slot) {
        if (!t) return false;
        if (t->ne[3] > 1) return true;
        if (for_op == GGML_OP_FLASH_ATTN_EXT) {
            // FA Q (src[0]) and dst exist in two shape conventions depending
            // on how the FA result is permuted: [D, N, H, B] (n_tokens in
            // ne[1], heads in ne[2]) OR [D, H, N, B] (heads in ne[1],
            // n_tokens in ne[2]). At decode either way N=1, so accept as
            // long as the *product* of ne[1] and ne[2] is small.
            //   K/V (src[1..2]) grow seq-wise on ne[1] — KV cache, allowed.
            if (is_dst || src_slot == 0) {
                // Accept any FA Q / dst where total "token slots" across
                // ne[1]*ne[2] equals heads*1 (decode) — i.e. one of the two
                // dims is exactly 1. Multi-token prompt eval has both > 1.
                return (t->ne[1] > 1) && (t->ne[2] > 1);
            }
            return false;
        }
        if (for_op == GGML_OP_SET_ROWS && is_dst) {
            // dst is the KV-cache view; its ne[1..2] are layout, not tokens.
            return false;
        }
        return t->ne[2] > 1;
    };
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = node->src[s];
        if (!src) continue;
        if (is_multi_token(src, node->op, /*is_dst=*/false, s)) {
            if (debug_enabled()) {
                fprintf(stderr, "[VE-GC] refuse: '%s' src[%d] is multi-token [%ld,%ld,%ld,%ld]\n",
                        node->name ? node->name : "?", s,
                        src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
            }
            return false;
        }
    }
    if (is_multi_token(node, node->op, /*is_dst=*/true, -1)) {
        if (debug_enabled()) {
            fprintf(stderr, "[VE-GC] refuse: '%s' dst is multi-token [%ld,%ld,%ld,%ld]\n",
                    node->name ? node->name : "?",
                    node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
        }
        return false;
    }
    if (node->op == GGML_OP_MUL_MAT && node->src[1] && node->src[1]->ne[1] > 1) {
        if (debug_enabled()) {
            fprintf(stderr, "[VE-GC] refuse: '%s' is batched MUL_MAT N=%ld (prompt eval)\n",
                    node->name ? node->name : "?", node->src[1]->ne[1]);
        }
        return false;
    }

    switch (node->op) {
        case GGML_OP_GET_ROWS:
            op.type = OpType::GET_ROWS;
            break;
        case GGML_OP_RMS_NORM:
            op.type = OpType::RMS_NORM;
            std::memcpy(&op.p.rms_norm.eps, node->op_params, sizeof(float));
            break;
        case GGML_OP_MUL:
            op.type = OpType::MUL;
            break;
        case GGML_OP_ADD:
            op.type = OpType::ADD;
            break;
        case GGML_OP_MUL_MAT:
            if (op.src_type == GGML_TYPE_BF16)      op.type = OpType::MUL_MAT_BF16;
            else if (op.src_type == GGML_TYPE_Q4_K) op.type = OpType::MUL_MAT_Q4K;
            else if (op.src_type == GGML_TYPE_VEBP) op.type = OpType::MUL_MAT_VEBP;
            else                                     op.type = OpType::MUL_MAT_F32;
            // Q4_K / VEBP inners are matvec-only (N=1). The compiler already
            // refuses N>1 MUL_MAT above (prompt eval), but guard explicitly.
            if ((op.type == OpType::MUL_MAT_Q4K || op.type == OpType::MUL_MAT_VEBP)
                && node->src[1] && node->src[1]->ne[1] != 1) {
                return false;
            }
            break;
        case GGML_OP_ROPE: {
            int32_t mode = 0, n_dims = 0;
            std::memcpy(&n_dims, (int32_t *) node->op_params + 1, sizeof(int32_t));
            std::memcpy(&mode,   (int32_t *) node->op_params + 2, sizeof(int32_t));
            if (mode & 8 /* MROPE */) return false;
            op.type = OpType::ROPE;
            op.p.rope.n_dims = n_dims;
            op.p.rope.mode   = mode;
            float fb, fs, ef, af;
            std::memcpy(&fb, (float *) node->op_params + 5, sizeof(float));
            std::memcpy(&fs, (float *) node->op_params + 6, sizeof(float));
            std::memcpy(&ef, (float *) node->op_params + 7, sizeof(float));
            std::memcpy(&af, (float *) node->op_params + 8, sizeof(float));
            if (ef != 0.0f) return false;   // YaRN not in our generator
            op.p.rope.freq_base  = fb;
            op.p.rope.freq_scale = fs;
            op.p.rope.mscale     = (af != 0.0f) ? af : 1.0f;
            break;
        }
        case GGML_OP_SET_ROWS:
            op.type = OpType::SET_ROWS;
            break;
        case GGML_OP_FLASH_ATTN_EXT: {
            op.type = OpType::FLASH_ATTN;
            std::memcpy(&op.p.flash_attn.scale,    (float *) node->op_params + 0, sizeof(float));
            std::memcpy(&op.p.flash_attn.max_bias, (float *) node->op_params + 1, sizeof(float));
            std::memcpy(&op.p.flash_attn.softcap,  (float *) node->op_params + 2, sizeof(float));
            if (op.p.flash_attn.max_bias != 0.0f || op.p.flash_attn.softcap != 0.0f) return false;
            const ggml_tensor * K = node->src[1];
            const ggml_tensor * V = node->src[2];
            if (!K || !V) return false;
            op.p.flash_attn.n_kv_heads = K->ne[2];
            op.p.flash_attn.kv_type    = (int) K->type;
            op.p.flash_attn.nb_k1      = K->nb[1];
            op.p.flash_attn.nb_k2      = K->nb[2];
            op.p.flash_attn.nb_v1      = V->nb[1];
            op.p.flash_attn.nb_v2      = V->nb[2];
            if (op.p.flash_attn.kv_type != (int) GGML_TYPE_BF16 &&
                op.p.flash_attn.kv_type != (int) GGML_TYPE_F32) {
                return false;
            }
            break;
        }
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            op.type = OpType::CPY;
            break;
        case GGML_OP_SOFT_MAX:
            op.type = OpType::SOFT_MAX;
            break;
        case GGML_OP_GLU: {
            // Only the SWIGLU split-mode variant (Llama-3 FFN). gate=src[0],
            // up=src[1], output = silu(gate) * up. Falls back to interpreter
            // for other GLU variants.
            if (ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU) return false;
            if (!node->src[0] || !node->src[1]) return false;
            if (node->src[0]->type != GGML_TYPE_F32 ||
                node->src[1]->type != GGML_TYPE_F32 ||
                node->type        != GGML_TYPE_F32) return false;
            op.type = OpType::GLU_SWIGLU;
            break;
        }
        case GGML_OP_UNARY:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ADD_ID:
        case GGML_OP_ARGSORT:
        default:
            return false;   // unsupported by the compiler — fall back
    }

    // Colmajor companion for BF16 MUL_MAT: allocate a new slot (no tensor
    // backing it; populated at execute time via
    // colmajor_weight_cache::get_or_create). The codegen emits a CBLAS
    // sgemv against the F32 col-major copy instead of the hand-tuned BF16
    // matvec. Cost: 4x weight memory in HBM, one-time transpose per
    // weight (~25ms for 3072x3072), and ~2x memory traffic per matvec
    // (F32 weights vs BF16). Benefit: CBLAS sgemv when N > 1 (no current
    // compiled-graph caller yet, since we only fuse decode) and a
    // platform for future GEMM-style fusion.
    //
    // Opt-IN with GGML_VE_COMPILE_COLMAJOR=1 — the default off because at
    // decode-shape N=1 the BF16 _inner matvec (sgemv_packed_bf16_unr) is
    // ~2-3% faster than colmajor sgemv. The path is wired up so we have
    // somewhere to land prompt-eval fusion later.
    static const bool use_colmajor =
        (std::getenv("GGML_VE_COMPILE_COLMAJOR") != nullptr);
    if (use_colmajor && op.type == OpType::MUL_MAT_BF16 && op.src0_idx >= 0) {
        ColmajorSpec sp;
        sp.src0_slot      = op.src0_idx;
        sp.M              = op.src0_ne[1];
        sp.K              = op.src0_ne[0];
        sp.companion_slot = (int) tensor_slot_order_.size();
        tensor_slot_order_.push_back({nullptr, BufferKind::WEIGHT_COLMAJOR});
        colmajor_specs_.push_back(sp);
        op.colmajor_idx = sp.companion_slot;
    }

    // Q4_K MUL_MAT: register two companion slots (qs + decoded hdr).
    // The original Q4_K weight slot stays in the table but isn't actually
    // touched by the kernel (we use qs/hdr instead); execute() fills both
    // companion slots from hbm_cache::get_or_upload_q4k_canon.
    if (op.type == OpType::MUL_MAT_Q4K && op.src0_idx >= 0) {
        Q4KSpec sp;
        sp.src0_slot = op.src0_idx;
        sp.M         = op.src0_ne[1];
        sp.K         = op.src0_ne[0];
        sp.qs_slot   = (int) tensor_slot_order_.size();
        tensor_slot_order_.push_back({nullptr, BufferKind::WEIGHT_Q4K_QS});
        sp.hdr_slot  = (int) tensor_slot_order_.size();
        tensor_slot_order_.push_back({nullptr, BufferKind::WEIGHT_Q4K_HDR});
        q4k_specs_.push_back(sp);
        op.q4k_qs_idx  = sp.qs_slot;
        op.q4k_hdr_idx = sp.hdr_slot;
    }

    // VEBP MUL_MAT: register three companion slots (sign/nz planes + scales).
    // execute() fills them from hbm_cache::get_or_upload_vebp.
    if (op.type == OpType::MUL_MAT_VEBP && op.src0_idx >= 0) {
        VebpSpec sp;
        sp.src0_slot = op.src0_idx;
        sp.M         = op.src0_ne[1];
        sp.K         = op.src0_ne[0];
        sp.ws_slot   = (int) tensor_slot_order_.size();
        tensor_slot_order_.push_back({nullptr, BufferKind::WEIGHT_VEBP_WS});
        sp.wn_slot   = (int) tensor_slot_order_.size();
        tensor_slot_order_.push_back({nullptr, BufferKind::WEIGHT_VEBP_WN});
        sp.wsc_slot  = (int) tensor_slot_order_.size();
        tensor_slot_order_.push_back({nullptr, BufferKind::WEIGHT_VEBP_WSC});
        vebp_specs_.push_back(sp);
        op.vebp_ws_idx  = sp.ws_slot;
        op.vebp_wn_idx  = sp.wn_slot;
        op.vebp_wsc_idx = sp.wsc_slot;
    }

    traced_ops_.push_back(op);
    return true;
}

bool GraphCompiler::trace(ggml_cgraph * cgraph) {
    traced_ops_.clear();
    tensor_buf_idx_.clear();
    tensor_buf_kind_.clear();
    output_tensors_.clear();
    weight_tensors_.clear();
    kv_cache_tensors_.clear();
    intermediate_bufs_.clear();
    tensor_slot_order_.clear();
    colmajor_specs_.clear();
    q4k_specs_.clear();
    vebp_specs_.clear();
    trace_valid_ = true;

    // Pre-pass: refuse if any weight tensor referenced by this cgraph
    // isn't already on a VE_HBM buffer. The compiled kernel takes the
    // slot's HBM pointer at face value, and we don't yet have the legacy
    // port's special "GET_ROWS weight on CPU mmap" upload+rewrite path.
    //
    // Trying to compile anyway and uploading via the host weight cache
    // produces garbage output: the GGUF loader leaves the embedding
    // table as a CPU mmap until first touch, and our cache hands back a
    // raw row-major HBM copy, but the compiled kernel's GET_ROWS code
    // assumes a layout/stride the legacy generator only sets up for
    // weights it explicitly knows about.
    //
    // Punting to the interpreter for these cgraphs is the right answer:
    // the interpreter goes through resolve_in()/the same hbm_weight_cache
    // and uploads the embedding to HBM as a side effect. The next cgraph
    // that touches the same weight will see it on HBM and compile
    // cleanly.
    // execute() walks the slot table assuming every real tensor lives in
    // VE_HBM; if any operand sits on CPU memory the slot fills with NULL
    // and the launch aborts. Trivial subgraphs that the scheduler hands us
    // for warmup or small ubatches sometimes have CPU-resident operands
    // even when the weights ARE on HBM. Refuse those before we waste a
    // 30-second NCC compile on a graph the executor can't run.
    // One-shot diagnostic: dump EVERY host-resident operand in this cgraph
    // (not just the first) so the CPU-operand set can be designed against.
    if (std::getenv("GGML_VE_GC_DUMP")) {
        fprintf(stderr, "[VE-GC-DUMP] cgraph: %d nodes, %d leafs\n",
                cgraph->n_nodes, cgraph->n_leafs);
        // Full per-node view: op, dst name/buffer, and each src's name+buffer.
        // A src whose producing node is NOT in this cgraph is a split-INPUT
        // (must be uploaded host->HBM); a dst consumed only outside is a
        // split-OUTPUT (must be downloaded HBM->host).
        std::set<const ggml_tensor *> produced_here;
        for (int i = 0; i < cgraph->n_nodes; ++i) if (cgraph->nodes[i]) produced_here.insert(cgraph->nodes[i]);
        for (int i = 0; std::getenv("GGML_VE_GC_DUMP")[0] == '2' && i < cgraph->n_nodes; ++i) {
            const ggml_tensor * n = cgraph->nodes[i];
            if (!n) continue;
            auto bufname = [](const ggml_tensor * t) -> const char * {
                if (!t || !t->buffer) return "<none>";
                auto bt = ggml_backend_buffer_get_type(t->buffer);
                return bt ? ggml_backend_buft_name(bt) : "<none>";
            };
            fprintf(stderr, "[VE-GC-DUMP] #%2d %-14s dst '%s'[%s]", i,
                    ggml_op_name(n->op), n->name ? n->name : "?", bufname(n));
            for (int s = 0; s < GGML_MAX_SRC && n->src[s]; ++s) {
                const ggml_tensor * c = n->src[s];
                while (c && c->view_src) c = c->view_src;
                bool ext = c && produced_here.find(c) == produced_here.end() &&
                           (!c || c->op == GGML_OP_NONE || produced_here.find(n->src[s]) == produced_here.end());
                fprintf(stderr, "  s%d '%s'[%s]%s", s, n->src[s]->name ? n->src[s]->name : "?",
                        bufname(n->src[s]), ext ? "*IN" : "");
            }
            fprintf(stderr, "\n");
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * n = cgraph->nodes[i];
            if (!n) continue;
            const ggml_tensor * srcs[GGML_MAX_SRC + 1] = { n, n->src[0], n->src[1], n->src[2] };
            for (int s = 0; s < (int) (sizeof(srcs)/sizeof(srcs[0])); ++s) {
                const ggml_tensor * t = srcs[s];
                if (!t) continue;
                ggml_backend_buffer_type_t bt = t->buffer ? ggml_backend_buffer_get_type(t->buffer) : nullptr;
                const char * bnn = bt ? ggml_backend_buft_name(bt) : nullptr;
                const bool h = !bnn || std::strncmp(bnn, "VE", 2) != 0 || !std::strstr(bnn, "_HBM");
                if (!h) continue;
                fprintf(stderr, "[VE-GC-DUMP]  op#%d %-12s %s '%s' type=%s ne=[%ld,%ld,%ld,%ld] buf=%s weight=%d leaf=%d\n",
                        i, ggml_op_name(n->op), s == 0 ? "DST" : "src",
                        t->name ? t->name : "?", ggml_type_name(t->type),
                        (long)t->ne[0],(long)t->ne[1],(long)t->ne[2],(long)t->ne[3],
                        bnn ? bnn : "<none>", is_weight(t) ? 1 : 0,
                        (t->op == GGML_OP_NONE) ? 1 : 0);
            }
        }
    }

    // Tensors PRODUCED by a compute op in this graph (canonical). VIEW/RESHAPE/
    // PERMUTE/TRANSPOSE/NONE don't produce data — they reference it.
    std::set<const ggml_tensor *> pre_produced;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (!n) continue;
        if (n->op == GGML_OP_VIEW    || n->op == GGML_OP_RESHAPE ||
            n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE ||
            n->op == GGML_OP_NONE) continue;
        pre_produced.insert(canonical(n));
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (!n) continue;
        const ggml_tensor * srcs[GGML_MAX_SRC + 1] = { n, n->src[0], n->src[1], n->src[2] };
        for (int s = 0; s < (int) (sizeof(srcs)/sizeof(srcs[0])); ++s) {
            const ggml_tensor * t = srcs[s];
            if (!t) continue;
            // Pure-metadata ops (VIEW / RESHAPE / PERMUTE / TRANSPOSE) carry
            // no data of their own; their canonical tensor's buffer is what
            // matters. Skip them here so we don't reject a graph for a
            // VIEW-of-an-HBM-tensor that the canonical-resolver handles fine.
            if (t->op == GGML_OP_VIEW    || t->op == GGML_OP_RESHAPE ||
                t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
                continue;
            }
            ggml_backend_buffer_type_t buft = t->buffer ? ggml_backend_buffer_get_type(t->buffer) : nullptr;
            const char * bn = buft ? ggml_backend_buft_name(buft) : nullptr;
            const bool host = !bn || std::strncmp(bn, "VE", 2) != 0 || !std::strstr(bn, "_HBM");
            if (!host) continue;

            // Self-containment gate. CPU-resident operands execute() can supply:
            //   - WEIGHTS (token_embd.weight etc.) -> one-time HBM cache upload.
            //   - LEAF inputs (op == NONE: inp_pos / KV index / mask) and the
            //     graph's own outputs -> per-token host<->HBM scratch.
            // What it must NOT accept is a computed INTERMEDIATE produced by a
            // DIFFERENT subgraph and read here (s>0, op != NONE, not produced
            // in this graph) — i.e. this cgraph is a middle fragment of a
            // scheduler-split decode. Chaining those compiled fragments through
            // host staging produces wrong values (Qwen3 per-head-norm models
            // fragment this way). The whole-decode-graph case (Llama) has no
            // such cross-fragment intermediate inputs and compiles cleanly.
            // Refuse middle fragments and let the interpreter run them.
            if (is_weight(t)) continue;                      // cache upload
            if (s == 0) continue;                            // this op's own dst (output)
            if (t->op == GGML_OP_NONE) continue;             // leaf input
            if (pre_produced.count(canonical(t))) continue;  // produced here
            if (debug_enabled()) {
                fprintf(stderr, "[VE-GC] refuse: cross-fragment intermediate input '%s' (%s) at op #%d — interpreter will run this fragment\n",
                        t->name ? t->name : "?", bn ? bn : "<no-buffer>", i);
            }
            trace_valid_ = false;
            return false;
        }
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (!trace_one(cgraph->nodes[i])) {
            if (debug_enabled()) {
                fprintf(stderr, "[VE-GC] cannot compile op #%d '%s' (%s)\n",
                        i, cgraph->nodes[i]->name ? cgraph->nodes[i]->name : "?",
                        ggml_op_name(cgraph->nodes[i]->op));
            }
            trace_valid_ = false;
            return false;
        }
    }

    if (debug_enabled()) {
        fprintf(stderr, "[VE-GC] traced %zu ops (%zu weights, %zu kv, %zu intermediates)\n",
                traced_ops_.size(), weight_tensors_.size(),
                kv_cache_tensors_.size(), intermediate_bufs_.size());
    }
    return true;
}

// -------------------------------------------------------------------------
// Code generation
// -------------------------------------------------------------------------
std::string GraphCompiler::compute_hash(const std::string & source) {
    std::hash<std::string> hasher;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016zx", hasher(source));
    return std::string(buf);
}

std::string GraphCompiler::cache_dir() {
    const char * home = std::getenv("HOME");
    std::string dir = std::string(home ? home : "/tmp") + "/.cache/ggml-ve-compiled";
    mkdir(dir.c_str(), 0755);
    return dir;
}

namespace {

std::string buf_ref(BufferKind k, int idx) {
    // Everything is addressed through a single per-call HBM pointer array.
    // Weights, KV-cache slots, and intermediates all get unique slot
    // numbers; the host fills the array at execute time by walking the
    // current cgraph in the same order as the trace.
    switch (k) {
        case BufferKind::WEIGHT:        return "p[" + std::to_string(idx) + "]";
        case BufferKind::KV_CACHE:      return "p[" + std::to_string(idx) + "]";
        case BufferKind::INTERMEDIATE:  return "p[" + std::to_string(idx) + "]";
        case BufferKind::INPUT:         return "input";
        case BufferKind::OUTPUT:        return "output";
    }
    return "BUF?";
}

// element count assuming single-token decode
int64_t per_token_n(const TracedOp & op) {
    if (op.ne[1] > 1 || op.ne[2] > 1) {
        return op.ne[0] * op.ne[1];
    }
    return op.ne[0];
}

std::string flit(float v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.9ef", v);
    return std::string(buf);
}

}  // namespace

std::string GraphCompiler::gen_op_code(const TracedOp & op, int idx) const {
    std::ostringstream ss;
    std::string dst  = buf_ref(op.dst_kind,  op.dst_idx);
    std::string src0 = (op.src0_idx >= 0) ? buf_ref(op.src0_kind, op.src0_idx) : "NULL";
    std::string src1 = (op.src1_idx >= 0) ? buf_ref(op.src1_kind, op.src1_idx) : "NULL";
    std::string src2 = (op.src2_idx >= 0) ? buf_ref(op.src2_kind, op.src2_idx) : "NULL";

    int64_t n = per_token_n(op);
    if (n == 0) n = op.ne[0];

    ss << "    // op " << idx << ": " << op_type_name(op.type)
       << "  '" << op.name << "'  n=" << n << "\n";

    switch (op.type) {
        case OpType::GET_ROWS: {
            // Embedding lookup. Token id is the first i32 of the input HMEM.
            // Whole op runs on a single thread (small: ~3072 elements);
            // `#pragma omp single` gives an implicit barrier at end so the
            // next op sees the result.
            ss << "    #pragma omp single\n";
            ss << "    {\n";
            if (op.src0_kind != BufferKind::WEIGHT) {
                ss << "        memcpy(" << dst << ", " << src0 << ", "
                   << (op.ne[0] * sizeof(float)) << ");\n";
                ss << "    }\n";
                break;
            }
            ss << "        int32_t tok = ((int32_t*)input)[0];\n";
            ss << "        int  edim = " << op.ne[0] << ";\n";
            if (op.src_type == GGML_TYPE_BF16) {
                ss << "        const uint32_t* row32 = (const uint32_t*)"
                   << src0 << " + ((int64_t)tok * edim) / 2;\n";
                ss << "        uint32_t* dst32 = (uint32_t*)" << dst << ";\n";
                ss << "        int half = edim / 2;\n";
                ss << "        #pragma _NEC ivdep\n";
                ss << "        for (int i = 0; i < half; i++) {\n";
                ss << "            uint32_t p = row32[i];\n";
                ss << "            dst32[2*i  ] = (p & 0xFFFFu) << 16;\n";
                ss << "            dst32[2*i+1] = p & 0xFFFF0000u;\n";
                ss << "        }\n";
            } else {
                ss << "        memcpy(" << dst << ", (float*)" << src0
                   << " + (int64_t)tok * edim, edim * sizeof(float));\n";
            }
            ss << "    }\n";
            break;
        }

        case OpType::RMS_NORM: {
            int64_t cols  = op.ne[0];
            int64_t rows  = (op.ne[1] > 1) ? op.ne[1] : 1;
            bool per_head = (cols <= 256 && rows > 1);
            if (per_head) {
                // Per-row independent; share rows across the team via omp for.
                ss << "    {\n";
                ss << "        int rows = " << rows << ", cols = " << cols << ";\n";
                ss << "        float eps = " << flit(op.p.rms_norm.eps) << ";\n";
                ss << "        #pragma omp for\n";
                ss << "        for (int r = 0; r < rows; r++) {\n";
                ss << "            const float* x = (const float*)" << src0 << " + r * cols;\n";
                ss << "            float* y = (float*)" << dst << " + r * cols;\n";
                ss << "            float sumsq = 0.f;\n";
                ss << "            for (int j = 0; j < cols; j++) sumsq += x[j] * x[j];\n";
                ss << "            float inv = 1.f / sqrtf(sumsq / cols + eps);\n";
                ss << "            for (int j = 0; j < cols; j++) y[j] = inv * x[j];\n";
                ss << "        }\n";
                ss << "    }\n";
            } else {
                // Single-row reduction. NCC's openmp doesn't reliably do
                // a worksharing-reduction when the reduction variable is
                // declared inside the enclosing parallel region (and the
                // outer wrapper is `#pragma omp parallel`, so every scope
                // here IS inside the parallel). We compute the reduction
                // by hand via a shared array, then the normalize loop is
                // a normal #pragma omp for.
                ss << "    {\n";
                ss << "        int cols = " << cols << ";\n";
                ss << "        float eps = " << flit(op.p.rms_norm.eps) << ";\n";
                ss << "        const float* x = (const float*)" << src0 << ";\n";
                ss << "        float* y = (float*)" << dst << ";\n";
                ss << "        static float __ssq[8];  /* per-thread partials */\n";
                ss << "        int tid = omp_get_thread_num();\n";
                ss << "        int nt  = omp_get_num_threads();\n";
                ss << "        float local = 0.f;\n";
                ss << "        #pragma omp for nowait\n";
                ss << "        for (int j = 0; j < cols; j++) local += x[j] * x[j];\n";
                ss << "        __ssq[tid] = local;\n";
                ss << "        #pragma omp barrier\n";
                ss << "        float sumsq = 0.f;\n";
                ss << "        for (int k = 0; k < nt; k++) sumsq += __ssq[k];\n";
                ss << "        float inv = 1.f / sqrtf(sumsq / cols + eps);\n";
                ss << "        #pragma omp for\n";
                ss << "        for (int j = 0; j < cols; j++) y[j] = inv * x[j];\n";
                ss << "    }\n";
            }
            break;
        }

        case OpType::MUL: {
            int64_t src1_n = op.src1_ne[0];
            if (op.src1_ne[1] > 1 && op.src1_ne[2] > 1) src1_n = op.src1_ne[0] * op.src1_ne[1];
            if (src1_n > 0 && src1_n < n && n % src1_n == 0) {
                int64_t repeats = n / src1_n;
                ss << "    #pragma omp for\n";
                ss << "    for (int64_t r = 0; r < " << repeats << "; r++) {\n";
                ss << "        for (int64_t i = 0; i < " << src1_n << "; i++) {\n";
                ss << "            ((float*)" << dst << ")[r*" << src1_n << "+i] = "
                   << "((float*)" << src0 << ")[r*" << src1_n << "+i] * "
                   << "((float*)" << src1 << ")[i];\n";
                ss << "        }\n";
                ss << "    }\n";
            } else {
                ss << "    #pragma omp for\n";
                ss << "    for (int i = 0; i < " << n << "; i++) {\n";
                ss << "        ((float*)" << dst << ")[i] = "
                   << "((float*)" << src0 << ")[i] * ((float*)" << src1 << ")[i];\n";
                ss << "    }\n";
            }
            break;
        }

        case OpType::ADD:
            ss << "    #pragma omp for\n";
            ss << "    for (int i = 0; i < " << n << "; i++) {\n";
            ss << "        ((float*)" << dst << ")[i] = "
               << "((float*)" << src0 << ")[i] + ((float*)" << src1 << ")[i];\n";
            ss << "    }\n";
            break;

        case OpType::MUL_MAT_F32: {
            // Single-thread external call. Wrap in omp single so the team
            // synchronises after.
            int64_t M = op.src0_ne[1], K = op.src0_ne[0];
            ss << "    #pragma omp single\n";
            ss << "    {\n";
            ss << "        ve_f32_matvec_ptr((float*)" << dst << ", (const float*)"
               << src0 << ", (const float*)" << src1
               << ", " << M << ", " << K << ");\n";
            ss << "    }\n";
            break;
        }

        case OpType::MUL_MAT_BF16: {
            int64_t M = op.src0_ne[1], K = op.src0_ne[0];
            if (op.colmajor_idx >= 0) {
                // Colmajor + CBLAS fast path. W_colmajor is M-by-K stored
                // column-major (lda=M), so column k is at &W[k*M]. CBLAS
                // sgemv with CblasColMajor / CblasNoTrans computes
                //   y[i] = sum_k W[i + k*M] * x[k]   (i in [imin, imax)).
                // We partition rows ourselves with #pragma omp for so the
                // outer omp team does the work; otherwise cblas_sgemv runs
                // single-threaded per-call.
                std::string cm = "p[" + std::to_string(op.colmajor_idx) + "]";
                ss << "    {\n";
                ss << "        int M = " << M << ", K = " << K << ";\n";
                ss << "        const float* W = (const float*)" << cm << ";\n";
                ss << "        const float* xv = (const float*)" << src1 << ";\n";
                ss << "        float* yv = (float*)" << dst << ";\n";
                ss << "        int nt = omp_get_num_threads();\n";
                ss << "        int chunk = (M + nt - 1) / nt;\n";
                ss << "        #pragma omp for\n";
                ss << "        for (int g = 0; g < nt; g++) {\n";
                ss << "            int imin = g * chunk;\n";
                ss << "            int imax = imin + chunk;\n";
                ss << "            if (imax > M) imax = M;\n";
                ss << "            if (imin < imax) {\n";
                ss << "                cblas_sgemv(CblasColMajor, CblasNoTrans,\n";
                ss << "                            imax - imin, K, 1.0f,\n";
                ss << "                            W + imin, M, xv, 1, 0.0f,\n";
                ss << "                            yv + imin, 1);\n";
                ss << "            }\n";
                ss << "        }\n";
                ss << "    }\n";
            } else {
                // _inner variant uses `#pragma omp for` internally — the
                // implicit barrier at end-of-for synchronises the team.
                ss << "    ve_bf16_matvec_rowmajor_ptr_inner((float*)" << dst
                   << ", (const uint16_t*)" << src0 << ", (const float*)" << src1
                   << ", " << M << ", " << K << ");\n";
            }
            break;
        }

        case OpType::MUL_MAT_Q4K: {
            // Q4_K matvec using the canonical-split + pre-decoded-header
            // inner. qs and hdr come from the Q4KSpec companion slots
            // (filled at execute time from hbm_cache::get_or_upload_q4k_canon).
            int64_t M = op.src0_ne[1], K = op.src0_ne[0];
            std::string qs_p  = "p[" + std::to_string(op.q4k_qs_idx)  + "]";
            std::string hdr_p = "p[" + std::to_string(op.q4k_hdr_idx) + "]";
            ss << "    ve_q4k_matvec_rowmajor_ptr_inner((float*)" << dst
               << ", (const unsigned char*)" << qs_p
               << ", (const unsigned char*)" << hdr_p
               << ", (const float*)" << src1
               << ", " << M << ", " << K << ");\n";
            break;
        }

        case OpType::MUL_MAT_VEBP: {
            // VEBP ternary matvec. Interleaved sign/nz planes + per-group
            // scales come from the VebpSpec companion slots (filled at
            // execute via hbm_cache::get_or_upload_vebp).
            int64_t M = op.src0_ne[1], K = op.src0_ne[0];
            std::string ws_p  = "p[" + std::to_string(op.vebp_ws_idx)  + "]";
            std::string wn_p  = "p[" + std::to_string(op.vebp_wn_idx)  + "]";
            std::string wsc_p = "p[" + std::to_string(op.vebp_wsc_idx) + "]";
            ss << "    ve_vebp_matvec_ptr_inner((float*)" << dst
               << ", (const unsigned long*)" << ws_p
               << ", (const unsigned long*)" << wn_p
               << ", (const float*)" << wsc_p
               << ", (const float*)" << src1
               << ", " << M << ", " << K << ");\n";
            break;
        }

        case OpType::ROPE: {
            int n_heads = (op.ne[1] > 1) ? op.ne[1] : 1;
            int head_sz = (int) op.ne[0];
            bool neox = (op.p.rope.mode & 2) != 0;
            // The host passes the token position via the `pos` function arg.
            // Don't dereference src1 — leaf-tensor classification is not
            // stable across the positional weight-resolution we do at execute
            // time, and src1's slot might point at a model weight.
            ss << "    {\n";
            ss << "        int32_t pos_i = (int32_t) pos;\n";
            (void) src1;
            ss << "        int head_size = " << head_sz << ";\n";
            ss << "        int n_heads = " << n_heads << ";\n";
            ss << "        float freq_base  = " << flit(op.p.rope.freq_base) << ";\n";
            ss << "        float freq_scale = " << flit(op.p.rope.freq_scale) << ";\n";
            ss << "        float mscale     = " << flit(op.p.rope.mscale) << ";\n";
            ss << "        const float* in  = (const float*)" << src0 << ";\n";
            ss << "        float*       out = (float*)"      << dst  << ";\n";
            if (neox) {
                ss << "        int half = head_size / 2;\n";
                ss << "        float theta_scale = powf(freq_base, -2.f / head_size);\n";
                ss << "        #pragma omp for\n";
                ss << "        for (int h = 0; h < n_heads; h++) {\n";
                ss << "            float theta = (float)pos_i * freq_scale;\n";
                ss << "            for (int i = 0; i < half; i++) {\n";
                ss << "                float c = cosf(theta) * mscale;\n";
                ss << "                float s = sinf(theta) * mscale;\n";
                ss << "                int a = h * head_size + i;\n";
                ss << "                int b = a + half;\n";
                ss << "                float v0 = in[a], v1 = in[b];\n";
                ss << "                out[a] = v0 * c - v1 * s;\n";
                ss << "                out[b] = v0 * s + v1 * c;\n";
                ss << "                theta *= theta_scale;\n";
                ss << "            }\n";
                ss << "        }\n";
            } else {
                ss << "        #pragma omp for\n";
                ss << "        for (int h = 0; h < n_heads; h++) {\n";
                ss << "            for (int i = 0; i < head_size; i += 2) {\n";
                ss << "                float freq = 1.f / powf(freq_base, (float)i / (float)head_size);\n";
                ss << "                float val  = (float)pos_i * freq * freq_scale;\n";
                ss << "                float c = cosf(val) * mscale;\n";
                ss << "                float s = sinf(val) * mscale;\n";
                ss << "                int idx = h * head_size + i;\n";
                ss << "                float v0 = in[idx], v1 = in[idx + 1];\n";
                ss << "                out[idx    ] = v0 * c - v1 * s;\n";
                ss << "                out[idx + 1] = v0 * s + v1 * c;\n";
                ss << "            }\n";
                ss << "        }\n";
            }
            ss << "    }\n";
            break;
        }

        case OpType::SET_ROWS: {
            // For single-token decode the write row is the current position.
            // We pass `pos` as a kernel arg so we don't have to dereference
            // src1's leaf index tensor (whose slot wouldn't survive the
            // positional weight-resolution at execute time).
            int64_t cols = op.src0_ne[0];
            size_t  dst_row_bytes = (size_t) op.nb[1];
            size_t  elem_bytes;
            switch (op.dst_type) {
                case GGML_TYPE_F32:  elem_bytes = 4; break;
                case GGML_TYPE_BF16:
                case GGML_TYPE_F16:  elem_bytes = 2; break;
                default:             elem_bytes = 4; break;
            }
            (void) elem_bytes;
            (void) src1;
            if (op.dst_type == GGML_TYPE_F32) {
                ss << "    #pragma omp single\n";
                ss << "    {\n";
                ss << "        int64_t idx0 = (int64_t) pos;\n";
                ss << "        const float* src = (const float*)" << src0 << ";\n";
                ss << "        char* base = (char*)" << dst << ";\n";
                ss << "        memcpy(base + idx0 * " << dst_row_bytes << ", src, "
                   << cols << " * 4);\n";
                ss << "    }\n";
            } else if (op.dst_type == GGML_TYPE_BF16) {
                ss << "    {\n";
                ss << "        int64_t idx0 = (int64_t) pos;\n";
                ss << "        int cols = " << cols << ";\n";
                ss << "        const float* src = (const float*)" << src0 << ";\n";
                ss << "        char* row = (char*)" << dst << " + idx0 * " << dst_row_bytes << ";\n";
                ss << "        uint16_t* drow = (uint16_t*)row;\n";
                ss << "        #pragma omp for\n";
                ss << "        for (int j = 0; j < cols; j++) {\n";
                ss << "            uint32_t u; memcpy(&u, &src[j], 4);\n";
                ss << "            drow[j] = (uint16_t)(u >> 16);\n";
                ss << "        }\n";
                ss << "    }\n";
            } else {
                ss << "    #pragma omp single\n";
                ss << "    {\n";
                ss << "        // F16 SET_ROWS not in compiled path\n";
                ss << "    }\n";
            }
            break;
        }

        case OpType::FLASH_ATTN: {
            // GGML FLASH_ATTN_EXT Q tensor layout is [D, N, H, B]:
            //   ne[0]=head_dim, ne[1]=n_tokens, ne[2]=n_heads, ne[3]=batch.
            // We only compile decode (N=1), so n_q_heads lives in ne[2].
            // Some old/synthetic shapes had heads collapsed into ne[1]; fall
            // back to ne[1] when ne[2] is 1 so we don't silently produce a
            // single-head kernel.
            int head_dim  = (int) op.src0_ne[0];
            int n_q_heads = (op.src0_ne[2] > 1) ? (int) op.src0_ne[2]
                                                : (int) op.src0_ne[1];
            int n_kv_heads= (int) op.p.flash_attn.n_kv_heads;
            int seq_len_curr_pos = -1;  // pass position+1 from runtime
            (void) seq_len_curr_pos;
            ss << "    {\n";
            ss << "        int head_dim   = " << head_dim   << ";\n";
            ss << "        int n_q_heads  = " << n_q_heads  << ";\n";
            ss << "        int n_kv_heads = " << n_kv_heads << ";\n";
            ss << "        int seq_len    = (int)(pos + 1);\n";
            ss << "        float scale    = " << flit(op.p.flash_attn.scale) << ";\n";
            ss << "        size_t nb_k1=" << op.p.flash_attn.nb_k1 << "u, nb_k2=" << op.p.flash_attn.nb_k2 << "u;\n";
            ss << "        size_t nb_v1=" << op.p.flash_attn.nb_v1 << "u, nb_v2=" << op.p.flash_attn.nb_v2 << "u;\n";
            if (std::getenv("GGML_VE_KERNEL_TRACE")) {
                ss << "        #pragma omp single\n"
                   << "        {fprintf(stderr,\"[VE-K]   FA args: hd=%d nqh=%d nkvh=%d seq=%d pos=%ld\\n\","
                   << "head_dim,n_q_heads,n_kv_heads,seq_len,(long)pos);fflush(stderr);}\n";
            }
            ss << "        float* outp = (float*)" << dst  << ";\n";
            ss << "        const float* qp = (const float*)" << src0 << ";\n";
            ss << "        const void*  kp = (const void*)"  << src1 << ";\n";
            ss << "        const void*  vp = (const void*)"  << src2 << ";\n";
            if (op.p.flash_attn.kv_type == (int) GGML_TYPE_BF16) {
                // _inner: uses `#pragma omp for` internally, implicit barrier.
                ss << "        attention_f32q_bf16kv_fused_gqa_inner(outp, qp, kp, vp, "
                   << "head_dim, n_q_heads, n_kv_heads, seq_len, scale, "
                   << "nb_k1, nb_k2, nb_v1, nb_v2);\n";
            } else {
                // F32 KV path doesn't have an _inner variant yet — fall back
                // to omp single. Bumping this is task #16 follow-up.
                ss << "        #pragma omp single\n";
                ss << "        {\n";
                ss << "            attention_f32_raw_gqa_stride_omp(outp, qp, kp, vp, "
                   << "head_dim, n_q_heads, n_kv_heads, seq_len, scale, "
                   << "nb_k1, nb_k2, nb_v1, nb_v2);\n";
                ss << "        }\n";
            }
            ss << "    }\n";
            break;
        }

        case OpType::CPY: {
            int elem_bytes = (op.dst_type == GGML_TYPE_F32  ? 4
                             : op.dst_type == GGML_TYPE_BF16 ? 2
                             : op.dst_type == GGML_TYPE_F16  ? 2 : 4);
            int64_t total_elems = op.ne[0] * op.ne[1] * op.ne[2] * op.ne[3];
            if (total_elems == 0) total_elems = op.ne[0];
            ss << "    #pragma omp single\n";
            ss << "    memcpy(" << dst << ", " << src0 << ", "
               << (size_t) (total_elems * elem_bytes) << "u);\n";
            break;
        }

        case OpType::GLU_SWIGLU: {
            // SWIGLU: y = silu(gate) * up. gate/up/dst are F32 with the
            // same shape; row count is ggml_nrows(gate) = ne[1]*ne[2]*ne[3].
            int64_t nc = op.ne[0];
            int64_t nr = (op.ne[1] > 0 ? op.ne[1] : 1) *
                         (op.ne[2] > 0 ? op.ne[2] : 1) *
                         (op.ne[3] > 0 ? op.ne[3] : 1);
            ss << "    {\n";
            ss << "        int nc = " << nc << ", nr = " << nr << ";\n";
            ss << "        float* y    = (float*)" << dst  << ";\n";
            ss << "        float* gate = (float*)" << src0 << ";\n";
            ss << "        float* up   = (float*)" << src1 << ";\n";
            // Inline SWIGLU over the flat element range. gate/up/y are
            // contiguous row-major [nc,nr], so we parallelise the whole
            // nc*nr range with one `#pragma omp for` (all 8 threads share it;
            // the external swiglu_hbm_full_inner only split over nr, leaving
            // 7 threads idle at decode where nr==1). Clamp before expf — the
            // VE's vectorised expf returns NaN past |x|~88 (see CLAUDE.md).
            ss << "        long total = (long)nc * (nr > 0 ? nr : 1);\n";
            ss << "        #pragma omp for\n";
            ss << "        for (long i = 0; i < total; i++) {\n";
            ss << "            float g = gate[i];\n";
            ss << "            if (g < -80.0f) g = -80.0f;\n";
            ss << "            if (g >  80.0f) g =  80.0f;\n";
            ss << "            y[i] = (g / (1.0f + expf(-g))) * up[i];\n";
            ss << "        }\n";
            ss << "    }\n";
            break;
        }

        case OpType::SOFT_MAX: {
            int rows = (op.ne[1] > 1) ? (int) op.ne[1] : 1;
            int cols = (int) op.ne[0];
            ss << "    #pragma omp single\n";
            ss << "    {\n";
            ss << "        int rows = " << rows << ", cols = " << cols << ";\n";
            ss << "        for (int r = 0; r < rows; r++) {\n";
            ss << "            float* x = (float*)" << src0 << " + r * cols;\n";
            ss << "            float* y = (float*)" << dst  << " + r * cols;\n";
            ss << "            float m = x[0];\n";
            ss << "            for (int i = 1; i < cols; i++) if (x[i] > m) m = x[i];\n";
            ss << "            float sum = 0.f;\n";
            ss << "            for (int i = 0; i < cols; i++) {\n";
            ss << "                float d = x[i] - m;\n";
            ss << "                if (d < -80.f) d = -80.f;\n";
            ss << "                y[i] = expf(d);\n";
            ss << "                sum += y[i];\n";
            ss << "            }\n";
            ss << "            float inv = 1.f / sum;\n";
            ss << "            for (int i = 0; i < cols; i++) y[i] *= inv;\n";
            ss << "        }\n";
            ss << "    }\n";
            break;
        }

        default:
            ss << "    /* UNSUPPORTED op " << op_type_name(op.type) << " */\n";
            break;
    }
    return ss.str();
}

std::string GraphCompiler::generate_source(const std::string & func_name, int64_t n_ctx) const {
    std::ostringstream ss;

    ss << "// Auto-generated VE graph kernel\n";
    ss << "// ops=" << traced_ops_.size()
       << " weights=" << weight_tensors_.size()
       << " kv=" << kv_cache_tensors_.size()
       << " inter=" << intermediate_bufs_.size()
       << " n_ctx=" << n_ctx << "\n\n";

    ss << "#include <stdint.h>\n";
    ss << "#include <stdio.h>\n";
    ss << "#include <string.h>\n";
    ss << "#include <math.h>\n";
    ss << "#include <omp.h>\n";
    ss << "#include <cblas.h>\n";   // NEC NLC CBLAS — for the colmajor MUL_MAT fast path
    ss << "#include <veda/device.h>\n\n";

    // External kernels from libve_sgemv.so. The "_inner" variants assume the
    // caller is inside a #pragma omp parallel region and use `#pragma omp for`
    // to share work — implicit barrier at end-of-for synchronises the team.
    // The "_omp" variants spawn their own region and are used only by the
    // OpType::FLASH_ATTN F32 fallback (kept until we add an _inner there).
    ss << "extern void ve_bf16_matvec_rowmajor_ptr_inner(float* y, const uint16_t* W, const float* x, int M, int K);\n";
    ss << "extern void ve_q4k_matvec_rowmajor_ptr_inner(float* y, const unsigned char* qs, const unsigned char* hdr, const float* x, int M, int K);\n";
    ss << "extern void ve_vebp_matvec_ptr_inner(float* y, const unsigned long* ws, const unsigned long* wn, const float* wsc, const float* x, int M, int K);\n";
    ss << "extern void ve_f32_matvec_ptr(float* y, const float* W, const float* x, int M, int K);\n";
    ss << "extern void attention_f32_raw_gqa_stride_omp(float* out, const float* q, const void* k, const void* v,"
       << " int head_dim, int n_q_heads, int n_kv_heads, int seq_len, float scale,"
       << " size_t nb_k1, size_t nb_k2, size_t nb_v1, size_t nb_v2);\n";
    ss << "extern void attention_f32q_bf16kv_fused_gqa_inner(float* out, const float* q, const void* k, const void* v,"
       << " int head_dim, int n_q_heads, int n_kv_heads, int seq_len, float scale,"
       << " size_t nb_k1, size_t nb_k2, size_t nb_v1, size_t nb_v2);\n";
    ss << "extern void swiglu_hbm_full_inner(float* y, float* gate, float* up, int nc, int nr);\n\n";

    int n_slots = (int) tensor_slot_order_.size();

    // Entry point. Single tensor-pointer array `tptr_hbm` of length n_slots
    // — host array passed by-pointer via vedaArgsSetPtr (legacy pattern).
    // Slot i = HBM pointer for the tensor encountered at index i during
    // trace; the host fills the same order when it walks the current
    // cgraph at execute time. Weights, KV cache, and intermediates all
    // share this array.
    ss << "uint64_t " << func_name << "(\n";
    ss << "    VEDAdeviceptr* tptr_hbm,  // [" << n_slots << "] tensor HBM ptrs\n";
    ss << "    void* input,              // HMEM token id (i32)\n";
    ss << "    void* output,             // HMEM logits out (f32)\n";
    ss << "    int64_t pos,              // current decode position\n";
    ss << "    int64_t n_ctx) {\n";
    ss << "    (void)n_ctx;\n";

    // Convert all tensor HBM pointers to raw VE addresses. We re-resolve
    // every call (cheap: just translates an opaque handle to a flat VE
    // pointer) because both the ggml allocator and the KV cache can rebind
    // buffers between iterations.
    // GGML_VE_COMPILE_NOOP=1 → emit a kernel that does literally nothing.
    // Helps isolate whether the SIGSEGV we see is in the launch / arg /
    // module path vs in our op generators or in vedaMemPtr() on the slots.
    bool gen_noop = (std::getenv("GGML_VE_COMPILE_NOOP") != nullptr);
    if (gen_noop) {
        ss << "    (void)tptr_hbm; (void)input; (void)output; (void)pos;\n";
        ss << "    /* truly empty body */\n";
        ss << "    return 0;\n";
        ss << "}\n";
        return ss.str();
    }

    // Slot resolution.
    //
    // Weight HBM pointers are stable across calls (the ggml allocator
    // places weights once at model load), so vedaMemPtr resolutions for
    // those slots can be cached in a function-local `static` array.
    // KV cache and intermediate slots can rebind across cgraph computes
    // (the allocator reuses scratch ranges), so we re-resolve those
    // every call.
    //
    // The legacy backend's graph compiler did the same thing; that's
    // codex finding #4 — ~10-30 ms/token of vedaMemPtr overhead the
    // un-cached version pays.
    ss << "    void* p[" << (n_slots > 0 ? n_slots : 1) << "];\n";
    ss << "    static void*           w_cached[" << (n_slots > 0 ? n_slots : 1) << "];\n";
    ss << "    static VEDAdeviceptr   w_hbm_cached[" << (n_slots > 0 ? n_slots : 1) << "];\n";
    ss << "    static int             w_initialized = 0;\n";
    ss << "    if (!w_initialized) {\n";
    for (int i = 0; i < n_slots; ++i) {
        BufferKind k = tensor_slot_order_[i].second;
        // Weight + colmajor-companion entries are stable; cache them.
        if (k == BufferKind::WEIGHT || k == BufferKind::WEIGHT_COLMAJOR) {
            ss << "        w_hbm_cached[" << i << "] = tptr_hbm[" << i << "];\n";
            ss << "        if (tptr_hbm[" << i << "] != 0) {\n";
            ss << "            if (vedaMemPtr(&w_cached[" << i << "], tptr_hbm[" << i << "]) != 0) return 1;\n";
            ss << "        } else w_cached[" << i << "] = 0;\n";
        }
    }
    ss << "        w_initialized = 1;\n";
    ss << "    }\n";

    // Each call: walk slots, re-resolve non-weight ones, copy weight ones
    // from the static cache. If a weight HBM pointer ever changes (e.g.
    // ggml allocator re-binds the weight buffer), fall back to a fresh
    // vedaMemPtr instead of trusting the cache.
    for (int i = 0; i < n_slots; ++i) {
        BufferKind k = tensor_slot_order_[i].second;
        bool cacheable = (k == BufferKind::WEIGHT || k == BufferKind::WEIGHT_COLMAJOR);
        if (cacheable) {
            ss << "    if (tptr_hbm[" << i << "] == w_hbm_cached[" << i << "]) {\n";
            ss << "        p[" << i << "] = w_cached[" << i << "];\n";
            ss << "    } else if (tptr_hbm[" << i << "] != 0) {\n";
            ss << "        if (vedaMemPtr(&p[" << i << "], tptr_hbm[" << i << "]) != 0) return 1;\n";
            ss << "        w_hbm_cached[" << i << "] = tptr_hbm[" << i << "];\n";
            ss << "        w_cached[" << i << "]     = p[" << i << "];\n";
            ss << "    } else p[" << i << "] = 0;\n";
        } else {
            ss << "    if (tptr_hbm[" << i << "] != 0) {\n";
            ss << "        if (vedaMemPtr(&p[" << i << "], tptr_hbm[" << i << "]) != 0) return 1;\n";
            ss << "    } else p[" << i << "] = 0;\n";
        }
    }
    ss << "\n";

    // One outer #pragma omp parallel for the whole cgraph. Every op inside
    // either calls an "_inner" external (which uses `#pragma omp for`) or
    // emits its own work-sharing directive (`#pragma omp for`, `single`).
    // The implicit barrier at end-of-construct synchronises before the
    // next op runs — no explicit `#pragma omp barrier` needed.
    //
    // Threads fork once per kernel launch instead of once per op (~18×
    // for the attention+FFN block on Llama-3.2-3B).
    ss << "    #pragma omp parallel num_threads(8)\n";
    ss << "    {\n";

    const bool ktrace = (std::getenv("GGML_VE_KERNEL_TRACE") != nullptr);
    for (size_t i = 0; i < traced_ops_.size(); ++i) {
        if (ktrace) {
            // Synchronised checkpoint: barrier forces the whole team to finish
            // the previous op before one thread prints, so the last line before
            // a VE fault is exactly the op that crashed.
            ss << "    #pragma omp barrier\n";
            ss << "    #pragma omp single\n    {fprintf(stderr,\"[VE-K] op " << i
               << " " << op_type_name(traced_ops_[i].type) << " dst_slot="
               << traced_ops_[i].dst_idx << " START\\n\");fflush(stderr);}\n";
        }
        ss << gen_op_code(traced_ops_[i], (int) i);
        if (ktrace) {
            ss << "    #pragma omp barrier\n";
            ss << "    #pragma omp single\n    {fprintf(stderr,\"[VE-K] op " << i
               << " DONE\\n\");fflush(stderr);}\n";
        }
    }

    ss << "    }\n";   // end #pragma omp parallel

    // Copy the very last op's output into the host-supplied `output` HMEM.
    // Only useful when this is the very last compute (logits); for layer
    // subgraphs the caller doesn't need this since the result already lives
    // in HBM via the corresponding slot.
    if (!traced_ops_.empty()) {
        const TracedOp & last = traced_ops_.back();
        ss << "\n    if (output) memcpy(output, p[" << last.dst_idx
           << "], " << (last.ne[0] * sizeof(float)) << "u);\n";
    }

    ss << "    return 0;\n";
    ss << "}\n";
    return ss.str();
}

// -------------------------------------------------------------------------
// Compile / load
// -------------------------------------------------------------------------
bool GraphCompiler::compile_source(const std::string & source, const std::string & so_path) {
    std::string src_path = so_path + ".c";
    {
        std::ofstream f(src_path);
        if (!f) {
            fprintf(stderr, "[VE-GC] cannot write %s\n", src_path.c_str());
            return false;
        }
        f << source;
    }

    const char * kernels_dir = std::getenv("GGML_VE_VEDA_KERNELS_DIR");
    std::string  kdir = kernels_dir ? kernels_dir
                                    : (std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp")
                                       + "/claude_workspace/ggml-ve-veda");

    // Link NEC NLC CBLAS so the colmajor MUL_MAT fast path can call
    // cblas_sgemv from the compiled .so. We use blas_openmp to avoid a
    // separate thread team — the outer wrapper already pins us inside a
    // #pragma omp parallel of size 8.
    const char * nlc_dir = std::getenv("NLC_HOME");  // set by nlcvars.sh
    std::string nlc_lib;
    if (nlc_dir && *nlc_dir) {
        nlc_lib = std::string(nlc_dir) + "/lib";
    } else {
        nlc_lib = "/opt/nec/ve/nlc/3.1.0/lib";
    }

    std::ostringstream cmd;
    cmd << "/opt/nec/ve/bin/ncc -O4 -fopenmp -fpic -shared "
        << "-I/opt/nec/ve/share/veoffload-veda/include "
        << "-I" << nlc_lib << "/../include "
        << "-L" << kdir << " -Wl,-rpath," << kdir << " "
        << "-L" << nlc_lib << " -Wl,-rpath," << nlc_lib << " "
        << "-o " << so_path << " " << src_path << " "
        << kdir << "/libve_sgemv.so -lcblas -lblas_openmp -lm 2>&1";

    fprintf(stderr, "[VE-GC] compiling %s ...\n", so_path.c_str());
    FILE * pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) return false;

    std::string out;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
    int rc = pclose(pipe);
    if (rc != 0) {
        fprintf(stderr, "[VE-GC] compilation failed:\n%s\n", out.c_str());
        return false;
    }
    return true;
}

CompiledGraph * GraphCompiler::load_compiled(const std::string & so_path, const std::string & hash) {
    VEDAmodule mod = 0;
    VEDAresult err = vedaModuleLoad(&mod, so_path.c_str());
    if (err != VEDA_SUCCESS) {
        fprintf(stderr, "[VE-GC] vedaModuleLoad(%s) failed: %s\n",
                so_path.c_str(), ggml_ve_err_str(err));
        return nullptr;
    }
    std::string func_name = "ve_graph_run_" + hash;
    VEDAfunction fn = 0;
    err = vedaModuleGetFunction(&fn, mod, func_name.c_str());
    if (err != VEDA_SUCCESS) {
        fprintf(stderr, "[VE-GC] vedaModuleGetFunction(%s) failed: %s\n",
                func_name.c_str(), ggml_ve_err_str(err));
        vedaModuleUnload(mod);
        return nullptr;
    }

    auto * cg = new CompiledGraph();
    cg->module    = mod;
    cg->run_func  = fn;
    cg->src_hash  = hash;
    cg->so_path   = so_path;
    cg->num_weights   = (int) weight_tensors_.size();
    cg->num_kv_caches = (int) kv_cache_tensors_.size();
    cg->num_slots     = (int) tensor_slot_order_.size();
    if (!traced_ops_.empty()) {
        cg->output_bytes = (size_t) traced_ops_.back().ne[0] * sizeof(float);
        cg->last_slot    = traced_ops_.back().dst_idx;
    }
    cg->slot_kinds.resize(tensor_slot_order_.size());
    for (size_t i = 0; i < tensor_slot_order_.size(); ++i) {
        cg->slot_kinds[i] = tensor_slot_order_[i].second;
    }
    cg->colmajor_specs = colmajor_specs_;
    cg->q4k_specs      = q4k_specs_;
    cg->vebp_specs     = vebp_specs_;
    return cg;
}

CompiledGraph * GraphCompiler::compile(int64_t n_ctx) {
    if (traced_ops_.empty() || !trace_valid_) return nullptr;

    // Emit with a fixed placeholder so the hash captures the graph
    // structure, not any caller-supplied namespace token. Earlier
    // revisions hashed source that already contained the cgraph
    // signature in the function name, which meant two cgraphs with
    // different signatures but identical bodies generated two
    // separate .so files and forced redundant NCC compiles.
    static const std::string placeholder = "ve_graph_run_PLACEHOLDER";
    std::string source = generate_source(placeholder, n_ctx);
    std::string hash   = compute_hash(source);

    // Substitute the real, hash-derived function name into the source
    // before handing it off to NCC. The symbol name in the .so must
    // match what load_compiled() looks up.
    std::string func_name = "ve_graph_run_" + hash;
    size_t      pos       = source.find(placeholder);
    if (pos != std::string::npos) {
        source.replace(pos, placeholder.size(), func_name);
    }

    std::string dir = cache_dir();
    std::string so  = dir + "/graph_" + hash + ".so";

    struct stat st;
    if (stat(so.c_str(), &st) == 0) {
        if (debug_enabled()) {
            fprintf(stderr, "[VE-GC] loading cached %s\n", so.c_str());
        }
        return load_compiled(so, hash);
    }

    if (!compile_source(source, so)) return nullptr;
    return load_compiled(so, hash);
}

// -------------------------------------------------------------------------
// Execute
// -------------------------------------------------------------------------
namespace {

// Resolve HBM device pointer for a tensor in the current graph.
VEDAdeviceptr hbm_ptr_for_tensor(const ggml_tensor * t) {
    if (!t || !t->buffer) return 0;
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    const char * bn = buft ? ggml_backend_buft_name(buft) : nullptr;
    if (!bn) return 0;
    if (std::strncmp(bn, "VE", 2) != 0 || !std::strstr(bn, "_HBM")) return 0;
    return reinterpret_cast<VEDAdeviceptr>(reinterpret_cast<uintptr_t>(t->data));
}

// Build a name -> HBM ptr table from the current cgraph.
void build_name_to_hbm(const ggml_cgraph * cg,
                       std::unordered_map<std::string, VEDAdeviceptr> & out) {
    for (int i = 0; i < cg->n_nodes; ++i) {
        const ggml_tensor * n = cg->nodes[i];
        if (n && n->name && n->name[0]) {
            VEDAdeviceptr p = hbm_ptr_for_tensor(n);
            if (p && !out.count(n->name)) out[n->name] = p;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = n ? n->src[s] : nullptr;
            if (src && src->name && src->name[0]) {
                VEDAdeviceptr p = hbm_ptr_for_tensor(src);
                if (p && !out.count(src->name)) out[src->name] = p;
            }
        }
    }
}

}  // namespace

bool GraphCompiler::execute(CompiledGraph * graph,
                            backend_context * bctx,
                            ggml_cgraph * current_graph) {
    if (!graph || !graph->run_func || !bctx || !current_graph) return false;

    // Drain anything the interpreter left pending before we hand the GPU
    // off to a custom-compiled function. Otherwise async kernels from the
    // previous graph_compute (e.g. the per-token warmup that still runs
    // through the interpreter) can race with our launch and trash VEDA
    // state on the VE side.
    bctx->flush("gcomp_execute_prelude");
    vedaCtxSynchronize();  // belt-and-braces — flush only syncs if dirty

    // Walk current_graph in the same encounter order as trace() used to
    // assign slots, then push each tensor's HBM pointer into the slot it was
    // assigned. ggml's allocator has already placed every tensor in HBM, so
    // we don't have to allocate or copy anything — we just hand the kernel
    // the right addresses.
    //
    // DEFENSIVE: use the *view-aware* tensor pointer (tensor->data already
    // includes the view's start offset). Earlier versions read the
    // *canonical* tensor's data, losing the view offset and producing
    // wrong-address kernel reads.
    std::vector<VEDAdeviceptr> tptrs(graph->num_slots, 0);
    std::set<const ggml_tensor *> seen_canon;

    // Walk the slot table from the front. Colmajor companion slots have
    // no tensor backing them — they're zero placeholders here, populated
    // after this loop from the colmajor cache. try_push advances past
    // such slots so real tensors land at the slot index trace assigned.
    size_t slot_pos = 0;
    auto advance_past_colmajor = [&]() {
        while (slot_pos < (size_t) graph->num_slots
               && (graph->slot_kinds[slot_pos] == BufferKind::WEIGHT_COLMAJOR
                   || graph->slot_kinds[slot_pos] == BufferKind::WEIGHT_Q4K_QS
                   || graph->slot_kinds[slot_pos] == BufferKind::WEIGHT_Q4K_HDR
                   || graph->slot_kinds[slot_pos] == BufferKind::WEIGHT_VEBP_WS
                   || graph->slot_kinds[slot_pos] == BufferKind::WEIGHT_VEBP_WN
                   || graph->slot_kinds[slot_pos] == BufferKind::WEIGHT_VEBP_WSC)) {
            ++slot_pos;
        }
    };

    // --- host-operand staging -------------------------------------------
    // A few small boundary tensors land on a CPU buffer even though the rest
    // of the subgraph is HBM-resident (cross-split hidden states, the mask,
    // the pre-attention Q view). For those we stage host<->HBM each token so
    // the fused kernel can dereference p[] uniformly. Inputs are uploaded
    // before launch; tensors PRODUCED by this subgraph are also downloaded
    // afterwards (so later splits / the sampler see them).
    if ((int) graph->stage_hbm.size() != graph->num_slots) {
        graph->stage_hbm.assign(graph->num_slots, 0);
        graph->stage_cap.assign(graph->num_slots, 0);
    }
    std::set<const ggml_tensor *> produced_here;
    for (int i = 0; i < current_graph->n_nodes; ++i) {
        const ggml_tensor * n = current_graph->nodes[i];
        if (!n) continue;
        // Only COMPUTE ops produce data. VIEW/RESHAPE/PERMUTE/TRANSPOSE nodes
        // merely reference a tensor produced elsewhere — counting their
        // canonical as "produced here" wrongly tags a cross-split INPUT
        // (e.g. 'Qcur-0', read by FA via a permuted view) as an OUTPUT,
        // triggering a needless copy-back.
        if (n->op == GGML_OP_VIEW    || n->op == GGML_OP_RESHAPE ||
            n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE ||
            n->op == GGML_OP_NONE) {
            continue;
        }
        produced_here.insert(canonical(n));
    }
    struct OutCopy { void * host_dst; VEDAdeviceptr hbm_src; size_t bytes; };
    std::vector<OutCopy> out_copies;
    bool stage_failed = false;

    auto ensure_scratch = [&](int s, size_t bytes) -> VEDAdeviceptr {
        if (graph->stage_hbm[s] && graph->stage_cap[s] >= bytes) return graph->stage_hbm[s];
        if (graph->stage_hbm[s]) { vedaMemFreeAsync(graph->stage_hbm[s], 0); graph->stage_hbm[s] = 0; }
        // Trailing guard: VE vectorised kernels (FA dot, SET_ROWS narrowing,
        // matvec) round the element count up to MVL=256 and over-access a few
        // hundred elements past the logical end. ggml's own buffers carry
        // slack; our exact-sized scratch did not, so the over-access fell into
        // the adjacent HBM heap chunk -> "malloc corruption" (when mapped) or
        // "DMA missing space" (when not). 64 KiB covers any reasonable
        // unroll/pack width.
        size_t alloc = (bytes + (64u << 10) + 4095) & ~size_t(4095);
        VEDAdeviceptr p = 0;
        if (vedaMemAllocAsync(&p, alloc, 0) != VEDA_SUCCESS) return 0;
        vedaCtxSynchronize();
        graph->stage_hbm[s] = p;
        graph->stage_cap[s] = alloc;
        return p;
    };

    // Stage the CPU-resident operand at slot `s` and return its offset-correct
    // HBM pointer (mirrors the canonical's whole host buffer so view offsets
    // stay valid). Returns 0 on failure.
    auto stage_host_operand = [&](int s, const ggml_tensor * raw, const ggml_tensor * c) -> VEDAdeviceptr {
        if (!c->data) return 0;                       // nothing to copy
        if (is_weight(c)) return 0;                   // never bounce a weight per-token
        const size_t nb = ggml_nbytes(c);
        if (nb == 0 || nb > (64u << 20)) {            // sanity cap (64 MiB)
            if (debug_enabled())
                fprintf(stderr, "[VE-GC] won't stage '%s' (%zu bytes)\n", c->name ? c->name : "?", nb);
            return 0;
        }
        VEDAdeviceptr scratch = ensure_scratch(s, nb);
        if (!scratch) return 0;
        // Mirror current host contents (also preserves the un-written part of
        // a partial-view output).
        if (vedaMemcpyHtoD(scratch, c->data, nb) != VEDA_SUCCESS) return 0;
        if (produced_here.count(c)) out_copies.push_back({ c->data, scratch, nb });
        const ggml_tensor * addr = raw->data ? raw : c;
        size_t off = (const uint8_t *) addr->data - (const uint8_t *) c->data;
        if (std::getenv("GGML_VE_STAGE_DEBUG")) {
            fprintf(stderr, "[VE-STAGE] slot=%d '%s' canon='%s' nb=%zu off=%zu %s ne=[%ld,%ld,%ld,%ld] raw_ne=[%ld,%ld,%ld,%ld]\n",
                    s, raw->name ? raw->name : "?", c->name ? c->name : "?", nb, off,
                    produced_here.count(c) ? "OUT" : "in",
                    (long)c->ne[0],(long)c->ne[1],(long)c->ne[2],(long)c->ne[3],
                    (long)raw->ne[0],(long)raw->ne[1],(long)raw->ne[2],(long)raw->ne[3]);
        }
        return scratch + off;
    };

    auto try_push = [&](const ggml_tensor * raw) {
        if (!raw) return;
        const ggml_tensor * c = canonical(raw);
        if (seen_canon.count(c)) return;
        seen_canon.insert(c);
        advance_past_colmajor();
        if (slot_pos >= (size_t) graph->num_slots) return;
        // Prefer the raw tensor's data (includes view offset). If it's a
        // pure metadata tensor with no buffer/data, the slot stays 0.
        const ggml_tensor * src_for_addr = raw->data ? raw : c;
        VEDAdeviceptr hbm = hbm_ptr_for_tensor(src_for_addr);
        if (hbm == 0 && src_for_addr->data) {
            if (is_weight(c)) {
                // CPU-resident WEIGHT (e.g. token_embd.weight, which llama.cpp
                // leaves on CPU_Mapped since only GET_ROWS touches it). Upload
                // ONCE to HBM via the weight cache (keyed by name) and reuse the
                // cached pointer every token — never a per-token bounce. This is
                // the "GET_ROWS weight on CPU mmap" path; without it the whole
                // 875-node decode graph was refused over this one weight.
                size_t nb = ggml_nbytes(c);
                const char * nm = (c->name && c->name[0]) ? c->name : nullptr;
                VEDAdeviceptr w_hbm = nm
                    ? bctx->cache().get_or_upload_by_name(nm, c->data, nb)
                    : bctx->cache().get_or_upload(c->data, nb);
                if (w_hbm) {
                    size_t off = (const uint8_t *) src_for_addr->data - (const uint8_t *) c->data;
                    hbm = w_hbm + off;
                } else {
                    stage_failed = true;
                }
            } else {
                // CPU-resident intermediate / leaf / boundary: per-token scratch.
                hbm = stage_host_operand((int) slot_pos, raw, c);
                if (hbm == 0) stage_failed = true;
            }
        }
        tptrs[slot_pos++] = hbm;
    };

    for (int i = 0; i < current_graph->n_nodes; ++i) {
        ggml_tensor * n = current_graph->nodes[i];
        if (!n) continue;
        try_push(n);
        try_push(n->src[0]);
        if (n->op != GGML_OP_ROPE && n->op != GGML_OP_SET_ROWS && n->op != GGML_OP_GET_ROWS) {
            try_push(n->src[1]);
        }
        try_push(n->src[2]);
    }

    if (stage_failed) {
        if (debug_enabled()) {
            fprintf(stderr, "[VE-GC] host-operand staging failed — abort (interpreter will run this graph)\n");
        }
        return false;
    }

    // Colmajor companion slots aren't tensors in the cgraph — we resize
    // tptrs to make room for them, then populate from the colmajor cache.
    // try_push only filled real tensor slots; the colmajor slots come at
    // the end of the slot table (allocated during trace, after every real
    // tensor was seen).
    if (!graph->colmajor_specs.empty()) {
        auto * dev = bctx ? bctx->dev() : nullptr;
        auto * cache = dev ? dev->colmajor : nullptr;
        VEDAfunction transpose_fn = bctx ? bctx->fn(K_BF16_TO_F32_COLMAJOR_HBM) : 0;
        if (!cache || transpose_fn == 0) {
            if (debug_enabled()) {
                fprintf(stderr, "[VE-GC] colmajor cache or transpose fn unavailable — abort\n");
            }
            return false;
        }
        for (const auto & sp : graph->colmajor_specs) {
            if (sp.src0_slot < 0 || sp.src0_slot >= (int) tptrs.size()) continue;
            if (sp.companion_slot < 0 || sp.companion_slot >= (int) tptrs.size()) continue;
            VEDAdeviceptr bf16_ptr = tptrs[sp.src0_slot];
            if (bf16_ptr == 0) continue;
            VEDAdeviceptr cm_ptr = cache->get_or_create(bf16_ptr, sp.M, sp.K, transpose_fn);
            if (cm_ptr == 0) {
                if (debug_enabled()) {
                    fprintf(stderr, "[VE-GC] colmajor lookup failed for slot %d (M=%ld K=%ld)\n",
                            sp.src0_slot, (long) sp.M, (long) sp.K);
                }
                return false;
            }
            tptrs[sp.companion_slot] = cm_ptr;
        }
    }

    // Q4_K companion slots (qs + decoded hdr per Q4_K MUL_MAT op). We need
    // the cgraph tensor's w->data and w->name to call get_or_upload_q4k_canon.
    // Walk the cgraph in traced order, find Q4_K MUL_MATs and match to specs.
    if (!graph->q4k_specs.empty() && bctx) {
        size_t spec_i = 0;
        for (int i = 0; i < current_graph->n_nodes && spec_i < graph->q4k_specs.size(); i++) {
            const ggml_tensor * node = current_graph->nodes[i];
            if (!node || node->op != GGML_OP_MUL_MAT) continue;
            const ggml_tensor * w = node->src[0];
            if (!w || w->type != GGML_TYPE_Q4_K) continue;
            const Q4KSpec & sp = graph->q4k_specs[spec_i++];
            VEDAdeviceptr qs_v = 0, hdr_v = 0;
            const char * name = (w->name && w->name[0]) ? w->name : nullptr;
            if (!bctx->cache().get_or_upload_q4k_canon(
                    name, w->data, (uint64_t) sp.M, (uint64_t) sp.K, &qs_v, &hdr_v)) {
                if (debug_enabled()) {
                    fprintf(stderr, "[VE-GC] q4k canon upload failed for %s (M=%ld K=%ld)\n",
                            name ? name : "?", (long) sp.M, (long) sp.K);
                }
                return false;
            }
            if (sp.qs_slot  >= 0 && sp.qs_slot  < (int) tptrs.size()) tptrs[sp.qs_slot]  = qs_v;
            if (sp.hdr_slot >= 0 && sp.hdr_slot < (int) tptrs.size()) tptrs[sp.hdr_slot] = hdr_v;
        }
    }

    // VEBP companion slots (interleaved sign/nz planes + group scales).
    if (!graph->vebp_specs.empty() && bctx) {
        size_t spec_i = 0;
        for (int i = 0; i < current_graph->n_nodes && spec_i < graph->vebp_specs.size(); i++) {
            const ggml_tensor * node = current_graph->nodes[i];
            if (!node || node->op != GGML_OP_MUL_MAT) continue;
            const ggml_tensor * w = node->src[0];
            if (!w || w->type != GGML_TYPE_VEBP) continue;
            const VebpSpec & sp = graph->vebp_specs[spec_i++];
            VEDAdeviceptr ws = 0, wn = 0, wsc = 0;
            const char * name = (w->name && w->name[0]) ? w->name : nullptr;
            // weights are device-resident here -> peek cache, bounce only on miss
            if (!bctx->cache().vebp_lookup(name, (uint64_t) sp.M, (uint64_t) sp.K,
                                           &ws, &wn, &wsc)) {
                const void * src = w->data;
                std::unique_ptr<uint8_t[]> bounce;
                const int64_t wbytes = (int64_t) sp.M * (sp.K / 256) * 68;
                if (w->buffer && !ggml_backend_buffer_is_host(w->buffer)) {
                    bounce.reset(new uint8_t[wbytes]);
                    if (vedaMemcpyDtoH(bounce.get(), (VEDAdeviceptr)(uintptr_t) w->data,
                                       wbytes) != VEDA_SUCCESS) return false;
                    src = bounce.get();
                }
                if (!bctx->cache().get_or_upload_vebp(name, src, (uint64_t) sp.M,
                                                      (uint64_t) sp.K, &ws, &wn, &wsc)) {
                    if (debug_enabled())
                        fprintf(stderr, "[VE-GC] vebp upload failed for %s (M=%ld K=%ld)\n",
                                name ? name : "?", (long) sp.M, (long) sp.K);
                    return false;
                }
            }
            if (sp.ws_slot  >= 0 && sp.ws_slot  < (int) tptrs.size()) tptrs[sp.ws_slot]  = ws;
            if (sp.wn_slot  >= 0 && sp.wn_slot  < (int) tptrs.size()) tptrs[sp.wn_slot]  = wn;
            if (sp.wsc_slot >= 0 && sp.wsc_slot < (int) tptrs.size()) tptrs[sp.wsc_slot] = wsc;
        }
    }

    // After walking the cgraph + populating colmajor slots, slot_pos should
    // have stepped past every WEIGHT/KV/INTERMEDIATE slot in the table —
    // anything less means try_push ran out of cgraph tensors before filling
    // all expected real slots.
    advance_past_colmajor();
    if (slot_pos != (size_t) graph->num_slots) {
        if (debug_enabled()) {
            fprintf(stderr, "[VE-GC] slot fill stopped at %zu / %d — abort\n",
                    slot_pos, graph->num_slots);
        }
        return false;
    }

    // Every slot must be a real HBM pointer. A null slot means the
    // ggml allocator never placed that tensor in VE_HBM (likely a CPU
    // mmap weight). Launching with a null pointer is undefined behaviour
    // on the VE — and almost always SIGSEGVs.
    for (size_t i = 0; i < tptrs.size(); ++i) {
        // Companion slots are populated by their own loops above; skip the
        // null check for them. Real tensor slots must be non-null.
        if (graph->slot_kinds[i] == BufferKind::WEIGHT_COLMAJOR) continue;
        if (graph->slot_kinds[i] == BufferKind::WEIGHT_Q4K_QS)   continue;
        if (graph->slot_kinds[i] == BufferKind::WEIGHT_Q4K_HDR)  continue;
        if (graph->slot_kinds[i] == BufferKind::WEIGHT_VEBP_WS)  continue;
        if (graph->slot_kinds[i] == BufferKind::WEIGHT_VEBP_WN)  continue;
        if (graph->slot_kinds[i] == BufferKind::WEIGHT_VEBP_WSC) continue;
        if (tptrs[i] == 0) {
            if (debug_enabled()) {
                fprintf(stderr, "[VE-GC] slot %zu kind=%d is NULL — abort\n",
                        i, (int) graph->slot_kinds[i]);
            }
            return false;
        }
    }
    // Optional slot-address dump for diagnosing wrong-address SIGSEGVs.
    if (debug_enabled()) {
        static int dumped = 0;
        if (dumped < 3) {
            fprintf(stderr, "[VE-GC] slots for n=%d compiled (%d slots):\n",
                    current_graph->n_nodes, (int) tptrs.size());
            for (size_t i = 0; i < tptrs.size(); ++i) {
                fprintf(stderr, "    slot %2zu = 0x%016lx (kind=%d)\n",
                        i, (unsigned long) tptrs[i],
                        (int) graph->slot_kinds[i]);
            }
            ++dumped;
        }
    }

    // Reject obvious host pointers (x86_64 user-space addrs ≥ 0x7000_…).
    // ggml's allocator gives VE_HBM tensors addresses in the 0x2000_… and
    // 0x6000_… bands on our hardware. Anything above 0x7000_0000_0000_0000
    // is host memory and would SIGSEGV inside vedaMemPtr.
    for (size_t i = 0; i < tptrs.size(); ++i) {
        uint64_t p = (uint64_t) tptrs[i];
        if (p >= 0x7000000000000000ULL) {
            if (debug_enabled()) {
                fprintf(stderr, "[VE-GC] slot %zu addr=0x%lx looks like a host ptr — abort\n",
                        i, (unsigned long) p);
            }
            return false;
        }
    }

    // Token id and decode position — for graphs that include GET_ROWS / ROPE
    // we look them up from the leaf tensors, reading through vedaMemcpyDtoH
    // when the leaf actually lives in HBM.
    auto read_leaf0 = [&](const ggml_tensor * t) -> int64_t {
        // Read element [0] of a position/index leaf, honouring its width
        // (i32 inp_pos vs i64 SET_ROWS index) and HBM vs host residency.
        if (!t || !t->data) return 0;
        const size_t w = ggml_type_size(t->type);  // 4 (i32) or 8 (i64)
        VEDAdeviceptr hbm = hbm_ptr_for_tensor(t);
        int64_t v64 = 0; int32_t v32 = 0;
        void * dst = (w == 8) ? (void *) &v64 : (void *) &v32;
        if (hbm) {
            if (vedaMemcpyDtoH(dst, hbm, w) != VEDA_SUCCESS) return 0;
        } else {
            std::memcpy(dst, t->data, w);
        }
        return (w == 8) ? v64 : (int64_t) v32;
    };

    int32_t token_id = 0;
    int64_t position = 0;
    bool    have_pos = false;
    for (int i = 0; i < current_graph->n_nodes; ++i) {
        const ggml_tensor * n = current_graph->nodes[i];
        if (n && n->op == GGML_OP_GET_ROWS && n->src[1]) {
            token_id = (int32_t) read_leaf0(n->src[1]);
            break;
        }
    }
    // Decode position drives ROPE's angle, SET_ROWS' KV-cell row, and FA's
    // seq_len (= pos+1). Prefer ROPE's inp_pos; fall back to the SET_ROWS
    // index leaf (i64) for attention subgraphs that contain SET_ROWS/FA but
    // no ROPE — without this they baked pos=0, writing the wrong KV row and
    // attending to a single position (garbage decode output).
    for (int i = 0; i < current_graph->n_nodes; ++i) {
        const ggml_tensor * n = current_graph->nodes[i];
        if (n && n->op == GGML_OP_ROPE && n->src[1]) {
            position = read_leaf0(n->src[1]);
            have_pos = true;
            break;
        }
    }
    if (!have_pos) {
        for (int i = 0; i < current_graph->n_nodes; ++i) {
            const ggml_tensor * n = current_graph->nodes[i];
            if (n && n->op == GGML_OP_SET_ROWS && n->src[1]) {
                position = read_leaf0(n->src[1]);
                have_pos = true;
                break;
            }
        }
    }

    // HMEM in/out — sized once, reused across every execute() of the same
    // compiled graph. Most layer subgraphs use neither (their data flows
    // entirely through HBM slot pointers), but allocating them lazily up
    // front is cheap and saves the pool acquire/release pair on every
    // token.
    if (graph->in_hmem == 0) {
        if (!ggml_ve_ok(vedaHMemAlloc(&graph->in_hmem, sizeof(int32_t)),
                        "vedaHMemAlloc(gcomp in)")) {
            return false;
        }
    }
    if (graph->out_hmem == 0) {
        size_t out_bytes = graph->output_bytes ? graph->output_bytes : 16;
        if (!ggml_ve_ok(vedaHMemAlloc(&graph->out_hmem, out_bytes),
                        "vedaHMemAlloc(gcomp out)")) {
            return false;
        }
    }
    VEDAhmemptr in_hmem  = graph->in_hmem;
    VEDAhmemptr out_hmem = graph->out_hmem;

    if (!ggml_ve_ok(vedaHMemcpy(reinterpret_cast<void *>(in_hmem),
                                 &token_id, sizeof(int32_t)),
                    "vedaHMemcpy(token in)")) {
        return false;
    }

    // Pass the host-side VEDAdeviceptr array via vedaArgsSetStack with
    // INTENT_IN — VEDA *copies* the bytes to a VE-side stack buffer and
    // the kernel receives a pointer it can actually dereference. This is
    // the bit the old "vedaArgsSetPtr with a (VEDAdeviceptr)(uintptr_t)
    // host-pointer" pattern silently gets wrong: vedaArgsSetPtr stores
    // the host address itself, so the kernel reads garbage / SIGSEGVs.
    // Verified with /tmp/test_launch_min.cpp (modes "read" vs "stack").
    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(gcomp)")) {
        bctx->pool().release(in_hmem);
        bctx->pool().release(out_hmem);
        return false;
    }
    vedaArgsSetStack(args, 0, tptrs.data(),
                     VEDA_ARGS_INTENT_IN,
                     tptrs.size() * sizeof(VEDAdeviceptr));
    vedaArgsSetHMEM (args, 1, in_hmem);
    vedaArgsSetHMEM (args, 2, out_hmem);
    vedaArgsSetI64  (args, 3, position);
    vedaArgsSetI64  (args, 4, /*n_ctx*/ 4096);

    // vedaLaunchKernel (without Ex) auto-destroys args on success. We sync
    // after to surface VE-side errors at the right point.
    if (!ggml_ve_ok(vedaLaunchKernel(graph->run_func, 0, args),
                    "vedaLaunchKernel(gcomp)")) {
        return false;
    }
    if (vedaCtxSynchronize() != VEDA_SUCCESS) {
        fprintf(stderr, "[VE-GC] sync after kernel returned non-success\n");
        return false;
    }

    // Download CPU-resident outputs this subgraph produced back to their host
    // buffers, so a later split (or the sampler) that reads them from host
    // sees the computed values. Inputs needed no copy-back (read-only).
    for (const auto & oc : out_copies) {
        if (vedaMemcpyDtoH(oc.host_dst, oc.hbm_src, oc.bytes) != VEDA_SUCCESS) {
            fprintf(stderr, "[VE-GC] output copy-back (%zu bytes) failed\n", oc.bytes);
            return false;
        }
    }

    return true;
}

bool GraphCompiler::matches_trace(const CompiledGraph * g) const {
    if (!g) return false;
    return (int) weight_tensors_.size() == g->num_weights
        && (int) kv_cache_tensors_.size() == g->num_kv_caches;
}

}  // namespace gcomp
}  // namespace ggml_ve
