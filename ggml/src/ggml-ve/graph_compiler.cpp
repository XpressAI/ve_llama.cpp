// Phase-7 graph compiler implementation. See graph_compiler.hpp for design.

#include "graph_compiler.hpp"

#include "backend_ctx.hpp"
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
        case OpType::ROPE:          return "ROPE";
        case OpType::SET_ROWS:      return "SET_ROWS";
        case OpType::FLASH_ATTN:    return "FLASH_ATTN";
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
    if (v < 0) v = (std::getenv("GGML_VE_COMPILE_DEBUG") != nullptr) ? 1 : 0;
    return v != 0;
}

}  // namespace

// -------------------------------------------------------------------------
// Enable / disable
// -------------------------------------------------------------------------
bool GraphCompiler::enabled() {
    static int v = -1;
    if (v < 0) {
        v = (std::getenv("GGML_VE_COMPILE_GRAPH") != nullptr) ? 1 : 0;
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

    // Per-op refusals while we incrementally bring up the codegen.
    switch (node->op) {
        case GGML_OP_GET_ROWS:
            // Token id lives in a small i32 leaf tensor on the host —
            // the codegen would need to bring it in via the kernel's
            // `input` HMEM arg, which is only wired for the first
            // GET_ROWS in a graph today.
            if (debug_enabled()) {
                fprintf(stderr, "[VE-GC] refuse: '%s' is GET_ROWS (not yet compiled)\n",
                        node->name ? node->name : "?");
            }
            return false;
        default:
            break;
    }

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
            // FA Q (src[0]): n_tokens=ne[1]. K/V (src[1..2]): n_kv_tokens=ne[1] (always grows; we accept).
            if (src_slot == 0) return t->ne[1] > 1;
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
            op.type = (op.src_type == GGML_TYPE_BF16)
                        ? OpType::MUL_MAT_BF16
                        : OpType::MUL_MAT_F32;
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
        case GGML_OP_UNARY:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ADD_ID:
        case GGML_OP_ARGSORT:
        default:
            return false;   // unsupported by the compiler — fall back
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
    trace_valid_ = true;

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
            // The kernel expects BF16 source; we widen to F32 inline.
            if (op.src0_kind != BufferKind::WEIGHT) {
                ss << "    memcpy(" << dst << ", " << src0 << ", "
                   << (op.ne[0] * sizeof(float)) << ");\n";
                break;
            }
            ss << "    {\n";
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
            ss << "    {\n";
            if (per_head) {
                ss << "        int rows = " << rows << ", cols = " << cols << ";\n";
                ss << "        float eps = " << flit(op.p.rms_norm.eps) << ";\n";
                ss << "        for (int r = 0; r < rows; r++) {\n";
                ss << "            const float* x = (const float*)" << src0 << " + r * cols;\n";
                ss << "            float* y = (float*)" << dst << " + r * cols;\n";
                ss << "            float sumsq = 0.f;\n";
                ss << "            for (int j = 0; j < cols; j++) sumsq += x[j] * x[j];\n";
                ss << "            float inv = 1.f / sqrtf(sumsq / cols + eps);\n";
                ss << "            for (int j = 0; j < cols; j++) y[j] = inv * x[j];\n";
                ss << "        }\n";
            } else {
                ss << "        int cols = " << cols << ";\n";
                ss << "        float eps = " << flit(op.p.rms_norm.eps) << ";\n";
                ss << "        const float* x = (const float*)" << src0 << ";\n";
                ss << "        float* y = (float*)" << dst << ";\n";
                ss << "        float sumsq = 0.f;\n";
                ss << "        for (int j = 0; j < cols; j++) sumsq += x[j] * x[j];\n";
                ss << "        float inv = 1.f / sqrtf(sumsq / cols + eps);\n";
                ss << "        for (int j = 0; j < cols; j++) y[j] = inv * x[j];\n";
            }
            ss << "    }\n";
            break;
        }

        case OpType::MUL: {
            int64_t src1_n = op.src1_ne[0];
            if (op.src1_ne[1] > 1 && op.src1_ne[2] > 1) src1_n = op.src1_ne[0] * op.src1_ne[1];
            if (src1_n > 0 && src1_n < n && n % src1_n == 0) {
                int64_t repeats = n / src1_n;
                ss << "    for (int64_t r = 0; r < " << repeats << "; r++) {\n";
                ss << "        for (int64_t i = 0; i < " << src1_n << "; i++) {\n";
                ss << "            ((float*)" << dst << ")[r*" << src1_n << "+i] = "
                   << "((float*)" << src0 << ")[r*" << src1_n << "+i] * "
                   << "((float*)" << src1 << ")[i];\n";
                ss << "        }\n";
                ss << "    }\n";
            } else {
                ss << "    for (int i = 0; i < " << n << "; i++) {\n";
                ss << "        ((float*)" << dst << ")[i] = "
                   << "((float*)" << src0 << ")[i] * ((float*)" << src1 << ")[i];\n";
                ss << "    }\n";
            }
            break;
        }

        case OpType::ADD:
            ss << "    for (int i = 0; i < " << n << "; i++) {\n";
            ss << "        ((float*)" << dst << ")[i] = "
               << "((float*)" << src0 << ")[i] + ((float*)" << src1 << ")[i];\n";
            ss << "    }\n";
            break;

        case OpType::MUL_MAT_F32: {
            int64_t M = op.src0_ne[1], K = op.src0_ne[0];
            ss << "    ve_f32_matvec_ptr((float*)" << dst << ", (const float*)"
               << src0 << ", (const float*)" << src1
               << ", " << M << ", " << K << ");\n";
            break;
        }

        case OpType::MUL_MAT_BF16: {
            int64_t M = op.src0_ne[1], K = op.src0_ne[0];
            ss << "    ve_bf16_matvec_rowmajor_ptr_omp((float*)" << dst
               << ", (const uint16_t*)" << src0 << ", (const float*)" << src1
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
            ss << "    {\n";
            ss << "        int64_t idx0 = (int64_t) pos;\n";
            (void) src1;
            ss << "        int cols = " << cols << ";\n";
            ss << "        const float* src = (const float*)" << src0 << ";\n";
            ss << "        char* base = (char*)" << dst << ";\n";
            ss << "        char* row = base + idx0 * " << dst_row_bytes << ";\n";
            if (op.dst_type == GGML_TYPE_F32) {
                ss << "        memcpy(row, src, cols * 4);\n";
            } else if (op.dst_type == GGML_TYPE_BF16) {
                ss << "        uint16_t* drow = (uint16_t*)row;\n";
                ss << "        #pragma _NEC ivdep\n";
                ss << "        for (int j = 0; j < cols; j++) {\n";
                ss << "            uint32_t u; memcpy(&u, &src[j], 4);\n";
                ss << "            drow[j] = (uint16_t)(u >> 16);\n";
                ss << "        }\n";
            } else {  // F16 — kept for completeness but we reject in supports
                ss << "        // F16 SET_ROWS not in compiled path\n";
                ss << "        (void)src; (void)cols;\n";
            }
            (void) elem_bytes;
            ss << "    }\n";
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
            ss << "        float* outp = (float*)" << dst  << ";\n";
            ss << "        const float* qp = (const float*)" << src0 << ";\n";
            ss << "        const void*  kp = (const void*)"  << src1 << ";\n";
            ss << "        const void*  vp = (const void*)"  << src2 << ";\n";
            if (op.p.flash_attn.kv_type == (int) GGML_TYPE_BF16) {
                ss << "        attention_f32q_bf16kv_fused_gqa_omp(outp, qp, kp, vp, "
                   << "head_dim, n_q_heads, n_kv_heads, seq_len, scale, "
                   << "nb_k1, nb_k2, nb_v1, nb_v2);\n";
            } else {
                ss << "        attention_f32_raw_gqa_stride_omp(outp, qp, kp, vp, "
                   << "head_dim, n_q_heads, n_kv_heads, seq_len, scale, "
                   << "nb_k1, nb_k2, nb_v1, nb_v2);\n";
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
            ss << "    memcpy(" << dst << ", " << src0 << ", "
               << (size_t) (total_elems * elem_bytes) << "u);\n";
            break;
        }

        case OpType::SOFT_MAX: {
            int rows = (op.ne[1] > 1) ? (int) op.ne[1] : 1;
            int cols = (int) op.ne[0];
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
    ss << "#include <veda/device.h>\n\n";

    // External kernels from libve_sgemv.so
    ss << "extern void ve_bf16_matvec_rowmajor_ptr_omp(float* y, const uint16_t* W, const float* x, int M, int K);\n";
    ss << "extern void ve_f32_matvec_ptr(float* y, const float* W, const float* x, int M, int K);\n";
    ss << "extern void attention_f32_raw_gqa_stride_omp(float* out, const float* q, const void* k, const void* v,"
       << " int head_dim, int n_q_heads, int n_kv_heads, int seq_len, float scale,"
       << " size_t nb_k1, size_t nb_k2, size_t nb_v1, size_t nb_v2);\n";
    ss << "extern void attention_f32q_bf16kv_fused_gqa_omp(float* out, const float* q, const void* k, const void* v,"
       << " int head_dim, int n_q_heads, int n_kv_heads, int seq_len, float scale,"
       << " size_t nb_k1, size_t nb_k2, size_t nb_v1, size_t nb_v2);\n\n";

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

    ss << "    void* p[" << (n_slots > 0 ? n_slots : 1) << "];\n";
    ss << "    for (int i = 0; i < " << n_slots << "; i++) {\n";
    ss << "        if (tptr_hbm[i] != 0) {\n";
    ss << "            if (vedaMemPtr(&p[i], tptr_hbm[i]) != 0) return 1;\n";
    ss << "        } else p[i] = 0;\n";
    ss << "    }\n";
    ss << "\n";

    for (size_t i = 0; i < traced_ops_.size(); ++i) {
        ss << gen_op_code(traced_ops_[i], (int) i);
    }

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

    std::ostringstream cmd;
    cmd << "/opt/nec/ve/bin/ncc -O4 -fopenmp -fpic -shared "
        << "-I/opt/nec/ve/share/veoffload-veda/include "
        << "-L" << kdir << " -Wl,-rpath," << kdir << " "
        << "-o " << so_path << " " << src_path << " "
        << kdir << "/libve_sgemv.so -lm 2>&1";

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
    return cg;
}

CompiledGraph * GraphCompiler::compile(const std::string & model_hash, int64_t n_ctx) {
    if (traced_ops_.empty() || !trace_valid_) return nullptr;

    std::string func_name = "ve_graph_run_" + model_hash;
    std::string source    = generate_source(func_name, n_ctx);
    std::string hash      = compute_hash(source);

    std::string dir = cache_dir();
    // Per-source .so: many cgraphs share n_nodes (every attention block of
    // every layer hits n=30) but emit different code. Keying only by
    // n_nodes had them clobber each other's .so on disk and force a fresh
    // NCC compile per layer, per call.
    std::string so  = dir + "/graph_" + model_hash + "_ctx"
                    + std::to_string(n_ctx) + "_" + hash + ".so";

    struct stat st;
    if (stat(so.c_str(), &st) == 0) {
        fprintf(stderr, "[VE-GC] loading cached %s\n", so.c_str());
        return load_compiled(so, model_hash);
    }

    if (!compile_source(source, so)) return nullptr;
    return load_compiled(so, model_hash);
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
    std::vector<VEDAdeviceptr> tptrs;
    tptrs.reserve(graph->num_slots);
    std::set<const ggml_tensor *> seen_canon;

    auto try_push = [&](const ggml_tensor * raw) {
        if (!raw) return;
        const ggml_tensor * c = canonical(raw);
        if (seen_canon.count(c)) return;
        seen_canon.insert(c);
        if ((int) tptrs.size() >= graph->num_slots) return;
        // Prefer the raw tensor's data (includes view offset). If it's a
        // pure metadata tensor with no buffer/data, the slot stays 0.
        const ggml_tensor * src_for_addr = raw->data ? raw : c;
        VEDAdeviceptr hbm = hbm_ptr_for_tensor(src_for_addr);
        tptrs.push_back(hbm);
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

    if ((int) tptrs.size() != graph->num_slots) {
        if (debug_enabled()) {
            fprintf(stderr, "[VE-GC] slot-count mismatch: compiled=%d current=%d — abort\n",
                    graph->num_slots, (int) tptrs.size());
        }
        return false;
    }

    // Every slot must be a real HBM pointer. A null slot means the
    // ggml allocator never placed that tensor in VE_HBM (likely a CPU
    // mmap weight). Launching with a null pointer is undefined behaviour
    // on the VE — and almost always SIGSEGVs.
    for (size_t i = 0; i < tptrs.size(); ++i) {
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
    auto read_i32_leaf = [&](const ggml_tensor * t) -> int32_t {
        if (!t || !t->data) return 0;
        VEDAdeviceptr hbm = hbm_ptr_for_tensor(t);
        int32_t v = 0;
        if (hbm) {
            if (vedaMemcpyDtoH(&v, hbm, sizeof(int32_t)) != VEDA_SUCCESS) v = 0;
        } else {
            v = ((const int32_t *) t->data)[0];
        }
        return v;
    };

    int32_t token_id = 0;
    int64_t position = 0;
    for (int i = 0; i < current_graph->n_nodes; ++i) {
        const ggml_tensor * n = current_graph->nodes[i];
        if (n && n->op == GGML_OP_GET_ROWS && n->src[1]) {
            token_id = read_i32_leaf(n->src[1]);
            break;
        }
    }
    for (int i = 0; i < current_graph->n_nodes; ++i) {
        const ggml_tensor * n = current_graph->nodes[i];
        if (n && n->op == GGML_OP_ROPE && n->src[1]) {
            position = (int64_t) read_i32_leaf(n->src[1]);
            break;
        }
    }

    // Optional HMEM in/out — only needed if the graph reads token id from
    // `input` or has to publish logits to `output`. Most layer subgraphs use
    // neither (their data flows entirely through HBM slot pointers).
    VEDAhmemptr in_hmem  = bctx->pool().acquire(sizeof(int32_t));
    VEDAhmemptr out_hmem = bctx->pool().acquire(graph->output_bytes ? graph->output_bytes : 16);
    if (in_hmem == 0 || out_hmem == 0) {
        if (in_hmem)  bctx->pool().release(in_hmem);
        if (out_hmem) bctx->pool().release(out_hmem);
        return false;
    }
    if (!ggml_ve_ok(vedaHMemcpy(reinterpret_cast<void *>(in_hmem),
                                 &token_id, sizeof(int32_t)),
                    "vedaHMemcpy(token in)")) {
        bctx->pool().release(in_hmem);
        bctx->pool().release(out_hmem);
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
        bctx->pool().release(in_hmem);
        bctx->pool().release(out_hmem);
        return false;
    }
    if (vedaCtxSynchronize() != VEDA_SUCCESS) {
        fprintf(stderr, "[VE-GC] sync after kernel returned non-success\n");
        bctx->pool().release(in_hmem);
        bctx->pool().release(out_hmem);
        return false;
    }

    bctx->pool().release(in_hmem);
    bctx->pool().release(out_hmem);
    return true;
}

bool GraphCompiler::matches_trace(const CompiledGraph * g) const {
    if (!g) return false;
    return (int) weight_tensors_.size() == g->num_weights
        && (int) kv_cache_tensors_.size() == g->num_kv_caches;
}

}  // namespace gcomp
}  // namespace ggml_ve
