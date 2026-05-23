#ifndef GGML_VE_GRAPH_COMPILER_HPP
#define GGML_VE_GRAPH_COMPILER_HPP

// Phase-7 graph compiler. On the first graph_compute, traces every op in
// the cgraph, generates a single C function that performs all of them
// sequentially, compiles it with ncc, and dispatches that single function
// via one VEDA launch per subsequent token. Eliminates per-op kernel-launch
// overhead (~50–100 µs × ~170 ops = 8–17 ms per token in the interpreter).
//
// Enable with GGML_VE_COMPILE_GRAPH=1. Falls back to the interpreter if
// any op in the graph isn't compilable. Compiled .so cached under
// ~/.cache/ggml-ve-compiled/ keyed by the source hash; first run after a
// code or model change pays the ncc compile cost (~30-60s) and subsequent
// runs load instantly.

#include "common.hpp"

#include "ggml.h"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_cgraph;
struct ggml_backend_buffer;

namespace ggml_ve {
class backend_context;
}

namespace ggml_ve {
namespace gcomp {

enum class OpType {
    UNKNOWN,
    GET_ROWS,
    RMS_NORM,
    MUL,
    ADD,
    MUL_MAT_F32,
    MUL_MAT_BF16,
    ROPE,             // mode 0 (normal) and 2 (NeoX)
    SET_ROWS,
    FLASH_ATTN,
    CPY,
    SOFT_MAX,
};

enum class BufferKind {
    WEIGHT,           // loaded once, never changes (model weights)
    WEIGHT_COLMAJOR,  // F32 column-major copy of a BF16 weight (cache populated lazily)
    KV_CACHE,         // persistent across tokens, mutates each step
    INTERMEDIATE,     // scratch, reused inside the kernel
    INPUT,            // pseudo: input embedding row
    OUTPUT,           // pseudo: output logits
};

struct TracedOp {
    OpType  type    = OpType::UNKNOWN;
    std::string name;

    int64_t ne[4]      = {0,0,0,0};
    int64_t nb[4]      = {0,0,0,0};

    int     src_type   = 0;
    int     dst_type   = 0;

    int     dst_idx    = -1;
    int     src0_idx   = -1;
    int     src1_idx   = -1;
    int     src2_idx   = -1;
    // For MUL_MAT_BF16 ops opted into the colmajor+CBLAS fast path,
    // the slot holding the F32 col-major companion of src0. -1 = use
    // the original BF16 path through `_inner` matvec.
    int     colmajor_idx = -1;
    BufferKind dst_kind  = BufferKind::INTERMEDIATE;
    BufferKind src0_kind = BufferKind::INTERMEDIATE;
    BufferKind src1_kind = BufferKind::INTERMEDIATE;
    BufferKind src2_kind = BufferKind::INTERMEDIATE;

    int64_t src0_ne[4] = {0,0,0,0};
    int64_t src1_ne[4] = {0,0,0,0};

    union {
        struct { float eps; }                                         rms_norm;
        struct { float freq_base, freq_scale, mscale; int n_dims, mode; } rope;
        struct { float scale, max_bias, softcap;
                 int   kv_type; int64_t n_kv_heads;
                 size_t nb_k1, nb_k2, nb_v1, nb_v2; }                 flash_attn;
    } p;
};

// Spec for a colmajor companion slot: at execute time we look up
// (or create) the F32 col-major copy of the BF16 weight at src0_slot,
// keyed by its HBM address, and stash the colmajor HBM pointer at
// companion_slot in the per-call slot array.
struct ColmajorSpec {
    int     src0_slot      = -1;
    int     companion_slot = -1;
    int64_t M              = 0;
    int64_t K              = 0;
};

struct CompiledGraph {
    VEDAmodule   module    = 0;
    VEDAfunction run_func  = 0;
    std::string  src_hash;
    std::string  so_path;

    int          num_weights      = 0;
    int          num_kv_caches    = 0;
    int          num_slots        = 0;          // total p[] slots
    int          last_slot        = -1;         // slot index of the last op's dst
    size_t       output_bytes     = 0;

    // Each slot's role; resolved at execute time by walking the current
    // cgraph and matching the same encounter order.
    std::vector<BufferKind> slot_kinds;

    // Colmajor companion slots populated at execute time from
    // ctx->dev()->colmajor->get_or_create(tptrs[src0_slot], M, K, ...).
    // Empty when no MUL_MAT in the graph opted into the colmajor path.
    std::vector<ColmajorSpec> colmajor_specs;

    // Reusable HMEM staging buffers for the kernel's `input` (token id)
    // and `output` (logits row) args. Allocated lazily on first execute
    // and reused for every subsequent call of the same graph — the
    // alternative was acquire-from-pool + release per call, which costs
    // a syscall pair (codex finding #4). Freed when the CompiledGraph
    // is destroyed.
    VEDAhmemptr  in_hmem  = 0;
    VEDAhmemptr  out_hmem = 0;
};

class GraphCompiler {
public:
    static bool enabled();

    GraphCompiler();
    ~GraphCompiler();

    // Trace a whole cgraph. Returns false if any op is unsupported.
    bool trace(ggml_cgraph * cgraph);

    // Compile the traced graph. model_hash distinguishes between different
    // models / configurations; n_ctx is part of the cache key.
    CompiledGraph * compile(const std::string & model_hash, int64_t n_ctx);

    // Execute a previously-compiled graph for the current cgraph.
    // - resolves weight HBM pointers via name lookup from current_graph
    // - resolves KV HBM pointers via name lookup from current_graph
    // - reads the token id from src[0] of the first GET_ROWS op
    // - writes the lm_head logits row into output_logits
    bool execute(CompiledGraph * graph,
                 backend_context * ctx,
                 ggml_cgraph * current_graph);

    // Compatibility check between a compiled graph and a fresh trace:
    // op count, op types, dst dims. Used to decide whether to recompile.
    bool matches_trace(const CompiledGraph * graph) const;

    int num_traced_ops() const { return (int) traced_ops_.size(); }

private:
    std::vector<TracedOp>               traced_ops_;
    std::unordered_map<const ggml_tensor *, int> tensor_buf_idx_;
    std::unordered_map<const ggml_tensor *, BufferKind> tensor_buf_kind_;
    std::set<const ggml_tensor *>       output_tensors_;
    std::vector<const ggml_tensor *>    weight_tensors_;
    std::vector<const ggml_tensor *>    kv_cache_tensors_;
    std::vector<std::pair<const ggml_tensor *, size_t>> intermediate_bufs_;
    // Order-of-encounter slot assignment for the unified p[N] array.
    std::vector<std::pair<const ggml_tensor *, BufferKind>> tensor_slot_order_;
    // Colmajor companion slots created during trace, populated at execute.
    std::vector<ColmajorSpec>           colmajor_specs_;

    bool trace_valid_ = false;

    bool        is_weight(const ggml_tensor * t) const;
    int         assign_buffer(const ggml_tensor * t, BufferKind & kind_out);
    bool        trace_one(ggml_tensor * node);
    std::string generate_source(const std::string & func_name, int64_t n_ctx) const;
    std::string gen_op_code(const TracedOp & op, int idx) const;
    static std::string compute_hash(const std::string & source);
    static std::string cache_dir();
    bool        compile_source(const std::string & source, const std::string & so_path);
    CompiledGraph * load_compiled(const std::string & so_path, const std::string & hash);
};

GraphCompiler & get_compiler();

}  // namespace gcomp
}  // namespace ggml_ve

#endif  // GGML_VE_GRAPH_COMPILER_HPP
