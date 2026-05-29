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
    MUL_MAT_Q4K,      // Q4_K weights, F32 activations; canonical-split + pre-decoded hdr
    MUL_MAT_VEBP,     // VEBP ternary weights, F32 activations; vpcnt matvec
    ROPE,             // mode 0 (normal) and 2 (NeoX)
    SET_ROWS,
    FLASH_ATTN,
    CPY,
    SOFT_MAX,
    GLU_SWIGLU,       // GGML_OP_GLU with GGML_GLU_OP_SWIGLU, split gate/up
};

enum class BufferKind {
    WEIGHT,           // loaded once, never changes (model weights)
    WEIGHT_COLMAJOR,  // F32 column-major copy of a BF16 weight (cache populated lazily)
    WEIGHT_Q4K_QS,    // Canonical-packed qs bytes for Q4_K (lazy from hbm cache)
    WEIGHT_Q4K_HDR,   // Pre-decoded fp32 d_sub+m_sub for Q4_K (lazy from hbm cache)
    WEIGHT_VEBP_WS,   // VEBP interleaved sign plane    (lazy from hbm cache)
    WEIGHT_VEBP_WN,   // VEBP interleaved nonzero plane (lazy from hbm cache)
    WEIGHT_VEBP_WSC,  // VEBP interleaved group scales  (lazy from hbm cache)
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
    // For MUL_MAT_Q4K ops, the slots holding the canonical-packed qs
    // bytes and the pre-decoded fp32 headers (populated at execute via
    // hbm_cache::get_or_upload_q4k_canon). -1 = not a Q4_K op.
    int     q4k_qs_idx   = -1;
    int     q4k_hdr_idx  = -1;
    // For MUL_MAT_VEBP ops, the slots holding the interleaved sign/nz
    // planes and per-group scales (populated at execute via
    // hbm_cache::get_or_upload_vebp). -1 = not a VEBP op.
    int     vebp_ws_idx  = -1;
    int     vebp_wn_idx  = -1;
    int     vebp_wsc_idx = -1;
    BufferKind dst_kind  = BufferKind::INTERMEDIATE;
    BufferKind src0_kind = BufferKind::INTERMEDIATE;
    BufferKind src1_kind = BufferKind::INTERMEDIATE;
    BufferKind src2_kind = BufferKind::INTERMEDIATE;

    int64_t src0_ne[4] = {0,0,0,0};
    int64_t src1_ne[4] = {0,0,0,0};

    union {
        struct { float eps; }                                         rms_norm;
        struct { float freq_base, freq_scale, mscale; int n_dims, mode;
                 float ext_factor, beta_fast, beta_slow; int n_ctx_orig; } rope;
        struct { float scale, max_bias, softcap;
                 int   kv_type; int64_t n_kv_heads;
                 size_t nb_k1, nb_k2, nb_v1, nb_v2;
                 // Q byte strides (src0 nb): per-token (nb1) and per-head (nb2).
                 // For prompt eval Q is permuted [D,N,H] so the head stride is
                 // N*D*4, not D*4 — the decode _inner assumed contiguous heads.
                 size_t q_nb1, q_nb2;
                 // dst byte strides: per-head (nb1) and per-token (nb2).
                 size_t o_nb1, o_nb2; }                               flash_attn;
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

// Spec for the Q4_K canonical-split companion slots: at execute time
// we look up (or build) the pre-decoded qs+hdr pair for the Q4_K weight,
// keyed by tensor name, and stash both HBM pointers in the slot array.
struct Q4KSpec {
    int     src0_slot   = -1;   // original weight slot (used as identity key)
    int     qs_slot     = -1;   // populated with canonical qs pointer
    int     hdr_slot    = -1;   // populated with pre-decoded hdr pointer
    int64_t M           = 0;
    int64_t K           = 0;
};

// Spec for the VEBP interleaved companion slots: at execute time we look
// up (or build) the interleaved sign/nz planes + group scales for the VEBP
// weight, keyed by tensor name, and stash all three HBM pointers.
struct VebpSpec {
    int     src0_slot = -1;     // original weight slot (identity key)
    int     ws_slot   = -1;     // populated with interleaved sign-plane pointer
    int     wn_slot   = -1;     // populated with interleaved nz-plane pointer
    int     wsc_slot  = -1;     // populated with interleaved group-scale pointer
    int64_t M         = 0;
    int64_t K         = 0;
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

    // Q4_K companion slots populated at execute time via
    // hbm_cache::get_or_upload_q4k_canon. Empty if no Q4_K MUL_MAT.
    std::vector<Q4KSpec>      q4k_specs;

    // VEBP companion slots populated at execute time via
    // hbm_cache::get_or_upload_vebp. Empty if no VEBP MUL_MAT.
    std::vector<VebpSpec>     vebp_specs;

    // Reusable HMEM staging buffers for the kernel's `input` (token id)
    // and `output` (logits row) args. Allocated lazily on first execute
    // and reused for every subsequent call of the same graph — the
    // alternative was acquire-from-pool + release per call, which costs
    // a syscall pair (codex finding #4). Freed when the CompiledGraph
    // is destroyed.
    VEDAhmemptr  in_hmem  = 0;
    size_t       in_cap   = 0;   // bytes allocated for in_hmem (n_tok token ids)
    VEDAhmemptr  out_hmem = 0;

    // Per-token positions (i32, n_tok entries) staged from the ROPE inp_pos /
    // SET_ROWS index leaf each execute(). Drives ROPE angles, SET_ROWS KV-cell
    // rows and (for N>1) per-query causal seq_len. One entry for decode.
    VEDAhmemptr  pos_hmem = 0;
    size_t       pos_cap  = 0;   // bytes allocated for pos_hmem

    // Per-slot HBM scratch for CPU-resident operands. The ggml scheduler
    // parks a handful of small boundary tensors (cross-split hidden states,
    // the attention mask, etc.) on a CPU buffer even though the rest of the
    // subgraph lives in HBM. For those slots we stage host<->HBM each token
    // (upload before launch, and download after for graph-produced ones) so
    // the fused kernel can address everything uniformly through p[]. Sized
    // num_slots, allocated lazily on first use, reused every token. This is
    // what lets the compiler engage on the main decode graph instead of
    // bailing to the interpreter on the first CPU operand.
    std::vector<VEDAdeviceptr> stage_hbm;   // scratch base per slot (0 = none)
    std::vector<size_t>        stage_cap;   // bytes allocated per slot
};

class GraphCompiler {
public:
    static bool enabled();

    GraphCompiler();
    ~GraphCompiler();

    // Trace a whole cgraph. Returns false if any op is unsupported.
    bool trace(ggml_cgraph * cgraph);

    // Compile the traced graph. The on-disk cache key is the hash of
    // the emitted source itself, so two cgraphs that produce identical
    // code share a single .so.
    CompiledGraph * compile();

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
    // n_tokens of the traced cgraph (1 = decode, >1 = prompt eval). Used to
    // divide baked element counts down to per-token constants so the codegen
    // can scale by the runtime n_tok arg (size-independent across N).
    int64_t                             n_tok_baked_ = 1;
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
    // Q4_K companion slots created during trace, populated at execute.
    std::vector<Q4KSpec>                q4k_specs_;
    // VEBP companion slots created during trace, populated at execute.
    std::vector<VebpSpec>               vebp_specs_;

    bool trace_valid_ = false;

    bool        is_weight(const ggml_tensor * t) const;
    int         assign_buffer(const ggml_tensor * t, BufferKind & kind_out);
    bool        trace_one(ggml_tensor * node);
    std::string generate_source(const std::string & func_name) const;
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
