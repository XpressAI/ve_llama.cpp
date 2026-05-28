#include "ops.hpp"
#include "common.hpp"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <veda.h>

namespace ggml_ve {

// Set GGML_VE_DEBUG_DISPATCH=1 to log the first 200 op dispatches with
// tensor types and buffer types — handy when an op produces garbage in a
// real model but passes unit tests.
namespace {
int debug_dispatch_count = 0;

// Per-op profiling. Enabled by GGML_VE_PROFILE_PEROP=1. Syncs the VE
// context after every op so the elapsed time reflects actual kernel
// runtime (most VE ops are queued + deferred-sync). The sync itself
// adds overhead — this mode is for diagnosis, not production.
struct perop_stat {
    uint64_t ns_total = 0;
    uint64_t calls    = 0;
    // For MUL_MAT we also bucket by the inner-batch dim N to expose
    // O(N²)-ish scaling in cblas / FA paths.
    uint64_t ns_by_nbucket[6] = {0};  // N: 1, 2-7, 8-31, 32-127, 128-511, 512+
    uint64_t calls_by_nbucket[6] = {0};
};

std::mutex                                  g_perop_mu;
std::unordered_map<std::string, perop_stat> g_perop;
std::atomic<bool>                           g_perop_registered{false};

int nbucket(int64_t N) {
    if (N <= 1)   return 0;
    if (N < 8)    return 1;
    if (N < 32)   return 2;
    if (N < 128)  return 3;
    if (N < 512)  return 4;
    return 5;
}

const char * nbucket_name(int b) {
    static const char * names[6] = { "N=1", "N=2-7", "N=8-31", "N=32-127", "N=128-511", "N>=512" };
    return names[b];
}

void perop_print() {
    std::lock_guard<std::mutex> lk(g_perop_mu);
    if (g_perop.empty()) return;

    std::vector<std::pair<std::string, perop_stat>> rows(g_perop.begin(), g_perop.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto & a, const auto & b) { return a.second.ns_total > b.second.ns_total; });

    uint64_t grand_total_ns = 0;
    uint64_t grand_total_calls = 0;
    for (const auto & r : rows) { grand_total_ns += r.second.ns_total; grand_total_calls += r.second.calls; }

    fprintf(stderr, "\n========== VE per-op profile ==========\n");
    fprintf(stderr, "%-22s %10s %12s %12s %7s\n", "op", "calls", "total_ms", "avg_us", "pct");
    for (const auto & r : rows) {
        const auto & s = r.second;
        double total_ms = s.ns_total / 1e6;
        double avg_us   = s.calls ? (s.ns_total / 1e3) / (double) s.calls : 0;
        double pct      = grand_total_ns ? 100.0 * s.ns_total / grand_total_ns : 0;
        fprintf(stderr, "%-22s %10llu %12.2f %12.2f %6.1f%%\n",
                r.first.c_str(),
                (unsigned long long) s.calls, total_ms, avg_us, pct);
        // Per-N breakdown if any bucket has calls.
        for (int b = 0; b < 6; ++b) {
            if (!s.calls_by_nbucket[b]) continue;
            double bt_ms = s.ns_by_nbucket[b] / 1e6;
            double ba_us = (s.ns_by_nbucket[b] / 1e3) / (double) s.calls_by_nbucket[b];
            fprintf(stderr, "    %-18s %10llu %12.2f %12.2f\n",
                    nbucket_name(b),
                    (unsigned long long) s.calls_by_nbucket[b], bt_ms, ba_us);
        }
    }
    fprintf(stderr, "------\n%-22s %10llu %12.2f\n",
            "TOTAL", (unsigned long long) grand_total_calls, grand_total_ns / 1e6);
    fprintf(stderr, "=======================================\n\n");
}

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}
}

bool supports_op(const device * dev, const ggml_tensor * op) {
    if (dev == nullptr || !dev->initialized) return false;

    static const bool dbg_support = std::getenv("GGML_VE_DEBUG_SUPPORT") != nullptr;
    auto trace = [&](bool ok) {
        if (dbg_support) {
            const char * buft_name = "?";
            if (op->buffer) {
                ggml_backend_buffer_type_t bt = ggml_backend_buffer_get_type(op->buffer);
                if (bt) buft_name = ggml_backend_buft_name(bt);
            }
            fprintf(stderr, "[VE-SUPPORT] %-20s dst=%s name='%s' buft=%s -> %s\n",
                    ggml_op_name(op->op), ggml_type_name(op->type),
                    op->name[0]?op->name:"?", buft_name, ok ? "YES" : "no");
        }
        return ok;
    };

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_ADD:
            if (std::getenv("GGML_VE_NO_ADD") != nullptr) return false;
            return ops::add_supports(op);
        case GGML_OP_MUL:
            if (std::getenv("GGML_VE_NO_MUL") != nullptr) return false;
            return ops::mul_supports(op);
        case GGML_OP_SCALE:
            if (std::getenv("GGML_VE_NO_SCALE") != nullptr) return false;
            return ops::scale_supports(op);
        case GGML_OP_UNARY:
            if (std::getenv("GGML_VE_NO_UNARY") != nullptr) return false;
            return ops::silu_supports(op);
        case GGML_OP_GLU:
            if (std::getenv("GGML_VE_NO_GLU") != nullptr) return false;
            return ops::glu_supports(op);
        case GGML_OP_RMS_NORM:
            if (std::getenv("GGML_VE_NO_RMS_NORM") != nullptr) return false;
            return ops::rms_norm_supports(op);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            if (std::getenv("GGML_VE_NO_CPY") != nullptr) return false;
            return ops::cpy_supports(op);
        case GGML_OP_GET_ROWS:
            if (std::getenv("GGML_VE_NO_GET_ROWS") != nullptr) return false;
            return ops::get_rows_supports(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ops::flash_attn_supports(op);
        case GGML_OP_SET_ROWS:
            if (std::getenv("GGML_VE_NO_SET_ROWS") != nullptr) return false;
            return ops::set_rows_supports(op);
        case GGML_OP_ROPE:
            if (std::getenv("GGML_VE_NO_ROPE") != nullptr) return false;
            return trace(ops::rope_supports(op));
        case GGML_OP_MUL_MAT_ID:
            return ops::mul_mat_id_supports(op);
        case GGML_OP_ADD_ID:
            return ops::add_id_supports(op);
        case GGML_OP_ARGSORT:
            return ops::argsort_supports(op);
        case GGML_OP_SOFT_MAX:
            if (std::getenv("GGML_VE_NO_SOFTMAX") != nullptr) return false;
            return ops::soft_max_supports(op);
        case GGML_OP_MUL_MAT:
            if (std::getenv("GGML_VE_NO_MUL_MAT") != nullptr) return false;
            return trace(ops::mul_mat_supports(op) || ops::mul_mat_q_supports(op));
        case GGML_OP_SSM_CONV:
            if (std::getenv("GGML_VE_NO_SSM_CONV") != nullptr) return false;
            return ops::ssm_conv_supports(op);
        case GGML_OP_GATED_DELTA_NET:
            if (std::getenv("GGML_VE_NO_GDN") != nullptr) return false;
            return ops::gated_delta_net_supports(op);
        case GGML_OP_SUB:
            if (std::getenv("GGML_VE_NO_SUB") != nullptr) return false;
            return trace(ops::sub_supports(op));
        case GGML_OP_SQR:
            if (std::getenv("GGML_VE_NO_SQR") != nullptr) return false;
            return trace(ops::sqr_supports(op));
        case GGML_OP_L2_NORM:
            if (std::getenv("GGML_VE_NO_L2_NORM") != nullptr) return false;
            return trace(ops::l2_norm_supports(op));
        case GGML_OP_SUM_ROWS:
            if (std::getenv("GGML_VE_NO_SUM_ROWS") != nullptr) return false;
            return trace(ops::sum_rows_supports(op));
        case GGML_OP_REPEAT:
            if (std::getenv("GGML_VE_NO_REPEAT") != nullptr) return false;
            return trace(ops::repeat_supports(op));
        case GGML_OP_CONCAT:
            if (std::getenv("GGML_VE_NO_CONCAT") != nullptr) return false;
            return trace(ops::concat_supports(op));
        default:
            return false;
    }
}

static bool compute_forward_inner(backend_context * ctx, ggml_tensor * node);

bool compute_forward(backend_context * ctx, ggml_tensor * node) {
    static const bool profile = (std::getenv("GGML_VE_PROFILE_PEROP") != nullptr);
    static const bool quant_safe = (std::getenv("GGML_VE_QUANT_SAFE_MODE") != nullptr);
    // Plain fast path — no per-op work.
    if (!profile && !quant_safe) {
        return compute_forward_inner(ctx, node);
    }
    // QUANT_SAFE_MODE: per-op flush. Required for correctness on Q-quant
    // models (see backend_ctx.cpp fresh-hit KNOWN-BUG comment + task #60).
    // Single POST-flush guarantees CPU memory is fresh by the time the
    // next op starts -- which means the next op's resolve_in_slow can
    // read from CPU and get valid bytes. A separate pre-flush is
    // redundant: the prior op's post-flush already drained the queue.
    // (Saves ~50us of vedaCtxSynchronize per op = ~2x throughput on
    // Q-quant models compared to the previous pre+post variant.)
    if (quant_safe && !profile) {
        bool ok = compute_forward_inner(ctx, node);
        ctx->flush("quant_safe post");
        return ok;
    }
    // PROFILE_PEROP (with or without quant_safe): timed per-op sync.
    if (!g_perop_registered.exchange(true)) {
        std::atexit(perop_print);
    }
    // Drain prior ops so the timer measures THIS op's runtime, not the
    // queue tail of whatever was launched earlier in this graph.
    ctx->flush("perop pre");
    uint64_t t0 = now_ns();
    bool ok = compute_forward_inner(ctx, node);
    // Sync again so the op we just launched actually finishes before we
    // stop the clock.
    ctx->flush("perop post");
    vedaCtxSynchronize();
    uint64_t dt = now_ns() - t0;

    // Pick a label. For MUL_MAT we also bucket by N (src1->ne[1]) so we
    // can see how cost scales with the prompt batch.
    const char * op_name = ggml_op_name(node->op);
    std::string label    = op_name;
    if (node->op == GGML_OP_UNARY) {
        // GGML_OP_UNARY hides a sub-op (SILU/SIGMOID/EXP/...) we want to
        // distinguish in the histogram.
        label = std::string("UNARY:") + ggml_unary_op_name(ggml_get_unary_op(node));
    } else if (node->op == GGML_OP_GLU) {
        label = std::string("GLU:") + ggml_glu_op_name(ggml_get_glu_op(node));
    }
    int b = -1;
    if (node->op == GGML_OP_MUL_MAT && node->src[1]) {
        b = nbucket(node->src[1]->ne[1]);
    } else if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[0]) {
        // Q rows = batch dimension for FA prefill.
        b = nbucket(node->src[0]->ne[1]);
    }
    {
        std::lock_guard<std::mutex> lk(g_perop_mu);
        auto & s = g_perop[label];
        s.ns_total += dt;
        s.calls    += 1;
        if (b >= 0 && b < 6) {
            s.ns_by_nbucket[b]    += dt;
            s.calls_by_nbucket[b] += 1;
        }
    }
    return ok;
}

static bool compute_forward_inner(backend_context * ctx, ggml_tensor * node) {
    static bool debug = (std::getenv("GGML_VE_DEBUG_DISPATCH") != nullptr);
    if (debug && debug_dispatch_count < 5000) {
        debug_dispatch_count++;
        const char * src0_buft = "?";
        const char * src1_buft = "?";
        if (node->src[0] && node->src[0]->buffer) {
            auto * b = ggml_backend_buffer_get_type(node->src[0]->buffer);
            src0_buft = b ? ggml_backend_buft_name(b) : "?";
        }
        if (node->src[1] && node->src[1]->buffer) {
            auto * b = ggml_backend_buffer_get_type(node->src[1]->buffer);
            src1_buft = b ? ggml_backend_buft_name(b) : "?";
        }
        fprintf(stderr, "[VE-DISPATCH %3d] %-18s dst=%s name=%-30s src0=%s/%s src1=%s/%s\n",
                debug_dispatch_count, ggml_op_name(node->op),
                ggml_type_name(node->type), node->name[0] ? node->name : "(noname)",
                node->src[0] ? ggml_type_name(node->src[0]->type) : "-", src0_buft,
                node->src[1] ? ggml_type_name(node->src[1]->type) : "-", src1_buft);
    }
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
    }
    // Zero-size tensors: scheduler dry-run / reserve passes hand us 1-node
    // chunks whose dst (and inputs) have ne[k]=0 in some dimension. There
    // is no work to do — matches the CPU backend's behavior of silently
    // doing nothing. Must come before the per-op dispatch so empty inputs
    // don't reach kernel launch code with nbytes=0.
    if (ggml_nbytes(node) == 0) {
        return true;
    }
    switch (node->op) {
        case GGML_OP_ADD:
            return ops::add_f32(ctx, node);
        case GGML_OP_SUB:
            return ops::sub_f32(ctx, node);
        case GGML_OP_SQR:
            return ops::sqr_f32(ctx, node);
        case GGML_OP_SUM_ROWS:
            return ops::sum_rows_f32(ctx, node);
        case GGML_OP_REPEAT:
            return ops::repeat_f32(ctx, node);
        case GGML_OP_CONCAT:
            return ops::concat_f32(ctx, node);
        case GGML_OP_MUL:
            return ops::mul_f32(ctx, node);
        case GGML_OP_SCALE:
            return ops::scale_f32(ctx, node);
        case GGML_OP_UNARY:
            return ops::silu_f32(ctx, node);
        case GGML_OP_L2_NORM:
            return ops::l2_norm_f32(ctx, node);
        case GGML_OP_GLU:
            return ops::glu_f32(ctx, node);
        case GGML_OP_RMS_NORM:
            return ops::rms_norm_f32(ctx, node);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return ops::cpy_f32(ctx, node);
        case GGML_OP_GET_ROWS:
            return ops::get_rows(ctx, node);
        case GGML_OP_FLASH_ATTN_EXT:
            return ops::flash_attn(ctx, node);
        case GGML_OP_SET_ROWS:
            return ops::set_rows(ctx, node);
        case GGML_OP_ROPE:
            return ops::rope(ctx, node);
        case GGML_OP_MUL_MAT_ID:
            return ops::mul_mat_id(ctx, node);
        case GGML_OP_ADD_ID:
            return ops::add_id(ctx, node);
        case GGML_OP_ARGSORT:
            return ops::argsort(ctx, node);
        case GGML_OP_SOFT_MAX:
            return ops::soft_max_f32(ctx, node);
        case GGML_OP_MUL_MAT:
            // Try the quant path first; if the op doesn't match, fall through
            // to the dense (F32/BF16) handler.
            if (ops::mul_mat_q_supports(node)) {
                return ops::mul_mat_q(ctx, node);
            }
            return ops::mul_mat(ctx, node);
        case GGML_OP_SSM_CONV:
            return ops::ssm_conv_f32(ctx, node);
        case GGML_OP_GATED_DELTA_NET:
            return ops::gated_delta_net_f32(ctx, node);
        default:
            GGML_LOG_ERROR("ggml-ve: unsupported op assigned to backend: %s\n",
                           ggml_op_name(node->op));
            return false;
    }
}

} // namespace ggml_ve
