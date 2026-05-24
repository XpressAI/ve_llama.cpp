// Registry, device-vtable, and public-API glue for the NEC SX-Aurora TSUBASA
// Vector Engine backend. All the actual functionality lives in:
//
//   common.hpp        - VEDA context guard, macros
//   device.{hpp,cpp}  - per-device state, kernel module loading
//   hbm_cache.hpp     - HBM weight cache
//   hmem_pool.hpp     - HMEM buffer pool
//   backend_ctx.hpp   - per-backend instance state + deferred sync
//   buffer.{hpp,cpp}  - HBM buffer-type vtable
//   ops.{hpp,cpp}     - op dispatcher
//   ops/<name>.cpp    - per-op handlers
//
// This file is intentionally small: it just wires those modules into ggml's
// backend / device / registry vtables.

#include "ggml-ve.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "backend_ctx.hpp"
#include "buffer.hpp"
#include "common.hpp"
#include "device.hpp"
#include "graph_compiler.hpp"
#include "ops.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml_ve {
namespace {

std::array<ggml_backend_device, GGML_VE_MAX_DEVICES> g_backend_devices;
ggml_backend_reg                                     g_backend_reg;
bool                                                 g_backend_inited = false;

ggml_guid_t backend_guid() {
    static const char * guid = "NEC-VE-AURORA-2!";
    return reinterpret_cast<ggml_guid_t>(const_cast<char *>(guid));
}

//
// ggml_backend_i  (per-instance backend)
//
const char * backend_get_name(ggml_backend_t backend) {
    auto * ctx = (backend_context *) backend->context;
    return ctx && ctx->dev() ? ctx->dev()->name : GGML_VE_NAME;
}

void backend_free(ggml_backend_t backend) {
    auto * ctx = (backend_context *) backend->context;
    if (ctx) {
        const device * d = ctx->dev();
        GGML_LOG_INFO("ggml-ve: %s total_ops=%ld mul_mat=%ld flash_attn=%ld syncs=%ld\n",
                      d ? d->name : "(no dev)",
                      ctx->ops_total(), ctx->ops_mul_mat(), ctx->ops_flash_attn(),
                      ctx->sync_count());
        size_t allocated = 0; int64_t hits = 0, misses = 0, whits = 0;
        ctx->pool().stats(&allocated, &hits, &misses, &whits);
        if (allocated || hits || misses) {
            GGML_LOG_INFO("ggml-ve: HMEM pool: %.2f MiB, hits=%ld misses=%ld weight-hits=%ld\n",
                          allocated / (1024.0 * 1024.0), hits, misses, whits);
        }
        delete ctx;
    }
    delete backend;
}

// In-memory cache of compiled graphs, keyed by a structural signature of
// the cgraph (op + dst dtype + dst shape + per-src dtype + per-src shape
// over every node). Two cgraphs with the same signature generate the same
// source and the same on-disk .so; they should hit the same in-memory
// CompiledGraph too — otherwise we needlessly recompile per call.
//
// `executable` tracks whether execute() succeeded last time. Some cgraph
// shapes (small GET_ROWS chunks, intermediate-source ops) trace + compile
// fine but fail in execute (typically because a slot tensor has no HBM
// backing). We KEEP the compiled handle anyway so we don't reload the
// .so from disk every call — just skip the execute attempt next time
// and fall through to the interpreter.
struct cached_graph {
    gcomp::CompiledGraph * cg         = nullptr;
    bool                   executable = true;
};
// NOT thread_local — ggml's scheduler can invoke backend_graph_compute
// from different worker threads, and a thread_local cache loses the
// CompiledGraph every time. Two cgraphs back-to-back from different
// threads would each force a fresh trace+compile (or .so load) even
// though the signature is identical.
static std::mutex                                       g_cg_cache_mu;
static std::unordered_map<uint64_t, cached_graph>       g_cg_cache;

// True for the KV cache tensors and the attention mask. The growing
// dimensions of these tensors (the seq dim and, for mask, the masked-kv
// dim) change mid-decode as ggml's allocator expands the KV view in
// power-of-2 chunks. The compiled kernel takes the actual sequence
// length from the runtime `pos` arg and doesn't care about the view's
// per-call shape, so excluding those dims keeps a single .so reusable
// across all decode steps in a session.
static bool is_kv_or_mask_tensor(const ggml_tensor * t) {
    if (!t || !t->name) return false;
    const char * n = t->name;
    return std::strncmp(n, "cache_k", 7) == 0
        || std::strncmp(n, "cache_v", 7) == 0
        || std::strncmp(n, "attn_inp_kq_mask", 16) == 0;
}

// 64-bit structural fingerprint. Hashes everything that affects
// generated source. Treats KV-cache and mask tensors as if their
// growing dimensions are wildcards — see is_kv_or_mask_tensor above.
//
// Note vs the legacy backend's compute_model_fingerprint (first 4
// MUL_MAT weight dims + n_nodes only): legacy could get away with a
// model-level fingerprint because it received one big cgraph per
// token. Upstream's current scheduler hands us several small
// cgraphs per token (6, 23, 853 nodes on Llama-class decode), and
// they have genuinely distinct shapes, so we need per-cgraph-shape
// granularity. Hence the broader hash.
static uint64_t cgraph_signature(const ggml_cgraph * g) {
    std::hash<uint64_t> H;
    uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset basis
    auto mix = [&](uint64_t v) { h ^= H(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    auto mix_tensor = [&](const ggml_tensor * t, int slot) {
        if (t == nullptr) { mix(0); return; }
        mix((uint64_t) t->type | (uint64_t)(slot + 1) << 32);
        const bool wild = is_kv_or_mask_tensor(t);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            // Hash ne[0] always (head_dim / embed_dim — stable per model).
            // For KV / mask tensors skip ne[1..2] (the seq + mask dims
            // that grow as KV occupancy crosses power-of-2 chunk
            // boundaries); ne[3] back in since batch is meaningful.
            if (wild && (d == 1 || d == 2)) {
                mix((uint64_t) 0xFFFFFFFFFFFFFFFFULL);  // wildcard marker
            } else {
                mix((uint64_t) t->ne[d]);
            }
        }
    };
    mix((uint64_t) g->n_nodes);
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * n = g->nodes[i];
        mix((uint64_t) n->op);
        mix_tensor(n, /*slot=*/-1);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            mix_tensor(n->src[s], s);
        }
    }
    return h;
}

// GGML_VE_DIAGNOSE_SIG=1 — diff per-tensor ne against the previous cgraph
// with the same n_nodes. On each compile-triggering miss, print which
// tensor's shape changed since the last seen cgraph of that node count.
// Answers "what's causing the second compile mid-decode" in one run.
struct sig_diag_node {
    int     op       = -1;
    int     type     = -1;
    int64_t dst_ne[4] = { -1, -1, -1, -1 };
    int     src_type[GGML_MAX_SRC] = { -1, -1, -1, -1 };
    int64_t src_ne  [GGML_MAX_SRC][4] = {{0}};
};
struct sig_diag_state {
    int                          n_nodes = -1;
    std::vector<sig_diag_node>   nodes;
};
static std::mutex                                       g_sig_diag_mu;
static std::unordered_map<int, sig_diag_state>          g_sig_diag_by_nnodes;

static void diagnose_signature(const ggml_cgraph * g, uint64_t sig) {
    sig_diag_state cur;
    cur.n_nodes = g->n_nodes;
    cur.nodes.resize(g->n_nodes);
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * n = g->nodes[i];
        cur.nodes[i].op   = (int) n->op;
        cur.nodes[i].type = (int) n->type;
        for (int d = 0; d < 4; ++d) cur.nodes[i].dst_ne[d] = n->ne[d];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = n->src[s];
            cur.nodes[i].src_type[s] = src ? (int) src->type : -1;
            for (int d = 0; d < 4; ++d) cur.nodes[i].src_ne[s][d] = src ? src->ne[d] : -1;
        }
    }

    std::lock_guard<std::mutex> lk(g_sig_diag_mu);
    auto & prev = g_sig_diag_by_nnodes[g->n_nodes];
    if (prev.n_nodes != g->n_nodes || prev.nodes.empty()) {
        fprintf(stderr, "[VE-SIG-DIAG] sig=%016lx n_nodes=%d — first cgraph of this size, no diff baseline\n",
                (unsigned long) sig, g->n_nodes);
        prev = std::move(cur);
        return;
    }

    fprintf(stderr, "[VE-SIG-DIAG] sig=%016lx n_nodes=%d — differences vs previous cgraph:\n",
            (unsigned long) sig, g->n_nodes);
    int diff_count = 0;
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * n = g->nodes[i];
        const char * nm = n->name && n->name[0] ? n->name : "?";
        if (cur.nodes[i].op != prev.nodes[i].op) {
            fprintf(stderr, "  node %3d '%s' op: %d -> %d (%s)\n",
                    i, nm, prev.nodes[i].op, cur.nodes[i].op, ggml_op_name(n->op));
            ++diff_count;
        }
        if (cur.nodes[i].type != prev.nodes[i].type) {
            fprintf(stderr, "  node %3d '%s' (%s) dst.type: %d -> %d\n",
                    i, nm, ggml_op_name(n->op),
                    prev.nodes[i].type, cur.nodes[i].type);
            ++diff_count;
        }
        for (int d = 0; d < 4; ++d) {
            if (cur.nodes[i].dst_ne[d] != prev.nodes[i].dst_ne[d]) {
                fprintf(stderr, "  node %3d '%s' (%s) dst.ne[%d]: %ld -> %ld\n",
                        i, nm, ggml_op_name(n->op),
                        d, (long) prev.nodes[i].dst_ne[d], (long) cur.nodes[i].dst_ne[d]);
                ++diff_count;
            }
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = n->src[s];
            const char * sn = (src && src->name && src->name[0]) ? src->name : "?";
            if (cur.nodes[i].src_type[s] != prev.nodes[i].src_type[s]) {
                fprintf(stderr, "  node %3d '%s' (%s) src[%d]='%s' type: %d -> %d\n",
                        i, nm, ggml_op_name(n->op), s, sn,
                        prev.nodes[i].src_type[s], cur.nodes[i].src_type[s]);
                ++diff_count;
            }
            for (int d = 0; d < 4; ++d) {
                if (cur.nodes[i].src_ne[s][d] != prev.nodes[i].src_ne[s][d]) {
                    fprintf(stderr, "  node %3d '%s' (%s) src[%d]='%s' ne[%d]: %ld -> %ld\n",
                            i, nm, ggml_op_name(n->op), s, sn,
                            d, (long) prev.nodes[i].src_ne[s][d], (long) cur.nodes[i].src_ne[s][d]);
                    ++diff_count;
                }
            }
        }
    }
    if (diff_count == 0) {
        fprintf(stderr, "  (no per-tensor differences — signature collision? signature impl bug?)\n");
    }

    prev = std::move(cur);
}

ggml_status backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (backend_context *) backend->context;
    VEDAContextGuard guard(ctx->dev() ? ctx->dev()->context : nullptr);
    if (!guard.is_valid()) return GGML_STATUS_FAILED;

    ctx->set_cgraph(cgraph);

    // GGML_VE_PROFILE=1 — dump the op-type histogram of the cgraphs that
    // reach this backend. Cheap way to see what the ggml scheduler is
    // actually handing us per token, which directly answers "why are
    // cgraphs so small" / "what op causes the split".
    static const bool profile_enabled = std::getenv("GGML_VE_PROFILE") != nullptr;
    if (profile_enabled) {
        int counts[GGML_OP_COUNT] = {0};
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_op op = cgraph->nodes[i]->op;
            if ((int) op < GGML_OP_COUNT) counts[op]++;
        }
        fprintf(stderr, "[VE-PROFILE] n_nodes=%d:", cgraph->n_nodes);
        for (int i = 0; i < GGML_OP_COUNT; ++i) {
            if (counts[i] > 0) {
                fprintf(stderr, " %s=%d", ggml_op_name((ggml_op) i), counts[i]);
            }
        }
        fprintf(stderr, "\n");
    }

    // GGML_VE_TIME_GRAPHS=1 — accumulate wall-clock per cgraph, grouped by
    // a coarse signature (op-type histogram). Prints a totals table at
    // process exit. This is the easiest way to answer "where is decode
    // spending its time" without poking the VE-side kernels.
    struct gtime_entry {
        std::string sig;
        int64_t     count = 0;
        double      ns_total = 0;
    };
    static bool time_enabled = std::getenv("GGML_VE_TIME_GRAPHS") != nullptr;
    static std::vector<gtime_entry> gtimes;
    static std::once_flag gtime_once;
    if (time_enabled) {
        std::call_once(gtime_once, []() {
            std::atexit([]() {
                fprintf(stderr, "\n[VE-TIME] graph-compute totals (sig = op-type histogram)\n");
                fprintf(stderr, "  %-8s %-12s %-12s   %s\n",
                        "calls", "total_ms", "avg_us", "sig");
                // bubble sort by total descending — fewer items than 50
                auto sorted = gtimes;
                std::sort(sorted.begin(), sorted.end(),
                          [](const gtime_entry & a, const gtime_entry & b){
                              return a.ns_total > b.ns_total;
                          });
                for (const auto & e : sorted) {
                    fprintf(stderr, "  %-8ld %-12.3f %-12.3f   %s\n",
                            (long) e.count, e.ns_total / 1e6,
                            (e.ns_total / e.count) / 1e3,
                            e.sig.c_str());
                }
            });
        });
    }
    auto gtime_start = std::chrono::steady_clock::now();
    std::string gtime_sig;
    if (time_enabled) {
        char buf[256];
        int counts[GGML_OP_COUNT] = {0};
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_op op = cgraph->nodes[i]->op;
            if ((int) op < GGML_OP_COUNT) counts[op]++;
        }
        int off = std::snprintf(buf, sizeof(buf), "n=%d", cgraph->n_nodes);
        for (int i = 0; i < GGML_OP_COUNT && off < (int) sizeof(buf) - 32; ++i) {
            if (counts[i] > 0) {
                off += std::snprintf(buf + off, sizeof(buf) - off,
                                     ",%s=%d",
                                     ggml_op_name((ggml_op) i), counts[i]);
            }
        }
        gtime_sig = buf;
    }

    // --- Compiled-graph fast path (opt-in via GGML_VE_COMPILE_GRAPH=1) ----
    // No node-count threshold by default. ncc compile is ~30 s per unique
    // cgraph signature, but a per-token decode chunk repeats ~50× per
    // second — even a 6-node chunk amortises in under a minute. The
    // previous default of 100 prevented the compiler from ever engaging
    // on Llama-class decode, where the scheduler hands us 6- and 23-node
    // chunks. Set GGML_VE_COMPILE_MIN_NODES=N to override.
    static const int gc_min_nodes = []{
        const char * env = std::getenv("GGML_VE_COMPILE_MIN_NODES");
        return env ? std::atoi(env) : 1;
    }();
    static const bool gc_verbose = []() {
        const char * env = std::getenv("GGML_VE_COMPILE_DEBUG");
        return !(env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0);
    }();
    if (gc_verbose) {
        fprintf(stderr, "[VE-GC] graph_compute called n_nodes=%d (min=%d)\n",
                cgraph->n_nodes, gc_min_nodes);
    }
    if (gcomp::GraphCompiler::enabled() && cgraph->n_nodes >= gc_min_nodes) {
        auto & gc = gcomp::get_compiler();
        const uint64_t sig = cgraph_signature(cgraph);
        static const bool diag_sig = (std::getenv("GGML_VE_DIAGNOSE_SIG") != nullptr);

        // Look up the cached entry under lock. We copy it out into a
        // local so the actual execute() / compile() / store happen
        // outside the critical section — ncc compile takes ~30 s and
        // we don't want to hold the cache mutex that long.
        cached_graph entry;
        bool         entry_present = false;
        {
            std::lock_guard<std::mutex> lk(g_cg_cache_mu);
            auto it = g_cg_cache.find(sig);
            if (it != g_cg_cache.end()) {
                entry         = it->second;
                entry_present = true;
            }
        }

        if (diag_sig && !entry_present) {
            diagnose_signature(cgraph, sig);
        }

        if (entry_present) {
            // Three possible cached outcomes: (1) executable=true with a
            // valid cg → execute; (2) executable=true but cg=null → trace
            // was rejected previously, fall through to interpreter; (3)
            // executable=false → execute regressed previously, ditto.
            if (entry.executable && entry.cg) {
                if (gc.execute(entry.cg, ctx, cgraph)) {
                    ctx->ops_total() += cgraph->n_nodes;
                    return GGML_STATUS_SUCCESS;
                }
                // Execute regressed — remember and stop trying.
                {
                    std::lock_guard<std::mutex> lk(g_cg_cache_mu);
                    auto it = g_cg_cache.find(sig);
                    if (it != g_cg_cache.end()) it->second.executable = false;
                }
                if (gc_verbose) {
                    fprintf(stderr, "[VE-GC] execute failed for sig=%016lx — marked un-executable, falling back\n",
                            (unsigned long) sig);
                }
            }
            // Anything else: silently fall through to the interpreter.
        } else {
            // Trace: succeed (compile + try execute) or fail (cache the
            // failure so the next call doesn't re-trace).
            bool traced = gc.trace(cgraph);
            cached_graph new_entry;  // cg=null, executable=true by default
            if (traced) {
                char hashbuf[32];
                std::snprintf(hashbuf, sizeof(hashbuf), "%016lx", (unsigned long) sig);
                gcomp::CompiledGraph * cg2 = gc.compile(hashbuf, /*n_ctx=*/ 4096);
                if (cg2) {
                    new_entry.cg         = cg2;
                    new_entry.executable = gc.execute(cg2, ctx, cgraph);
                } else if (gc_verbose) {
                    fprintf(stderr, "[VE-GC] compile failed for sig=%016lx\n", (unsigned long) sig);
                }
            } else if (gc_verbose) {
                fprintf(stderr, "[VE-GC] trace rejected the graph sig=%016lx; cached as interpreter-only\n",
                        (unsigned long) sig);
            }
            // Cache the outcome either way (cg may be null, executable
            // may be false). Future calls with the same sig skip trace
            // and either execute (if cg+executable) or fall straight
            // through to the interpreter.
            {
                std::lock_guard<std::mutex> lk(g_cg_cache_mu);
                g_cg_cache[sig] = new_entry;
            }
            if (new_entry.cg && new_entry.executable) {
                ctx->ops_total() += cgraph->n_nodes;
                return GGML_STATUS_SUCCESS;
            }
        }
    }

    // --- Interpreter path -------------------------------------------------
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!compute_forward(ctx, node)) {
            ctx->abort_pending();
            return GGML_STATUS_FAILED;
        }
        ctx->ops_total()++;
    }
    // Do NOT flush here. ggml's scheduler invokes the .synchronize hook
    // (backend_synchronize, below) at boundaries where the host actually
    // needs to read VE-side state (e.g. when handing logits back to CPU
    // for sampling, or before a tensor transitions to another backend).
    // The legacy ve_llama backend's graph_compute also doesn't flush —
    // syncing here forces a vedaCtxSynchronize per cgraph chunk, and on
    // models where the scheduler hands us many small chunks per token
    // (Llama-3-class: ~19 chunks/token vs legacy's 1) the per-chunk sync
    // becomes the dominant overhead. Measured ~3x decode regression vs
    // legacy. Trust the scheduler — anything that legitimately needs to
    // be sync'd will route through backend_synchronize.

    if (time_enabled) {
        auto end = std::chrono::steady_clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - gtime_start).count();
        // Linear search — there are < 50 distinct cgraph shapes per token.
        bool found = false;
        for (auto & e : gtimes) {
            if (e.sig == gtime_sig) { e.count++; e.ns_total += ns; found = true; break; }
        }
        if (!found) {
            gtime_entry e; e.sig = gtime_sig; e.count = 1; e.ns_total = ns;
            gtimes.push_back(std::move(e));
        }
    }
    return GGML_STATUS_SUCCESS;
}

void backend_synchronize(ggml_backend_t backend) {
    auto * ctx = (backend_context *) backend->context;
    VEDAContextGuard guard(ctx->dev() ? ctx->dev()->context : nullptr);
    if (guard.is_valid()) {
        ctx->flush("backend_synchronize");
    }
}

const ggml_backend_i backend_iface = {
    /* .get_name                = */ backend_get_name,
    /* .free                    = */ backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

//
// ggml_backend_device_i
//
const char * dev_get_name(ggml_backend_dev_t dev) {
    auto * d = (device *) dev->context;
    return d->name[0] ? d->name : GGML_VE_NAME;
}

const char * dev_get_description(ggml_backend_dev_t dev) {
    auto * d = (device *) dev->context;
    return d->description[0] ? d->description : "NEC SX-Aurora TSUBASA Vector Engine";
}

void dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * d = (device *) dev->context;
    *free  = d->free_memory;
    *total = d->total_memory;
}

enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = dev_get_name(dev);
    props->description = dev_get_description(dev);
    props->type        = dev_get_type(dev);
    props->device_id   = nullptr;
    dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * d = (device *) dev->context;
    if (d == nullptr || !d->initialized) return nullptr;

    auto * ctx = new backend_context(d);
    d->ref_count++;

    auto * backend = new ggml_backend{
        /* .guid    = */ backend_guid(),
        /* .iface   = */ backend_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    return backend;
}

ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t dev) {
    auto * d = (device *) dev->context;
    return hbm_buffer_type(d->ve_device);
}

bool dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto * d = (device *) dev->context;
    return supports_op(d, op);
}

bool dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Multi-device isolation: each VE_HBM buffer type belongs to exactly one device.
    if (buft == nullptr) return false;
    auto * d = (device *) dev->context;
    if (d == nullptr) return false;

    const char * name = ggml_backend_buft_name(buft);
    if (name == nullptr) return false;

    if (std::strncmp(name, "VE", 2) == 0 && std::strstr(name, "_HBM") != nullptr) {
        int buft_dev = std::atoi(name + 2);  // "VE0_HBM" -> 0
        return buft_dev == d->ve_device;
    }

    // Claim CPU buffers too. Without this the ggml scheduler splits the
    // cgraph at every host-side tensor (token id, position, mask, ...)
    // and a Llama-3.2-3B decode arrives as ~56 fragments instead of one
    // big graph. Every op handler now routes CPU-side tensors through
    // backend_context::resolve_in/out (weights cached via
    // hbm_weight_cache, activations staged through temp HBM), so the
    // scheduler can safely place anything on us.
    //
    // Set GGML_VE_NO_CPU_BUFFERS=1 to revert to the old VE-HBM-only
    // behaviour (useful for bisecting a regression).
    if (std::getenv("GGML_VE_NO_CPU_BUFFERS") == nullptr &&
        std::strncmp(name, "CPU", 3) == 0) {
        return true;
    }
    return false;
}

bool dev_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    // Ops VE wants to claim from CPU memory even when the scheduler's first
    // choice would be CPU. MUL_MAT is the heaviest op; ADD is bandwidth-bound
    // but trivial.
    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_UNARY:
        case GGML_OP_RMS_NORM:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_GET_ROWS:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_SET_ROWS:
        case GGML_OP_ROPE:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ADD_ID:
        case GGML_OP_ARGSORT:
        case GGML_OP_SOFT_MAX:
            return true;
        default:
            return false;
    }
}

const ggml_backend_device_i device_iface = {
    /* .get_name             = */ dev_get_name,
    /* .get_description      = */ dev_get_description,
    /* .get_memory           = */ dev_get_memory,
    /* .get_type             = */ dev_get_type,
    /* .get_props            = */ dev_get_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ dev_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// ggml_backend_reg_i
//
const char * reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_VE_NAME;
}

void init_backend_objects_once() {
    if (g_backend_inited) return;
    init_devices_once();
    buffer_types_init();

    // Wire each VE_HBM buffer type to its owning ggml_backend_device.
    for (int i = 0; i < GGML_VE_MAX_DEVICES; ++i) {
        device * d = device_at(i);
        g_backend_devices[i] = ggml_backend_device{
            /* .iface   = */ device_iface,
            /* .reg     = */ &g_backend_reg,
            /* .context = */ d,
        };
        ggml_backend_buffer_type_t buft = hbm_buffer_type(i);
        if (buft) buft->device = &g_backend_devices[i];
    }
    g_backend_inited = true;
}

size_t reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    init_backend_objects_once();
    return (size_t) device_count();
}

ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    init_backend_objects_once();
    size_t visible = 0;
    for (int i = 0; i < GGML_VE_MAX_DEVICES; ++i) {
        device * d = device_at(i);
        if (d == nullptr || !d->initialized) continue;
        if (visible == index) return &g_backend_devices[i];
        ++visible;
    }
    return nullptr;
}

void * reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

const ggml_backend_reg_i reg_iface = {
    /* .get_name         = */ reg_get_name,
    /* .get_device_count = */ reg_get_device_count,
    /* .get_device       = */ reg_get_device,
    /* .get_proc_address = */ reg_get_proc_address,
};

} // namespace
} // namespace ggml_ve

//
// Public API
//
ggml_backend_buffer_type_t ggml_backend_ve_buffer_type(void) {
    ggml_ve::init_backend_objects_once();
    return ggml_ve::device_count() > 0 ? ggml_ve::hbm_buffer_type(0) : nullptr;
}

ggml_backend_reg_t ggml_backend_ve_reg(void) {
    ggml_ve::init_backend_objects_once();
    ggml_ve::g_backend_reg = ggml_backend_reg{
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_ve::reg_iface,
        /* .context     = */ nullptr,
    };
    return &ggml_ve::g_backend_reg;
}

bool ggml_backend_is_ve(ggml_backend_t backend) {
    return backend != nullptr &&
           ggml_guid_matches(backend->guid, ggml_ve::backend_guid());
}

int ggml_backend_ve_get_device_count(void) {
    ggml_ve::init_backend_objects_once();
    return ggml_ve::device_count();
}

void ggml_backend_ve_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_ve::init_backend_objects_once();
    *free  = 0;
    *total = 0;
    auto * d = ggml_ve::device_at(device);
    if (d == nullptr || !d->initialized) return;
    *free  = d->free_memory;
    *total = d->total_memory;
}

bool ggml_backend_ve_get_stats(ggml_backend_t backend, struct ggml_backend_ve_stats * out) {
    if (out == nullptr || !ggml_backend_is_ve(backend)) return false;
    auto * ctx = (ggml_ve::backend_context *) backend->context;
    if (ctx == nullptr) return false;
    int64_t cache_hits = 0, cache_misses = 0;
    ctx->cache().stats(/*allocated=*/nullptr, &cache_hits, &cache_misses);
    out->hbm_cache_hits   = cache_hits;
    out->hbm_cache_misses = cache_misses;
    out->syncs            = ctx->sync_count();
    out->ops_total        = ctx->ops_total();
    out->ops_mul_mat      = ctx->ops_mul_mat();
    out->ops_hbm          = ctx->ops_hbm();
    return true;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ve_reg)
