// Phase-5 GGML_OP_SET_ROWS — KV-cache writes. dst is the cache tensor in HBM
// (BF16, F16, or F32), src is the freshly computed value to write (F32).
// Indices come via an HBM tensor, staged through HMEM by the pool.
//
// ggml's SET_ROWS semantics:
//   dst[idx[i]] = src[i]
// The all-HBM kernels also do the precision conversion F32 -> {F16, BF16, F32}.
//
// IMPORTANT: ggml's SET_ROWS uses int64 indices, but the VE kernel reads
// indices as int32*. We narrow i64 -> i32 in the staging copy. Indices for
// KV caches fit comfortably in 32 bits (context length << 2^31). For i32
// inputs we pass through directly.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../kv_shadow_cache.hpp"
#include "../ops.hpp"

#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ggml_ve {
namespace ops {

bool set_rows_supports(const ggml_tensor * op) {
    if (op->op != GGML_OP_SET_ROWS) return false;
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * idx = op->src[1];
    if (src == nullptr || idx == nullptr) return false;
    if (src->type != GGML_TYPE_F32) return false;
    if (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16 && op->type != GGML_TYPE_F32) {
        return false;
    }
    if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) return false;
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(idx)) return false;
    if (op->ne[2] != 1 || op->ne[3] != 1) return false;
    return true;
}

bool set_rows(backend_context * ctx, ggml_tensor * dst) {
    if (!set_rows_supports(dst)) return false;
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * idx = dst->src[1];

    const VEDAdeviceptr src_hbm = ctx->resolve_in(src);
    const VEDAdeviceptr dst_hbm = ctx->resolve_out(dst);
    if (src_hbm == 0 || dst_hbm == 0) return false;

    const uint64_t nc     = (uint64_t) src->ne[0];
    const uint64_t nr     = (uint64_t) ggml_nelements(idx);
    const uint64_t nb_dst = (uint64_t) dst->nb[1];
    const uint64_t nb_src = (uint64_t) src->nb[1];

    // Stage indices into temp HBM (NOT HMEM; HMEM is for inter-VE/MPI).
    // i64 → i32 narrowing happens on host before the upload.
    const size_t idx32_bytes = nr * sizeof(int32_t);
    if (idx32_bytes == 0) return true;  // zero rows -> no-op
    VEDAdeviceptr idx_tmp = 0;
    if (vedaMemAllocAsync(&idx_tmp, idx32_bytes, 0) != VEDA_SUCCESS || idx_tmp == 0) return false;

    if (idx->type == GGML_TYPE_I32) {
        VEDAresult err;
        if (tensor_is_hbm(idx)) {
            err = vedaMemcpyDtoDAsync(idx_tmp, tensor_hbm_ptr(idx), idx32_bytes, 0);
        } else if (idx->data != nullptr) {
            err = vedaMemcpyHtoDAsync(idx_tmp, idx->data, idx32_bytes, 0);
        } else {
            vedaMemFreeAsync(idx_tmp, 0);
            return false;
        }
        if (!ggml_ve_ok(err, "vedaMemcpy* (set_rows i32 idx)")) {
            vedaMemFreeAsync(idx_tmp, 0);
            return false;
        }
    } else {  // GGML_TYPE_I64 — narrow on host.
        std::vector<int64_t> host_i64(nr);
        bool i64_ok = false;
        if (tensor_is_hbm(idx)) {
            i64_ok = ggml_ve_ok(vedaMemcpyDtoH(host_i64.data(),
                                                tensor_hbm_ptr(idx),
                                                nr * sizeof(int64_t)),
                                "vedaMemcpyDtoH (set_rows i64 idx)");
        } else if (idx->data != nullptr) {
            std::memcpy(host_i64.data(), idx->data, nr * sizeof(int64_t));
            i64_ok = true;
        }
        if (!i64_ok) { vedaMemFreeAsync(idx_tmp, 0); return false; }
        std::vector<int32_t> host_i32(nr);
        for (uint64_t i = 0; i < nr; ++i) host_i32[i] = (int32_t) host_i64[i];
        if (!ggml_ve_ok(vedaMemcpyHtoDAsync(idx_tmp, host_i32.data(), idx32_bytes, 0),
                        "vedaMemcpyHtoDAsync (set_rows i32 narrowed idx)")) {
            vedaMemFreeAsync(idx_tmp, 0);
            return false;
        }
        // host_i32 destructor must outlive the async HtoD -- the i64
        // narrowing is short-lived so we explicitly sync here. This is
        // a slow path (i64 indices are rare); the common i32 path above
        // stays fully async.
        vedaCtxSynchronize();
    }
    ctx->enqueue_hbm_free(idx_tmp);

    kernel_id kid;
    switch (dst->type) {
        case GGML_TYPE_F16:  kid = K_SET_ROWS_F16_HBM_FULL;  break;
        case GGML_TYPE_BF16: kid = K_SET_ROWS_BF16_HBM_FULL; break;
        case GGML_TYPE_F32:  kid = K_SET_ROWS_F32_HBM_FULL;  break;
        default: return false;
    }
    VEDAfunction fn = ctx->fn(kid);
    if (fn == 0) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(set_rows)")) return false;
    vedaArgsSetVPtr(args, 0, dst_hbm);
    vedaArgsSetVPtr(args, 1, src_hbm);
    vedaArgsSetVPtr(args, 2, idx_tmp);
    vedaArgsSetU64 (args, 3, nc);
    vedaArgsSetU64 (args, 4, nr);
    vedaArgsSetU64 (args, 5, nb_dst);
    vedaArgsSetU64 (args, 6, nb_src);

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(set_rows_hbm_full)")) {
        return false;
    }
    ctx->mark_sync_pending();

    /* ------------------------------------------------------------------ *
     * Stage-3 column-major KV shadow: after a successful SET_ROWS to a
     * KV cache tensor, mirror the same row(s) into our col-major shadow
     * so FA can read seq-unit-stride.
     *
     * Mirror policy is "contiguous-from-watermark". We mirror a SET_ROWS
     * call when the indices form a run [watermark, watermark+nr). This
     * matches both the prompt-prefill pattern (one big SET_ROWS with
     * indices [0..nr)) and the decode pattern (per-token SET_ROWS with
     * the next sequential index). Anything else (gap, slot reuse, cache
     * shift) invalidates the shadow and FA falls back to row-major.
     * ------------------------------------------------------------------ */
    static const bool shadow_enabled =
        std::getenv("GGML_VE_NO_KV_SHADOW") == nullptr;
    if (shadow_enabled
        && dst->type == GGML_TYPE_BF16
        && dst->name && dst->name[0]
        && (std::strncmp(dst->name, "cache_k", 7) == 0
            || std::strncmp(dst->name, "cache_v", 7) == 0)
        && ctx->dev() && ctx->dev()->kv_shadow
        && ctx->fn(K_KVCACHE_MIRROR_TO_COLMAJOR_HBM) != 0) {

        /* Walk the view_src chain to the bottom-most tensor for a stable
         * shadow key. Permuted KV views have view-of-view ancestry. */
        const ggml_tensor * dst_canon = dst;
        while (dst_canon->view_src) dst_canon = dst_canon->view_src;
        VEDAdeviceptr dst_canon_hbm = tensor_is_hbm(dst_canon)
                                          ? tensor_hbm_ptr(dst_canon) : 0;
        const int64_t channels = (int64_t) nc;
        const int64_t seq_max  = (int64_t) dst_canon->ne[1];

        /* Read indices host-side. The i32 path stages from host or HBM
         * above (we have a copy in idx_hmem) but the source-of-truth for
         * the contiguity check is whatever's still on host. For an HBM-
         * resident index we'd need a dtoh — skip the mirror in that case
         * rather than block on the copy. */
        const int32_t * idx_host_i32 = nullptr;
        const int64_t * idx_host_i64 = nullptr;
        if (idx->type == GGML_TYPE_I32 && idx->data) {
            idx_host_i32 = (const int32_t *) idx->data;
        } else if (idx->type == GGML_TYPE_I64 && idx->data) {
            idx_host_i64 = (const int64_t *) idx->data;
        }

        if (dst_canon_hbm != 0
            && (idx_host_i32 != nullptr || idx_host_i64 != nullptr)) {

            auto * sh = ctx->dev()->kv_shadow->get_or_create(
                dst_canon_hbm, channels, seq_max);

            if (sh != nullptr) {
                int64_t row_start = idx_host_i32
                    ? (int64_t) idx_host_i32[0]
                    : idx_host_i64[0];

                /* Contiguity check: idx[i] == row_start + i for all i;
                 * (row_start == watermark) OR (row_start == 0, fresh start);
                 * whole run in [0, seq_max). */
                bool contiguous_from_watermark =
                    (row_start == sh->watermark || row_start == 0)
                    && (row_start >= 0)
                    && (row_start + (int64_t) nr <= seq_max);

                for (uint64_t i = 1; contiguous_from_watermark && i < nr; ++i) {
                    int64_t want = row_start + (int64_t) i;
                    int64_t got  = idx_host_i32
                        ? (int64_t) idx_host_i32[i]
                        : idx_host_i64[i];
                    if (got != want) contiguous_from_watermark = false;
                }

                if (contiguous_from_watermark) {
                    VEDAargs margs = nullptr;
                    if (ggml_ve_ok(vedaArgsCreate(&margs),
                                   "vedaArgsCreate(kv_mirror)")) {
                        vedaArgsSetVPtr(margs, 0, src_hbm);
                        vedaArgsSetVPtr(margs, 1, sh->shadow_hbm);
                        vedaArgsSetU64 (margs, 2, (uint64_t) row_start);
                        vedaArgsSetU64 (margs, 3, (uint64_t) nr);
                        vedaArgsSetU64 (margs, 4, (uint64_t) channels);
                        vedaArgsSetU64 (margs, 5, (uint64_t) seq_max);
                        uint64_t mrc = 0;
                        if (ggml_ve_ok(vedaLaunchKernelEx(
                                           ctx->fn(K_KVCACHE_MIRROR_TO_COLMAJOR_HBM),
                                           0, margs, /*destroyArgs=*/1, &mrc),
                                       "vedaLaunchKernelEx(kv_mirror)")) {
                            ctx->dev()->kv_shadow->note_rows_mirrored(
                                sh, row_start, (int64_t) nr);
                        }
                    }
                } else {
                    /* Non-monotonic write — invalidate so the watermark
                     * doesn't pretend rows are mirrored when they aren't. */
                    ctx->dev()->kv_shadow->invalidate(sh);
                }
            }
        }
    }

    return true;
}

}  // namespace ops
}  // namespace ggml_ve
