// Phase-5 GGML_OP_GET_ROWS — embedding-table lookup. The src0 (embeddings)
// is BF16 or F32 in HBM; src1 (indices) is i32 in HBM. We stage the indices
// to HMEM since the kernel reads them as a host pointer.
//
// The HBM→HBM kernel variant means we don't need an HMEM staging for the
// output — it writes directly to dst's HBM buffer.

#include "../backend_ctx.hpp"
#include "../device.hpp"
#include "../ops.hpp"

#include "ggml.h"

namespace ggml_ve {
namespace ops {

bool get_rows_supports(const ggml_tensor * op) {

    if (op->op != GGML_OP_GET_ROWS) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * src = op->src[0];  // embeddings
    const ggml_tensor * idx = op->src[1];  // i32 indices
    if (src == nullptr || idx == nullptr) return false;
    // F16 src (e.g. a tied/F16 token_embd) is accepted: the VE has no F16 path,
    // so it's converted to BF16 once at HBM upload and served by the BF16 kernel.
    if (src->type != GGML_TYPE_BF16 && src->type != GGML_TYPE_F32 &&
        src->type != GGML_TYPE_F16) return false;
    if (idx->type != GGML_TYPE_I32) return false;
    // Need contiguous indices and output.
    if (!ggml_is_contiguous(idx) || !ggml_is_contiguous(op)) return false;
    // 2D outputs only for now (cols=ne[0], rows=ne[1]).
    if (op->ne[2] != 1 || op->ne[3] != 1) return false;
    return true;
}

bool get_rows(backend_context * ctx, ggml_tensor * dst) {
    if (!get_rows_supports(dst)) return false;
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * idx = dst->src[1];

    // src (embedding table) is typically a CPU-side BF16 weight on first
    // call; resolve_in uploads it via hbm_weight_cache so subsequent calls
    // re-use the cached HBM ptr. F16 weights are converted to BF16 once on
    // upload (same byte size) and served by the BF16 kernel below.
    const VEDAdeviceptr src_hbm =
        (src->type == GGML_TYPE_F16)
            ? ctx->cache().get_or_upload_f16_as_bf16(src->name, src->data, ggml_nbytes(src),
                                                     tensor_is_hbm(src))
            : ctx->resolve_in(src);
    const VEDAdeviceptr dst_hbm = ctx->resolve_out(dst);
    if (src_hbm == 0 || dst_hbm == 0) return false;

    const uint64_t nc      = (uint64_t) dst->ne[0];      // cols / embedding dim
    const uint64_t nr      = (uint64_t) ggml_nelements(idx);  // n indices
    const uint64_t nb_src  = (uint64_t) src->nb[1];
    const uint64_t nb_dst  = (uint64_t) dst->nb[1];

    // Stage indices into a temp HBM (NOT HMEM -- see kb: HMEM is for
    // inter-VE / NEC MPI only, not single-VE VH↔VE staging). For HBM
    // sources, we do D→D; for host sources, H→D async; both freed after
    // the next sync.
    const size_t idx_bytes = nr * sizeof(int32_t);
    if (idx_bytes == 0) return true;   // zero rows -> no-op
    VEDAdeviceptr idx_tmp = 0;
    if (vedaMemAllocAsync(&idx_tmp, idx_bytes, 0) != VEDA_SUCCESS || idx_tmp == 0) {
        return false;
    }
    VEDAresult idx_err;
    if (tensor_is_hbm(idx)) {
        idx_err = vedaMemcpyDtoDAsync(idx_tmp, tensor_hbm_ptr(idx), idx_bytes, 0);
    } else if (idx->data != nullptr) {
        idx_err = vedaMemcpyHtoDAsync(idx_tmp, idx->data, idx_bytes, 0);
    } else {
        vedaMemFreeAsync(idx_tmp, 0);
        return false;
    }
    if (!ggml_ve_ok(idx_err, "vedaMemcpy* (get_rows idx)")) {
        vedaMemFreeAsync(idx_tmp, 0);
        return false;
    }
    ctx->enqueue_hbm_free(idx_tmp);

    kernel_id kid = (src->type == GGML_TYPE_BF16 || src->type == GGML_TYPE_F16)
        ? K_GET_ROWS_BF16_F32_HBM_HBM
        : K_GET_ROWS_F32_F32_HBM_HBM;
    VEDAfunction fn = ctx->fn(kid);
    if (fn == 0) return false;

    VEDAargs args = nullptr;
    if (!ggml_ve_ok(vedaArgsCreate(&args), "vedaArgsCreate(get_rows)")) return false;
    vedaArgsSetVPtr(args, 0, dst_hbm);
    vedaArgsSetVPtr(args, 1, src_hbm);
    vedaArgsSetVPtr(args, 2, idx_tmp);
    vedaArgsSetU64 (args, 3, nc);
    vedaArgsSetU64 (args, 4, nr);
    vedaArgsSetU64 (args, 5, nb_src);
    vedaArgsSetU64 (args, 6, nb_dst);

    if (!ggml_ve_ok(vedaLaunchKernelEx(fn, 0, args, /*destroyArgs=*/1, nullptr),
                    "vedaLaunchKernelEx(get_rows_hbm_hbm)")) {
        return false;
    }
    ctx->mark_sync_pending();
    return true;
}

}  // namespace ops
}  // namespace ggml_ve
