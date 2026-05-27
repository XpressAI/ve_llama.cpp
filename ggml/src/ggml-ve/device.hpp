#ifndef GGML_VE_DEVICE_HPP
#define GGML_VE_DEVICE_HPP

// Per-device state. One ggml_ve_device instance per physical VE card. The
// device owns the VEDA context, the loaded kernel modules, and the table of
// VEDAfunction handles indexed by enum. The table layout is fixed (size
// known at compile time) but mostly populated lazily — handles for kernels
// that aren't shipped in the .so default to 0 and the op handlers check.

#include "common.hpp"

namespace ggml_ve {

// Kernel identifiers. Add new entries here as ops land in later phases.
// Keep groups contiguous and grouped by feature so it's easy to spot
// what's missing for a given op.
enum kernel_id {
    // ---- Phase 1: sanity check ----
    K_ADD_HBM_FULL,

    // ---- Phase 2: MUL_MAT ----
    K_BF16_MATVEC_HMEM,
    K_BF16_MATVEC_OMP_HMEM,
    K_BF16_MATVEC_HBM,
    K_BF16_MATVEC_HBM_OMP,
    K_BF16_MATVEC_HBM_FULL,
    K_BF16_MATVEC_HBM_HMEM,
    K_BF16_MATMUL_HBM_FULL,
    K_BF16_SGEMM_COLMAJOR_HBM,
    K_BF16_SGEMM_COLMAJOR_HBM_OMP,
    K_BF16_SGEMM_BATCHED_CBLAS_HBM,
    K_BF16_TO_F32_COLMAJOR_HBM,
    K_BF16_TO_F32_ROWMAJOR_HBM,

    K_F32_MATVEC_HMEM,
    K_F32_MATVEC_HBM_OMP,
    K_F32_MATVEC_HBM_FULL,
    K_F32_MATMUL_HBM_OMP,
    K_F32_MATMUL_HBM_OMP_V2,
    K_F32_MATMUL_HMEM,
    K_F32_SGEMM_BATCHED_CBLAS_HBM,         // row-major CBLAS
    K_F32_SGEMM_BATCHED_CBLAS_HBM_NOTRANS, // column-major CBLAS (fast prompt eval)

    K_CBLAS_SGEMV_HBM_HMEM,
    K_CBLAS_SGEMM_HBM_HMEM,
    K_CBLAS_SGEMV_COLMAJOR_HBM_HMEM,
    K_CBLAS_SGEMM_COLMAJOR_HBM_HMEM,

    // ---- Phase 3: quantized ----
    K_Q2K_MATVEC_OMP_HMEM,
    K_Q4K_MATVEC_OMP_HMEM,
    K_Q5K_MATVEC_OMP_HMEM,
    K_Q6K_MATVEC_OMP_HMEM,
    K_Q8_0_FUSED_MATVEC_HBM,
    K_Q8_0_FUSED_MATVEC_HBM_FULL,
    K_Q8_0_FUSED_SGEMM_HBM,
    K_Q2K_BF16_MATVEC_HBM,
    K_DEQUANT_Q2K_BF16,
    K_DEQUANT_Q8_0_BF16,
    K_DEQUANT_MXFP4_BF16_HMEM,

    // ---- Phase 4: flash attention ----
    K_FLASH_ATTN_F32_HBM,
    K_FLASH_ATTN_BF16_INTRINSICS_HBM,
    K_FLASH_ATTN_EXT_F32Q_BF16KV_HBM,
    K_FLASH_ATTN_EXT_F32Q_BF16KV_HBM_F32MASK,
    K_FLASH_ATTN_EXT_F32Q_BF16KV_HMEM,
    K_FLASH_ATTN_EXT_F32Q_F16KV_HMEM,
    K_FLASH_ATTN_EXT_F32_F32MASK_HMEM,
    K_FLASH_ATTN_EXT_F32_F16MASK_HMEM,
    K_FLASH_ATTN_EXT_BF16_F16MASK_HMEM,

    // ---- Phase 5: norm / rope / softmax / get_rows / set_rows / elementwise ----
    K_RMS_NORM_HBM_SIMPLE,
    K_RMS_NORM_HBM_OMP,
    K_RMS_NORM_F32_HMEM,
    K_RMS_NORM_OMP_HMEM,
    K_RMS_NORM_INPLACE_HMEM,

    K_ROPE_NORMAL_HBM_OMP,
    K_ROPE_NEOX_HBM_OMP,
    K_ROPE_IMROPE_HBM_OMP,
    K_ROPE_NORMAL_HBM_OMP_NOCACHE,
    K_ROPE_NEOX_HBM_OMP_NOCACHE,
    K_ROPE_NORMAL_HBM_SINGLE_POS,
    K_ROPE_NEOX_HBM_SINGLE_POS,
    K_ROPE_IMROPE_HBM_SINGLE_POS,
    K_ROPE_NORMAL_F32_HMEM,
    K_ROPE_NORMAL_F32_OMP_HMEM,
    K_ROPE_NEOX_F32_HMEM,
    K_ROPE_NEOX_F32_OMP_HMEM,

    K_SOFT_MAX_F32_HMEM,
    K_SOFT_MAX_F32_OMP_HMEM,
    K_SOFT_MAX_F32_MASKED_HMEM,
    K_SOFT_MAX_F32_MASKED_ATTN_HMEM,
    K_SOFT_MAX_F32_MASKED_ATTN_SINKS_HMEM,
    K_SOFT_MAX_F32_MASKED_FULL_HMEM,

    K_GET_ROWS_F32_HMEM,
    K_GET_ROWS_BF16_F32_HMEM,
    K_GET_ROWS_F16_F32_HMEM,
    K_GET_ROWS_F32_HBM,
    K_GET_ROWS_BF16_F32_HBM,
    K_GET_ROWS_F32_F32_HBM_HBM,
    K_GET_ROWS_BF16_F32_HBM_HBM,

    K_SET_ROWS_F16_HBM,
    K_SET_ROWS_BF16_HBM,
    K_SET_ROWS_F32_HBM,
    K_SET_ROWS_F16_HBM_FULL,
    K_SET_ROWS_BF16_HBM_FULL,
    K_SET_ROWS_F32_HBM_FULL,
    K_SET_ROW_F16_HBM_SINGLE,
    K_SET_ROW_BF16_HBM_SINGLE,
    K_SET_ROW_F32_HBM_SINGLE,

    K_ADD_F32_HMEM,
    K_ADD_F32_OMP_HMEM,
    K_ADD_HBM_FULL_BROADCAST,
    K_ADD_HBM_HMEM,
    K_MUL_F32_HMEM,
    K_MUL_F32_OMP_HMEM,
    K_MUL_HBM_FULL,
    K_MUL_HBM_FULL_BCAST,
    K_MUL_HBM_HMEM,
    K_SCALE_F32_HMEM,
    K_SCALE_HBM_FULL,
    K_SILU_F32_HMEM,
    K_SILU_F32_OMP_HMEM,
    K_SILU_HBM_FULL,
    K_GELU_F32_HMEM,
    K_GELU_F32_OMP_HMEM,
    K_SWIGLU_HBM_FULL,
    K_SWIGLU_HBM_FULL_OMP,
    K_SWIGLU_OAI_HBM_FULL,

    K_CPY_F32_F32_HMEM,
    K_CPY_F32_F32_OMP_HMEM,
    K_CPY_STRIDED_F32_F32_HMEM,
    K_CPY_STRIDED_F32_F32_HBM,
    K_CPY_BF16_F32_HMEM,
    K_CPY_F32_BF16_HMEM,
    K_CPY_F32_F16_HMEM,
    K_COPY_HBM_FULL,

    K_DIAG_MASK_INF_F32_HMEM,
    K_DIAG_MASK_ZERO_F32_HMEM,

    // ---- Phase 6: MoE ----
    K_ARGSORT_F32_HMEM,
    K_ARGSORT_F32_OMP_HMEM,
    K_MUL_MAT_ID_BF16_F32_HMEM,
    K_MUL_MAT_ID_BF16_F32_HBM,
    K_MUL_MAT_ID_BF16_F32_HBM_FULL,
    K_ADD_ID_F32_HMEM,
    K_ADD_ID_F32_HBM_FULL,

    /* Column-major KV cache fast path (Stage 3). FA reads from a
     * persistent BF16 col-major shadow we maintain alongside the
     * standard cache; mirror kernel writes each new row to that shadow
     * after every SET_ROWS. Append-only at the end of the enum so old
     * cached .so files keep loading correctly. */
    K_FLASH_ATTN_EXT_F32Q_BF16KV_COLMAJOR_HBM,
    K_KVCACHE_MIRROR_TO_COLMAJOR_HBM,

    /* Recurrent-layer ops for Qwen3.5 / Qwen3.6 (hybrid SSM+attention).
     * SSM_CONV is the 1D causal conv that feeds the gated delta net;
     * GATED_DELTA_NET is the fused linear-attention + state update.
     * Both pure-F32, fully HBM-resident. */
    K_SSM_CONV_F32_HBM,
    K_COPY_STRIDED_F32_HBM,
    K_COPY_BYTES_F32_HBM,
    K_GATED_DELTA_NET_F32_HBM,
    K_F32_TRUNCATE_TO_BF16_INPLACE,

    /* Element-wise / unary additions (Qwen3.5 GDN block). All pure F32,
     * fully HBM-resident. Append-only to keep cached .so files compatible. */
    K_SIGMOID_HBM_FULL,
    K_SOFTPLUS_HBM_FULL,
    K_EXP_HBM_FULL,
    K_NEG_HBM_FULL,
    K_SQR_HBM_FULL,
    K_SUB_HBM_FULL,
    K_L2_NORM_HBM_FULL,

    K_SUM_ROWS_F32_HBM,
    K_REPEAT_F32_HBM,
    K_CONCAT_F32_HBM,
    K_ROPE_MROPE_F32_HBM,
    K_RMS_NORM_STRIDED_HBM,
    K_CPY_F32_TO_F16_HBM,

    K_COUNT,
};

// Symbol name in libve_*.so for each kernel_id. Defined once, in device.cpp.
const char * kernel_symbol(kernel_id id);

// Which module a kernel lives in.
enum kernel_module {
    KMOD_SGEMV,    // libve_sgemv.so (main library)
    KMOD_KERNELS,  // libve_kernels.so (K-quant kernels)
};

// Module that owns each kernel, also defined in device.cpp.
kernel_module kernel_owner(kernel_id id);

// Forward decl: defined in colmajor_cache.hpp
class colmajor_weight_cache;
// Forward decl: defined in kv_shadow_cache.hpp
class kv_shadow_cache;

struct device {
    int          ve_device   = 0;     // VE device ID (0..3 on a 4-card system)
    int          ref_count   = 0;     // backend instances using this device

    VEDAdevice   handle      = 0;
    VEDAcontext  context     = nullptr;

    VEDAmodule   module_sgemv   = nullptr;
    VEDAmodule   module_kernels = nullptr;

    VEDAfunction fns[K_COUNT] = {};   // kernel handle table; 0 if unavailable

    size_t  total_memory = 0;
    size_t  free_memory  = 0;
    char    name[128]    = {};
    char    description[128] = {};

    bool    initialized  = false;

    // Per-device F32 column-major weight cache. Shared across all backend
    // instances on this device so the on-VE transpose only runs once per
    // weight tensor for the entire process lifetime. Lazily heap-allocated
    // in init_devices_once() to keep this header free of the cache impl.
    colmajor_weight_cache * colmajor = nullptr;

    // Column-major shadow of the BF16 KV cache (Stage 3 colmajor FA).
    // One shadow per cache_k_lN / cache_v_lN tensor, keyed by HBM addr.
    // Populated lazily on first SET_ROWS mirror; FA reads it when the
    // watermark covers the requested seq_len.
    kv_shadow_cache *       kv_shadow = nullptr;

    // Convenience accessor with bounds check.
    VEDAfunction fn(kernel_id id) const {
        return (unsigned) id < K_COUNT ? fns[id] : 0;
    }
};

// Lazy global init. Idempotent. Reads `VE_SGEMV_PATH` / `VE_KERNELS_PATH`
// env vars to override the compile-time defaults.
void init_devices_once();

int            device_count();   // number of *initialized* devices
device *       device_at(int index);

} // namespace ggml_ve

#endif // GGML_VE_DEVICE_HPP
