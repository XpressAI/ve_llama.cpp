#include "device.hpp"
#include "colmajor_cache.hpp"
#include "kv_shadow_cache.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifndef GGML_VE_SGEMV_PATH
#define GGML_VE_SGEMV_PATH "libve_sgemv.so"
#endif

#ifndef GGML_VE_KERNELS_PATH
#define GGML_VE_KERNELS_PATH "libve_kernels.so"
#endif

namespace ggml_ve {

namespace {

std::once_flag                                       g_init_once;
int                                                  g_device_count = 0;
bool                                                 g_veda_inited  = false;
std::array<device, GGML_VE_MAX_DEVICES>              g_devices;

// Map kernel_id -> (module, symbol name). The table is sized at compile time;
// adding a kernel is one line here plus its enum entry in device.hpp.
struct kernel_meta {
    const char *  symbol;
    kernel_module mod;
};

const kernel_meta & meta(kernel_id id) {
    static const kernel_meta table[K_COUNT] = {
        /* K_ADD_HBM_FULL                             */ { "ve_add_hbm_full",                          KMOD_SGEMV  },

        /* K_BF16_MATVEC_HMEM                         */ { "ve_bf16_matvec_hmem",                      KMOD_SGEMV  },
        /* K_BF16_MATVEC_OMP_HMEM                     */ { "ve_bf16_matvec_omp_hmem",                  KMOD_SGEMV  },
        /* K_BF16_MATVEC_HBM                          */ { "ve_bf16_matvec_hbm",                       KMOD_SGEMV  },
        /* K_BF16_MATVEC_HBM_OMP                      */ { "ve_bf16_matvec_hbm_omp",                   KMOD_SGEMV  },
        /* K_BF16_MATVEC_HBM_FULL                     */ { "ve_bf16_matvec_hbm_full",                  KMOD_SGEMV  },
        /* K_BF16_MATVEC_HBM_HMEM                     */ { "ve_bf16_matvec_hbm_hmem",                  KMOD_SGEMV  },
        /* K_BF16_MATMUL_HBM_FULL                     */ { "ve_bf16_matmul_hbm_full",                  KMOD_SGEMV  },
        /* K_BF16_SGEMM_COLMAJOR_HBM                  */ { "ve_bf16_sgemm_colmajor_hbm",               KMOD_SGEMV  },
        /* K_BF16_SGEMM_COLMAJOR_HBM_OMP              */ { "ve_bf16_sgemm_colmajor_hbm_omp",           KMOD_SGEMV  },
        /* K_BF16_SGEMM_BATCHED_CBLAS_HBM             */ { "ve_bf16_sgemm_batched_cblas_hbm",          KMOD_SGEMV  },
        /* K_BF16_TO_F32_COLMAJOR_HBM                 */ { "ve_bf16_to_f32_colmajor_hbm",              KMOD_SGEMV  },
        /* K_BF16_TO_F32_ROWMAJOR_HBM                 */ { "ve_bf16_to_f32_rowmajor_hbm",              KMOD_SGEMV  },

        /* K_F32_MATVEC_HMEM                          */ { "ve_f32_matvec_hmem",                       KMOD_SGEMV  },
        /* K_F32_MATVEC_HBM_OMP                       */ { "ve_f32_matvec_hbm_omp",                    KMOD_SGEMV  },
        /* K_F32_MATVEC_HBM_FULL                      */ { "ve_f32_matvec_hbm_full",                   KMOD_SGEMV  },
        /* K_F32_MATMUL_HBM_OMP                       */ { "ve_f32_matmul_hbm_omp",                    KMOD_SGEMV  },
        /* K_F32_MATMUL_HBM_OMP_V2                    */ { "ve_f32_matmul_hbm_omp_v2",                 KMOD_SGEMV  },
        /* K_F32_MATMUL_HMEM                          */ { "ve_f32_matmul_hmem",                       KMOD_SGEMV  },
        /* K_F32_SGEMM_BATCHED_CBLAS_HBM              */ { "ve_f32_sgemm_batched_cblas_hbm",           KMOD_SGEMV  },
        /* K_F32_SGEMM_BATCHED_CBLAS_HBM_NOTRANS      */ { "ve_f32_sgemm_batched_cblas_hbm_notrans",   KMOD_SGEMV  },

        /* K_CBLAS_SGEMV_HBM_HMEM                     */ { "ve_cblas_sgemv_hbm_hmem",                  KMOD_SGEMV  },
        /* K_CBLAS_SGEMM_HBM_HMEM                     */ { "ve_cblas_sgemm_hbm_hmem",                  KMOD_SGEMV  },
        /* K_CBLAS_SGEMV_COLMAJOR_HBM_HMEM            */ { "ve_cblas_sgemv_colmajor_hbm_hmem",         KMOD_SGEMV  },
        /* K_CBLAS_SGEMM_COLMAJOR_HBM_HMEM            */ { "ve_cblas_sgemm_colmajor_hbm_hmem",         KMOD_SGEMV  },

        /* K_Q2K_MATVEC_OMP_HMEM                      */ { "ve_q2k_matvec_f32_omp_hmem",               KMOD_SGEMV  },
        /* K_Q4K_MATVEC_OMP_HMEM                      */ { "ve_q4k_matvec_f32_omp_hmem",               KMOD_SGEMV  },
        /* K_Q5K_MATVEC_OMP_HMEM                      */ { "ve_q5k_matvec_f32_omp_hmem",               KMOD_SGEMV  },
        /* K_Q6K_MATVEC_OMP_HMEM                      */ { "ve_q6k_matvec_f32_omp_hmem",               KMOD_SGEMV  },
        /* K_Q8_0_FUSED_MATVEC_HBM                    */ { "ve_q8_0_fused_matvec_hbm",                 KMOD_SGEMV  },
        /* K_Q8_0_FUSED_MATVEC_HBM_FULL               */ { "ve_q8_0_fused_matvec_hbm_full",            KMOD_SGEMV  },
        /* K_Q8_0_FUSED_SGEMM_HBM                     */ { "ve_q8_0_fused_sgemm_hbm",                  KMOD_SGEMV  },
        /* K_Q2K_BF16_MATVEC_HBM                      */ { "ve_q2k_bf16_matvec_hbm",                   KMOD_SGEMV  },
        /* K_DEQUANT_Q2K_BF16                         */ { "ve_dequant_q2k_bf16",                      KMOD_SGEMV  },
        /* K_DEQUANT_Q8_0_BF16                        */ { "ve_dequant_q8_0_bf16",                     KMOD_SGEMV  },
        /* K_DEQUANT_MXFP4_BF16_HMEM                  */ { "ve_dequant_mxfp4_bf16_hmem",               KMOD_SGEMV  },

        /* K_FLASH_ATTN_F32_HBM                       */ { "ve_flash_attn_f32_hbm",                    KMOD_SGEMV  },
        /* K_FLASH_ATTN_BF16_INTRINSICS_HBM           */ { "ve_flash_attn_bf16_intrinsics_hbm",        KMOD_SGEMV  },
        /* K_FLASH_ATTN_EXT_F32Q_BF16KV_HBM           */ { "ve_flash_attn_ext_f32q_bf16kv_hbm",        KMOD_SGEMV  },
        /* K_FLASH_ATTN_EXT_F32Q_BF16KV_HBM_F32MASK   */ { "ve_flash_attn_ext_f32q_bf16kv_hbm_f32mask", KMOD_SGEMV },
        /* K_FLASH_ATTN_EXT_F32Q_BF16KV_HMEM          */ { "ve_flash_attn_ext_f32q_bf16kv_hmem",       KMOD_SGEMV  },
        /* K_FLASH_ATTN_EXT_F32Q_F16KV_HMEM           */ { "ve_flash_attn_ext_f32q_f16kv_hmem",        KMOD_SGEMV  },
        /* K_FLASH_ATTN_EXT_F32_F32MASK_HMEM          */ { "ve_flash_attn_ext_f32_f32mask_hmem",       KMOD_SGEMV  },
        /* K_FLASH_ATTN_EXT_F32_F16MASK_HMEM          */ { "ve_flash_attn_ext_f32_f16mask_hmem",       KMOD_SGEMV  },
        /* K_FLASH_ATTN_EXT_BF16_F16MASK_HMEM         */ { "ve_flash_attn_ext_bf16_f16mask_hmem",      KMOD_SGEMV  },

        /* K_RMS_NORM_HBM_SIMPLE                      */ { "ve_rms_norm_hbm_simple",                   KMOD_SGEMV  },
        /* K_RMS_NORM_HBM_OMP                         */ { "ve_rms_norm_hbm_omp",                      KMOD_SGEMV  },
        /* K_RMS_NORM_F32_HMEM                        */ { "ve_rms_norm_f32_hmem",                     KMOD_SGEMV  },
        /* K_RMS_NORM_OMP_HMEM                        */ { "ve_rms_norm_f32_omp_hmem",                 KMOD_SGEMV  },
        /* K_RMS_NORM_INPLACE_HMEM                    */ { "ve_rms_norm_f32_inplace_hmem",             KMOD_SGEMV  },

        /* K_ROPE_NORMAL_HBM_OMP                      */ { "ve_rope_normal_hbm_omp",                   KMOD_SGEMV  },
        /* K_ROPE_NEOX_HBM_OMP                        */ { "ve_rope_neox_hbm_omp",                     KMOD_SGEMV  },
        /* K_ROPE_IMROPE_HBM_OMP                      */ { "ve_rope_imrope_hbm_omp",                   KMOD_SGEMV  },
        /* K_ROPE_NORMAL_HBM_OMP_NOCACHE              */ { "ve_rope_normal_hbm_omp_nocache",           KMOD_SGEMV  },
        /* K_ROPE_NEOX_HBM_OMP_NOCACHE                */ { "ve_rope_neox_hbm_omp_nocache",             KMOD_SGEMV  },
        /* K_ROPE_NORMAL_HBM_SINGLE_POS               */ { "ve_rope_normal_hbm_single_pos",            KMOD_SGEMV  },
        /* K_ROPE_NEOX_HBM_SINGLE_POS                 */ { "ve_rope_neox_hbm_single_pos",              KMOD_SGEMV  },
        /* K_ROPE_IMROPE_HBM_SINGLE_POS               */ { "ve_rope_imrope_hbm_single_pos",            KMOD_SGEMV  },
        /* K_ROPE_NORMAL_F32_HMEM                     */ { "ve_rope_normal_f32_hmem",                  KMOD_SGEMV  },
        /* K_ROPE_NORMAL_F32_OMP_HMEM                 */ { "ve_rope_normal_f32_omp_hmem",              KMOD_SGEMV  },
        /* K_ROPE_NEOX_F32_HMEM                       */ { "ve_rope_neox_f32_hmem",                    KMOD_SGEMV  },
        /* K_ROPE_NEOX_F32_OMP_HMEM                   */ { "ve_rope_neox_f32_omp_hmem",                KMOD_SGEMV  },

        /* K_SOFT_MAX_F32_HMEM                        */ { "ve_soft_max_f32_hmem",                     KMOD_SGEMV  },
        /* K_SOFT_MAX_F32_OMP_HMEM                    */ { "ve_soft_max_f32_omp_hmem",                 KMOD_SGEMV  },
        /* K_SOFT_MAX_F32_MASKED_HMEM                 */ { "ve_soft_max_f32_masked_hmem",              KMOD_SGEMV  },
        /* K_SOFT_MAX_F32_MASKED_ATTN_HMEM            */ { "ve_soft_max_f32_masked_attn_hmem",         KMOD_SGEMV  },
        /* K_SOFT_MAX_F32_MASKED_ATTN_SINKS_HMEM      */ { "ve_soft_max_f32_masked_attn_sinks_hmem",   KMOD_SGEMV  },
        /* K_SOFT_MAX_F32_MASKED_FULL_HMEM            */ { "ve_soft_max_f32_masked_full_hmem",         KMOD_SGEMV  },

        /* K_GET_ROWS_F32_HMEM                        */ { "ve_get_rows_f32_hmem",                     KMOD_SGEMV  },
        /* K_GET_ROWS_BF16_F32_HMEM                   */ { "ve_get_rows_bf16_f32_hmem",                KMOD_SGEMV  },
        /* K_GET_ROWS_F16_F32_HMEM                    */ { "ve_get_rows_f16_f32_hmem",                 KMOD_SGEMV  },
        /* K_GET_ROWS_F32_HBM                         */ { "ve_get_rows_f32_hbm",                      KMOD_SGEMV  },
        /* K_GET_ROWS_BF16_F32_HBM                    */ { "ve_get_rows_bf16_f32_hbm",                 KMOD_SGEMV  },
        /* K_GET_ROWS_F32_F32_HBM_HBM                 */ { "ve_get_rows_f32_f32_hbm_hbm",              KMOD_SGEMV  },
        /* K_GET_ROWS_BF16_F32_HBM_HBM                */ { "ve_get_rows_bf16_f32_hbm_hbm",             KMOD_SGEMV  },

        /* K_SET_ROWS_F16_HBM                         */ { "ve_set_rows_f16_hbm",                      KMOD_SGEMV  },
        /* K_SET_ROWS_BF16_HBM                        */ { "ve_set_rows_bf16_hbm",                     KMOD_SGEMV  },
        /* K_SET_ROWS_F32_HBM                         */ { "ve_set_rows_f32_hbm",                      KMOD_SGEMV  },
        /* K_SET_ROWS_F16_HBM_FULL                    */ { "ve_set_rows_f16_hbm_full",                 KMOD_SGEMV  },
        /* K_SET_ROWS_BF16_HBM_FULL                   */ { "ve_set_rows_bf16_hbm_full",                KMOD_SGEMV  },
        /* K_SET_ROWS_F32_HBM_FULL                    */ { "ve_set_rows_f32_hbm_full",                 KMOD_SGEMV  },
        /* K_SET_ROW_F16_HBM_SINGLE                   */ { "ve_set_row_f16_hbm_single",                KMOD_SGEMV  },
        /* K_SET_ROW_BF16_HBM_SINGLE                  */ { "ve_set_row_bf16_hbm_single",               KMOD_SGEMV  },
        /* K_SET_ROW_F32_HBM_SINGLE                   */ { "ve_set_row_f32_hbm_single",                KMOD_SGEMV  },

        /* K_ADD_F32_HMEM                             */ { "ve_add_f32_hmem",                          KMOD_SGEMV  },
        /* K_ADD_F32_OMP_HMEM                         */ { "ve_add_f32_omp_hmem",                      KMOD_SGEMV  },
        /* K_ADD_HBM_FULL_BROADCAST                   */ { "ve_add_hbm_full_broadcast",                KMOD_SGEMV  },
        /* K_ADD_HBM_HMEM                             */ { "ve_add_hbm_hmem",                          KMOD_SGEMV  },
        /* K_MUL_F32_HMEM                             */ { "ve_mul_f32_hmem",                          KMOD_SGEMV  },
        /* K_MUL_F32_OMP_HMEM                         */ { "ve_mul_f32_omp_hmem",                      KMOD_SGEMV  },
        /* K_MUL_HBM_FULL                             */ { "ve_mul_hbm_full",                          KMOD_SGEMV  },
        /* K_MUL_HBM_FULL_BCAST                       */ { "ve_mul_hbm_full_bcast",                    KMOD_SGEMV  },
        /* K_MUL_HBM_HMEM                             */ { "ve_mul_hbm_hmem",                          KMOD_SGEMV  },
        /* K_SCALE_F32_HMEM                           */ { "ve_scale_f32_hmem",                        KMOD_SGEMV  },
        /* K_SCALE_HBM_FULL                           */ { "ve_scale_hbm_full",                        KMOD_SGEMV  },
        /* K_SILU_F32_HMEM                            */ { "ve_silu_f32_hmem",                         KMOD_SGEMV  },
        /* K_SILU_F32_OMP_HMEM                        */ { "ve_silu_f32_omp_hmem",                     KMOD_SGEMV  },
        /* K_SILU_HBM_FULL                            */ { "ve_silu_hbm_full",                         KMOD_SGEMV  },
        /* K_GELU_F32_HMEM                            */ { "ve_gelu_f32_hmem",                         KMOD_SGEMV  },
        /* K_GELU_F32_OMP_HMEM                        */ { "ve_gelu_f32_omp_hmem",                     KMOD_SGEMV  },
        /* K_SWIGLU_HBM_FULL                          */ { "ve_swiglu_hbm_full",                       KMOD_SGEMV  },
        /* K_SWIGLU_HBM_FULL_OMP                      */ { "ve_swiglu_hbm_full_omp",                   KMOD_SGEMV  },
        /* K_SWIGLU_OAI_HBM_FULL                      */ { "ve_swiglu_oai_f32_hbm_full",               KMOD_SGEMV  },

        /* K_CPY_F32_F32_HMEM                         */ { "ve_cpy_f32_f32_hmem",                      KMOD_SGEMV  },
        /* K_CPY_F32_F32_OMP_HMEM                     */ { "ve_cpy_f32_f32_omp_hmem",                  KMOD_SGEMV  },
        /* K_CPY_STRIDED_F32_F32_HMEM                 */ { "ve_cpy_strided_f32_f32_hmem",              KMOD_SGEMV  },
        /* K_CPY_STRIDED_F32_F32_HBM                  */ { "ve_cpy_strided_f32_f32_hbm",               KMOD_SGEMV  },
        /* K_CPY_BF16_F32_HMEM                        */ { "ve_cpy_bf16_f32_hmem",                     KMOD_SGEMV  },
        /* K_CPY_F32_BF16_HMEM                        */ { "ve_cpy_f32_bf16_hmem",                     KMOD_SGEMV  },
        /* K_CPY_F32_F16_HMEM                         */ { "ve_cpy_f32_f16_hmem",                      KMOD_SGEMV  },
        /* K_COPY_HBM_FULL                            */ { "ve_copy_hbm_full",                         KMOD_SGEMV  },

        /* K_DIAG_MASK_INF_F32_HMEM                   */ { "ve_diag_mask_inf_f32_hmem",                KMOD_SGEMV  },
        /* K_DIAG_MASK_ZERO_F32_HMEM                  */ { "ve_diag_mask_zero_f32_hmem",               KMOD_SGEMV  },

        /* K_ARGSORT_F32_HMEM                         */ { "ve_argsort_f32_hmem",                      KMOD_SGEMV  },
        /* K_ARGSORT_F32_OMP_HMEM                     */ { "ve_argsort_f32_omp_hmem",                  KMOD_SGEMV  },
        /* K_MUL_MAT_ID_BF16_F32_HMEM                 */ { "ve_mul_mat_id_bf16_f32_hmem",              KMOD_SGEMV  },
        /* K_MUL_MAT_ID_BF16_F32_HBM                  */ { "ve_mul_mat_id_bf16_f32_hbm",               KMOD_SGEMV  },
        /* K_MUL_MAT_ID_BF16_F32_HBM_FULL             */ { "ve_mul_mat_id_bf16_f32_hbm_full",          KMOD_SGEMV  },
        /* K_ADD_ID_F32_HMEM                          */ { "ve_add_id_f32_hmem",                       KMOD_SGEMV  },
        /* K_ADD_ID_F32_HBM_FULL                      */ { "ve_add_id_f32_hbm_full",                   KMOD_SGEMV  },

        /* K_FLASH_ATTN_EXT_F32Q_BF16KV_COLMAJOR_HBM  */ { "ve_flash_attn_ext_f32q_bf16kv_colmajor_hbm", KMOD_SGEMV },
        /* K_KVCACHE_MIRROR_TO_COLMAJOR_HBM           */ { "ve_kvcache_mirror_to_colmajor_hbm",        KMOD_SGEMV  },

        /* K_SSM_CONV_F32_HBM                         */ { "ve_ssm_conv_f32_hbm",                      KMOD_SGEMV  },
        /* K_COPY_STRIDED_F32_HBM                     */ { "ve_copy_strided_f32_hbm",                  KMOD_SGEMV  },
        /* K_COPY_BYTES_F32_HBM                       */ { "ve_copy_bytes_f32_hbm",                    KMOD_SGEMV  },
        /* K_GATED_DELTA_NET_F32_HBM                  */ { "ve_gated_delta_net_f32_hbm",               KMOD_SGEMV  },
        /* K_F32_TRUNCATE_TO_BF16_INPLACE             */ { "ve_f32_truncate_to_bf16_precision_inplace", KMOD_SGEMV },
        /* K_SIGMOID_HBM_FULL                         */ { "ve_sigmoid_hbm_full",                      KMOD_SGEMV  },
        /* K_SOFTPLUS_HBM_FULL                        */ { "ve_softplus_hbm_full",                     KMOD_SGEMV  },
        /* K_EXP_HBM_FULL                             */ { "ve_exp_hbm_full",                          KMOD_SGEMV  },
        /* K_NEG_HBM_FULL                             */ { "ve_neg_hbm_full",                          KMOD_SGEMV  },
        /* K_SQR_HBM_FULL                             */ { "ve_sqr_hbm_full",                          KMOD_SGEMV  },
        /* K_SUB_HBM_FULL                             */ { "ve_sub_hbm_full",                          KMOD_SGEMV  },
        /* K_L2_NORM_HBM_FULL                         */ { "ve_l2_norm_hbm_full",                      KMOD_SGEMV  },
    };
    return table[id];
}

void resolve_all_kernels(device * dev) {
    const bool debug = (std::getenv("GGML_VE_DEBUG_KERNELS") != nullptr);
    for (int i = 0; i < K_COUNT; ++i) {
        const kernel_meta & m = meta((kernel_id) i);
        VEDAmodule mod = (m.mod == KMOD_SGEMV) ? dev->module_sgemv : dev->module_kernels;
        if (mod == nullptr) {
            dev->fns[i] = 0;
            continue;
        }
        VEDAresult err = vedaModuleGetFunction(&dev->fns[i], mod, m.symbol);
        if (err != VEDA_SUCCESS) {
            dev->fns[i] = 0;
            if (debug) {
                GGML_LOG_WARN("ggml-ve: kernel '%s' not found in module: %s\n",
                              m.symbol, ggml_ve_err_str(err));
            }
        }
    }
}

bool load_modules(device * dev) {
    const char * sgemv_path   = std::getenv("VE_SGEMV_PATH");
    const char * kernels_path = std::getenv("VE_KERNELS_PATH");
    if (!sgemv_path)   sgemv_path   = GGML_VE_SGEMV_PATH;
    if (!kernels_path) kernels_path = GGML_VE_KERNELS_PATH;

    VEDAresult err = vedaModuleLoad(&dev->module_sgemv, sgemv_path);
    if (err != VEDA_SUCCESS) {
        GGML_LOG_ERROR("ggml-ve: failed to load %s: %s\n", sgemv_path, ggml_ve_err_str(err));
        dev->module_sgemv = nullptr;
        return false;
    }
    GGML_LOG_INFO("ggml-ve: loaded SGEMV kernels: %s\n", sgemv_path);

    err = vedaModuleLoad(&dev->module_kernels, kernels_path);
    if (err != VEDA_SUCCESS) {
        GGML_LOG_WARN("ggml-ve: optional %s not loaded: %s\n",
                      kernels_path, ggml_ve_err_str(err));
        dev->module_kernels = nullptr;
    } else {
        GGML_LOG_INFO("ggml-ve: loaded Q-kernels: %s\n", kernels_path);
    }

    resolve_all_kernels(dev);

    // The Phase 1 sanity-check kernel is required.
    if (dev->fn(K_ADD_HBM_FULL) == 0) {
        GGML_LOG_ERROR("ggml-ve: required symbol ve_add_hbm_full not found in %s\n", sgemv_path);
        return false;
    }
    return true;
}

void init_one_device(int i) {
    device & dev = g_devices[i];
    dev.ve_device = i;

    if (!ggml_ve_ok(vedaDeviceGet(&dev.handle, i), "vedaDeviceGet")) return;

    // Short name: "VE0", "VE1", ... (used by ggml's backend filtering).
    std::snprintf(dev.name, sizeof(dev.name), "VE%d", i);

    // Long description: the VEDA-reported model name, e.g. "NEC SX-Aurora
    // Tsubasa VE20B". Falls back to a generic string if vedaDeviceGetName fails.
    char model[128] = {};
    if (vedaDeviceGetName(model, sizeof(model), dev.handle) == VEDA_SUCCESS && model[0]) {
        std::snprintf(dev.description, sizeof(dev.description), "%s", model);
    } else {
        std::snprintf(dev.description, sizeof(dev.description),
                      "NEC SX-Aurora TSUBASA Vector Engine");
    }

    size_t total_mem = 0;
    vedaDeviceTotalMem(&total_mem, i);
    dev.total_memory = total_mem;
    dev.free_memory  = total_mem;

    // Primary context: much faster than vedaCtxCreate (~8x kernel launch perf).
    if (!ggml_ve_ok(vedaDevicePrimaryCtxRetain(&dev.context, dev.handle),
                    "vedaDevicePrimaryCtxRetain")) {
        return;
    }

    {
        VEDAContextGuard guard(dev.context);
        if (!guard.is_valid()) return;
        if (!load_modules(&dev)) return;
    }

    GGML_LOG_INFO("ggml-ve: VE%d: %s (%.2f GiB HBM)\n",
                  i, dev.name, total_mem / (1024.0 * 1024.0 * 1024.0));
    dev.colmajor = new colmajor_weight_cache();
    dev.colmajor->set_context(dev.context);
    dev.kv_shadow = new kv_shadow_cache();
    dev.kv_shadow->set_context(dev.context);
    dev.initialized = true;
}

void init_impl() {
    if (g_veda_inited) return;

    VEDAresult err = vedaInit(0);
    if (err != VEDA_SUCCESS) {
        GGML_LOG_WARN("ggml-ve: vedaInit failed: %s\n", ggml_ve_err_str(err));
        return;
    }
    g_veda_inited = true;

    int count = 0;
    err = vedaDeviceGetCount(&count);
    if (err != VEDA_SUCCESS) {
        GGML_LOG_WARN("ggml-ve: vedaDeviceGetCount failed: %s\n", ggml_ve_err_str(err));
        return;
    }
    g_device_count = count > GGML_VE_MAX_DEVICES ? GGML_VE_MAX_DEVICES : count;

    for (int i = 0; i < g_device_count; ++i) {
        init_one_device(i);
    }
}

} // namespace

void init_devices_once() {
    std::call_once(g_init_once, init_impl);
}

int device_count() {
    int n = 0;
    for (int i = 0; i < g_device_count; ++i) {
        if (g_devices[i].initialized) n++;
    }
    return n;
}

device * device_at(int index) {
    if (index < 0 || index >= g_device_count) return nullptr;
    return &g_devices[index];
}

const char * kernel_symbol(kernel_id id) {
    return ((unsigned) id < K_COUNT) ? meta(id).symbol : "(unknown)";
}

kernel_module kernel_owner(kernel_id id) {
    return ((unsigned) id < K_COUNT) ? meta(id).mod : KMOD_SGEMV;
}

} // namespace ggml_ve
