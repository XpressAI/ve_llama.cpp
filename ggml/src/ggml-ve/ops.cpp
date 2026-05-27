#include "ops.hpp"
#include <cstdio>
#include <cstdlib>

namespace ggml_ve {

// Set GGML_VE_DEBUG_DISPATCH=1 to log the first 200 op dispatches with
// tensor types and buffer types — handy when an op produces garbage in a
// real model but passes unit tests.
namespace {
int debug_dispatch_count = 0;
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
            return ops::rope_supports(op);
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
            return ops::mul_mat_supports(op) || ops::mul_mat_q_supports(op);
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

bool compute_forward(backend_context * ctx, ggml_tensor * node) {
    static bool debug = (std::getenv("GGML_VE_DEBUG_DISPATCH") != nullptr);
    if (debug && debug_dispatch_count < 200) {
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
