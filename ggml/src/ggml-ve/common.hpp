#ifndef GGML_VE_COMMON_HPP
#define GGML_VE_COMMON_HPP

// Shared building blocks for the NEC SX-Aurora TSUBASA Vector Engine backend:
// the VEDA error-check macro, the RAII context guard, and the HBM-format
// enum used by the weight cache. Everything else lives in a dedicated header.

#include "ggml-impl.h"

#include <veda.h>

#include <cstdio>

#define GGML_VE_NAME    "VE"
#define GGML_VE_VERSION 1

// Maximum number of VE devices the backend will enumerate.
#define GGML_VE_MAX_DEVICES 8

// Abort on any unexpected VEDA error. Use only in code paths that can't
// reasonably recover (registry setup, etc.); op handlers return bool.
#define VEDA_CHECK(stmt)                                                    \
    do {                                                                    \
        VEDAresult _err = (stmt);                                           \
        if (_err != VEDA_SUCCESS) {                                         \
            const char * _msg = nullptr;                                    \
            vedaGetErrorString(_err, &_msg);                                \
            GGML_LOG_ERROR("%s: VEDA error: %s\n", __func__,                \
                           _msg ? _msg : "unknown");                        \
            GGML_ABORT("VEDA error");                                       \
        }                                                                   \
    } while (0)

inline const char * ggml_ve_err_str(VEDAresult err) {
    const char * s = nullptr;
    vedaGetErrorString(err, &s);
    return s ? s : "unknown VEDA error";
}

inline bool ggml_ve_ok(VEDAresult err, const char * call) {
    if (err == VEDA_SUCCESS) return true;
    GGML_LOG_ERROR("ggml-ve: %s failed: %s\n", call, ggml_ve_err_str(err));
    return false;
}

// RAII guard that pushes a VEDA context on entry and pops it on exit.
// VEDA contexts are thread-local; every entry point into a kernel or alloc
// must have one active. See kb/veda-api/common-pitfalls.md (#5).
class VEDAContextGuard {
public:
    explicit VEDAContextGuard(VEDAcontext ctx) : valid_(false), prev_ctx_(nullptr) {
        if (ctx == nullptr) {
            return;
        }
        VEDAresult err = vedaCtxPushCurrent(ctx);
        if (err == VEDA_SUCCESS) {
            valid_ = true;
        } else {
            fprintf(stderr, "VEDAContextGuard: vedaCtxPushCurrent failed: %s\n",
                    ggml_ve_err_str(err));
        }
    }

    ~VEDAContextGuard() {
        if (valid_) {
            vedaCtxPopCurrent(&prev_ctx_);
        }
    }

    VEDAContextGuard(const VEDAContextGuard &) = delete;
    VEDAContextGuard & operator=(const VEDAContextGuard &) = delete;

    bool is_valid() const { return valid_; }

private:
    bool         valid_;
    VEDAcontext  prev_ctx_;
};

// In which layout / precision a weight tensor is cached on the VE side.
enum ggml_ve_hbm_format {
    GGML_VE_HBM_FP32          = 0,
    GGML_VE_HBM_BF16          = 1,
    GGML_VE_HBM_FP32_COLMAJOR = 2,
    GGML_VE_HBM_BF16_COLMAJOR = 3,
    GGML_VE_HBM_Q8_0          = 4,
};

#endif // GGML_VE_COMMON_HPP
