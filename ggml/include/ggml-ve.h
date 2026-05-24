#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_ve_buffer_type(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ve_reg(void);

GGML_BACKEND_API bool ggml_backend_is_ve(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_ve_get_device_count(void);
GGML_BACKEND_API void ggml_backend_ve_get_device_memory(int device, size_t * free, size_t * total);

// Per-backend-instance counters. Mostly exposed for testing — lets the
// VE-specific tests assert that the HBM weight cache is actually hitting
// (a regression would push it back to per-call uploads) and that the
// deferred-sync path is still batching (a regression would explode the
// sync count from O(graphs) to O(ops)).
struct ggml_backend_ve_stats {
    int64_t hbm_cache_hits;     // host-resident weight resolved from cache
    int64_t hbm_cache_misses;   // host-resident weight uploaded fresh
    int64_t syncs;              // vedaCtxSynchronize calls
    int64_t ops_total;          // ops dispatched to VE
    int64_t ops_mul_mat;        // of which MUL_MAT
    int64_t ops_hbm;            // of which used an all-HBM kernel
};

GGML_BACKEND_API bool ggml_backend_ve_get_stats(ggml_backend_t backend, struct ggml_backend_ve_stats * out);

#ifdef __cplusplus
}
#endif
