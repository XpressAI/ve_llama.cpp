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

#ifdef __cplusplus
}
#endif
