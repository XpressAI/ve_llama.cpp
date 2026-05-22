#ifndef GGML_VE_BUFFER_HPP
#define GGML_VE_BUFFER_HPP

// HBM buffer type. tensor->data is the (cast) VEDAdeviceptr, so kernels pass
// it via vedaArgsSetVPtr (see kb/veda-api/common-pitfalls.md #1).

#include "ggml-backend.h"

namespace ggml_ve {

// Initialise the static per-device buffer-type tables. Idempotent.
void buffer_types_init();

// Get the per-device VE_HBM buffer type ("VE0_HBM", "VE1_HBM", ...).
ggml_backend_buffer_type_t hbm_buffer_type(int device_id);

} // namespace ggml_ve

#endif // GGML_VE_BUFFER_HPP
