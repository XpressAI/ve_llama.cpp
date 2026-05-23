find_path(VEDA_INCLUDE_DIR
    NAMES veda.h
    PATHS /opt/nec/ve/share/veda /opt/nec/ve /opt/nec/ve/share/veoffload-veda
    PATH_SUFFIXES include share/veda/include share/veoffload-veda/include
)

find_library(VEDA_LIBRARY
    NAMES veda
    PATHS /opt/nec/ve/veos/lib64 /opt/nec/ve/lib64
)

if (NOT VEDA_INCLUDE_DIR)
    message(FATAL_ERROR "VE backend requires VEDA headers")
endif()

if (NOT VEDA_LIBRARY)
    message(FATAL_ERROR "VE backend requires libveda")
endif()

# External VE kernel libraries (built out-of-tree from ~/claude_workspace/ggml-ve-veda/)
#   libve_sgemv.so   - main kernel library (BF16/F32 matvec, attention, RoPE, RMS, ...)
#   libve_kernels.so - secondary library (K-quant kernels)
# Override at configure time with -DGGML_VE_VEDA_KERNELS_DIR=...
# or at runtime with the VE_SGEMV_PATH / VE_KERNELS_PATH env vars.
if (NOT GGML_VE_VEDA_KERNELS_DIR)
    set(GGML_VE_VEDA_KERNELS_DIR "$ENV{HOME}/claude_workspace/ggml-ve-veda" CACHE PATH
        "Directory containing libve_sgemv.so / libve_kernels.so")
endif()

set(GGML_VE_SGEMV_SO   "${GGML_VE_VEDA_KERNELS_DIR}/libve_sgemv.so")
set(GGML_VE_KERNELS_SO "${GGML_VE_VEDA_KERNELS_DIR}/libve_kernels.so")

if (NOT EXISTS "${GGML_VE_SGEMV_SO}")
    message(WARNING "VE backend: libve_sgemv.so not found at ${GGML_VE_SGEMV_SO}. "
                    "Build it from ${GGML_VE_VEDA_KERNELS_DIR} or set VE_SGEMV_PATH at runtime.")
endif()

ggml_add_backend_library(ggml-ve
    ggml-ve.cpp
    backend_ctx.cpp
    device.cpp
    buffer.cpp
    graph_compiler.cpp
    ops.cpp
    ops/add.cpp
    ops/cpy.cpp
    ops/elementwise.cpp
    ops/flash_attn.cpp
    ops/get_rows.cpp
    ops/moe.cpp
    ops/mul_mat.cpp
    ops/mul_mat_q.cpp
    ops/rms_norm.cpp
    ops/rope.cpp
    ops/set_rows.cpp
    ops/softmax.cpp
)

target_include_directories(ggml-ve PRIVATE ${VEDA_INCLUDE_DIR})
target_link_libraries(ggml-ve PRIVATE ${VEDA_LIBRARY})
target_compile_definitions(ggml-ve PRIVATE
    GGML_USE_VE
    GGML_VE_SGEMV_PATH="${GGML_VE_SGEMV_SO}"
    GGML_VE_KERNELS_PATH="${GGML_VE_KERNELS_SO}"
)

message(STATUS "VE backend: VEDA include dir: ${VEDA_INCLUDE_DIR}")
message(STATUS "VE backend: VEDA library:     ${VEDA_LIBRARY}")
message(STATUS "VE backend: SGEMV kernel .so: ${GGML_VE_SGEMV_SO}")
message(STATUS "VE backend: Q-kernel .so:     ${GGML_VE_KERNELS_SO}")
