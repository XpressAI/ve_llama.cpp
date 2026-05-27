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

# VE-side kernel libraries (libve_sgemv.so, libve_kernels.so).
#
# By default we build them from the vendored sources in kernels-veda/
# using NCC + LLVM-VE-RV. That subdirectory's CMakeLists.txt detects
# the toolchains and produces the .so files in ${CMAKE_BINARY_DIR}/bin/
# next to libggml-ve.so.
#
# If the toolchains aren't installed, the in-tree build silently skips
# (the subdir's CMakeLists returns early with a WARNING). In that case
# you can still point at an externally-built .so by setting
# -DGGML_VE_VEDA_KERNELS_DIR=... at configure time, or override the
# path at runtime with VE_SGEMV_PATH / VE_KERNELS_PATH.
add_subdirectory(kernels-veda)

if (DEFINED GGML_VE_VEDA_KERNELS_DIR)
    # External override: user provided a directory with prebuilt .so files.
    set(GGML_VE_SGEMV_SO   "${GGML_VE_VEDA_KERNELS_DIR}/libve_sgemv.so")
    set(GGML_VE_KERNELS_SO "${GGML_VE_VEDA_KERNELS_DIR}/libve_kernels.so")
elseif (DEFINED GGML_VE_SGEMV_SO_BUILT)
    # In-tree build (subdir set these in our scope).
    set(GGML_VE_SGEMV_SO   "${GGML_VE_SGEMV_SO_BUILT}")
    set(GGML_VE_KERNELS_SO "${GGML_VE_KERNELS_SO_BUILT}")
else()
    # No toolchain, no override. Compile in the canonical bin/ paths so
    # VE_SGEMV_PATH / VE_KERNELS_PATH env vars can still pick something up
    # at runtime.
    set(GGML_VE_SGEMV_SO   "${CMAKE_BINARY_DIR}/bin/libve_sgemv.so")
    set(GGML_VE_KERNELS_SO "${CMAKE_BINARY_DIR}/bin/libve_kernels.so")
    message(WARNING "VE backend: no NCC/LLVM-VE-RV toolchain found and no GGML_VE_VEDA_KERNELS_DIR set. "
                    "libve_sgemv.so will be loaded from ${GGML_VE_SGEMV_SO} at runtime — "
                    "ensure it exists there, or override with VE_SGEMV_PATH.")
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
    ops/ssm_conv.cpp
    ops/gated_delta_net.cpp
    ops/sum_rows.cpp
    ops/repeat.cpp
    ops/concat.cpp
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
