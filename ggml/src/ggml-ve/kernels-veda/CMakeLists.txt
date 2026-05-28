# VE-side kernel libraries: libve_sgemv.so (the main library) and
# libve_kernels.so (the K-quant secondary library). These are built by
# the NEC NCC compiler (host CMake doesn't touch them) and loaded by
# the VE backend via VEDA at runtime.
#
# Two toolchains are involved:
#   - NCC      : NEC's vector compiler. Builds the wrapper TUs that drive
#                OMP parallelism and use NEC vector pragmas.
#   - LLVM-VE-RV : LLVM with Region Vectorizer, the only thing that
#                  understands the velintrin.h intrinsics header. Builds
#                  the inner BF16/colmajor intrinsics TUs.
# The two-toolchain build is annoying but unavoidable — NCC has no
# intrinsics, LLVM-VE-RV has no OpenMP parallelism the way we want it.
#
# Locate the toolchains. Defaults match a stock NEC SDK install.

set(GGML_VE_NCC     "/opt/nec/ve/bin/ncc"                              CACHE FILEPATH "NEC C compiler for VE")
set(GGML_VE_LLVM    "/usr/local/ve/llvm-ve-rv-2.2.0/bin/clang"         CACHE FILEPATH "LLVM-VE-RV clang for VE intrinsics")
set(GGML_VE_NLC_LIB "/opt/nec/ve/nlc/3.1.0/lib"                        CACHE PATH     "NEC Numeric Library Collection lib dir (cblas)")
set(GGML_VE_VEDA_INCLUDE_DIR
    "/opt/nec/ve/share/veoffload-veda/include"                         CACHE PATH     "VEDA headers for VE device code")

if (NOT EXISTS "${GGML_VE_NCC}")
    message(WARNING "VE backend: ncc not found at ${GGML_VE_NCC}; skipping libve_sgemv.so build. "
                    "Install NEC SDK or set -DGGML_VE_NCC=...")
    return()
endif()
if (NOT EXISTS "${GGML_VE_LLVM}")
    message(WARNING "VE backend: llvm-ve-rv clang not found at ${GGML_VE_LLVM}; skipping libve_sgemv.so build. "
                    "Install llvm-ve-rv or set -DGGML_VE_LLVM=...")
    return()
endif()

# Where to drop the built .so so the host backend can load it. We put it
# next to the host backend's shared library so a single -Wl,-rpath,$ORIGIN
# would find it, and so that the build/bin/ layout matches how llama.cpp
# arranges other shared libs.
set(GGML_VE_KERNEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${GGML_VE_KERNEL_OUTPUT_DIR}")

set(NCC_INCLUDES  -I${GGML_VE_VEDA_INCLUDE_DIR})
set(LLVM_INCLUDES -I${GGML_VE_VEDA_INCLUDE_DIR})
set(NCC_FLAGS     -O4 -fopenmp -fpic -ftrace ${NCC_INCLUDES})
set(LLVM_FLAGS    --target=ve-linux -O3 -fPIC ${LLVM_INCLUDES})

# -----------------------------------------------------------------------------
# Per-TU rules. Each .c is compiled to a .o in the build dir. NCC-built TUs use
# ${GGML_VE_NCC}; LLVM-VE-RV-built TUs use ${GGML_VE_LLVM}.
# -----------------------------------------------------------------------------
function(ggml_ve_add_ncc_object SRC OBJ_VAR)
    set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}")
    set(OBJ_PATH "${CMAKE_CURRENT_BINARY_DIR}/${SRC}.o")
    add_custom_command(
        OUTPUT  ${OBJ_PATH}
        COMMAND ${GGML_VE_NCC} ${NCC_FLAGS} -c -o ${OBJ_PATH} ${SRC_PATH}
        DEPENDS ${SRC_PATH}
        COMMENT "NCC ${SRC}"
        VERBATIM)
    set(${OBJ_VAR} ${OBJ_PATH} PARENT_SCOPE)
endfunction()

function(ggml_ve_add_llvm_object SRC OBJ_VAR)
    set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}")
    set(OBJ_PATH "${CMAKE_CURRENT_BINARY_DIR}/${SRC}.o")
    add_custom_command(
        OUTPUT  ${OBJ_PATH}
        COMMAND ${GGML_VE_LLVM} ${LLVM_FLAGS} -c -o ${OBJ_PATH} ${SRC_PATH}
        DEPENDS ${SRC_PATH}
        COMMENT "LLVM-VE-RV ${SRC}"
        VERBATIM)
    set(${OBJ_VAR} ${OBJ_PATH} PARENT_SCOPE)
endfunction()

# Wrapper objects (NCC: drive OMP, use NEC pragmas).
ggml_ve_add_ncc_object(ve_sgemv_wrapper.c    OBJ_SGEMV_WRAPPER)
ggml_ve_add_ncc_object(ve_kvcolmajor_wrapper.c OBJ_KVCOL_WRAPPER)
ggml_ve_add_ncc_object(ve_recurrent_wrapper.c OBJ_RECUR_WRAPPER)
ggml_ve_add_ncc_object(q4k_full_dispatch.c   OBJ_Q4K_DISPATCH)

# Intrinsics objects (LLVM-VE-RV: velintrin.h).
ggml_ve_add_llvm_object(flash_attn_bf16_intrinsics.c OBJ_FA_INTRIN)
ggml_ve_add_llvm_object(flash_attn_bf16_fused.c      OBJ_FA_FUSED)
ggml_ve_add_llvm_object(flash_attn_bf16_colmajor.c   OBJ_FA_COL)
ggml_ve_add_llvm_object(flash_attn_bf16kv_tile_fast.c OBJ_FA_TILE_FAST)
ggml_ve_add_llvm_object(ve_sgemv_bf16.c              OBJ_SGEMV_BF16)
ggml_ve_add_llvm_object(bf16_sgemm_intrinsics.c      OBJ_BF16_SGEMM)
ggml_ve_add_llvm_object(sgemv_packed_bf16_unr.c      OBJ_SGEMV_PACKED)
ggml_ve_add_llvm_object(q4k_full_intrin.c            OBJ_Q4K_FULL)

# Link with NCC because we need its OpenMP runtime symbols. ftrace flag
# is on so per-function profiling works in shadow benchmark runs; turn
# off at release time if it's eating measurement.
set(LIBVE_SGEMV "${GGML_VE_KERNEL_OUTPUT_DIR}/libve_sgemv.so")
add_custom_command(
    OUTPUT  ${LIBVE_SGEMV}
    COMMAND ${GGML_VE_NCC} -shared -fopenmp -ftrace -o ${LIBVE_SGEMV}
        ${OBJ_SGEMV_WRAPPER} ${OBJ_KVCOL_WRAPPER} ${OBJ_RECUR_WRAPPER}
        ${OBJ_Q4K_DISPATCH}
        ${OBJ_FA_INTRIN} ${OBJ_FA_FUSED} ${OBJ_FA_COL} ${OBJ_FA_TILE_FAST}
        ${OBJ_SGEMV_BF16} ${OBJ_BF16_SGEMM} ${OBJ_SGEMV_PACKED}
        ${OBJ_Q4K_FULL}
        -lm -L${GGML_VE_NLC_LIB} -lcblas -lblas_openmp
        -Wl,-rpath,${GGML_VE_NLC_LIB}
    DEPENDS
        ${OBJ_SGEMV_WRAPPER} ${OBJ_KVCOL_WRAPPER} ${OBJ_RECUR_WRAPPER}
        ${OBJ_Q4K_DISPATCH}
        ${OBJ_FA_INTRIN} ${OBJ_FA_FUSED} ${OBJ_FA_COL} ${OBJ_FA_TILE_FAST}
        ${OBJ_SGEMV_BF16} ${OBJ_BF16_SGEMM} ${OBJ_SGEMV_PACKED}
        ${OBJ_Q4K_FULL}
    COMMENT "NCC link libve_sgemv.so"
    VERBATIM)

# Secondary library: libve_kernels.so. Currently just ve_kernels.c (small
# K-quant entry points). May grow.
ggml_ve_add_ncc_object(ve_kernels.c OBJ_VE_KERNELS)
set(LIBVE_KERNELS "${GGML_VE_KERNEL_OUTPUT_DIR}/libve_kernels.so")
add_custom_command(
    OUTPUT  ${LIBVE_KERNELS}
    COMMAND ${GGML_VE_NCC} -shared -fopenmp -ftrace -o ${LIBVE_KERNELS}
        ${OBJ_VE_KERNELS}
        -lm -L${GGML_VE_NLC_LIB} -lcblas -lblas_openmp
        -Wl,-rpath,${GGML_VE_NLC_LIB}
    DEPENDS ${OBJ_VE_KERNELS}
    COMMENT "NCC link libve_kernels.so"
    VERBATIM)

# Drive the build via a custom target the parent CMake depends on.
add_custom_target(ggml-ve-kernels-veda ALL
    DEPENDS ${LIBVE_SGEMV} ${LIBVE_KERNELS})

# Tell the parent scope where the .so files landed so the host backend
# can use the same path at runtime by default.
set(GGML_VE_SGEMV_SO_BUILT   "${LIBVE_SGEMV}"   PARENT_SCOPE)
set(GGML_VE_KERNELS_SO_BUILT "${LIBVE_KERNELS}" PARENT_SCOPE)
