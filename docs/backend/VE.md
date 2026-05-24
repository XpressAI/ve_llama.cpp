# llama.cpp for NEC SX-Aurora TSUBASA Vector Engine

- [Background](#background)
- [Hardware Support](#hardware-support)
- [Data Types Supported](#data-types-supported)
- [Quantization Support](#quantization-support)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Running](#running)
- [Environment Variables](#environment-variables)
- [Performance](#performance)
- [Status & Known Limitations](#status--known-limitations)

## Background

The **NEC SX-Aurora TSUBASA** is a PCIe-form-factor vector processor sold by NEC for HPC workloads. Each Vector Engine (VE) card has 8 vector cores at ~1.4 GHz, 48 GB of on-card HBM2 at 1.2 TB/s, and a 256-element 64-bit vector unit with a "packed" mode that processes 512 × FP32 ops per instruction. It also has 16 × 256-bit vector mask registers with 1-instruction bitwise operations and single-instruction popcount — a primitive that is structurally absent on contemporary GPUs and that makes the architecture interesting for low-bit quantization.

The VE runs without an OS on-card and is driven from the host (the "VH" — Vector Host, an x86_64 server) via VEDA, NEC's CUDA-Driver-API-shaped hybrid programming interface.

### llama.cpp + VE

This backend offloads compute-heavy ops (MUL_MAT, FLASH_ATTN_EXT, RMS_NORM, ROPE, GET_ROWS, SET_ROWS, SOFT_MAX, elementwise, MoE dispatch) to the VE through VEDA. Weights live in HBM after the first request; the deferred-sync host loop batches one `vedaCtxSynchronize` per compute graph rather than per kernel launch (a 3× decode improvement). An optional JIT graph compiler (`GGML_VE_COMPILE_GRAPH=1`) further fuses per-token subgraphs into a single VE-side kernel for an additional ~20% decode speedup once the cache is warm.

## Hardware Support

| Vector Engine | Status | Notes |
|---|---|---|
| VE 2.0 (VE20B) | Supported | 8 cores, 48 GB HBM2, 1.2 TB/s. Reference hardware. |
| VE 3.0 (VE30B) | Untested | 16 cores, 96 GB HBM2, 2.45 TB/s. Expected to work with the same code path; some BF16-on-VE3 intrinsics are not yet wired. |
| VE 1.0 | Not supported | No HBM, lacks instructions the backend relies on. |

Multi-card (up to 4 cards per host) is plumbed but only single-card configurations have been exercised end-to-end.

## Data Types Supported

| Data Type | Compute | Storage | Notes |
|---|---|---|---|
| F32 | Supported | Supported | Native FP32 vector ops. |
| BF16 | Emulated via FP32 | Supported | The VE has no native BF16 ALU; the matvec kernels load BF16 into packed-FP32 vector registers (`vldu` + shift/mask/or) and run FMAs at FP32 precision. This is the well-trodden path from NEC's `ve-llama2.c` reference (50–60 t/s on Llama-2-7B BF16). |
| F16 | Conversion to F32 | Storage only (KV cache) | No native compute. |

## Quantization Support

| Format | Status |
|---|---|
| Q8_0 | Supported (full-HBM kernel) |
| Q4_K_M / Q5_K_M / Q6_K / Q2_K / Q3_K | **Not yet supported** |
| MXFP4 | Not yet supported |

The K-quants and MXFP4 paths are deliberately deferred. NCC cannot vectorize int8/int16 arithmetic, and the legacy "dequantize-to-HBM at load time" workaround inflates the resident weight size 4–8× which defeats the purpose of running large models on the 48 GB card. VE-native quant formats designed around the mask-register popcount primitive are the planned path forward.

## Prerequisites

- **VEDA + NLC** from NEC's SDK. On RHEL/CentOS hosts the packages are `veoffload-veda` and `nlc-3.1.0` (or newer). On Ubuntu use NEC's official `.deb`s.
- **NCC compiler** (`/opt/nec/ve/bin/ncc`) — for the OpenMP/VEDA wrapper code that lives in the VE-side kernel `.so`.
- **LLVM-VE-RV** (`/usr/local/ve/llvm-ve-rv-2.2.0/bin/clang`) — required for the BF16 intrinsics kernels. NCC does not support `velintrin.h`.
- **Environment** — source NEC's environment script before building or running:
  ```bash
  source /opt/nec/ve/nlc/3.1.0/bin/nlcvars.sh
  export PATH=/opt/nec/ve/bin:$PATH
  ```

## Build

The VE backend depends on a separately-built shared library (`libve_sgemv.so`) that contains the VE-side kernels. Build it once from the kernel source tree:

```bash
cd /path/to/ggml-ve-veda
make libve_sgemv.so
```

This produces `libve_sgemv.so` in the same directory.

Then build llama.cpp with the VE backend enabled:

```bash
cd /path/to/llama.cpp
cmake -B build -DGGML_VE=ON \
    -DGGML_VE_VEDA_KERNELS_DIR=/path/to/ggml-ve-veda
cmake --build build -j8 --target llama-cli
```

`GGML_VE_VEDA_KERNELS_DIR` tells the backend where to find `libve_sgemv.so` at runtime. If unset, it defaults to the build-time path and can be overridden later with `VE_SGEMV_PATH=...`.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `GGML_VE` | `OFF` | Compile the VE backend |
| `GGML_VE_VEDA_KERNELS_DIR` | `~/claude_workspace/ggml-ve-veda` | Search path for `libve_sgemv.so` (compiled in as a default; overridable with `VE_SGEMV_PATH` at runtime) |

## Running

A first run looks just like any other backend:

```bash
VE_NODE_NUMBER=0 ./build/bin/llama-cli \
    -m models/Llama-3.2-3B-Instruct-BF16.gguf \
    -p "The capital of France is" -n 50 \
    --temp 0 -ngl 99 --single-turn \
    -ctk bf16 -ctv bf16
```

`-ngl 99` offloads every layer to the VE. `-ctk bf16 -ctv bf16` keeps the KV cache in BF16 (the format the BF16 attention kernel reads directly).

### Selecting a Device

`VE_NODE_NUMBER=N` selects VE card N (0-indexed). The backend respects this at registration time; if the env var is unset and multiple cards are present, card 0 is used.

## Environment Variables

| Variable | Purpose |
|---|---|
| `VE_NODE_NUMBER` | Select which VE card to use (per NEC SDK convention) |
| `VE_SGEMV_PATH` | Override the path to `libve_sgemv.so` |
| `VE_KERNELS_PATH` | Override the path to `libve_kernels.so` (optional, K-quant kernels — not yet built) |
| `GGML_VE_COMPILE_GRAPH=1` | Enable the JIT graph compiler (~20% decode speedup after warm cache) |
| `GGML_VE_NO_COLMAJOR=1` | Disable the F32 col-major + CBLAS-NoTrans fast path for prompt eval (debug only) |
| `GGML_VE_DEBUG_DISPATCH=1` | Log the first 200 op dispatches with tensor / buffer info |
| `GGML_VE_DEBUG_SYNC=1` | Log every deferred-sync flush |
| `GGML_VE_DEBUG_KERNELS=1` | Log kernel-load failures at init |
| `VE_PROGINF=DETAIL` | NEC-side profiler — V.Op.Ratio, A.V.Length, MFLOPS, LLC hit ratio |
| `VEDA_FTRACE=1` | Per-function ftrace for the VE-side `.so` (combine with `-ftrace` on the wrapper compile) |

## Performance

Reference numbers on a single VE 2.0 card, Llama-3.2-3B-Instruct BF16:

| Configuration | Decode (tg) | Prompt eval (pp32) |
|---|---|---|
| `GGML_VE_HBM=1` (interpreter) | 14–20 t/s | ~60 t/s |
| `GGML_VE_COMPILE_GRAPH=1` (JIT) | 25–35 t/s | ~75 t/s |

For comparison, NEC's standalone `ve-llama2.c` reference (BF16 Llama-2-7B) achieves ~50 t/s. The gap to that number on this backend is mostly per-op kernel launch overhead, which the JIT graph compiler exists to close.

## Status & Known Limitations

- **Supported end-to-end:** F32 and BF16 weights, Q8_0 quantization, full attention (FA with BF16 KV at standard head sizes), MoE dispatch, RoPE / RMS_NORM / GET_ROWS / SET_ROWS / SoftMax / elementwise. 515/515 per-op tests pass on the reference card.
- **Not yet supported:** K-quants (Q2_K through Q6_K), MXFP4, multi-card sharding, masked SoftMax broadcast (falls back to CPU).
- **CPU_REPACK:** the upstream CPU backend repacks quantized weights into a custom `block_q*_Kx8` layout when no other backend claims them. The VE backend explicitly rejects CPU_REPACK buffer types in `supports_op` so the standard layout reaches us.
- **VE 3.0 hardware** is not yet covered by the BF16 intrinsics; the kernel falls back to a generic path on VE 3.0.

## References

- [SX-Aurora TSUBASA architecture overview](https://www.nec.com/en/global/solutions/hpc/sx/index.html)
- [`ve-llama2.c`](https://github.com/efocht/ve-llama2.c) — NEC engineer's BF16 reference port
- [BF16 on Vector Engine](https://sx-aurora.github.io/posts/Llama2-on-VE-bf16/) — efocht's blog post on the packed-FP32 BF16 matvec trick this backend's BF16 path is built on
- [NEC AVEO + VEDA introduction](https://sxauroratsubasa.sakura.ne.jp/documents/aveo-veda-hybrid-programming/) — the host-device API the backend dispatches through
