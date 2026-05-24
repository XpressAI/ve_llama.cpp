# llama.cpp for NEC SX-Aurora TSUBASA Vector Engine

- [Background](#background)
- [Hardware](#hardware)
- [Model & Data Type Support](#model--data-type-support)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Running](#running)
- [Environment Variables](#environment-variables)
- [Performance](#performance)
- [Known Limitations](#known-limitations)

## Background

The **NEC SX-Aurora TSUBASA** is a PCIe vector processor sold by NEC for HPC workloads. Each Vector Engine (VE) card has 8 vector cores, 48 GB of on-card HBM2, and 1.2 TB/s of memory bandwidth.

The VE backend offloads compute-heavy llama.cpp ops (matrix multiplies, flash attention, RMS norm, RoPE, embeddings, attention, MoE dispatch, and elementwise ops) to the VE through NEC's VEDA programming interface. Weights are uploaded to HBM once on the first request and reused across all subsequent requests.

## Hardware

| Vector Engine | Status     | Notes                                       |
| ------------- | ---------- | ------------------------------------------- |
| VE 2.0        | Supported  | Reference hardware. 8 cores, 48 GB HBM2.    |
| VE 3.0        | Untested   | Expected to work; not verified end-to-end.  |
| VE 1.0        | Unsupported| Lacks instructions the backend relies on.   |

Multi-card hosts are recognised at registration time; pick the card with `VE_NODE_NUMBER=N`. Sharding a single model across cards is not yet implemented.

## Model & Data Type Support

| Weight format        | Status        |
| -------------------- | ------------- |
| F32                  | Supported     |
| BF16                 | Supported     |
| Q8_0                 | Supported     |
| Q2_K / Q3_K / Q4_K_M / Q5_K_M / Q6_K | Not supported — falls back to CPU |
| MXFP4                | Not supported — falls back to CPU |

| KV cache type | Status    |
| ------------- | --------- |
| BF16          | Supported — recommended (`-ctk bf16 -ctv bf16`) |
| F16           | Supported |
| F32           | Supported |

If you load a model in an unsupported quantization format, the unsupported ops will run on CPU and performance will collapse to CPU rates. BF16 is the format that exercises the fast paths.

## Prerequisites

The VE backend depends on NEC's SX-Aurora SDK. Install (or have your system administrator install) the following packages from NEC:

- **VEOS + VEDA** — the driver stack and host-side API (`veoffload-veda`).
- **NLC 3.1.0 or newer** — NEC Numeric Library Collection, used for CBLAS.
- **NCC** — NEC's C compiler for VE (`/opt/nec/ve/bin/ncc`).
- **LLVM-VE-RV 2.2.0 or newer** — LLVM-VE with the Region Vectorizer (`/usr/local/ve/llvm-ve-rv-2.2.0/bin/clang`), required for the BF16 intrinsics in the kernel library.

Source the NEC environment script once per shell before building or running:

```bash
source /opt/nec/ve/nlc/3.1.0/bin/nlcvars.sh
export PATH=/opt/nec/ve/bin:$PATH
```

## Build

```bash
cmake -B build -DGGML_VE=ON
cmake --build build -j8 --target llama-cli
```

That's it — the VE-side kernel library (`libve_sgemv.so`) is built from the in-tree sources at `ggml/src/ggml-ve/kernels-veda/` as part of the same invocation and dropped into `build/bin/` next to the host backend. There is no separate kernel build step.

If the SDK is installed somewhere other than the canonical paths, point the build at it with `-DGGML_VE_NCC=...`, `-DGGML_VE_LLVM=...`, `-DGGML_VE_NLC_LIB=...`. If you have a pre-built kernel library elsewhere, point CMake at it with `-DGGML_VE_VEDA_KERNELS_DIR=<dir>` to skip the in-tree build entirely.

## Running

```bash
VE_NODE_NUMBER=0 ./build/bin/llama-cli \
    -m models/Llama-3.2-3B-Instruct-BF16.gguf \
    -p "The capital of France is" -n 50 \
    --temp 0 -ngl 99 --single-turn \
    -fa on -ctk bf16 -ctv bf16
```

- `-ngl 99` offloads every layer to the VE.
- `-fa on -ctk bf16 -ctv bf16` enables flash attention with a BF16 KV cache. These are the recommended defaults — the BF16 attention path is the most heavily optimised.

If you have multiple VE cards, set `VE_NODE_NUMBER` to the card index you want (0-based).

## Environment Variables

| Variable                       | Purpose                                              |
| ------------------------------ | ---------------------------------------------------- |
| `VE_NODE_NUMBER=N`             | Select which VE card to use (NEC SDK convention).    |
| `GGML_VE_COMPILE_GRAPH=1`      | Enable the JIT graph compiler. First-token latency increases on a cold cache; subsequent tokens get ~20% faster. The compiled graphs are cached on disk so the cold cost is paid only once per shape. |
| `VE_SGEMV_PATH=/path/to/libve_sgemv.so` | Override the location of the kernel library at runtime. |

## Performance

Single VE 2.0 card, Llama-3.2-3B-Instruct BF16, `-fa on -ctk bf16 -ctv bf16`:

| Decode length | Tokens / second |
| ------------- | --------------- |
| 100 tokens    | ~29             |
| 200 tokens    | ~29             |
| 400 tokens    | ~29             |

Decode speed stays roughly flat as the context grows, because the flash-attention path reads the KV cache along its unit-stride axis. Larger BF16 models (Llama-3-8B, Llama-3.1-70B with multi-card support once available) scale proportionally to model bandwidth.

With `GGML_VE_COMPILE_GRAPH=1`, decode improves by an additional 15–25% once the graph cache is warm.

## Known Limitations

- **K-quants and MXFP4 are not supported.** Models in those formats will technically load but fall back to the CPU for matrix multiplies, which is the dominant cost — performance will be poor. Use F32, BF16, or Q8_0 weights for now.
- **No multi-card sharding.** Single-card decode only. A multi-card model fits if it fits in 48 GB.
- **VE 3.0** has not been validated end-to-end. It is expected to work but some BF16 intrinsics are tuned for VE 2.0.

## References

- [SX-Aurora TSUBASA architecture overview](https://www.nec.com/en/global/solutions/hpc/sx/index.html)
- [`ve-llama2.c`](https://github.com/efocht/ve-llama2.c) — NEC engineer's standalone BF16 Llama-2 port; the inspiration for the BF16 matvec implementation used here.
