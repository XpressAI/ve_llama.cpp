# GGML NEC SX-Aurora Vector Engine (VE) backend

Runs llama.cpp inference on NEC SX-Aurora TSUBASA Vector Engine cards via the
VEDA offload API. Host-side dispatch is compiled with the system C++ compiler;
the actual VE kernels live in a prebuilt `libve_sgemv.so` (built separately with
`ncc` + LLVM-VE-RV from `ggml-ve-veda/`).

Enable with `-DGGML_VE=ON`. The KV cache **must** be BF16 (`-ctk bf16 -ctv bf16`)
— the VE backend has no F16 path.

## Kernel library location

The backend loads `libve_sgemv.so` (and `libve_kernels.so` when present) at init.

| Variable | Meaning |
|---|---|
| `GGML_VE_KERNELS_PATH` | Directory holding the VE kernel `.so`s. Defaults to the build-time path baked from `ggml-ve-veda/`. |

## Device selection

| Variable | Meaning |
|---|---|
| `VE_NODE_NUMBER` | Which VE card to use (`0`..`3`). |
| `OMP_NUM_THREADS` | OpenMP threads on the VE (8 cores/card). |

## Performance knobs

These change throughput, not correctness. All have safe defaults; override only
when tuning.

| Variable | Default | Meaning |
|---|---|---|
| `GGML_VE_HBM` | off | Cache model weights in VE HBM (uploaded once, reused). The main inference fast path — strongly recommended. |
| `GGML_VE_COMPILE_GRAPH` | off | JIT the whole decode graph into a single fused VE kernel (one launch/token instead of ~one-per-op). On a model whose decode is one self-contained cgraph (e.g. Llama-3.2-3B-BF16) this is ~2.1× over the interpreter. First run per unique graph pays a ~30–60 s `ncc` compile; the `.so` is cached under `~/.cache/ggml-ve-compiled/`. |
| `GGML_VE_COMPILE_MIN_NODES` | `1` | Minimum cgraph node count the graph compiler will attempt. Default `1` = no threshold; the self-containment check (below) decides what actually compiles. Raise it (e.g. `64`) to skip JIT-compiling small graphs not worth the `ncc` cost. *This replaces the old hard-coded threshold — it is now purely an opt-in perf knob.* |
| `GGML_VE_COMPILE_CHUNK` | `48` | Ops per generated chunk function. The fused kernel is split into `static` chunk functions of this many ops so NCC doesn't overflow its optimizer tables on large (8B-class, ~800-op) graphs and silently fall back to scalar/serial code. All chunks still run inside one `#pragma omp parallel`. Smaller = safer but more call-boundary overhead; larger risks the overflow. |
| `GGML_VE_COLMAJOR_FA_MIN` | `96` | KV length at/above which flash-attention switches to the column-major CBLAS path (the Stage-1 crossover). Lower = use colmajor sooner. |
| `GGML_VE_COLMAJOR_N1` | off | Force the N=1 (decode) matmul onto the colmajor F32 CBLAS path. |

### Graph compiler: what compiles

The compiler fuses a cgraph only when it is **self-contained**: every operand is
either a model weight (uploaded once to HBM via the weight cache — including
`token_embd.weight`, which llama.cpp keeps on a CPU buffer), a per-token leaf
input (token id / positions / KV-cell index / attention mask, staged host↔HBM),
or a tensor produced **within the same graph**.

A cgraph that reads a *computed intermediate produced by a different subgraph*
is a **middle fragment** of a scheduler-split decode (this is how Qwen3-family
models with per-head Q/K-norm get split). Those are refused and run on the
interpreter — correct, just without the fusion speedup. Whole-decode-graph
models (no such fragmentation) compile and get the full win.

## Debug / diagnostics

Off by default; set to any value to enable.

| Variable | Meaning |
|---|---|
| `GGML_VE_COMPILE_DEBUG` | Graph-compiler trace/compile/execute logging, including refusal reasons. |
| `GGML_VE_GC_DUMP` | List every CPU-resident operand in each cgraph (weight/leaf/intermediate) — the "why didn't this compile" diagnostic. |
| `GGML_VE_STAGE_DEBUG` | Log each host↔HBM operand staging (name, size, in/out). |
| `GGML_VE_KERNEL_TRACE` | Emit a per-op checkpoint (and FA arg dump) into the generated kernel to pinpoint a faulting op. Adds barriers — for debugging only. |
| `GGML_VE_COMPILE_FTRACE` | Build the generated kernel with `-ftrace -report-all` so the fused function shows up in `ftrace.out` (analyse with `/opt/nec/ve/bin/ftrace`) and NCC writes a `.L` vectorisation/parallelisation listing. For profiling the compiled kernel. |
| `GGML_VE_DIAGNOSE_SIG` | On a compile-triggering cgraph-signature miss, print what changed vs the previous graph of the same node count. |
| `GGML_VE_DEBUG_*` | Per-op debug logging (`_MUL_MAT`, `_ROPE`, `_FA`, `_CPY`, `_RMS_NORM`, `_SYNC`, …). |
| `GGML_VE_NO_*` | Disable a specific op on VE so it falls back to CPU (`_MUL_MAT`, `_FA_TILE`, `_GLU`, `_GET_ROWS`, `_CPY`, `_KV_SHADOW`, `_COLMAJOR`, …) — for bisecting correctness. |

## Testing the graph compiler

```bash
source /opt/nec/ve/nlc/3.1.0/bin/nlcvars.sh
export PATH=/opt/nec/ve/bin:$PATH

# Run 1: compile + cache (slow, ignore timing)
GGML_VE_HBM=1 GGML_VE_COMPILE_GRAPH=1 VE_NODE_NUMBER=0 ./build/bin/llama-completion \
  -m models/Llama-3.2-3B-Instruct-BF16.gguf -p "Hello" -n 5 \
  --temp 0 -ub 1 -ctk bf16 -ctv bf16 -fa on

# Run 2: cached .so (fast, real performance)
GGML_VE_HBM=1 GGML_VE_COMPILE_GRAPH=1 VE_NODE_NUMBER=0 ./build/bin/llama-completion \
  -m models/Llama-3.2-3B-Instruct-BF16.gguf -p "Tell me about France." -n 50 \
  --temp 0 -ub 1 -ctk bf16 -ctv bf16 -fa on
```

Clear `~/.cache/ggml-ve-compiled/` only when testing code changes, never between
performance runs.
