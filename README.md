# Mini Tensor Engine

Mini Tensor Engine is a custom C++ machine learning inference engine built from scratch.

Simple meaning:

- a model already has learned weights
- inference means using those weights to produce an output
- this project builds the software that runs that output path

This project is about **running** models correctly and efficiently, not training them.

## Goal

The goal is to understand how ML inference works at a low level and to show that process clearly.

The project focuses on:

- tensor storage
- matrix multiplication
- neural-network layers
- model execution
- correctness checking
- performance benchmarking

## Why It Matters

Most ML tools hide the low-level details. This project shows what happens underneath:

- how model data is stored
- how a forward pass runs
- why matrix multiplication is the main cost
- how cache-aware code improves speed
- how multithreading helps larger workloads

So this project connects machine learning with systems programming and performance engineering.

## Key Concepts

- **C++ Systems Programming**: custom tensor structure, memory layout, modular design  
- **Performance Engineering**: cache-aware optimization, benchmarking, kernel vs end-to-end tradeoffs  
- **Parallel Computing**: multithreaded matrix multiplication, scalability considerations  
- **ML Infrastructure**: inference pipeline, forward-pass execution, layer abstractions  
- **Numerical Computing**: matrix multiplication, floating-point correctness, validation  
- **Systems + ML Integration**: Python ↔ C++ verification, precomputation (cached weights)  
- **Benchmarking Discipline**: backend comparison, performance measurement, regression safety

## What Was Built

- a custom rank-2 `Tensor` class
- three matmul backends: `naive`, `transpose_rhs`, `threaded_transpose_rhs`
- common layers: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- a manifest-driven `FeedForwardModel`
- Python reference generation and C++ correctness validation
- benchmarking with CSV/JSON export
- multiple example model bundles
- a local UI for guided explanation, inference, and benchmark visualization

## Results

The C++ engine matches the Python reference output. The AVX2 dot-product
kernel is present in the benchmark binary, and the assembly check finds
`vmulps` and `vaddps` instructions in `build/mte_benchmark`.

The latest large matrix benchmark was run with:

```bash
./build/mte_benchmark --iterations 5 --warmup 2 --threads 1,4 --skip-model --csv-out build/results.csv --json-out build/results.json
```

On this Apple Silicon machine the AVX2 benchmark binary is built as x86_64 so
the AVX2 path can be compiled and inspected. Naive is skipped for `2048` and
`4096` because those sizes are too slow for this benchmark run.

| Case | Backend | Threads | Avg ns | Speedup vs naive |
| --- | --- | ---: | ---: | ---: |
| `128x128x128` | `naive` | 1 | `4057925.00` | `1.00x` |
| `128x128x128` | `transpose_rhs` | 1 | `1540833.40` | `2.63x` |
| `128x128x128` | `threaded_transpose_rhs` | 1 | `1517291.60` | `2.67x` |
| `128x128x128` | `threaded_transpose_rhs` | 4 | `526833.20` | `7.70x` |
| `256x256x256` | `naive` | 1 | `32117716.60` | `1.00x` |
| `256x256x256` | `transpose_rhs` | 1 | `12039883.40` | `2.67x` |
| `256x256x256` | `threaded_transpose_rhs` | 1 | `12061800.00` | `2.66x` |
| `256x256x256` | `threaded_transpose_rhs` | 4 | `3276425.00` | `9.80x` |
| `512x512x512` | `naive` | 1 | `259484433.40` | `1.00x` |
| `512x512x512` | `transpose_rhs` | 1 | `95735258.40` | `2.71x` |
| `512x512x512` | `threaded_transpose_rhs` | 1 | `95574416.80` | `2.71x` |
| `512x512x512` | `threaded_transpose_rhs` | 4 | `28063441.60` | `9.25x` |
| `1024x1024x1024` | `naive` | 1 | `2630605008.40` | `1.00x` |
| `1024x1024x1024` | `transpose_rhs` | 1 | `777458233.40` | `3.38x` |
| `1024x1024x1024` | `threaded_transpose_rhs` | 1 | `776299408.20` | `3.39x` |
| `1024x1024x1024` | `threaded_transpose_rhs` | 4 | `211434725.00` | `12.44x` |
| `2048x2048x2048` | `transpose_rhs` | 1 | `6231755500.00` | `n/a` |
| `2048x2048x2048` | `threaded_transpose_rhs` | 1 | `6230773675.00` | `n/a` |
| `2048x2048x2048` | `threaded_transpose_rhs` | 4 | `1805852341.60` | `n/a` |
| `4096x4096x4096` | `transpose_rhs` | 1 | `49378770275.00` | `n/a` |
| `4096x4096x4096` | `threaded_transpose_rhs` | 1 | `49353226900.00` | `n/a` |
| `4096x4096x4096` | `threaded_transpose_rhs` | 4 | `16093085308.40` | `n/a` |

The cache-locality benefit of `transpose_rhs` over naive peaks at
`1024x1024x1024`, where it is `3.38x` faster than naive. L2/L3 pressure becomes
visible at `1024`: doubling from `512` to `1024` should cost about `8x` for
cubic work, but naive grows by `10.14x` while `transpose_rhs` grows by `8.12x`.
At `1024`, each matrix is about `4 MiB`, so the working set has moved beyond
typical private L2 capacity and is leaning harder on shared cache. At `2048`
and `4096`, naive is skipped, but the transposed backend continues scaling
close to cubic work: `8.02x` from `1024` to `2048`, then `7.92x` from `2048`
to `4096`. Threading starts helping meaningfully at `128x128x128`: the
4-thread backend is `2.88x` faster than the 1-thread threaded backend there.

## Tiling

The `tiled_transpose_rhs` backend transposes the right-hand side once, then
walks the output matrix in row and column tiles so nearby output cells reuse
nearby slices of the same contiguous input rows and transposed RHS rows. With
the current dot-product kernel, tiling gives only a modest gain over plain
`transpose_rhs`: the best measured result was tile size `32` at `512x512x512`,
where tiled ran in `95575066.70 ns` versus `95721795.90 ns` for plain
transpose, a `1.0015x` speedup. At `1024x1024x1024`, tile sizes `32`, `64`,
and `128` were effectively tied with plain transpose, ranging from `0.9987x`
to `1.0003x`. That small gain is expected here because the pretransposed AVX2
kernel already streams both dot-product operands contiguously, and the largest
cases are mostly limited by memory bandwidth rather than simple L2/L3 locality.

## Main Folders

- `src`: C++ engine, inference CLI, benchmark CLI
- `include/mte`: C++ headers
- `python`: reference generation and Python baseline
- `tests`: correctness tests
- `data/examples`: sample model bundles
- `ui`: local browser dashboard

## Try It

```bash
python3 python/export_reference.py
python3 python/baseline.py
./build/mte_infer --backend threaded_transpose_rhs --threads 4
./build/mte_benchmark --iterations 200 --warmup 20 --threads 1,2,4,8 --csv-out build/results.csv --json-out build/results.json
python3 ui/server.py
```

Then open `http://127.0.0.1:8000`.
