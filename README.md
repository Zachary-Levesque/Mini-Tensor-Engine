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

The C++ engine matches the Python reference output, and the AVX2 dot-product
kernel is present in the benchmark binary. The assembly check finds `vmulps`
and `vaddps` instructions in `build/mte_benchmark`.

The latest benchmark was run with:

```bash
./build/mte_benchmark --iterations 200 --warmup 20 --threads 1,2,4,8 --csv-out build/results.csv --json-out build/results.json
```

On this Apple Silicon machine the AVX2 benchmark binary is built as x86_64 so
the AVX2 path can be compiled and inspected. The new single-thread
`transpose_rhs` AVX2 path compares against the old checked-in `results.json`
numbers as follows:

- `32x64x32`: old `488250.00 ns`, new `57523.96 ns`, `8.49x` faster
- `64x64x64`: old `1729833.00 ns`, new `194613.96 ns`, `8.89x` faster
- `128x128x128`: old `12811459.00 ns`, new `1516633.33 ns`, `8.45x` faster
- `256x256x256`: old `98634500.00 ns`, new `12245588.54 ns`, `8.05x` faster

Within the new benchmark run, the AVX2 `transpose_rhs` backend is faster than
the naive backend by:

- `32x64x32`: `2.47x` faster than naive
- `64x64x64`: `2.57x` faster than naive
- `128x128x128`: `2.71x` faster than naive
- `256x256x256`: `4.10x` faster than naive

The threaded AVX2 backend improves larger workloads further. At 8 threads,
`threaded_transpose_rhs` is `21.15x` faster than naive at `256x256x256`
(`2377007.08 ns` versus `50262629.79 ns`). For the smallest case,
`32x64x32`, 8 threads are slower than naive (`0.61x`) because thread overhead
dominates.

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
