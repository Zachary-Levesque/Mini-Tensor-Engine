# Mini-Tensor-Engine

High-performance ML inference engine in C++ with a custom tensor implementation, multiple matrix-multiplication backends, simple neural-network layers, a minimal model abstraction, and validation against a Python reference pipeline.

## Current Scope

The initial milestone is correctness-first:

- contiguous rank-2 tensor storage
- naive and cache-friendlier CPU matrix multiplication backends
- `Linear`, `ReLU`, and `Softmax` layers
- a reusable two-layer MLP model object
- Python-generated reference weights and outputs
- C++ inference executable with configurable reference-data path
- separate benchmark executable for backend comparison and latency measurement

## Layout

- `include/mte`: public headers
- `src`: C++ implementation and inference executable
- `src/benchmark.cpp`: simple microbenchmark entry point
- `tests`: lightweight C++ correctness tests
- `python`: baseline inference and reference-data export scripts
- `data/reference`: deterministic model weights, input, and expected output

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
./build/mte_infer --backend transpose_rhs
./build/mte_benchmark --iterations 200 --warmup 20
```

## Reference Data

Generate the deterministic Python reference tensors with:

```bash
python3 python/export_reference.py
```

The inference executable accepts `--data-dir <path>` if you want to point it at a different exported reference bundle.

`mte_infer` also accepts `--backend naive|transpose_rhs` to validate different matrix-multiplication implementations. `mte_benchmark` compares those backends on several matrix sizes and a small end-to-end model path.
