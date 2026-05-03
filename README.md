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

- the C++ engine matches the Python reference output
- the cache-aware backend is much faster than the naive backend on larger matrix multiplies
- the threaded backend improves performance further on larger workloads
- the model system can run multiple example networks
- the UI makes the project easy to demo and understand

One important result is that threading helps big workloads much more than tiny ones, because extra threads also add overhead.

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
