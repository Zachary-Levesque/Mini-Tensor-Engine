# Mini Tensor Engine

Mini Tensor Engine is a custom machine learning inference engine built from scratch in C++.

Simple version:

- a neural network is a set of learned numbers called weights
- inference means using those weights to make a prediction
- this project builds the software that runs that prediction path

This project does **not** train models. It focuses on **running** models correctly and efficiently.

## Goal

The goal of this project is to understand and demonstrate how ML inference works under the hood.

Instead of relying on a large framework to hide the details, this engine implements the important parts directly:

- tensor storage
- matrix multiplication
- neural-network layers
- model execution
- correctness validation
- performance benchmarking

There is also a second goal: make the project easy to explain and demo. That is why the repo includes a local UI that shows the model flow, tensors, validation status, and performance results.

## Why This Matters

Most people use ML frameworks at a high level. That is useful, but it hides the core engineering work.

This project is important because it shows:

- how tensors are stored in memory
- how a model forward pass is executed
- why matrix multiplication is usually the most expensive operation
- how cache-aware design changes performance
- how multithreading can speed up large workloads
- how to improve speed without breaking correctness

In short, this project connects:

- machine learning
- systems programming
- performance engineering
- software design

## What Was Implemented

The project currently includes:

- a custom rank-2 `Tensor` class with contiguous memory storage
- three matrix multiplication backends:
  - `naive`
  - `transpose_rhs`
  - `threaded_transpose_rhs`
- neural-network layers:
  - `Linear`
  - `ReLU`
  - `Sigmoid`
  - `Tanh`
  - `Softmax`
- a manifest-driven `FeedForwardModel` that loads models from a simple `model.txt`
- Python reference generation for trusted outputs
- C++ validation against Python outputs
- a benchmark executable for kernel and end-to-end model timing
- benchmark export to CSV and JSON
- multiple example model bundles
- a local UI for guided walkthrough, inference testing, tensor inspection, and benchmark viewing

## What The Engine Can Do

Right now, the engine can:

- load a model definition from files
- load the input tensor and model parameters
- run a forward pass in C++
- compare the result against Python
- switch between different matrix multiplication implementations
- benchmark performance across different problem sizes and thread counts
- visualize all of this in a browser UI

So this is no longer just a code experiment. It is a working mini inference runtime plus a demo interface.

## Results

The project achieved several useful results:

- the C++ engine matches the Python reference output
- the optimized cache-aware backend is much faster than the naive backend on larger matrix multiplies
- the multithreaded backend improves performance further on larger workloads
- the flexible model system can now run different example networks, not just one hardcoded architecture
- the UI makes the system understandable to a non-expert viewer

One important lesson from the benchmarks:

- threading helps large workloads
- threading does not always help tiny workloads because thread management has overhead

That is a real systems engineering result, not just a coding detail.

## Why This Is Useful

This project is useful in three ways.

### 1. Technical learning

It teaches how inference engines actually work:

- data layout
- linear algebra kernels
- activations
- model execution
- validation
- benchmarking

### 2. Engineering practice

It demonstrates:

- C++ design
- debugging
- performance measurement
- correctness-first development
- modular architecture

### 3. Portfolio value

It is a strong interview project because it shows that the builder can:

- create a real technical system from scratch
- explain how it works
- measure and improve performance
- present the work clearly

## Project Structure

- `include/mte`: public C++ headers
- `src`: C++ engine implementation, inference CLI, and benchmark CLI
- `tests`: C++ correctness tests
- `python`: Python reference generation and baseline execution
- `data/reference`: default reference bundle used by the main demo path
- `data/examples`: multiple example model bundles for the UI and flexible-model demos
- `ui`: local dashboard for explanation, inference, and benchmarks

## Try It

Generate the reference data:

```bash
python3 python/export_reference.py
```

Run the Python reference:

```bash
python3 python/baseline.py
```

Run the C++ engine:

```bash
./build/mte_infer --backend threaded_transpose_rhs --threads 4
```

Run the benchmark:

```bash
./build/mte_benchmark --iterations 200 --warmup 20 --threads 1,2,4,8 --csv-out build/results.csv --json-out build/results.json
```

Run the UI:

```bash
python3 ui/server.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Final Summary

Mini Tensor Engine is a small but serious inference engine.

It proves that a model can be:

- represented clearly
- executed correctly
- optimized carefully
- benchmarked honestly
- explained visually

That is what makes the project valuable.
