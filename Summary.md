# Mini Tensor Engine: Interview Summary

## 1. What is this project?

This project is a custom **C++ machine learning inference engine** built from scratch.

Simple meaning:

- A trained neural network already has learned weights
- Inference means using those weights to make a prediction
- This project builds the software that runs that prediction path in C++

It is **not** about training models.

It is about:

- running a model
- understanding the low-level math
- controlling memory and performance
- benchmarking different implementations

---

## 2. What is the goal of the project?

The goal is to build a small but serious inference engine that demonstrates both **systems engineering** and **ML infrastructure** skill.

There are two layers to the goal.

### Technical goal

Build a CPU-based engine that can:

- store tensor data efficiently
- run matrix multiplication
- apply neural-network layers
- load model parameters
- execute inference correctly
- compare performance between implementations

### Professional goal

Demonstrate:

- strong C++ programming
- low-level systems thinking
- performance engineering
- understanding of ML inference internals
- clean architecture and benchmarking discipline

That is why this project is relevant for software, systems, ML infrastructure, and performance-oriented roles.

---

## 3. Why is this project relevant?

Most people use frameworks like PyTorch at a high level.

This project is relevant because it shows understanding below that level:

- how tensors are stored in memory
- how a neural network forward pass is executed
- why matrix multiplication dominates runtime
- how cache behavior affects speed
- how to validate numerical correctness
- how to optimize without breaking correctness

So this project is valuable because it connects:

- machine learning
- systems programming
- performance optimization
- software engineering discipline

---

## 4. What does the engine currently do?

Right now the engine can:

- generate deterministic reference model data in Python
- load that data in C++
- run a small two-layer neural network
- validate that C++ output matches Python output
- benchmark multiple matrix multiplication backends

The current model is:

`input -> Linear -> ReLU -> Linear -> Softmax`

So the engine already performs real forward-pass inference correctly.

---

## 5. What are the main components?

### Tensor

The `Tensor` class is the core data structure.

It currently supports:

- rank-2 tensor shapes
- contiguous memory storage
- shape tracking
- row/column indexing
- direct access to underlying values

Why it matters:

- tensors are the basic data structure of neural networks
- memory layout strongly affects performance

### Matrix multiplication

This is the most important numerical kernel in the project.

Why it matters:

- linear layers rely on matrix multiplication
- in many ML systems, this is a major runtime cost

### Layers

The engine currently implements:

- `Linear`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `Softmax`

Why these matter:

- `Linear` performs the learned transformation
- `ReLU` adds nonlinearity
- `Sigmoid` squashes values into the range from 0 to 1
- `Tanh` squashes values into the range from -1 to 1
- `Softmax` converts final scores into probability-like outputs

### Model abstraction

The engine uses a `FeedForwardModel` object driven by a simple `model.txt` manifest.

Why it matters:

- groups weights and biases cleanly
- validates tensor shapes
- runs the forward pass
- supports backend selection
- caches optimized weight layouts when needed
- makes it easy to define different feed-forward networks without hardcoding one exact model

### IO

The project loads tensors from simple text files.

Why it matters:

- makes the data flow easy to inspect and debug
- provides a clean correctness path from Python to C++

### Testing and benchmarking

The project includes:

- correctness checks
- backend equivalence validation
- numerical tolerance comparisons
- runtime benchmarking
- multiple sample model bundles for demonstration

Why it matters:

- performance work is only useful if the results stay correct
- the engine is easier to explain when the architecture can be switched and observed

---

## 6. What implementations does matrix multiplication use?

There are currently two backends.

### `naive`

This is the simple triple-loop implementation.

Why it exists:

- easy to understand
- good correctness baseline
- useful comparison point

### `transpose_rhs`

This is a more cache-friendly implementation.

What it does:

- transposes the right-hand-side matrix
- makes the multiply access memory more sequentially
- improves cache behavior on larger problems

Why it matters:

- the math stays the same
- only the memory access pattern changes
- better memory access can make the operation much faster

---

## 7. What optimization was added and why?

An important optimization was added for the model path:

- when using the transpose-based backend, the model caches the transposed weight matrices

Why this matters:

- model weights are reused across many inferences
- re-transposing them every time wastes work
- caching moves that cost to model setup instead of repeated inference

This is a strong systems idea:

If data is reused often, precompute the expensive transformed version once.

---

## 8. How does the full pipeline work?

The current pipeline is:

1. Python creates reference input, weights, biases, and expected output
2. C++ loads those tensors
3. the model runs `Linear -> ReLU -> Linear -> Softmax`
4. the C++ output is compared to Python output
5. the benchmark executable measures backend performance

This gives both:

- correctness validation
- performance measurement

---

## 9. Why are Python and C++ both used?

### Python is used for:

- fast reference generation
- deterministic model data
- trusted correctness baseline

### C++ is used for:

- custom engine implementation
- low-level control over memory and execution
- performance-oriented development

This split is useful because Python makes correctness easy to verify, while C++ is the real implementation target.

---

## 10. What results do we currently have?

Current status:

- the C++ model output matches the Python reference output
- tests pass
- both matmul backends are numerically equivalent for the tested model path
- the optimized backend is measurably faster on larger matrix multiplications
- after caching transposed weights, the optimized backend also improves end-to-end model inference

Important engineering takeaway:

An optimization can look good in isolation but still hurt real inference if integrated poorly. We observed that, then fixed it by caching the transposed weights.

That is a strong interview talking point.

---

## 11. What technical skills does this project show?

### C++ and software engineering

- custom data structures
- modular design
- API/interface design
- error handling
- file IO
- testable architecture

### Systems and performance

- contiguous memory layout
- cache-aware optimization
- backend comparison
- benchmark-driven development
- reuse of precomputed transformed data
- thinking about kernel-level vs end-to-end performance

### ML infrastructure

- inference vs training distinction
- tensor-based computation
- forward-pass execution
- neural-network layer composition
- weight handling
- Python-to-C++ validation workflow

### Numerical correctness

- floating-point tolerance comparisons
- reference-based validation
- regression protection while optimizing

---

## 12. What should I say in an interview?

### Short explanation

"I built a small C++ machine learning inference engine from scratch to understand how neural networks run at a low level. I implemented custom tensor storage, matrix multiplication, neural-network layers, model execution, correctness validation against Python, and benchmarked multiple CPU backends."

### Why you built it

"I wanted a project that demonstrates both systems programming and ML infrastructure skill. The point was not just to run a model, but to understand memory layout, numerical kernels, correctness validation, and optimization tradeoffs."

### Most interesting technical point

"The most interesting part was separating correctness from optimization. I first built a naive working path, then added a more cache-friendly matrix multiplication backend, benchmarked it, and realized that kernel-level speedups do not automatically improve end-to-end inference. I then cached transposed weights inside the model so the optimized backend improved real inference too."

---

## 13. What resume line could I use?

Built a custom C++ machine learning inference engine from scratch with contiguous tensor storage, multiple matrix multiplication backends, neural-network forward-pass execution, validation against a Python reference pipeline, and benchmark-driven CPU optimization.

---

## 14. What is still left to build?

The project is still a foundation, not the final form.

Next strong steps would be:

- multithreaded matrix multiplication
- more realistic benchmark suites
- additional layer types
- more flexible model definitions
- better weight formats
- SIMD optimization
- possibly Apple Metal later

---

## 15. Final takeaway

This project matters because it shows that machine learning inference is not just model math. It is also a systems problem involving data layout, kernel implementation, correctness validation, and careful performance measurement.

The current version already proves:

- the engine runs a real model correctly
- C++ and Python outputs match
- the code has reusable structure
- optimization decisions are measured and justified

That is the key story to tell in an interview.
