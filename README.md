# Mini-Tensor-Engine

High-performance ML inference engine in C++ with a custom tensor implementation, multiple matrix-multiplication backends, simple neural-network layers, a minimal model abstraction, and validation against a Python reference pipeline.

## Current Scope

The initial milestone is correctness-first:

- contiguous rank-2 tensor storage
- naive, cache-friendlier, and multithreaded CPU matrix multiplication backends
- `Linear`, `ReLU`, `Sigmoid`, `Tanh`, and `Softmax` layers
- a reusable feed-forward model loader driven by a simple manifest format
- Python-generated reference weights and outputs
- C++ inference executable with configurable reference-data path
- separate benchmark executable for backend comparison, latency measurement, thread scaling, and larger synthetic inference workloads

## Roadmap

The project has completed the first correctness-first and early optimization stages. The next major phases are:

- multithreaded matrix multiplication and thread-scaling benchmarks
- broader benchmark suites and deeper performance analysis
- additional model and layer support
- a clean visualization UI for inference flow, correctness checks, and benchmark results

The UI phase is intended as the final presentation layer on top of the real engine so the project is both technically strong and easy to demo.

## Layout

- `include/mte`: public headers
- `src`: C++ implementation and inference executable
- `src/benchmark.cpp`: simple microbenchmark entry point
- `tests`: lightweight C++ correctness tests
- `python`: baseline inference and reference-data export scripts
- `data/reference`: deterministic model weights, input, and expected output
- `data/examples`: multiple sample model bundles used by the UI and flexible model demos

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
./build/mte_infer --backend threaded_transpose_rhs --threads 4
./build/mte_benchmark --iterations 200 --warmup 20 --threads 1,2,4,8
./build/mte_benchmark --iterations 200 --warmup 20 --threads 1,2,4,8 --csv-out build/results.csv --json-out build/results.json
```

## UI Demo

Run the local dashboard with:

```bash
python3 ui/server.py
```

Then open `http://127.0.0.1:8000` in your browser.

The dashboard opens with a simple project summary first, then lets the user move into an interactive playground.

The UI shows:

- a plain-language explanation of the project goal and why it matters
- multiple sample models you can switch between
- model architecture and tensor views
- layer-by-layer inference flow
- Python-reference validation status
- inference execution controls for different backends and thread counts
- benchmark controls and visual summaries

The UI reads the existing reference tensors and latest benchmark JSON output, and it can also trigger fresh inference and benchmark runs locally.

## Reference Data

Generate the deterministic Python reference tensors with:

```bash
python3 python/export_reference.py
```

This also generates multiple sample bundles under `data/examples`.

The inference executable accepts `--data-dir <path>` if you want to point it at a different exported reference bundle.

The Python baseline accepts the same pattern:

```bash
python3 python/baseline.py --data-dir data/examples/sigmoid_gate
python3 python/baseline.py --data-dir data/examples/tanh_mixer
```

`mte_infer` also accepts `--backend naive|transpose_rhs|threaded_transpose_rhs` and `--threads <count>` to validate different matrix-multiplication implementations. `mte_benchmark` compares those backends on several matrix sizes, reports thread scaling, and measures both the small demo model and larger synthetic inference workloads.

Reference bundles can include a `model.txt` manifest such as:

```text
linear w1.txt b1.txt
relu
linear w2.txt b2.txt
softmax
```

The current manifest parser supports `linear`, `relu`, `sigmoid`, `tanh`, and `softmax`.

`mte_benchmark` also accepts:

- `--csv-out <path>` to export all benchmark rows as CSV
- `--json-out <path>` to export structured benchmark data for future UI or analysis tooling
