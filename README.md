# Mini Tensor Engine

Mini Tensor Engine is a C++ inference runtime for studying how tensor layout, matrix multiplication kernels, SIMD, cache locality, threading, and int8 quantization change CPU inference latency.

## Why?

An inference engine moves input tensors and learned weights through matrix kernels, bias addition, activation functions, and model-level dispatch. At the CPU level, the hard part is not writing `C = A * B`; it is choosing memory layouts, avoiding strided cache-line fetches, preserving numerical correctness, and proving changes with measurements. PyTorch hides those details by design, while this project keeps them visible in C++.

The project targets systems engineering skills used in inference runtimes: row-major storage, backend selection, AVX2 dot products, pretransposed weights, cache scaling, thread partitioning, int8 quantization, and benchmark discipline. Each claim is tied to `results.json`, not intuition.

## Architecture

The `Tensor` class is rank-2 and row-major, backed by `std::vector<float>`. That limits generality, but it makes address calculation explicit: `row * cols + col` is the storage contract every kernel shares. The matmul backend enum keeps dispatch cheap and makes comparisons repeatable. `naive` is the baseline with strided RHS reads. `transpose_rhs` pays one layout conversion so both dot-product operands are contiguous. `threaded_transpose_rhs` partitions rows across workers, which keeps output writes disjoint but does not remove bandwidth pressure. `tiled_transpose_rhs` walks row and column blocks after transposing RHS; its best measured gain is only `1.0015x`, because the AVX2 pretransposed dot product already streams contiguous operands.

Layer code is intentionally separate from kernel code. `Linear` calls matmul plus bias, while `ReLU`, `Sigmoid`, `Tanh`, and `Softmax` operate over tensors without owning backend policy. The model loader reads a manifest and caches transposed weights when the chosen backend benefits from that format. Python generates deterministic reference tensors and expected outputs; C++ loads the same files and must match them before benchmark output is treated as valid.

## Performance results

| Case | Backend | Threads | Avg (ns) | vs Naive |
| --- | --- | ---: | ---: | ---: |
| `128x128x128` | `naive` | 1 | `4068833.35` | `1.00x` |
| `128x128x128` | `transpose_rhs` | 1 | `1512945.80` | `2.69x` |
| `128x128x128` | `tiled_transpose_rhs` | 1 | `1500064.60` | `2.71x` |
| `128x128x128` | `threaded_transpose_rhs` | 1 | `1517608.35` | `2.68x` |
| `256x256x256` | `naive` | 1 | `32082735.40` | `1.00x` |
| `256x256x256` | `transpose_rhs` | 1 | `12022047.90` | `2.67x` |
| `256x256x256` | `tiled_transpose_rhs` | 1 | `11997002.05` | `2.67x` |
| `256x256x256` | `threaded_transpose_rhs` | 1 | `12017931.25` | `2.67x` |
| `512x512x512` | `naive` | 1 | `266517072.90` | `1.00x` |
| `512x512x512` | `transpose_rhs` | 1 | `95725933.35` | `2.78x` |
| `512x512x512` | `tiled_transpose_rhs` | 1 | `96027616.70` | `2.78x` |
| `512x512x512` | `threaded_transpose_rhs` | 1 | `95685639.60` | `2.79x` |
| `1024x1024x1024` | `naive` | 1 | `2618671414.60` | `1.00x` |
| `1024x1024x1024` | `transpose_rhs` | 1 | `776541785.40` | `3.37x` |
| `1024x1024x1024` | `tiled_transpose_rhs` | 1 | `779939433.30` | `3.36x` |
| `1024x1024x1024` | `threaded_transpose_rhs` | 1 | `776688218.75` | `3.37x` |
| `2048x2048x2048` | `transpose_rhs` | 1 | `6172739629.20` | `n/a` |
| `2048x2048x2048` | `threaded_transpose_rhs` | 1 | `6194787968.75` | `n/a` |
| `4096x4096x4096` | `transpose_rhs` | 1 | `49272107562.50` | `n/a` |
| `4096x4096x4096` | `threaded_transpose_rhs` | 1 | `49469847595.85` | `n/a` |

Cache locality is clearest at `1024x1024x1024`: `transpose_rhs` is `3.37x` versus naive by replacing strided RHS loads with contiguous loads. The one-thread threaded backend tracks `transpose_rhs`, confirming that row partitioning does not change the single-thread memory path. At `2048` and `4096`, naive is skipped; `transpose_rhs` moves from `6172739629.20 ns` to `49272107562.50 ns`, close to cubic scaling and consistent with bandwidth pressure. The int8 path reaches `3.44x` at `1024x1024x1024`, showing that reduced data width matters when memory traffic dominates.

## Int8 quantization

Symmetric quantization maps floats to signed int8 with one scale and zero point `0`. Matmul accumulates int8 products into `int32_t`, then converts each result with `lhs.scale * rhs.scale`.

| Case | Float transpose_rhs (ns) | Int8 dequantized (ns) | Speedup |
| --- | ---: | ---: | ---: |
| `256x256x256` | `12003135.45` | `5174472.95` | `2.32x` |
| `512x512x512` | `95741181.25` | `32833693.75` | `2.92x` |
| `1024x1024x1024` | `781326560.40` | `227291441.70` | `3.44x` |

Tests require quantize/dequantize max absolute error below `0.02` and `32x64 * 64x32` int8 matmul max absolute error below `0.5`. This is scalar int8; production kernels need AVX-VNNI or AMX dot-product instructions, and without them scalar int8 can lose to a stronger AVX2 float implementation.

## How to run it

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/mte_tests
./build/mte_benchmark --iterations 20 --warmup 5 --threads 1 --skip-model --csv-out build/results.csv --json-out build/results.json
python3 python/export_reference.py
python3 python/baseline.py
./build/mte_infer --backend threaded_transpose_rhs --threads 4
```
