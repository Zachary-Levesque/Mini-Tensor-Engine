# Interview Preparation Summary

## The two-minute pitch

I built Mini Tensor Engine to understand CPU inference below the framework layer. It is a C++ inference runtime with a rank-2 row-major tensor type, multiple matmul backends, AVX2 SIMD, extended cache-scale benchmarks, a tiled backend, and scalar int8 quantization. I built it because PyTorch hides the address-level decisions that dominate CPU inference: strided versus contiguous loads, actual vector instructions, bandwidth pressure, and quantization error.

The core design is simple on purpose. I use row-major storage so I can reason about every memory access. My naive backend is the correctness baseline, `transpose_rhs` turns RHS column walks into contiguous dot-product reads, AVX2 makes the inner loop explicit, tiling tests cache blocking, and int8 tests reduced data width. The headline result is `1024x1024x1024`: float `transpose_rhs` is `776541785.40 ns` versus `2618671414.60 ns` for naive, or `3.37x`. The int8 path is `227291441.70 ns` versus `781326560.40 ns` for float `transpose_rhs`, or `3.44x`. The point is not just the numbers; it is that each result follows from a specific systems decision.

## The technical deep dives

### How does your tensor class work and why did you lay it out that way?

I kept `Tensor` rank-2 and row-major, backed by `std::vector<float>`. Element address calculation is `row * cols + col`. That is less general than a production tensor library, but it makes the memory model visible. I am studying kernel behavior, not arbitrary views or broadcasting. This layout lets me explain why naive RHS access jumps through memory and why pretransposing RHS makes the innermost loop contiguous.

### Walk me through your matmul backends — what does each one do at the memory level?

`naive` reads LHS rows contiguously but RHS columns with a stride of `rhs.cols()`. `transpose_rhs` stores RHS as `[cols, inner]`, so a dot product reads both `lhs_row[k]` and `rhs_transposed_row[k]` contiguously. `threaded_transpose_rhs` keeps that pattern and partitions output rows, so writes are disjoint. `tiled_transpose_rhs` walks output row and column blocks after transposing RHS; it tests cache blocking, but this kernel already streams both dot-product operands.

### You mentioned AVX2 — did you verify it actually worked, and how?

Yes. I added a guarded AVX2 dot product using `__m256`, `_mm256_loadu_ps`, `_mm256_mul_ps`, and `_mm256_add_ps`, with a scalar fallback. I built an x86_64 AVX2 benchmark binary and inspected disassembly. The check found `vmulps` and `vaddps` in `build/mte_benchmark`. On Apple Silicon, I had to target x86_64 because native arm64 cannot contain AVX2.

### What did your benchmark results show about cache behavior?

The cache story is clearest as size grows. `transpose_rhs` is `2.69x` versus naive at `128x128x128`, `2.78x` at `512x512x512`, and `3.37x` at `1024x1024x1024`, so the strided RHS penalty grows with working set size. At `2048` and `4096`, I skipped naive. `transpose_rhs` moves from `6172739629.20 ns` at `2048` to `49272107562.50 ns` at `4096`, close to cubic scaling, so bandwidth and total work dominate.

### Why didn't tiling help, and does that concern you?

It does not concern me because it matches the access pattern. `tiled_transpose_rhs` is effectively tied with `transpose_rhs`: `96027616.70 ns` versus `95725933.35 ns` at `512`, and `779939433.30 ns` versus `776541785.40 ns` at `1024`. Earlier tile sweeps found only `1.0015x` at best. Output row/column tiling does not reduce much traffic once RHS is pretransposed and AVX2 streams both operands. I would need `k` blocking and register accumulation for a serious tiled GEMM.

### What is int8 quantization and why did it give you a speedup even without VNNI?

I compute `scale = max_abs / 127.0f`, clamp to `[-127, 127]`, and keep zero point at `0`. For matmul, I quantize LHS and a pretransposed RHS, accumulate int8 products into `int32_t`, then multiply by `lhs.scale * rhs.scale`. It speeds up because the data footprint is smaller. I measured `2.32x` at `256x256x256`, `2.92x` at `512x512x512`, and `3.44x` at `1024x1024x1024`. Production int8 still wants AVX-VNNI or AMX.

### How do you validate correctness when you have multiple backends with different implementations?

I validate in layers. Unit tests compare naive, transposed, threaded, and tiled matmul on known matrices. The benchmark validates optimized outputs against naive when feasible. Python generates deterministic reference model data, and C++ must match it. Quantization uses looser tolerances: dequantization error below `0.02`, and `32x64 * 64x32` int8 matmul below `0.5` max absolute error versus float.

### What would you build next and why?

I would build a real blocked GEMM that tiles the `k` dimension and accumulates partial products, because my tiled backend only changes output traversal. I would also add AVX-VNNI or AMX int8 kernels, runtime CPU feature detection, and benchmark output with bandwidth estimates and variance. Those are the next steps toward production-style inference infrastructure.

## The numbers I must know cold

| Case | Backend | Threads | Avg ns | Speedup |
| --- | --- | ---: | ---: | ---: |
| `128x128x128` | `naive` | 1 | `4068833.35` | `1.00x vs naive` |
| `128x128x128` | `transpose_rhs` | 1 | `1512945.80` | `2.69x vs naive` |
| `128x128x128` | `tiled_transpose_rhs` | 1 | `1500064.60` | `2.71x vs naive` |
| `128x128x128` | `threaded_transpose_rhs` | 1 | `1517608.35` | `2.68x vs naive` |
| `256x256x256` | `naive` | 1 | `32082735.40` | `1.00x vs naive` |
| `256x256x256` | `transpose_rhs` | 1 | `12022047.90` | `2.67x vs naive` |
| `256x256x256` | `tiled_transpose_rhs` | 1 | `11997002.05` | `2.67x vs naive` |
| `256x256x256` | `threaded_transpose_rhs` | 1 | `12017931.25` | `2.67x vs naive` |
| `512x512x512` | `naive` | 1 | `266517072.90` | `1.00x vs naive` |
| `512x512x512` | `transpose_rhs` | 1 | `95725933.35` | `2.78x vs naive` |
| `512x512x512` | `tiled_transpose_rhs` | 1 | `96027616.70` | `2.78x vs naive` |
| `512x512x512` | `threaded_transpose_rhs` | 1 | `95685639.60` | `2.79x vs naive` |
| `1024x1024x1024` | `naive` | 1 | `2618671414.60` | `1.00x vs naive` |
| `1024x1024x1024` | `transpose_rhs` | 1 | `776541785.40` | `3.37x vs naive` |
| `1024x1024x1024` | `tiled_transpose_rhs` | 1 | `779939433.30` | `3.36x vs naive` |
| `1024x1024x1024` | `threaded_transpose_rhs` | 1 | `776688218.75` | `3.37x vs naive` |
| `2048x2048x2048` | `transpose_rhs` | 1 | `6172739629.20` | `n/a` |
| `2048x2048x2048` | `threaded_transpose_rhs` | 1 | `6194787968.75` | `n/a` |
| `4096x4096x4096` | `transpose_rhs` | 1 | `49272107562.50` | `n/a` |
| `4096x4096x4096` | `threaded_transpose_rhs` | 1 | `49469847595.85` | `n/a` |
| `256x256x256` | `int8_dequantized` | 1 | `5174472.95` | `2.32x vs float transpose_rhs` |
| `512x512x512` | `int8_dequantized` | 1 | `32833693.75` | `2.92x vs float transpose_rhs` |
| `1024x1024x1024` | `int8_dequantized` | 1 | `227291441.70` | `3.44x vs float transpose_rhs` |

## The one thing that surprised me

The most surprising result was that tiling did almost nothing while scalar int8 produced a meaningful speedup. I expected tiling to be the obvious systems win, but once RHS was pretransposed and AVX2 streamed both operands, simple output tiling had almost no room to help. Int8 taught me the opposite lesson: reducing data width alone reached `3.44x` at `1024x1024x1024`, which made the memory-pressure story concrete.
