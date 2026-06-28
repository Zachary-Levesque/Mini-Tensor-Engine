# Interview Preparation Summary

## The two-minute pitch

I built Mini Tensor Engine to understand CPU inference below the framework layer. It is a C++ inference runtime with a rank-2 row-major tensor type, multiple matrix multiplication backends, manifest-driven feed-forward layers, AVX2 dot-product code, extended cache-scale benchmarks, a tiled backend, and a scalar int8 quantized path. I built it because using PyTorch tells me what result a model produces, but it does not force me to confront why one memory layout turns into strided cache-line reads, why transposing weights can matter more than changing the math, or why correctness gates need to come before performance claims.

The key design decision was to keep the tensor representation simple enough that I could reason about every address touched by matmul. The naive backend is my baseline. The `transpose_rhs` backend changes RHS access from column-wise striding to contiguous dot-product reads. The AVX2 path makes that dot product explicit and I verified the binary contained `vmulps` and `vaddps`. I then extended benchmarks through `4096x4096x4096`, added tiling, and added int8 quantization. The headline result is that at `1024x1024x1024`, `transpose_rhs` runs in `776541785.40 ns` versus `2618671414.60 ns` for naive, a `3.37x` improvement. The int8 path runs `1024x1024x1024` in `227291441.70 ns` versus `781326560.40 ns` for float `transpose_rhs`, a `3.44x` speedup. The result I would talk about in an interview is not just the speedup; it is that each number maps to a specific memory-access decision.

## The technical deep dives

### How does your tensor class work and why did you lay it out that way?

If an interviewer asks me this, I would say: I intentionally kept `Tensor` rank-2 and row-major. Internally, it stores shape metadata and a contiguous `std::vector<float>`, so element access is `row * cols + col`. That is less general than a production tensor library with arbitrary rank, strides, views, and broadcasting, but it makes the memory model completely explicit. For this project, that was the right tradeoff because the main goal was to study matmul kernels, not to build a full NumPy replacement. With a simple row-major layout, I can explain exactly why reading `rhs.at(k, col)` in the naive backend jumps by `rhs.cols()` floats, and why reading a pretransposed RHS row lets the innermost loop stream contiguous memory.

### Walk me through your matmul backends — what does each one do at the memory level?

My answer would be: the naive backend uses the classic triple loop. For each output element, it walks across one LHS row contiguously, but it walks down one RHS column in row-major storage, so every `k` step jumps through memory. `transpose_rhs` first transforms RHS from `[inner, cols]` to `[cols, inner]`. After that, each output dot product reads `lhs_row[k]` and `rhs_transposed_row[k]`, both contiguous. `threaded_transpose_rhs` keeps the same access pattern and splits output rows across threads, so there are no overlapping writes. `tiled_transpose_rhs` transposes RHS once, then visits output cells by row and column blocks; the idea is to improve reuse of nearby rows and columns, although measured results show it barely changes this particular kernel because the pretransposed dot product is already streaming.

### You mentioned AVX2 — did you verify it actually worked, and how?

I did not rely on source code inspection alone. I added a guarded AVX2 dot product using `__m256`, `_mm256_loadu_ps`, `_mm256_mul_ps`, and `_mm256_add_ps`, with a scalar fallback for non-AVX2 builds. Then I rebuilt the benchmark binary for an x86_64 AVX2 target and inspected the disassembly. The check found `vmulps` and `vaddps` in `build/mte_benchmark`, which tells me the SIMD path was actually present in the binary. On Apple Silicon, the native target is arm64, so I had to be explicit about the x86_64 build when validating AVX2 instructions.

### What did your benchmark results show about cache behavior?

I measured sizes from `128` through `4096`. The clearest cache-locality signal is that `transpose_rhs` improves as the matrices grow: `2.69x` versus naive at `128x128x128`, `2.78x` at `512x512x512`, and `3.37x` at `1024x1024x1024`. That happens because the naive RHS access becomes increasingly costly as the working set grows, while the transposed backend keeps the innermost dot product contiguous. At `2048` and `4096`, I skipped naive because it would dominate runtime, but `transpose_rhs` moves from `6172739629.20 ns` to `49272107562.50 ns`, which is close to the expected `8x` cubic scaling. That tells me the transpose trick has removed the obvious strided-access penalty, and the remaining bottleneck is much closer to bandwidth and total work.

### Why didn't tiling help, and does that concern you?

It does not concern me because the result matches the kernel structure. The best tiling result I measured earlier was only `1.0015x` over plain transpose at `512x512x512`, and in the current results `tiled_transpose_rhs` is effectively tied with `transpose_rhs`: `96027616.70 ns` versus `95725933.35 ns` at `512`, and `779939433.30 ns` versus `776541785.40 ns` at `1024`. Tiling usually helps when it increases reuse that was otherwise being lost. In my implementation, each dot product already streams both operands contiguously after RHS transpose, so the simple row/column output tiling does not reduce the dominant traffic much. The interview value is that I can explain why an optimization did not help instead of pretending every optimization should produce a large number.

### What is int8 quantization and why did it give you a speedup even without VNNI?

I implemented symmetric int8 quantization: I find the max absolute value, set `scale = max_abs / 127.0f`, clamp values into `[-127, 127]`, and store zero point as `0`. For matmul, I quantize LHS and a pretransposed RHS, accumulate int8 products into `int32_t`, and multiply by `lhs.scale * rhs.scale` to return a float tensor. It gave speedups even without VNNI because the data footprint is smaller and the scalar integer loop does less memory traffic. The measured int8 speedups are `2.32x` at `256x256x256`, `2.92x` at `512x512x512`, and `3.44x` at `1024x1024x1024`. I would also be clear that a production int8 kernel should use AVX-VNNI or AMX; scalar int8 is educational, not the endpoint.

### How do you validate correctness when you have multiple backends with different implementations?

I validate at multiple levels. For backend equivalence, the tests compare naive, transposed, threaded, and tiled matmul outputs on known small matrices. The benchmark also validates optimized outputs against naive when naive is feasible. For model-level correctness, Python generates deterministic input, weights, biases, manifests, and expected outputs, and the C++ inference path has to match that reference within tolerance. For quantization, I use separate tolerances because quantization intentionally introduces error: dequantization max absolute error must be below `0.02`, and the `32x64 * 64x32` int8 matmul test must stay below `0.5` max absolute error versus float.

### What would you build next and why?

I would build a real blocked GEMM that tiles the `k` dimension and accumulates partial blocks, not just row/column output tiling. The current tiling backend is useful for explanation, but it does not materially reduce memory traffic. After that, I would add architecture-specific int8 dot-product kernels using AVX-VNNI where available, and keep the scalar path as a fallback. I would also add CPU feature detection so one binary can choose scalar, AVX2 float, or VNNI int8 at runtime. Finally, I would improve benchmark reporting to include bandwidth estimates and confidence intervals, because senior systems work needs repeatability, not just one timing table.

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

The most surprising result was that tiling did almost nothing while scalar int8 produced a meaningful speedup. I expected tiling to be the more obvious systems optimization, but after implementing it, the numbers made the real bottleneck clearer: once RHS is pretransposed and the AVX2 dot product streams both inputs contiguously, simple output row/column tiling does not change much. The int8 result taught me the opposite lesson from a different angle. Even without VNNI, reducing the data width was enough to get `3.44x` at `1024x1024x1024`, which tells me memory traffic and bandwidth pressure are central to this project’s performance profile. 
