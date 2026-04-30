#pragma once

#include "mte/tensor.hpp"

namespace mte {

Tensor Linear(const Tensor& input, const Tensor& weights, const Tensor& bias);
Tensor Linear(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    MatMulBackend backend);
Tensor Linear(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    MatMulBackend backend,
    std::size_t num_threads);
Tensor ReLU(const Tensor& input);
Tensor Softmax(const Tensor& input);

}  // namespace mte
