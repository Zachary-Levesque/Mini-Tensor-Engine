#include "mte/layers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mte {

Tensor Linear(const Tensor& input, const Tensor& weights, const Tensor& bias) {
    return Linear(input, weights, bias, MatMulBackend::kTransposeRhs);
}

Tensor Linear(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    MatMulBackend backend) {
    return Linear(input, weights, bias, backend, 1);
}

Tensor Linear(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    MatMulBackend backend,
    std::size_t num_threads) {
    return AddBias(MatMul(input, weights, backend, num_threads), bias);
}

Tensor ReLU(const Tensor& input) {
    Tensor output(input.shape(), input.data());
    for (float& value : output.data()) {
        value = std::max(0.0F, value);
    }
    return output;
}

Tensor Sigmoid(const Tensor& input) {
    Tensor output(input.shape(), input.data());
    for (float& value : output.data()) {
        value = 1.0F / (1.0F + std::exp(-value));
    }
    return output;
}

Tensor Tanh(const Tensor& input) {
    Tensor output(input.shape(), input.data());
    for (float& value : output.data()) {
        value = std::tanh(value);
    }
    return output;
}

Tensor Softmax(const Tensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument("Softmax currently supports only rank-2 tensors");
    }

    Tensor output(input.shape(), input.data());
    const std::size_t rows = input.rows();
    const std::size_t cols = input.cols();

    for (std::size_t row = 0; row < rows; ++row) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t col = 0; col < cols; ++col) {
            row_max = std::max(row_max, input.at(row, col));
        }

        float sum = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) {
            const float exp_value = std::exp(input.at(row, col) - row_max);
            output.at(row, col) = exp_value;
            sum += exp_value;
        }

        for (std::size_t col = 0; col < cols; ++col) {
            output.at(row, col) /= sum;
        }
    }

    return output;
}

}  // namespace mte
