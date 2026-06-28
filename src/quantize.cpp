#include "mte/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mte {

namespace {

std::size_t ComputeSize(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("quantized tensor shape must not be empty");
    }
    std::size_t size = 1;
    for (std::size_t dimension : shape) {
        size *= dimension;
    }
    return size;
}

void ValidateQuantizedTensor(const QuantizedTensor& input) {
    if (input.shape.size() != 2) {
        throw std::invalid_argument("quantized tensor must be rank-2");
    }
    if (input.data.size() != ComputeSize(input.shape)) {
        throw std::invalid_argument("quantized tensor data size does not match shape");
    }
    if (input.scale <= 0.0F) {
        throw std::invalid_argument("quantized tensor scale must be positive");
    }
    if (input.zero_point != 0) {
        throw std::invalid_argument("only symmetric int8 quantization is supported");
    }
}

}  // namespace

QuantizedTensor QuantizeSymmetric(const Tensor& input) {
    float max_abs = 0.0F;
    for (float value : input.data()) {
        max_abs = std::max(max_abs, std::fabs(value));
    }

    const float scale = max_abs == 0.0F ? 1.0F : max_abs / 127.0F;
    QuantizedTensor output{input.shape(), {}, scale, 0};
    output.data.reserve(input.size());

    for (float value : input.data()) {
        const int rounded = static_cast<int>(std::nearbyint(value / scale));
        const int clamped = std::clamp(rounded, -127, 127);
        output.data.push_back(static_cast<std::int8_t>(clamped));
    }

    return output;
}

Tensor DequantizeTensor(const QuantizedTensor& input) {
    ValidateQuantizedTensor(input);

    std::vector<float> values;
    values.reserve(input.data.size());
    for (std::int8_t value : input.data) {
        values.push_back(static_cast<float>(value) * input.scale);
    }
    return Tensor(input.shape, std::move(values));
}

Tensor MatMulInt8Dequantized(const QuantizedTensor& lhs, const QuantizedTensor& rhs_transposed) {
    ValidateQuantizedTensor(lhs);
    ValidateQuantizedTensor(rhs_transposed);

    const std::size_t rows = lhs.shape[0];
    const std::size_t inner = lhs.shape[1];
    const std::size_t cols = rhs_transposed.shape[0];

    if (rhs_transposed.shape[1] != inner) {
        throw std::invalid_argument("pretransposed quantized rhs shape mismatch");
    }

    Tensor output({rows, cols});
    const float output_scale = lhs.scale * rhs_transposed.scale;

    for (std::size_t row = 0; row < rows; ++row) {
        const std::int8_t* lhs_row = lhs.data.data() + row * inner;
        for (std::size_t col = 0; col < cols; ++col) {
            // rhs_transposed must be row-major with shape [rhs_cols, inner], matching
            // the pre-transposed float layout used by MatMulWithPretransposedRhs.
            const std::int8_t* rhs_row = rhs_transposed.data.data() + col * inner;
            std::int32_t accumulator = 0;
            for (std::size_t k = 0; k < inner; ++k) {
                accumulator += static_cast<std::int32_t>(lhs_row[k]) *
                               static_cast<std::int32_t>(rhs_row[k]);
            }
            output.data()[row * cols + col] =
                static_cast<float>(accumulator) * output_scale;
        }
    }

    return output;
}

}  // namespace mte
