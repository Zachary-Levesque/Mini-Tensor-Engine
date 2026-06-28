#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mte/tensor.hpp"

namespace mte {

struct QuantizedTensor {
    std::vector<std::size_t> shape;
    std::vector<std::int8_t> data;
    float scale = 1.0F;
    std::int32_t zero_point = 0;
};

QuantizedTensor QuantizeSymmetric(const Tensor& input);
Tensor DequantizeTensor(const QuantizedTensor& input);
Tensor MatMulInt8Dequantized(const QuantizedTensor& lhs, const QuantizedTensor& rhs_transposed);

}  // namespace mte
