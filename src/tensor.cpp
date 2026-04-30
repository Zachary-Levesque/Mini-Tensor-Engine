#include "mte/tensor.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace mte {

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), data_(ComputeSize(shape_), 0.0F) {}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> values)
    : shape_(std::move(shape)), data_(std::move(values)) {
    if (data_.size() != ComputeSize(shape_)) {
        throw std::invalid_argument("tensor data size does not match shape");
    }
}

Tensor::Tensor(std::initializer_list<std::size_t> shape, std::vector<float> values)
    : Tensor(std::vector<std::size_t>(shape), std::move(values)) {}

const std::vector<std::size_t>& Tensor::shape() const noexcept {
    return shape_;
}

std::size_t Tensor::rank() const noexcept {
    return shape_.size();
}

std::size_t Tensor::size() const noexcept {
    return data_.size();
}

bool Tensor::empty() const noexcept {
    return data_.empty();
}

std::size_t Tensor::rows() const {
    if (rank() != 2) {
        throw std::logic_error("rows() requires a rank-2 tensor");
    }
    return shape_[0];
}

std::size_t Tensor::cols() const {
    if (rank() != 2) {
        throw std::logic_error("cols() requires a rank-2 tensor");
    }
    return shape_[1];
}

float& Tensor::at(std::size_t row, std::size_t col) {
    return data_.at(FlattenIndex(row, col));
}

const float& Tensor::at(std::size_t row, std::size_t col) const {
    return data_.at(FlattenIndex(row, col));
}

std::vector<float>& Tensor::data() noexcept {
    return data_;
}

const std::vector<float>& Tensor::data() const noexcept {
    return data_;
}

std::string Tensor::DebugString() const {
    std::ostringstream stream;
    stream << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        stream << shape_[i];
        if (i + 1 < shape_.size()) {
            stream << ", ";
        }
    }
    stream << "], values=[";
    for (std::size_t i = 0; i < data_.size(); ++i) {
        stream << data_[i];
        if (i + 1 < data_.size()) {
            stream << ", ";
        }
    }
    stream << "])";
    return stream.str();
}

std::size_t Tensor::FlattenIndex(std::size_t row, std::size_t col) const {
    if (rank() != 2) {
        throw std::logic_error("2D indexing requires rank-2 tensor");
    }
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("tensor index out of range");
    }
    return row * shape_[1] + col;
}

std::size_t Tensor::ComputeSize(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("tensor shape must not be empty");
    }
    return std::accumulate(
        shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

Tensor MatMul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument("MatMul currently supports only rank-2 tensors");
    }
    if (lhs.cols() != rhs.rows()) {
        throw std::invalid_argument("MatMul shape mismatch");
    }

    const std::size_t rows = lhs.rows();
    const std::size_t inner = lhs.cols();
    const std::size_t cols = rhs.cols();

    Tensor output({rows, cols});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < inner; ++k) {
                sum += lhs.at(row, k) * rhs.at(k, col);
            }
            output.at(row, col) = sum;
        }
    }
    return output;
}

Tensor AddBias(const Tensor& input, const Tensor& bias) {
    if (input.rank() != 2 || bias.rank() != 2) {
        throw std::invalid_argument("AddBias currently supports only rank-2 tensors");
    }
    if (bias.rows() != 1 || bias.cols() != input.cols()) {
        throw std::invalid_argument("bias shape must be [1, input_cols]");
    }

    Tensor output(input.shape(), input.data());
    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t col = 0; col < input.cols(); ++col) {
            output.at(row, col) += bias.at(0, col);
        }
    }
    return output;
}

bool HasSameShape(const Tensor& lhs, const Tensor& rhs) noexcept {
    return lhs.shape() == rhs.shape();
}

}  // namespace mte
