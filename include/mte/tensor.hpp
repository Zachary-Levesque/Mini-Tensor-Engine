#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace mte {

enum class MatMulBackend {
    kNaive,
    kTransposeRhs,
};

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);
    Tensor(std::vector<std::size_t> shape, std::vector<float> values);
    Tensor(std::initializer_list<std::size_t> shape, std::vector<float> values);

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t rows() const;
    [[nodiscard]] std::size_t cols() const;

    [[nodiscard]] float& at(std::size_t row, std::size_t col);
    [[nodiscard]] const float& at(std::size_t row, std::size_t col) const;

    [[nodiscard]] std::vector<float>& data() noexcept;
    [[nodiscard]] const std::vector<float>& data() const noexcept;

    [[nodiscard]] std::string DebugString() const;

private:
    std::size_t FlattenIndex(std::size_t row, std::size_t col) const;
    static std::size_t ComputeSize(const std::vector<std::size_t>& shape);

    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

Tensor MatMul(const Tensor& lhs, const Tensor& rhs);
Tensor MatMul(const Tensor& lhs, const Tensor& rhs, MatMulBackend backend);
Tensor Transpose(const Tensor& input);
Tensor MatMulWithPretransposedRhs(const Tensor& lhs, const Tensor& rhs_transposed);
Tensor AddBias(const Tensor& input, const Tensor& bias);
bool HasSameShape(const Tensor& lhs, const Tensor& rhs) noexcept;
const char* MatMulBackendName(MatMulBackend backend) noexcept;

}  // namespace mte
