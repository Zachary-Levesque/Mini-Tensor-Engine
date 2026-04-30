#pragma once

#include <filesystem>

#include "mte/tensor.hpp"

namespace mte {

class TwoLayerPerceptron {
public:
    TwoLayerPerceptron(
        Tensor w1,
        Tensor b1,
        Tensor w2,
        Tensor b2,
        MatMulBackend backend = MatMulBackend::kTransposeRhs,
        std::size_t num_threads = 1);

    [[nodiscard]] Tensor Forward(const Tensor& input) const;
    [[nodiscard]] MatMulBackend backend() const noexcept;
    [[nodiscard]] std::size_t num_threads() const noexcept;

    [[nodiscard]] static TwoLayerPerceptron LoadFromDirectory(
        const std::filesystem::path& directory,
        MatMulBackend backend = MatMulBackend::kTransposeRhs,
        std::size_t num_threads = 1);

private:
    [[nodiscard]] Tensor ApplyFirstLinear(const Tensor& input) const;
    [[nodiscard]] Tensor ApplySecondLinear(const Tensor& input) const;

    Tensor w1_;
    Tensor b1_;
    Tensor w2_;
    Tensor b2_;
    Tensor w1_transposed_;
    Tensor w2_transposed_;
    MatMulBackend backend_;
    std::size_t num_threads_;
};

}  // namespace mte
