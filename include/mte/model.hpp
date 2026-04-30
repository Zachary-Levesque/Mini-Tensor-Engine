#pragma once

#include <filesystem>
#include <vector>

#include "mte/tensor.hpp"

namespace mte {

enum class LayerType {
    kLinear,
    kReLU,
    kSigmoid,
    kTanh,
    kSoftmax,
};

struct LayerDefinition {
    LayerType type;
    Tensor weights;
    Tensor bias;
    Tensor weights_transposed;
};

class FeedForwardModel {
public:
    FeedForwardModel(
        std::vector<LayerDefinition> layers,
        MatMulBackend backend = MatMulBackend::kTransposeRhs,
        std::size_t num_threads = 1);

    [[nodiscard]] Tensor Forward(const Tensor& input) const;
    [[nodiscard]] MatMulBackend backend() const noexcept;
    [[nodiscard]] std::size_t num_threads() const noexcept;
    [[nodiscard]] std::size_t input_width() const noexcept;
    [[nodiscard]] std::size_t num_layers() const noexcept;

    [[nodiscard]] static FeedForwardModel LoadFromDirectory(
        const std::filesystem::path& directory,
        MatMulBackend backend = MatMulBackend::kTransposeRhs,
        std::size_t num_threads = 1);

    [[nodiscard]] static FeedForwardModel MakeTwoLayerPerceptron(
        Tensor w1,
        Tensor b1,
        Tensor w2,
        Tensor b2,
        MatMulBackend backend = MatMulBackend::kTransposeRhs,
        std::size_t num_threads = 1);

private:
    [[nodiscard]] Tensor ApplyLinear(const Tensor& input, const LayerDefinition& layer) const;

    std::vector<LayerDefinition> layers_;
    MatMulBackend backend_;
    std::size_t num_threads_;
    std::size_t input_width_;
};

}  // namespace mte
