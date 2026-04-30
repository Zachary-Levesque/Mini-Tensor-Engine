#pragma once

#include <filesystem>

#include "mte/tensor.hpp"

namespace mte {

class TwoLayerPerceptron {
public:
    TwoLayerPerceptron(Tensor w1, Tensor b1, Tensor w2, Tensor b2);

    [[nodiscard]] Tensor Forward(const Tensor& input) const;

    [[nodiscard]] static TwoLayerPerceptron LoadFromDirectory(
        const std::filesystem::path& directory);

private:
    Tensor w1_;
    Tensor b1_;
    Tensor w2_;
    Tensor b2_;
};

}  // namespace mte
