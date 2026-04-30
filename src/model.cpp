#include "mte/model.hpp"

#include <stdexcept>

#include "mte/io.hpp"
#include "mte/layers.hpp"

namespace mte {

namespace {

void ValidateLinearLayer(const Tensor& weights, const Tensor& bias, const char* layer_name) {
    if (weights.rank() != 2 || bias.rank() != 2) {
        throw std::invalid_argument(std::string(layer_name) + " parameters must be rank-2");
    }
    if (bias.rows() != 1 || bias.cols() != weights.cols()) {
        throw std::invalid_argument(std::string(layer_name) + " bias width must match output width");
    }
}

}  // namespace

TwoLayerPerceptron::TwoLayerPerceptron(Tensor w1, Tensor b1, Tensor w2, Tensor b2)
    : w1_(std::move(w1)), b1_(std::move(b1)), w2_(std::move(w2)), b2_(std::move(b2)) {
    ValidateLinearLayer(w1_, b1_, "layer1");
    ValidateLinearLayer(w2_, b2_, "layer2");

    if (w1_.cols() != w2_.rows()) {
        throw std::invalid_argument("layer1 output width must match layer2 input width");
    }
}

Tensor TwoLayerPerceptron::Forward(const Tensor& input) const {
    if (input.rank() != 2) {
        throw std::invalid_argument("input must be rank-2");
    }
    if (input.cols() != w1_.rows()) {
        throw std::invalid_argument("input width does not match first layer input width");
    }

    const Tensor hidden = ReLU(Linear(input, w1_, b1_));
    return Softmax(Linear(hidden, w2_, b2_));
}

TwoLayerPerceptron TwoLayerPerceptron::LoadFromDirectory(
    const std::filesystem::path& directory) {
    return TwoLayerPerceptron(
        LoadTensorFromTextFile(directory / "w1.txt"),
        LoadTensorFromTextFile(directory / "b1.txt"),
        LoadTensorFromTextFile(directory / "w2.txt"),
        LoadTensorFromTextFile(directory / "b2.txt"));
}

}  // namespace mte
