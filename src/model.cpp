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

TwoLayerPerceptron::TwoLayerPerceptron(
    Tensor w1,
    Tensor b1,
    Tensor w2,
    Tensor b2,
    MatMulBackend backend,
    std::size_t num_threads)
    : w1_(std::move(w1)),
      b1_(std::move(b1)),
      w2_(std::move(w2)),
      b2_(std::move(b2)),
      backend_(backend),
      num_threads_(std::max<std::size_t>(1, num_threads)) {
    ValidateLinearLayer(w1_, b1_, "layer1");
    ValidateLinearLayer(w2_, b2_, "layer2");

    if (w1_.cols() != w2_.rows()) {
        throw std::invalid_argument("layer1 output width must match layer2 input width");
    }

    if (backend_ == MatMulBackend::kTransposeRhs ||
        backend_ == MatMulBackend::kThreadedTransposeRhs) {
        w1_transposed_ = Transpose(w1_);
        w2_transposed_ = Transpose(w2_);
    }
}

Tensor TwoLayerPerceptron::Forward(const Tensor& input) const {
    if (input.rank() != 2) {
        throw std::invalid_argument("input must be rank-2");
    }
    if (input.cols() != w1_.rows()) {
        throw std::invalid_argument("input width does not match first layer input width");
    }

    const Tensor hidden = ReLU(ApplyFirstLinear(input));
    return Softmax(ApplySecondLinear(hidden));
}

MatMulBackend TwoLayerPerceptron::backend() const noexcept {
    return backend_;
}

std::size_t TwoLayerPerceptron::num_threads() const noexcept {
    return num_threads_;
}

Tensor TwoLayerPerceptron::ApplyFirstLinear(const Tensor& input) const {
    if (backend_ == MatMulBackend::kTransposeRhs ||
        backend_ == MatMulBackend::kThreadedTransposeRhs) {
        return AddBias(MatMulWithPretransposedRhs(input, w1_transposed_, num_threads_), b1_);
    }
    return Linear(input, w1_, b1_, backend_, num_threads_);
}

Tensor TwoLayerPerceptron::ApplySecondLinear(const Tensor& input) const {
    if (backend_ == MatMulBackend::kTransposeRhs ||
        backend_ == MatMulBackend::kThreadedTransposeRhs) {
        return AddBias(MatMulWithPretransposedRhs(input, w2_transposed_, num_threads_), b2_);
    }
    return Linear(input, w2_, b2_, backend_, num_threads_);
}

TwoLayerPerceptron TwoLayerPerceptron::LoadFromDirectory(
    const std::filesystem::path& directory,
    MatMulBackend backend,
    std::size_t num_threads) {
    return TwoLayerPerceptron(
        LoadTensorFromTextFile(directory / "w1.txt"),
        LoadTensorFromTextFile(directory / "b1.txt"),
        LoadTensorFromTextFile(directory / "w2.txt"),
        LoadTensorFromTextFile(directory / "b2.txt"),
        backend,
        num_threads);
}

}  // namespace mte
