#include "mte/model.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

std::vector<std::string> SplitTokens(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<LayerDefinition> LoadLayerDefinitionsFromManifest(
    const std::filesystem::path& directory) {
    std::ifstream manifest(directory / "model.txt");
    if (!manifest) {
        throw std::runtime_error("failed to open model manifest: " + (directory / "model.txt").string());
    }

    std::vector<LayerDefinition> layers;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(manifest, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const std::vector<std::string> tokens = SplitTokens(line);
        if (tokens.empty()) {
            continue;
        }

        if (tokens[0] == "linear") {
            if (tokens.size() != 3) {
                throw std::runtime_error("linear manifest entry must be: linear <weights> <bias>");
            }
            layers.push_back(LayerDefinition{
                LayerType::kLinear,
                LoadTensorFromTextFile(directory / tokens[1]),
                LoadTensorFromTextFile(directory / tokens[2]),
                {},
            });
            continue;
        }
        if (tokens[0] == "relu") {
            if (tokens.size() != 1) {
                throw std::runtime_error("relu manifest entry must not take arguments");
            }
            layers.push_back(LayerDefinition{LayerType::kReLU, {}, {}, {}});
            continue;
        }
        if (tokens[0] == "softmax") {
            if (tokens.size() != 1) {
                throw std::runtime_error("softmax manifest entry must not take arguments");
            }
            layers.push_back(LayerDefinition{LayerType::kSoftmax, {}, {}, {}});
            continue;
        }

        throw std::runtime_error(
            "unknown layer type in model manifest at line " + std::to_string(line_number));
    }

    if (layers.empty()) {
        throw std::runtime_error("model manifest must contain at least one layer");
    }

    return layers;
}

std::vector<LayerDefinition> MakeLegacyTwoLayerManifest(const std::filesystem::path& directory) {
    return {
        {
            LayerType::kLinear,
            LoadTensorFromTextFile(directory / "w1.txt"),
            LoadTensorFromTextFile(directory / "b1.txt"),
            {},
        },
        {LayerType::kReLU, {}, {}, {}},
        {
            LayerType::kLinear,
            LoadTensorFromTextFile(directory / "w2.txt"),
            LoadTensorFromTextFile(directory / "b2.txt"),
            {},
        },
        {LayerType::kSoftmax, {}, {}, {}},
    };
}

}  // namespace

FeedForwardModel::FeedForwardModel(
    std::vector<LayerDefinition> layers,
    MatMulBackend backend,
    std::size_t num_threads)
    : layers_(std::move(layers)),
      backend_(backend),
      num_threads_(std::max<std::size_t>(1, num_threads)),
      input_width_(0) {
    if (layers_.empty()) {
        throw std::invalid_argument("model must contain at least one layer");
    }

    bool seen_linear = false;
    std::size_t current_width = 0;

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        LayerDefinition& layer = layers_[i];
        if (layer.type == LayerType::kLinear) {
            ValidateLinearLayer(layer.weights, layer.bias, "linear layer");
            if (!seen_linear) {
                input_width_ = layer.weights.rows();
                current_width = layer.weights.cols();
                seen_linear = true;
            } else {
                if (layer.weights.rows() != current_width) {
                    throw std::invalid_argument("linear layer input width does not match previous layer output width");
                }
                current_width = layer.weights.cols();
            }

            if (backend_ == MatMulBackend::kTransposeRhs ||
                backend_ == MatMulBackend::kThreadedTransposeRhs) {
                layer.weights_transposed = Transpose(layer.weights);
            }
            continue;
        }

        if (!seen_linear) {
            throw std::invalid_argument("model must start with a linear layer");
        }
    }

    if (!seen_linear) {
        throw std::invalid_argument("model must contain at least one linear layer");
    }
}

Tensor FeedForwardModel::Forward(const Tensor& input) const {
    if (input.rank() != 2) {
        throw std::invalid_argument("input must be rank-2");
    }
    if (input.cols() != input_width_) {
        throw std::invalid_argument("input width does not match model input width");
    }

    Tensor activations = input;
    for (const LayerDefinition& layer : layers_) {
        switch (layer.type) {
            case LayerType::kLinear:
                activations = ApplyLinear(activations, layer);
                break;
            case LayerType::kReLU:
                activations = ReLU(activations);
                break;
            case LayerType::kSoftmax:
                activations = Softmax(activations);
                break;
        }
    }
    return activations;
}

MatMulBackend FeedForwardModel::backend() const noexcept {
    return backend_;
}

std::size_t FeedForwardModel::num_threads() const noexcept {
    return num_threads_;
}

std::size_t FeedForwardModel::input_width() const noexcept {
    return input_width_;
}

std::size_t FeedForwardModel::num_layers() const noexcept {
    return layers_.size();
}

FeedForwardModel FeedForwardModel::LoadFromDirectory(
    const std::filesystem::path& directory,
    MatMulBackend backend,
    std::size_t num_threads) {
    const std::filesystem::path manifest_path = directory / "model.txt";
    if (std::filesystem::exists(manifest_path)) {
        return FeedForwardModel(LoadLayerDefinitionsFromManifest(directory), backend, num_threads);
    }
    return FeedForwardModel(MakeLegacyTwoLayerManifest(directory), backend, num_threads);
}

FeedForwardModel FeedForwardModel::MakeTwoLayerPerceptron(
    Tensor w1,
    Tensor b1,
    Tensor w2,
    Tensor b2,
    MatMulBackend backend,
    std::size_t num_threads) {
    return FeedForwardModel(
        {
            {LayerType::kLinear, std::move(w1), std::move(b1), {}},
            {LayerType::kReLU, {}, {}, {}},
            {LayerType::kLinear, std::move(w2), std::move(b2), {}},
            {LayerType::kSoftmax, {}, {}, {}},
        },
        backend,
        num_threads);
}

Tensor FeedForwardModel::ApplyLinear(const Tensor& input, const LayerDefinition& layer) const {
    if (backend_ == MatMulBackend::kTransposeRhs ||
        backend_ == MatMulBackend::kThreadedTransposeRhs) {
        return AddBias(MatMulWithPretransposedRhs(input, layer.weights_transposed, num_threads_), layer.bias);
    }
    return Linear(input, layer.weights, layer.bias, backend_, num_threads_);
}

}  // namespace mte
