#include "mte/io.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace mte {

Tensor LoadTensorFromTextFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open tensor file: " + path.string());
    }

    std::size_t rows = 0;
    std::size_t cols = 0;
    input >> rows >> cols;
    if (!input || rows == 0 || cols == 0) {
        throw std::runtime_error("invalid tensor header in file: " + path.string());
    }

    std::vector<float> values;
    values.reserve(rows * cols);

    float value = 0.0F;
    while (input >> value) {
        values.push_back(value);
    }

    if (values.size() != rows * cols) {
        throw std::runtime_error(
            "tensor value count does not match shape in file: " + path.string());
    }

    return Tensor({rows, cols}, std::move(values));
}

void SaveTensorToTextFile(const Tensor& tensor, const std::filesystem::path& path) {
    if (tensor.rank() != 2) {
        throw std::invalid_argument("SaveTensorToTextFile currently supports only rank-2 tensors");
    }

    std::filesystem::create_directories(path.parent_path());

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open tensor file for writing: " + path.string());
    }

    output << tensor.rows() << ' ' << tensor.cols() << '\n';
    output << std::fixed << std::setprecision(8);
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        if (i > 0) {
            output << ' ';
        }
        output << tensor.data()[i];
    }
    output << '\n';
}

ReferenceCase LoadReferenceCase(const std::filesystem::path& directory) {
    return ReferenceCase{
        LoadTensorFromTextFile(directory / "input.txt"),
        LoadTensorFromTextFile(directory / "output.txt"),
    };
}

}  // namespace mte
