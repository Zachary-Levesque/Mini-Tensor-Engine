#include "mte/io.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace mte {

Tensor LoadTensorFromTextFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open tensor file: " + path);
    }

    std::size_t rows = 0;
    std::size_t cols = 0;
    input >> rows >> cols;
    if (!input || rows == 0 || cols == 0) {
        throw std::runtime_error("invalid tensor header in file: " + path);
    }

    std::vector<float> values;
    values.reserve(rows * cols);

    float value = 0.0F;
    while (input >> value) {
        values.push_back(value);
    }

    if (values.size() != rows * cols) {
        throw std::runtime_error("tensor value count does not match shape in file: " + path);
    }

    return Tensor({rows, cols}, std::move(values));
}

}  // namespace mte
