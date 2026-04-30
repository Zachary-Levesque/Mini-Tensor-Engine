#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "mte/io.hpp"
#include "mte/layers.hpp"

namespace {

bool AlmostEqual(float lhs, float rhs, float tolerance = 1e-5F) {
    return std::fabs(lhs - rhs) <= tolerance;
}

void PrintTensor(const mte::Tensor& tensor, const std::string& name) {
    std::cout << name << " (" << tensor.shape()[0] << "x" << tensor.shape()[1] << ")\n";
    for (std::size_t row = 0; row < tensor.shape()[0]; ++row) {
        for (std::size_t col = 0; col < tensor.shape()[1]; ++col) {
            std::cout << std::fixed << std::setprecision(6) << tensor.at(row, col) << ' ';
        }
        std::cout << '\n';
    }
}

}  // namespace

int main() {
    try {
        const auto input = mte::LoadTensorFromTextFile("data/reference/input.txt");
        const auto w1 = mte::LoadTensorFromTextFile("data/reference/w1.txt");
        const auto b1 = mte::LoadTensorFromTextFile("data/reference/b1.txt");
        const auto w2 = mte::LoadTensorFromTextFile("data/reference/w2.txt");
        const auto b2 = mte::LoadTensorFromTextFile("data/reference/b2.txt");
        const auto expected = mte::LoadTensorFromTextFile("data/reference/output.txt");

        const auto hidden = mte::ReLU(mte::Linear(input, w1, b1));
        const auto output = mte::Softmax(mte::Linear(hidden, w2, b2));

        PrintTensor(output, "C++ output");

        bool matches = true;
        for (std::size_t i = 0; i < output.size(); ++i) {
            if (!AlmostEqual(output.data()[i], expected.data()[i])) {
                matches = false;
                std::cerr << "Mismatch at index " << i << ": got " << output.data()[i]
                          << ", expected " << expected.data()[i] << '\n';
            }
        }

        if (!matches) {
            return 1;
        }

        std::cout << "Validation passed against Python reference output.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << '\n';
        return 1;
    }
}
