#include <cmath>
#include <iostream>
#include <stdexcept>

#include "mte/layers.hpp"

namespace {

bool AlmostEqual(float lhs, float rhs, float tolerance = 1e-5F) {
    return std::fabs(lhs - rhs) <= tolerance;
}

void Expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        mte::Tensor lhs({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
        mte::Tensor rhs({3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
        mte::Tensor product = mte::MatMul(lhs, rhs);

        Expect(product.shape()[0] == 2 && product.shape()[1] == 2, "matmul shape mismatch");
        Expect(AlmostEqual(product.at(0, 0), 58.0F), "matmul value mismatch (0,0)");
        Expect(AlmostEqual(product.at(0, 1), 64.0F), "matmul value mismatch (0,1)");
        Expect(AlmostEqual(product.at(1, 0), 139.0F), "matmul value mismatch (1,0)");
        Expect(AlmostEqual(product.at(1, 1), 154.0F), "matmul value mismatch (1,1)");

        mte::Tensor relu_input({1, 4}, {-1.0F, 0.0F, 2.5F, -3.0F});
        mte::Tensor relu_output = mte::ReLU(relu_input);
        Expect(AlmostEqual(relu_output.at(0, 0), 0.0F), "relu mismatch at 0");
        Expect(AlmostEqual(relu_output.at(0, 2), 2.5F), "relu mismatch at 2");

        mte::Tensor softmax_input({1, 3}, {1.0F, 2.0F, 3.0F});
        mte::Tensor softmax_output = mte::Softmax(softmax_input);
        float sum = 0.0F;
        for (float value : softmax_output.data()) {
            sum += value;
        }
        Expect(AlmostEqual(sum, 1.0F), "softmax outputs must sum to 1");

        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
