#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "mte/io.hpp"
#include "mte/model.hpp"

namespace {

bool AlmostEqual(float lhs, float rhs, float tolerance = 1e-5F) {
    return std::fabs(lhs - rhs) <= tolerance;
}

struct Options {
    std::filesystem::path data_dir = "data/reference";
};

Options ParseArgs(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--data-dir") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--data-dir requires a value");
            }
            options.data_dir = argv[++i];
            continue;
        }
        throw std::invalid_argument("unknown argument: " + std::string(arg));
    }

    return options;
}

void PrintTensor(const mte::Tensor& tensor, const std::string& name) {
    std::cout << name << " (" << tensor.rows() << "x" << tensor.cols() << ")\n";
    for (std::size_t row = 0; row < tensor.rows(); ++row) {
        for (std::size_t col = 0; col < tensor.cols(); ++col) {
            std::cout << std::fixed << std::setprecision(6) << tensor.at(row, col) << ' ';
        }
        std::cout << '\n';
    }
}

bool MatchesWithinTolerance(
    const mte::Tensor& actual,
    const mte::Tensor& expected,
    float tolerance = 1e-5F) {
    if (!mte::HasSameShape(actual, expected)) {
        std::cerr << "Shape mismatch: got " << actual.DebugString()
                  << ", expected " << expected.DebugString() << '\n';
        return false;
    }

    bool matches = true;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!AlmostEqual(actual.data()[i], expected.data()[i], tolerance)) {
            matches = false;
            std::cerr << "Mismatch at index " << i << ": got " << actual.data()[i]
                      << ", expected " << expected.data()[i] << '\n';
        }
    }
    return matches;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        const mte::ReferenceCase reference_case = mte::LoadReferenceCase(options.data_dir);
        const mte::TwoLayerPerceptron model =
            mte::TwoLayerPerceptron::LoadFromDirectory(options.data_dir);
        const mte::Tensor output = model.Forward(reference_case.input);

        PrintTensor(output, "C++ output");

        if (!MatchesWithinTolerance(output, reference_case.expected_output)) {
            return 1;
        }

        std::cout << "Validation passed against Python reference output.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << '\n';
        return 1;
    }
}
