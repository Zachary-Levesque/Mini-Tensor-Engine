#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string_view>

#include "mte/io.hpp"
#include "mte/model.hpp"

namespace {

struct Options {
    std::filesystem::path data_dir = "data/reference";
    std::size_t iterations = 100000;
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
        if (arg == "--iterations") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--iterations requires a value");
            }
            options.iterations = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        throw std::invalid_argument("unknown argument: " + std::string(arg));
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        const mte::ReferenceCase reference_case = mte::LoadReferenceCase(options.data_dir);
        const mte::TwoLayerPerceptron model =
            mte::TwoLayerPerceptron::LoadFromDirectory(options.data_dir);

        volatile float sink = 0.0F;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < options.iterations; ++i) {
            const mte::Tensor output = model.Forward(reference_case.input);
            sink += output.data()[0];
        }
        const auto end = std::chrono::steady_clock::now();

        const auto total_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        const double avg_ns = static_cast<double>(total_ns) / options.iterations;

        std::cout << "Iterations: " << options.iterations << '\n';
        std::cout << "Total time (ms): " << std::fixed << std::setprecision(3)
                  << static_cast<double>(total_ns) / 1'000'000.0 << '\n';
        std::cout << "Average time per inference (ns): " << std::fixed << std::setprecision(2)
                  << avg_ns << '\n';
        std::cout << "Sink: " << sink << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
