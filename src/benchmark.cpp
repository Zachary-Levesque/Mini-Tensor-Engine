#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "mte/model.hpp"
#include "mte/tensor.hpp"

namespace {

struct MatMulCase {
    std::size_t rows;
    std::size_t inner;
    std::size_t cols;
};

struct Options {
    std::size_t iterations = 200;
    std::size_t warmup_iterations = 20;
    bool run_model_benchmark = true;
    std::vector<std::size_t> thread_counts = {1, 2, 4, 8};
};

Options ParseArgs(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--iterations") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--iterations requires a value");
            }
            options.iterations = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--warmup requires a value");
            }
            options.warmup_iterations = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--skip-model") {
            options.run_model_benchmark = false;
            continue;
        }
        if (arg == "--threads") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--threads requires a value");
            }
            options.thread_counts.clear();
            std::string raw = argv[++i];
            std::size_t start = 0;
            while (start < raw.size()) {
                const std::size_t comma = raw.find(',', start);
                const std::string token = raw.substr(start, comma - start);
                options.thread_counts.push_back(static_cast<std::size_t>(std::stoull(token)));
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
            continue;
        }
        throw std::invalid_argument("unknown argument: " + std::string(arg));
    }

    return options;
}

std::vector<float> GenerateValues(std::size_t count, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> values(count);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

[[nodiscard]] bool AlmostEqual(float lhs, float rhs, float tolerance = 1e-4F) {
    return std::fabs(lhs - rhs) <= tolerance;
}

void ValidateEquivalent(const mte::Tensor& lhs, const mte::Tensor& rhs) {
    if (!mte::HasSameShape(lhs, rhs)) {
        throw std::runtime_error("benchmark backend comparison failed: shape mismatch");
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!AlmostEqual(lhs.data()[i], rhs.data()[i])) {
            throw std::runtime_error("benchmark backend comparison failed: value mismatch");
        }
    }
}

template <typename Fn>
double MeasureAverageNanoseconds(Fn&& fn, std::size_t warmup_iterations, std::size_t iterations) {
    for (std::size_t i = 0; i < warmup_iterations; ++i) {
        fn();
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        fn();
    }
    const auto end = std::chrono::steady_clock::now();

    const auto total_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return static_cast<double>(total_ns) / iterations;
}

void RunMatMulBenchmarks(const Options& options) {
    const std::vector<MatMulCase> cases = {
        {32, 64, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
    };

    std::mt19937 generator(7);
    std::cout << "MatMul backend comparison\n";
    std::cout << "case,backend,threads,avg_ns\n";

    for (const MatMulCase& benchmark_case : cases) {
        mte::Tensor lhs(
            {benchmark_case.rows, benchmark_case.inner},
            GenerateValues(benchmark_case.rows * benchmark_case.inner, generator));
        mte::Tensor rhs(
            {benchmark_case.inner, benchmark_case.cols},
            GenerateValues(benchmark_case.inner * benchmark_case.cols, generator));

        const mte::Tensor naive_output = mte::MatMul(lhs, rhs, mte::MatMulBackend::kNaive);
        const mte::Tensor optimized_output =
            mte::MatMul(lhs, rhs, mte::MatMulBackend::kTransposeRhs);
        ValidateEquivalent(naive_output, optimized_output);

        volatile float sink = 0.0F;
        for (const mte::MatMulBackend backend :
             {mte::MatMulBackend::kNaive, mte::MatMulBackend::kTransposeRhs}) {
            const double avg_ns = MeasureAverageNanoseconds(
                [&]() {
                    const mte::Tensor output = mte::MatMul(lhs, rhs, backend);
                    sink += output.data()[0];
                },
                options.warmup_iterations,
                options.iterations);

            std::cout << benchmark_case.rows << 'x' << benchmark_case.inner << 'x'
                      << benchmark_case.cols << ',' << mte::MatMulBackendName(backend) << ','
                      << 1 << ',' << std::fixed << std::setprecision(2) << avg_ns << '\n';
        }

        for (std::size_t num_threads : options.thread_counts) {
            const double avg_ns = MeasureAverageNanoseconds(
                [&]() {
                    const mte::Tensor output = mte::MatMul(
                        lhs, rhs, mte::MatMulBackend::kThreadedTransposeRhs, num_threads);
                    sink += output.data()[0];
                },
                options.warmup_iterations,
                options.iterations);

            std::cout << benchmark_case.rows << 'x' << benchmark_case.inner << 'x'
                      << benchmark_case.cols << ','
                      << mte::MatMulBackendName(mte::MatMulBackend::kThreadedTransposeRhs)
                      << ',' << num_threads << ',' << std::fixed << std::setprecision(2)
                      << avg_ns << '\n';
        }
        std::cout << "case_sink," << sink << '\n';
    }
}

void RunModelBenchmark(const Options& options) {
    const mte::TwoLayerPerceptron naive_model(
        mte::Tensor({4, 5}, {0.2F, -0.4F, 0.1F, 0.5F, -0.3F,
                             0.7F, 0.6F, -0.2F, 0.1F, 0.8F,
                             -0.5F, 0.2F, 0.3F, -0.6F, 0.4F,
                             0.9F, -0.1F, 0.5F, 0.2F, -0.7F}),
        mte::Tensor({1, 5}, {0.1F, -0.2F, 0.05F, 0.3F, -0.4F}),
        mte::Tensor({5, 3}, {0.3F, -0.1F, 0.8F,
                             -0.6F, 0.4F, 0.2F,
                             0.5F, 0.7F, -0.3F,
                             0.1F, -0.5F, 0.9F,
                             -0.2F, 0.6F, 0.4F}),
        mte::Tensor({1, 3}, {0.05F, -0.15F, 0.25F}),
        mte::MatMulBackend::kNaive);

    const mte::TwoLayerPerceptron optimized_model(
        mte::Tensor({4, 5}, {0.2F, -0.4F, 0.1F, 0.5F, -0.3F,
                             0.7F, 0.6F, -0.2F, 0.1F, 0.8F,
                             -0.5F, 0.2F, 0.3F, -0.6F, 0.4F,
                             0.9F, -0.1F, 0.5F, 0.2F, -0.7F}),
        mte::Tensor({1, 5}, {0.1F, -0.2F, 0.05F, 0.3F, -0.4F}),
        mte::Tensor({5, 3}, {0.3F, -0.1F, 0.8F,
                             -0.6F, 0.4F, 0.2F,
                             0.5F, 0.7F, -0.3F,
                             0.1F, -0.5F, 0.9F,
                             -0.2F, 0.6F, 0.4F}),
        mte::Tensor({1, 3}, {0.05F, -0.15F, 0.25F}),
        mte::MatMulBackend::kTransposeRhs);

    std::vector<mte::TwoLayerPerceptron> models;
    models.push_back(naive_model);
    models.push_back(optimized_model);
    for (std::size_t num_threads : options.thread_counts) {
        models.emplace_back(
            mte::Tensor({4, 5}, {0.2F, -0.4F, 0.1F, 0.5F, -0.3F,
                                 0.7F, 0.6F, -0.2F, 0.1F, 0.8F,
                                 -0.5F, 0.2F, 0.3F, -0.6F, 0.4F,
                                 0.9F, -0.1F, 0.5F, 0.2F, -0.7F}),
            mte::Tensor({1, 5}, {0.1F, -0.2F, 0.05F, 0.3F, -0.4F}),
            mte::Tensor({5, 3}, {0.3F, -0.1F, 0.8F,
                                 -0.6F, 0.4F, 0.2F,
                                 0.5F, 0.7F, -0.3F,
                                 0.1F, -0.5F, 0.9F,
                                 -0.2F, 0.6F, 0.4F}),
            mte::Tensor({1, 3}, {0.05F, -0.15F, 0.25F}),
            mte::MatMulBackend::kThreadedTransposeRhs,
            num_threads);
    }

    const mte::Tensor input({1, 4}, {1.0F, -2.0F, 3.0F, 0.5F});
    const mte::Tensor baseline_output = naive_model.Forward(input);
    for (const mte::TwoLayerPerceptron& model : models) {
        ValidateEquivalent(baseline_output, model.Forward(input));
    }

    std::cout << "Model backend comparison\n";
    std::cout << "backend,threads,avg_ns\n";

    volatile float sink = 0.0F;
    for (const mte::TwoLayerPerceptron& model : models) {
        const double avg_ns = MeasureAverageNanoseconds(
            [&]() {
                const mte::Tensor output = model.Forward(input);
                sink += output.data()[0];
            },
            options.warmup_iterations,
            options.iterations);

        std::cout << mte::MatMulBackendName(model.backend()) << ',' << model.num_threads()
                  << ',' << std::fixed << std::setprecision(2) << avg_ns << '\n';
    }
    std::cout << "model_sink," << sink << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        RunMatMulBenchmarks(options);
        if (options.run_model_benchmark) {
            RunModelBenchmark(options);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
