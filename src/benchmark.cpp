#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
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

struct ModelCase {
    const char* name;
    std::size_t batch_size;
    std::size_t input_width;
    std::size_t hidden_width;
    std::size_t output_width;
};

struct MatMulResult {
    std::string case_name;
    std::string backend_name;
    std::size_t threads;
    double avg_ns;
};

struct ModelResult {
    std::string case_name;
    std::size_t batch_size;
    std::size_t input_width;
    std::size_t hidden_width;
    std::size_t output_width;
    std::string backend_name;
    std::size_t threads;
    double avg_ns;
};

struct Options {
    std::size_t iterations = 200;
    std::size_t warmup_iterations = 20;
    bool run_model_benchmark = true;
    std::vector<std::size_t> thread_counts = {1, 2, 4, 8};
    std::filesystem::path csv_out;
    std::filesystem::path json_out;
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
            if (options.iterations == 0) {
                throw std::invalid_argument("--iterations must be greater than 0");
            }
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
                const std::size_t num_threads = static_cast<std::size_t>(std::stoull(token));
                if (num_threads == 0) {
                    throw std::invalid_argument("--threads values must be greater than 0");
                }
                options.thread_counts.push_back(num_threads);
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
            if (options.thread_counts.empty()) {
                throw std::invalid_argument("--threads must provide at least one value");
            }
            continue;
        }
        if (arg == "--csv-out") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--csv-out requires a value");
            }
            options.csv_out = argv[++i];
            continue;
        }
        if (arg == "--json-out") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--json-out requires a value");
            }
            options.json_out = argv[++i];
            continue;
        }
        throw std::invalid_argument("unknown argument: " + std::string(arg));
    }

    return options;
}

std::string EscapeJson(const std::string& value) {
    std::ostringstream escaped;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            default:
                escaped << ch;
                break;
        }
    }
    return escaped.str();
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

std::vector<MatMulResult> RunMatMulBenchmarks(const Options& options) {
    const std::vector<MatMulCase> cases = {
        {32, 64, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
    };

    std::mt19937 generator(7);
    std::vector<MatMulResult> results;
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
            results.push_back(MatMulResult{
                std::to_string(benchmark_case.rows) + "x" +
                    std::to_string(benchmark_case.inner) + "x" +
                    std::to_string(benchmark_case.cols),
                mte::MatMulBackendName(backend),
                1,
                avg_ns,
            });
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
            results.push_back(MatMulResult{
                std::to_string(benchmark_case.rows) + "x" +
                    std::to_string(benchmark_case.inner) + "x" +
                    std::to_string(benchmark_case.cols),
                mte::MatMulBackendName(mte::MatMulBackend::kThreadedTransposeRhs),
                num_threads,
                avg_ns,
            });
        }
        std::cout << "case_sink," << sink << '\n';
    }
    return results;
}

struct ModelData {
    mte::Tensor input;
    mte::Tensor w1;
    mte::Tensor b1;
    mte::Tensor w2;
    mte::Tensor b2;
};

struct ModelBundle {
    mte::Tensor input;
    mte::FeedForwardModel model;
};

ModelData MakeSyntheticModelData(const ModelCase& model_case, std::mt19937& generator) {
    return ModelData{
        mte::Tensor(
            {model_case.batch_size, model_case.input_width},
            GenerateValues(model_case.batch_size * model_case.input_width, generator)),
        mte::Tensor(
            {model_case.input_width, model_case.hidden_width},
            GenerateValues(model_case.input_width * model_case.hidden_width, generator)),
        mte::Tensor(
            {1, model_case.hidden_width},
            GenerateValues(model_case.hidden_width, generator)),
        mte::Tensor(
            {model_case.hidden_width, model_case.output_width},
            GenerateValues(model_case.hidden_width * model_case.output_width, generator)),
        mte::Tensor(
            {1, model_case.output_width},
            GenerateValues(model_case.output_width, generator)),
    };
}

ModelBundle MakeSyntheticModel(
    const ModelData& model_data,
    mte::MatMulBackend backend,
    std::size_t num_threads) {
    return ModelBundle{
        model_data.input,
        mte::FeedForwardModel::MakeTwoLayerPerceptron(
            model_data.w1,
            model_data.b1,
            model_data.w2,
            model_data.b2,
            backend,
            num_threads),
    };
}

std::vector<ModelResult> RunModelBenchmark(const Options& options) {
    const std::vector<ModelCase> cases = {
        {"tiny_demo", 1, 4, 5, 3},
        {"batch32_hidden128", 32, 128, 128, 32},
        {"batch64_hidden256", 64, 256, 256, 64},
        {"batch128_hidden512", 128, 512, 512, 128},
    };

    std::mt19937 generator(17);
    std::vector<ModelResult> results;
    std::cout << "Model backend comparison\n";
    std::cout << "case,batch,input,hidden,output,backend,threads,avg_ns\n";

    for (const ModelCase& model_case : cases) {
        const ModelData model_data = MakeSyntheticModelData(model_case, generator);
        ModelBundle naive_bundle =
            MakeSyntheticModel(model_data, mte::MatMulBackend::kNaive, 1);
        ModelBundle optimized_bundle =
            MakeSyntheticModel(model_data, mte::MatMulBackend::kTransposeRhs, 1);

        std::vector<ModelBundle> threaded_bundles;
        threaded_bundles.reserve(options.thread_counts.size());
        for (std::size_t num_threads : options.thread_counts) {
            threaded_bundles.push_back(
                MakeSyntheticModel(
                    model_data,
                    mte::MatMulBackend::kThreadedTransposeRhs,
                    num_threads));
        }

        const mte::Tensor baseline_output = naive_bundle.model.Forward(naive_bundle.input);
        ValidateEquivalent(baseline_output, optimized_bundle.model.Forward(optimized_bundle.input));
        for (const ModelBundle& threaded_bundle : threaded_bundles) {
            ValidateEquivalent(baseline_output, threaded_bundle.model.Forward(threaded_bundle.input));
        }

        volatile float sink = 0.0F;

        const auto report_result = [&](const ModelBundle& bundle) {
            const double avg_ns = MeasureAverageNanoseconds(
                [&]() {
                    const mte::Tensor output = bundle.model.Forward(bundle.input);
                    sink += output.data()[0];
                },
                options.warmup_iterations,
                options.iterations);

            std::cout << model_case.name << ',' << model_case.batch_size << ','
                      << model_case.input_width << ',' << model_case.hidden_width << ','
                      << model_case.output_width << ','
                      << mte::MatMulBackendName(bundle.model.backend()) << ','
                      << bundle.model.num_threads() << ',' << std::fixed << std::setprecision(2)
                      << avg_ns << '\n';
            results.push_back(ModelResult{
                model_case.name,
                model_case.batch_size,
                model_case.input_width,
                model_case.hidden_width,
                model_case.output_width,
                mte::MatMulBackendName(bundle.model.backend()),
                bundle.model.num_threads(),
                avg_ns,
            });
        };

        report_result(naive_bundle);
        report_result(optimized_bundle);
        for (const ModelBundle& threaded_bundle : threaded_bundles) {
            report_result(threaded_bundle);
        }

        std::cout << "model_sink," << sink << '\n';
    }
    return results;
}

void WriteCsvReport(
    const std::filesystem::path& path,
    const std::vector<MatMulResult>& matmul_results,
    const std::vector<ModelResult>& model_results) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.string().empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open CSV output: " + path.string());
    }

    output << "section,case,batch,input,hidden,output,backend,threads,avg_ns\n";
    for (const MatMulResult& result : matmul_results) {
        output << "matmul," << result.case_name << ",,,,," << result.backend_name << ','
               << result.threads << ',' << std::fixed << std::setprecision(2) << result.avg_ns
               << '\n';
    }
    for (const ModelResult& result : model_results) {
        output << "model," << result.case_name << ',' << result.batch_size << ','
               << result.input_width << ',' << result.hidden_width << ','
               << result.output_width << ',' << result.backend_name << ',' << result.threads
               << ',' << std::fixed << std::setprecision(2) << result.avg_ns << '\n';
    }
}

void WriteJsonReport(
    const std::filesystem::path& path,
    const Options& options,
    const std::vector<MatMulResult>& matmul_results,
    const std::vector<ModelResult>& model_results) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.string().empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open JSON output: " + path.string());
    }

    output << "{\n";
    output << "  \"iterations\": " << options.iterations << ",\n";
    output << "  \"warmup_iterations\": " << options.warmup_iterations << ",\n";
    output << "  \"matmul_results\": [\n";
    for (std::size_t i = 0; i < matmul_results.size(); ++i) {
        const MatMulResult& result = matmul_results[i];
        output << "    {\"case\": \"" << EscapeJson(result.case_name)
               << "\", \"backend\": \"" << EscapeJson(result.backend_name)
               << "\", \"threads\": " << result.threads
               << ", \"avg_ns\": " << std::fixed << std::setprecision(2) << result.avg_ns
               << "}";
        if (i + 1 < matmul_results.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ],\n";
    output << "  \"model_results\": [\n";
    for (std::size_t i = 0; i < model_results.size(); ++i) {
        const ModelResult& result = model_results[i];
        output << "    {\"case\": \"" << EscapeJson(result.case_name)
               << "\", \"batch\": " << result.batch_size
               << ", \"input\": " << result.input_width
               << ", \"hidden\": " << result.hidden_width
               << ", \"output\": " << result.output_width
               << ", \"backend\": \"" << EscapeJson(result.backend_name)
               << "\", \"threads\": " << result.threads
               << ", \"avg_ns\": " << std::fixed << std::setprecision(2) << result.avg_ns
               << "}";
        if (i + 1 < model_results.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ]\n";
    output << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        const std::vector<MatMulResult> matmul_results = RunMatMulBenchmarks(options);
        std::vector<ModelResult> model_results;
        if (options.run_model_benchmark) {
            model_results = RunModelBenchmark(options);
        }
        if (!options.csv_out.empty()) {
            WriteCsvReport(options.csv_out, matmul_results, model_results);
            std::cout << "CSV report written to " << options.csv_out << '\n';
        }
        if (!options.json_out.empty()) {
            WriteJsonReport(options.json_out, options, matmul_results, model_results);
            std::cout << "JSON report written to " << options.json_out << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
