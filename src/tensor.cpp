#include "mte/tensor.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

namespace mte {

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), data_(ComputeSize(shape_), 0.0F) {}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> values)
    : shape_(std::move(shape)), data_(std::move(values)) {
    if (data_.size() != ComputeSize(shape_)) {
        throw std::invalid_argument("tensor data size does not match shape");
    }
}

Tensor::Tensor(std::initializer_list<std::size_t> shape, std::vector<float> values)
    : Tensor(std::vector<std::size_t>(shape), std::move(values)) {}

const std::vector<std::size_t>& Tensor::shape() const noexcept {
    return shape_;
}

std::size_t Tensor::rank() const noexcept {
    return shape_.size();
}

std::size_t Tensor::size() const noexcept {
    return data_.size();
}

bool Tensor::empty() const noexcept {
    return data_.empty();
}

std::size_t Tensor::rows() const {
    if (rank() != 2) {
        throw std::logic_error("rows() requires a rank-2 tensor");
    }
    return shape_[0];
}

std::size_t Tensor::cols() const {
    if (rank() != 2) {
        throw std::logic_error("cols() requires a rank-2 tensor");
    }
    return shape_[1];
}

float& Tensor::at(std::size_t row, std::size_t col) {
    return data_.at(FlattenIndex(row, col));
}

const float& Tensor::at(std::size_t row, std::size_t col) const {
    return data_.at(FlattenIndex(row, col));
}

std::vector<float>& Tensor::data() noexcept {
    return data_;
}

const std::vector<float>& Tensor::data() const noexcept {
    return data_;
}

std::string Tensor::DebugString() const {
    std::ostringstream stream;
    stream << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        stream << shape_[i];
        if (i + 1 < shape_.size()) {
            stream << ", ";
        }
    }
    stream << "], values=[";
    for (std::size_t i = 0; i < data_.size(); ++i) {
        stream << data_[i];
        if (i + 1 < data_.size()) {
            stream << ", ";
        }
    }
    stream << "])";
    return stream.str();
}

std::size_t Tensor::FlattenIndex(std::size_t row, std::size_t col) const {
    if (rank() != 2) {
        throw std::logic_error("2D indexing requires rank-2 tensor");
    }
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("tensor index out of range");
    }
    return row * shape_[1] + col;
}

std::size_t Tensor::ComputeSize(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("tensor shape must not be empty");
    }
    return std::accumulate(
        shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

namespace {

void ValidateMatMulInputs(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument("MatMul currently supports only rank-2 tensors");
    }
    if (lhs.cols() != rhs.rows()) {
        throw std::invalid_argument("MatMul shape mismatch");
    }
}

Tensor MatMulNaive(const Tensor& lhs, const Tensor& rhs) {
    ValidateMatMulInputs(lhs, rhs);

    const std::size_t rows = lhs.rows();
    const std::size_t inner = lhs.cols();
    const std::size_t cols = rhs.cols();

    Tensor output({rows, cols});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < inner; ++k) {
                sum += lhs.at(row, k) * rhs.at(k, col);
            }
            output.at(row, col) = sum;
        }
    }
    return output;
}

Tensor MatMulTransposeRhs(const Tensor& lhs, const Tensor& rhs) {
    ValidateMatMulInputs(lhs, rhs);

    return MatMulWithPretransposedRhs(lhs, Transpose(rhs));
}

Tensor MatMulThreadedTransposeRhs(
    const Tensor& lhs,
    const Tensor& rhs,
    std::size_t num_threads) {
    ValidateMatMulInputs(lhs, rhs);
    return MatMulWithPretransposedRhs(lhs, Transpose(rhs), num_threads);
}

}  // namespace

Tensor Transpose(const Tensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument("Transpose currently supports only rank-2 tensors");
    }

    Tensor output({input.cols(), input.rows()});
    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t col = 0; col < input.cols(); ++col) {
            output.at(col, row) = input.at(row, col);
        }
    }
    return output;
}

Tensor MatMulWithPretransposedRhs(const Tensor& lhs, const Tensor& rhs_transposed) {
    return MatMulWithPretransposedRhs(lhs, rhs_transposed, 1);
}

Tensor MatMulWithPretransposedRhs(
    const Tensor& lhs,
    const Tensor& rhs_transposed,
    std::size_t num_threads) {
    if (lhs.rank() != 2 || rhs_transposed.rank() != 2) {
        throw std::invalid_argument(
            "MatMulWithPretransposedRhs currently supports only rank-2 tensors");
    }

    const std::size_t rows = lhs.rows();
    const std::size_t inner = lhs.cols();
    const std::size_t cols = rhs_transposed.rows();

    if (rhs_transposed.cols() != inner) {
        throw std::invalid_argument("pretransposed rhs shape mismatch");
    }

    Tensor output({rows, cols});

    const std::size_t effective_threads =
        std::max<std::size_t>(1, std::min(num_threads, rows));

    auto compute_rows = [&](std::size_t row_begin, std::size_t row_end) {
        for (std::size_t row = row_begin; row < row_end; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                const float* lhs_row = lhs.data().data() + (row * inner);
                const float* rhs_row = rhs_transposed.data().data() + (col * inner);

                float sum = 0.0F;
                for (std::size_t k = 0; k < inner; ++k) {
                    sum += lhs_row[k] * rhs_row[k];
                }
                output.at(row, col) = sum;
            }
        }
    };

    if (effective_threads == 1) {
        compute_rows(0, rows);
        return output;
    }

    std::vector<std::thread> workers;
    workers.reserve(effective_threads - 1);

    const std::size_t rows_per_thread = rows / effective_threads;
    const std::size_t extra_rows = rows % effective_threads;

    std::size_t next_row = 0;
    for (std::size_t thread_index = 0; thread_index < effective_threads - 1; ++thread_index) {
        const std::size_t row_count = rows_per_thread + (thread_index < extra_rows ? 1 : 0);
        const std::size_t row_begin = next_row;
        const std::size_t row_end = row_begin + row_count;
        workers.emplace_back(compute_rows, row_begin, row_end);
        next_row = row_end;
    }

    compute_rows(next_row, rows);

    for (std::thread& worker : workers) {
        worker.join();
    }

    return output;
}

Tensor MatMul(const Tensor& lhs, const Tensor& rhs) {
    return MatMul(lhs, rhs, MatMulBackend::kTransposeRhs);
}

Tensor MatMul(const Tensor& lhs, const Tensor& rhs, MatMulBackend backend) {
    return MatMul(lhs, rhs, backend, 1);
}

Tensor MatMul(const Tensor& lhs, const Tensor& rhs, MatMulBackend backend, std::size_t num_threads) {
    switch (backend) {
        case MatMulBackend::kNaive:
            return MatMulNaive(lhs, rhs);
        case MatMulBackend::kTransposeRhs:
            return MatMulTransposeRhs(lhs, rhs);
        case MatMulBackend::kThreadedTransposeRhs:
            return MatMulThreadedTransposeRhs(lhs, rhs, num_threads);
    }

    throw std::invalid_argument("unknown MatMul backend");
}

Tensor AddBias(const Tensor& input, const Tensor& bias) {
    if (input.rank() != 2 || bias.rank() != 2) {
        throw std::invalid_argument("AddBias currently supports only rank-2 tensors");
    }
    if (bias.rows() != 1 || bias.cols() != input.cols()) {
        throw std::invalid_argument("bias shape must be [1, input_cols]");
    }

    Tensor output(input.shape(), input.data());
    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t col = 0; col < input.cols(); ++col) {
            output.at(row, col) += bias.at(0, col);
        }
    }
    return output;
}

bool HasSameShape(const Tensor& lhs, const Tensor& rhs) noexcept {
    return lhs.shape() == rhs.shape();
}

const char* MatMulBackendName(MatMulBackend backend) noexcept {
    switch (backend) {
        case MatMulBackend::kNaive:
            return "naive";
        case MatMulBackend::kTransposeRhs:
            return "transpose_rhs";
        case MatMulBackend::kThreadedTransposeRhs:
            return "threaded_transpose_rhs";
    }
    return "unknown";
}

}  // namespace mte
