#pragma once

#include <filesystem>

#include "mte/tensor.hpp"

namespace mte {

struct ReferenceCase {
    Tensor input;
    Tensor expected_output;
};

Tensor LoadTensorFromTextFile(const std::filesystem::path& path);
void SaveTensorToTextFile(const Tensor& tensor, const std::filesystem::path& path);
ReferenceCase LoadReferenceCase(const std::filesystem::path& directory);

}  // namespace mte
