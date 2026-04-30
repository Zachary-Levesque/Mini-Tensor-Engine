#pragma once

#include <string>

#include "mte/tensor.hpp"

namespace mte {

Tensor LoadTensorFromTextFile(const std::string& path);

}  // namespace mte
