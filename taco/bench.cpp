#include "bench.h"

#include <cstdlib>
#include <iostream>

#include "taco/tensor.h"
#include "taco/util/strings.h"

std::string constructRandomTensorKey(std::vector<int> dims, float sparsity) {
  auto path = std::getenv("TACO_RANDOM_TENSOR_PATH");
  if (path == nullptr) {
    std::cout << "TACO_RANDOM_TENSOR_PATH is unset" << std::endl;
    assert(false);
  }
  std::string pathStr(path);
  std::stringstream result;
  result << pathStr;
  if (pathStr[pathStr.size() - 1] != '/') {
    result << "/";
  }
  result << taco::util::join(dims, "x") << "-" << sparsity << ".tns";
  return result.str();
}

taco::TensorBase loadRandomTensor(std::string name, std::vector<int> dims, float sparsity, taco::Format format) {
  // For now, just say that the python code must generate the random
  // tensor before use.
  auto tensor = taco::read(constructRandomTensorKey(dims, sparsity), format, true);
  tensor.setName(name);
  return tensor;
}
