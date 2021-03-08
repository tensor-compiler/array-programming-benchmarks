#include "bench.h"

#include <cstdlib>
#include <iostream>

#include "taco/tensor.h"
#include "taco/util/strings.h"

std::string getTacoTensorPath() {
  auto path = std::getenv("TACO_TENSOR_PATH");
  if (path == nullptr) {
    std::cout << "TACO_TENSOR_PATH is unset" << std::endl;
    assert(false);
  }
  return std::string(path);
}

std::string constructRandomTensorKey(std::vector<int> dims, float sparsity) {
  auto path = getTacoTensorPath();
  std::stringstream result;
  result << path;
  if (path[path.size() - 1] != '/') {
    result << "/";
  }
  result << "random/";
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
