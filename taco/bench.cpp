#include "bench.h"

#include <cstdlib>
#include <iostream>

#include "taco/tensor.h"
#include "taco/util/strings.h"

std::string getEnvVar(std::string varname) {
  auto path = std::getenv(varname.c_str());
  if (path == nullptr) {
    return "";
  }
  return std::string(path);
}

std::string getTacoTensorPath() {
  std::string result = getEnvVar("TACO_TENSOR_PATH");
  if (result == "") {
    assert(false && "TACO_TENSOR_PATH is unset");
  }
  return cleanPath(result);
}

std::string getValidationOutputPath() {
  auto result = getEnvVar("VALIDATION_OUTPUT_PATH");
  if (result != "") {
    result = cleanPath(result);
  }
  return result;
}

std::string cleanPath(std::string path) {
  std::string result(path);
  if (result[result.size() - 1] != '/') {
    result += "/";
  }
  return result;
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
