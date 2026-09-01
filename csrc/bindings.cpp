#include <cuda.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>

namespace {

void check_cuda(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = "CUDA_ERROR_UNKNOWN";
  const char* description = "unknown CUDA driver error";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  TORCH_CHECK(false, operation, " failed: ", name, " (", description, ")");
}

std::string extension_version() { return "0.1.0"; }

std::int64_t cuda_driver_version() {
  check_cuda(cuInit(0), "cuInit");
  int version = 0;
  check_cuda(cuDriverGetVersion(&version), "cuDriverGetVersion");
  return static_cast<std::int64_t>(version);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("extension_version", &extension_version);
  module.def("cuda_driver_version", &cuda_driver_version);
}
