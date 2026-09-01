#include <cuda.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "runtime/stream_lifecycle.h"
#include "vmm/paged_region.h"

#include <cstdint>
#include <memory>
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
  pybind11::class_<flexmoe::PagedRegion,
                   std::shared_ptr<flexmoe::PagedRegion>>(module,
                                                          "PagedRegion")
      .def(pybind11::init<int, std::uint64_t>(), pybind11::arg("device"),
           pybind11::arg("virtual_bytes"))
      .def("create_block", &flexmoe::PagedRegion::create_block,
           pybind11::arg("nbytes"))
      .def("map", &flexmoe::PagedRegion::map, pybind11::arg("offset"),
           pybind11::arg("block_id"), pybind11::arg("nbytes"))
      .def("unmap", &flexmoe::PagedRegion::unmap, pybind11::arg("offset"),
           pybind11::arg("nbytes"))
      .def("tensor", &flexmoe::PagedRegion::tensor,
           pybind11::arg("offset"), pybind11::arg("shape"),
           pybind11::arg("dtype_code"))
      .def("snapshot", &flexmoe::PagedRegion::snapshot)
      .def_property_readonly("device", &flexmoe::PagedRegion::device)
      .def_property_readonly("base_address",
                             &flexmoe::PagedRegion::base_address)
      .def_property_readonly("virtual_bytes",
                             &flexmoe::PagedRegion::virtual_bytes)
      .def_property_readonly("granularity",
                             &flexmoe::PagedRegion::granularity);
  pybind11::class_<flexmoe::StreamLifecycle>(module, "StreamLifecycle")
      .def(pybind11::init<int, std::int64_t>(), pybind11::arg("device"),
           pybind11::arg("total_layers"))
      .def("record_load_done", &flexmoe::StreamLifecycle::record_load_done,
           pybind11::arg("layer_idx"))
      .def("wait_load_done", &flexmoe::StreamLifecycle::wait_load_done,
           pybind11::arg("layer_idx"),
           pybind11::arg("compute_stream_handle"))
      .def("record_compute_done",
           &flexmoe::StreamLifecycle::record_compute_done,
           pybind11::arg("layer_idx"),
           pybind11::arg("compute_stream_handle"))
      .def("synchronize_compute_done",
           &flexmoe::StreamLifecycle::synchronize_compute_done,
           pybind11::arg("layer_idx"))
      .def("synchronize_load_stream",
           &flexmoe::StreamLifecycle::synchronize_load_stream)
      .def("snapshot", &flexmoe::StreamLifecycle::snapshot)
      .def_property_readonly("device", &flexmoe::StreamLifecycle::device)
      .def_property_readonly("total_layers",
                             &flexmoe::StreamLifecycle::total_layers)
      .def_property_readonly("load_stream",
                             &flexmoe::StreamLifecycle::load_stream);
}
