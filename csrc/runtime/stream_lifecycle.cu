#include "stream_lifecycle.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace flexmoe {
namespace {

[[noreturn]] void throw_cuda(cudaError_t error, const char* operation) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorName(error) << " ("
          << cudaGetErrorString(error) << ")";
  throw std::runtime_error(message.str());
}

void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw_cuda(error, operation);
  }
}

class ScopedDevice final {
 public:
  explicit ScopedDevice(int device) {
    check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device) {
      check_cuda(cudaSetDevice(device), "cudaSetDevice");
      restore_ = true;
    }
  }

  ~ScopedDevice() noexcept {
    if (restore_) {
      (void)cudaSetDevice(previous_);
    }
  }

  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;

 private:
  int previous_{};
  bool restore_{false};
};

cudaStream_t stream_from_handle(std::uint64_t handle) noexcept {
  return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(handle));
}

}  // namespace

StreamLifecycle::StreamLifecycle(int device, std::int64_t total_layers)
    : device_(device), total_layers_(total_layers) {
  if (device < 0) {
    throw std::invalid_argument("device must be non-negative");
  }
  if (total_layers < 2) {
    throw std::invalid_argument("total_layers must be at least two");
  }
  const auto layer_count = static_cast<std::size_t>(total_layers);
  load_done_.assign(layer_count, nullptr);
  compute_done_.assign(layer_count, nullptr);
  load_recorded_.assign(layer_count, 0);
  compute_recorded_.assign(layer_count, 0);

  ScopedDevice device_guard(device_);
  try {
    check_cuda(cudaStreamCreateWithFlags(&load_stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
      check_cuda(cudaEventCreateWithFlags(&load_done_[layer],
                                          cudaEventDisableTiming),
                 "cudaEventCreateWithFlags(load_done)");
      check_cuda(cudaEventCreateWithFlags(
                     &compute_done_[layer],
                     cudaEventDisableTiming | cudaEventBlockingSync),
                 "cudaEventCreateWithFlags(compute_done)");
    }
  } catch (...) {
    close_noexcept();
    throw;
  }
}

StreamLifecycle::~StreamLifecycle() noexcept { close_noexcept(); }

std::size_t StreamLifecycle::layer_offset(std::int64_t layer_idx) const {
  if (layer_idx < 0 || layer_idx >= total_layers_) {
    throw std::out_of_range("layer_idx is outside the lifecycle range");
  }
  return static_cast<std::size_t>(layer_idx);
}

void StreamLifecycle::require_recorded(
    const std::vector<std::uint8_t>& flags, std::size_t offset,
    const char* event_kind) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    throw std::runtime_error("stream lifecycle is closed");
  }
  if (flags[offset] == 0) {
    throw std::runtime_error(std::string(event_kind) +
                             " event has not been recorded");
  }
}

void StreamLifecycle::record_load_done(std::int64_t layer_idx) {
  const std::size_t offset = layer_offset(layer_idx);
  ScopedDevice device_guard(device_);
  check_cuda(cudaEventRecord(load_done_[offset], load_stream_),
             "cudaEventRecord(load_done)");
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    throw std::runtime_error("stream lifecycle is closed");
  }
  load_recorded_[offset] = 1;
}

void StreamLifecycle::wait_load_done(std::int64_t layer_idx,
                                     std::uint64_t compute_stream_handle) {
  const std::size_t offset = layer_offset(layer_idx);
  require_recorded(load_recorded_, offset, "load");
  ScopedDevice device_guard(device_);
  check_cuda(cudaStreamWaitEvent(stream_from_handle(compute_stream_handle),
                                 load_done_[offset], 0),
             "cudaStreamWaitEvent(load_done)");
}

void StreamLifecycle::record_compute_done(
    std::int64_t layer_idx, std::uint64_t compute_stream_handle) {
  const std::size_t offset = layer_offset(layer_idx);
  ScopedDevice device_guard(device_);
  check_cuda(cudaEventRecord(compute_done_[offset],
                             stream_from_handle(compute_stream_handle)),
             "cudaEventRecord(compute_done)");
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    throw std::runtime_error("stream lifecycle is closed");
  }
  compute_recorded_[offset] = 1;
}

void StreamLifecycle::synchronize_compute_done(std::int64_t layer_idx) {
  const std::size_t offset = layer_offset(layer_idx);
  require_recorded(compute_recorded_, offset, "compute");
  ScopedDevice device_guard(device_);
  check_cuda(cudaEventSynchronize(compute_done_[offset]),
             "cudaEventSynchronize(compute_done)");
}

void StreamLifecycle::synchronize_load_stream() {
  ScopedDevice device_guard(device_);
  check_cuda(cudaStreamSynchronize(load_stream_), "cudaStreamSynchronize");
}

std::uint64_t StreamLifecycle::load_stream() const noexcept {
  return static_cast<std::uint64_t>(
      reinterpret_cast<std::uintptr_t>(load_stream_));
}

StreamLifecycle::Snapshot StreamLifecycle::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::uint64_t loads = 0;
  std::uint64_t computes = 0;
  for (const std::uint8_t recorded : load_recorded_) {
    loads += recorded != 0 ? 1 : 0;
  }
  for (const std::uint8_t recorded : compute_recorded_) {
    computes += recorded != 0 ? 1 : 0;
  }
  return {{"device", static_cast<std::uint64_t>(device_)},
          {"total_layers", static_cast<std::uint64_t>(total_layers_)},
          {"load_stream", load_stream()},
          {"load_events_recorded", loads},
          {"compute_events_recorded", computes}};
}

void StreamLifecycle::close_noexcept() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return;
  }
  closed_ = true;

  int previous_device = 0;
  const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
  const bool device_ready = cudaSetDevice(device_) == cudaSuccess;
  if (device_ready && load_stream_ != nullptr) {
    (void)cudaStreamSynchronize(load_stream_);
  }
  if (device_ready) {
    for (cudaEvent_t event : load_done_) {
      if (event != nullptr) {
        (void)cudaEventDestroy(event);
      }
    }
    for (cudaEvent_t event : compute_done_) {
      if (event != nullptr) {
        (void)cudaEventDestroy(event);
      }
    }
    if (load_stream_ != nullptr) {
      (void)cudaStreamDestroy(load_stream_);
      load_stream_ = nullptr;
    }
  }
  if (have_previous && previous_device != device_) {
    (void)cudaSetDevice(previous_device);
  }
}

}  // namespace flexmoe
