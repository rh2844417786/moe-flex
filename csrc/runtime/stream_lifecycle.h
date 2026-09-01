#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace flexmoe {

class StreamLifecycle final {
 public:
  using Snapshot = std::map<std::string, std::uint64_t>;

  StreamLifecycle(int device, std::int64_t total_layers);
  ~StreamLifecycle() noexcept;

  StreamLifecycle(const StreamLifecycle&) = delete;
  StreamLifecycle& operator=(const StreamLifecycle&) = delete;
  StreamLifecycle(StreamLifecycle&&) = delete;
  StreamLifecycle& operator=(StreamLifecycle&&) = delete;

  void record_load_done(std::int64_t layer_idx);
  void wait_load_done(std::int64_t layer_idx,
                      std::uint64_t compute_stream_handle);
  void record_compute_done(std::int64_t layer_idx,
                           std::uint64_t compute_stream_handle);
  void synchronize_compute_done(std::int64_t layer_idx);
  void synchronize_load_stream();

  [[nodiscard]] int device() const noexcept { return device_; }
  [[nodiscard]] std::int64_t total_layers() const noexcept {
    return total_layers_;
  }
  [[nodiscard]] std::uint64_t load_stream() const noexcept;
  [[nodiscard]] Snapshot snapshot() const;

 private:
  [[nodiscard]] std::size_t layer_offset(std::int64_t layer_idx) const;
  void require_recorded(const std::vector<std::uint8_t>& flags,
                        std::size_t offset, const char* event_kind) const;
  void close_noexcept() noexcept;

  int device_{};
  std::int64_t total_layers_{};
  cudaStream_t load_stream_{};
  std::vector<cudaEvent_t> load_done_;
  std::vector<cudaEvent_t> compute_done_;
  std::vector<std::uint8_t> load_recorded_;
  std::vector<std::uint8_t> compute_recorded_;
  bool closed_{false};
  mutable std::mutex mutex_;
};

}  // namespace flexmoe
