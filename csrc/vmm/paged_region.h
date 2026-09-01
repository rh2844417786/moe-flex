#pragma once

#include <cuda.h>
#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace flexmoe {

class PagedRegion final : public std::enable_shared_from_this<PagedRegion> {
 public:
  using Snapshot = std::map<std::string, std::uint64_t>;

  PagedRegion(int device, std::uint64_t virtual_bytes);
  ~PagedRegion() noexcept;

  PagedRegion(const PagedRegion&) = delete;
  PagedRegion& operator=(const PagedRegion&) = delete;
  PagedRegion(PagedRegion&&) = delete;
  PagedRegion& operator=(PagedRegion&&) = delete;

  std::uint64_t create_block(std::uint64_t bytes);
  void map(std::uint64_t offset, std::uint64_t block_id,
           std::uint64_t bytes);
  void unmap(std::uint64_t offset, std::uint64_t bytes);
  torch::Tensor tensor(std::uint64_t offset,
                       const std::vector<std::int64_t>& sizes,
                       std::int64_t dtype_code);

  [[nodiscard]] int device() const noexcept { return device_index_; }
  [[nodiscard]] std::uint64_t base_address() const noexcept {
    return static_cast<std::uint64_t>(base_);
  }
  [[nodiscard]] std::uint64_t virtual_bytes() const noexcept {
    return virtual_bytes_;
  }
  [[nodiscard]] std::uint64_t granularity() const noexcept {
    return granularity_;
  }
  [[nodiscard]] Snapshot snapshot() const;

 private:
  struct Block {
    CUmemGenericAllocationHandle handle{};
    std::uint64_t bytes{};
  };

  struct Mapping {
    std::uint64_t block_id{};
    std::uint64_t bytes{};
  };

  [[nodiscard]] CUmemAllocationProp allocation_properties() const noexcept;
  void validate_aligned_range(std::uint64_t offset,
                              std::uint64_t bytes) const;
  void validate_tensor_range(std::uint64_t offset,
                             std::uint64_t bytes) const;
  void close_noexcept() noexcept;

  int device_index_{};
  CUdevice device_{};
  CUcontext context_{};
  CUdeviceptr base_{};
  std::uint64_t virtual_bytes_{};
  std::uint64_t granularity_{};
  std::uint64_t next_block_id_{1};
  bool closed_{false};
  std::unordered_map<std::uint64_t, Block> blocks_;
  std::map<std::uint64_t, Mapping> mappings_;
  mutable std::mutex mutex_;
};

}  // namespace flexmoe
