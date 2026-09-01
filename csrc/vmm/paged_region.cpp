#include "paged_region.h"

#include <c10/core/ScalarType.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace flexmoe {
namespace {

[[noreturn]] void throw_cuda(CUresult result, const char* operation) {
  const char* name = "CUDA_ERROR_UNKNOWN";
  const char* description = "unknown CUDA driver error";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::ostringstream message;
  message << operation << " failed: " << name << " (" << description << ")";
  throw std::runtime_error(message.str());
}

void check_cuda(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    throw_cuda(result, operation);
  }
}

class ScopedContext final {
 public:
  explicit ScopedContext(CUcontext context) {
    check_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent");
    pushed_ = true;
  }

  ~ScopedContext() noexcept {
    if (!pushed_) {
      return;
    }
    CUcontext ignored = nullptr;
    (void)cuCtxPopCurrent(&ignored);
  }

  ScopedContext(const ScopedContext&) = delete;
  ScopedContext& operator=(const ScopedContext&) = delete;

 private:
  bool pushed_{false};
};

std::uint64_t checked_round_up(std::uint64_t value,
                               std::uint64_t alignment) {
  if (value == 0 || alignment == 0) {
    throw std::invalid_argument("virtual_bytes and alignment must be positive");
  }
  const std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::uint64_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::uint64_t>::max() - increment) {
    throw std::overflow_error("rounded virtual size overflows uint64");
  }
  return value + increment;
}

std::uint64_t checked_end(std::uint64_t offset, std::uint64_t bytes) {
  if (bytes == 0) {
    throw std::invalid_argument("mapping size must be positive");
  }
  if (offset > std::numeric_limits<std::uint64_t>::max() - bytes) {
    throw std::overflow_error("address range overflows uint64");
  }
  return offset + bytes;
}

at::ScalarType scalar_type_from_code(std::int64_t code) {
  switch (code) {
    case 0:
      return at::kBFloat16;
    case 1:
      return at::kHalf;
    case 2:
      return at::kFloat;
    case 3:
      return at::kByte;
    case 4:
      return at::kChar;
    case 5:
      return at::kInt;
    case 6:
      return at::kLong;
    default:
      throw std::invalid_argument("unsupported tensor dtype code");
  }
}

}  // namespace

PagedRegion::PagedRegion(int device, std::uint64_t virtual_bytes)
    : device_index_(device) {
  if (device < 0) {
    throw std::invalid_argument("device must be non-negative");
  }
  check_cuda(cuInit(0), "cuInit");
  check_cuda(cuDeviceGet(&device_, device), "cuDeviceGet");
  check_cuda(cuDevicePrimaryCtxRetain(&context_, device_),
             "cuDevicePrimaryCtxRetain");

  try {
    ScopedContext context_guard(context_);
    const CUmemAllocationProp properties = allocation_properties();
    std::size_t granularity = 0;
    check_cuda(cuMemGetAllocationGranularity(
                   &granularity, &properties,
                   CU_MEM_ALLOC_GRANULARITY_MINIMUM),
               "cuMemGetAllocationGranularity");
    granularity_ = static_cast<std::uint64_t>(granularity);
    virtual_bytes_ = checked_round_up(virtual_bytes, granularity_);
    if (virtual_bytes_ > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("virtual region does not fit size_t");
    }
    check_cuda(cuMemAddressReserve(
                   &base_, static_cast<std::size_t>(virtual_bytes_),
                   static_cast<std::size_t>(granularity_), 0, 0),
               "cuMemAddressReserve");
  } catch (...) {
    (void)cuDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
    throw;
  }
}

PagedRegion::~PagedRegion() noexcept { close_noexcept(); }

CUmemAllocationProp PagedRegion::allocation_properties() const noexcept {
  CUmemAllocationProp properties{};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device_;
  properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  return properties;
}

void PagedRegion::validate_aligned_range(std::uint64_t offset,
                                         std::uint64_t bytes) const {
  const std::uint64_t end = checked_end(offset, bytes);
  if (offset % granularity_ != 0 || bytes % granularity_ != 0) {
    throw std::invalid_argument(
        "mapping offset and size must be aligned to allocation granularity");
  }
  if (end > virtual_bytes_) {
    throw std::out_of_range("mapping range exceeds the virtual reservation");
  }
}

std::uint64_t PagedRegion::create_block(std::uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    throw std::runtime_error("paged region is closed");
  }
  validate_aligned_range(0, bytes);
  const CUmemAllocationProp properties = allocation_properties();
  CUmemGenericAllocationHandle handle{};
  ScopedContext context_guard(context_);
  check_cuda(cuMemCreate(&handle, static_cast<std::size_t>(bytes), &properties,
                         0),
             "cuMemCreate");
  const std::uint64_t block_id = next_block_id_++;
  try {
    blocks_.emplace(block_id, Block{handle, bytes});
  } catch (...) {
    (void)cuMemRelease(handle);
    throw;
  }
  return block_id;
}

void PagedRegion::map(std::uint64_t offset, std::uint64_t block_id,
                      std::uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    throw std::runtime_error("paged region is closed");
  }
  validate_aligned_range(offset, bytes);
  const std::uint64_t end = checked_end(offset, bytes);

  auto next = mappings_.lower_bound(offset);
  if (next != mappings_.end() && next->first < end) {
    throw std::runtime_error("mapping would overlap a live virtual range");
  }
  if (next != mappings_.begin()) {
    const auto previous = std::prev(next);
    if (checked_end(previous->first, previous->second.bytes) > offset) {
      throw std::runtime_error("mapping would overlap a live virtual range");
    }
  }

  const auto block = blocks_.find(block_id);
  if (block == blocks_.end()) {
    throw std::invalid_argument("unknown physical block id");
  }
  if (block->second.bytes != bytes) {
    throw std::invalid_argument(
        "mapping size must equal the physical block size");
  }
  if (std::any_of(mappings_.begin(), mappings_.end(),
                  [block_id](const auto& entry) {
                    return entry.second.block_id == block_id;
                  })) {
    throw std::runtime_error("physical block is already mapped");
  }

  ScopedContext context_guard(context_);
  check_cuda(cuMemMap(base_ + offset, static_cast<std::size_t>(bytes), 0,
                      block->second.handle, 0),
             "cuMemMap");
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device_;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  const CUresult access_result = cuMemSetAccess(
      base_ + offset, static_cast<std::size_t>(bytes), &access, 1);
  if (access_result != CUDA_SUCCESS) {
    (void)cuMemUnmap(base_ + offset, static_cast<std::size_t>(bytes));
    throw_cuda(access_result, "cuMemSetAccess");
  }
  mappings_.emplace(offset, Mapping{block_id, bytes});
}

void PagedRegion::unmap(std::uint64_t offset, std::uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    throw std::runtime_error("paged region is closed");
  }
  validate_aligned_range(offset, bytes);
  const auto mapping = mappings_.find(offset);
  if (mapping == mappings_.end() || mapping->second.bytes != bytes) {
    throw std::runtime_error("requested range is not an exact live mapping");
  }
  ScopedContext context_guard(context_);
  check_cuda(cuMemUnmap(base_ + offset, static_cast<std::size_t>(bytes)),
             "cuMemUnmap");
  mappings_.erase(mapping);
}

void PagedRegion::validate_tensor_range(std::uint64_t offset,
                                        std::uint64_t bytes) const {
  const std::uint64_t end = checked_end(offset, bytes);
  if (end > virtual_bytes_) {
    throw std::out_of_range("tensor view exceeds the virtual reservation");
  }

  std::uint64_t cursor = offset;
  auto mapping = mappings_.upper_bound(cursor);
  if (mapping != mappings_.begin()) {
    --mapping;
  }
  while (cursor < end) {
    if (mapping == mappings_.end() || mapping->first > cursor) {
      throw std::runtime_error("tensor view crosses an unmapped virtual range");
    }
    const std::uint64_t mapping_end =
        checked_end(mapping->first, mapping->second.bytes);
    if (mapping_end <= cursor) {
      ++mapping;
      continue;
    }
    cursor = std::min(end, mapping_end);
    ++mapping;
  }
}

torch::Tensor PagedRegion::tensor(
    std::uint64_t offset, const std::vector<std::int64_t>& sizes,
    std::int64_t dtype_code) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    throw std::runtime_error("paged region is closed");
  }
  if (sizes.empty()) {
    throw std::invalid_argument("tensor shape must contain at least one axis");
  }
  const at::ScalarType scalar_type = scalar_type_from_code(dtype_code);
  const std::uint64_t element_bytes = c10::elementSize(scalar_type);
  if (offset % element_bytes != 0) {
    throw std::invalid_argument("tensor offset is not dtype-aligned");
  }

  std::uint64_t elements = 1;
  for (const std::int64_t size : sizes) {
    if (size <= 0) {
      throw std::invalid_argument("tensor dimensions must be positive");
    }
    const auto unsigned_size = static_cast<std::uint64_t>(size);
    if (elements >
        std::numeric_limits<std::uint64_t>::max() / unsigned_size) {
      throw std::overflow_error("tensor element count overflows uint64");
    }
    elements *= unsigned_size;
  }
  if (elements > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
    throw std::overflow_error("tensor byte count overflows uint64");
  }
  const std::uint64_t tensor_bytes = elements * element_bytes;
  validate_tensor_range(offset, tensor_bytes);

  auto owner = shared_from_this();
  void* pointer = reinterpret_cast<void*>(base_ + offset);
  auto options = torch::TensorOptions()
                     .dtype(scalar_type)
                     .device(torch::kCUDA, device_index_);
  return torch::from_blob(
      pointer, sizes,
      [owner = std::move(owner)](void*) noexcept { (void)owner; }, options);
}

PagedRegion::Snapshot PagedRegion::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::uint64_t mapped_bytes = 0;
  for (const auto& entry : mappings_) {
    mapped_bytes += entry.second.bytes;
  }
  return {{"device", static_cast<std::uint64_t>(device_index_)},
          {"base_address", static_cast<std::uint64_t>(base_)},
          {"virtual_bytes", virtual_bytes_},
          {"granularity", granularity_},
          {"block_count", static_cast<std::uint64_t>(blocks_.size())},
          {"mapping_count", static_cast<std::uint64_t>(mappings_.size())},
          {"mapped_bytes", mapped_bytes}};
}

void PagedRegion::close_noexcept() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return;
  }
  closed_ = true;

  bool context_pushed = false;
  if (context_ != nullptr && cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
    context_pushed = true;
    (void)cuCtxSynchronize();
    for (const auto& entry : mappings_) {
      (void)cuMemUnmap(base_ + entry.first,
                       static_cast<std::size_t>(entry.second.bytes));
    }
    mappings_.clear();
    for (const auto& entry : blocks_) {
      (void)cuMemRelease(entry.second.handle);
    }
    blocks_.clear();
    if (base_ != 0 && virtual_bytes_ != 0) {
      (void)cuMemAddressFree(base_, static_cast<std::size_t>(virtual_bytes_));
      base_ = 0;
    }
  }
  if (context_pushed) {
    CUcontext ignored = nullptr;
    (void)cuCtxPopCurrent(&ignored);
  }
  if (context_ != nullptr) {
    (void)cuDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
  }
}

}  // namespace flexmoe
