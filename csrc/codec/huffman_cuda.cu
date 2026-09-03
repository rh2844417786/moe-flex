#include "huffman.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace flexmoe {
namespace {

constexpr int kThreads = 256;
constexpr std::int64_t kRequiredChunkElements = 4096;
constexpr int kSegmentThreads = 32;
constexpr std::int64_t kSegmentElements = 128;

cudaStream_t stream_from_handle(std::uint64_t handle) noexcept {
  return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(handle));
}

template <typename T>
void require_cuda_vector(const torch::Tensor& tensor, at::ScalarType dtype,
                         const char* name, int device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.get_device() == device, name, " is on the wrong device");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  (void)sizeof(T);
}

__global__ void decode_huffman_kernel(
    const std::uint8_t* sign_mantissa, const std::uint8_t* exponent_payload,
    const std::int64_t* chunk_byte_offsets,
    const std::int64_t* chunk_bit_lengths, const std::int16_t* trie_left,
    const std::int16_t* trie_right, const std::int16_t* trie_symbol,
    std::int64_t trie_nodes, std::uint16_t* destination, int* errors,
    std::int64_t element_count, std::int64_t chunk_elements) {
  extern __shared__ std::uint8_t exponent_cache[];
  __shared__ int chunk_error;
  const std::int64_t chunk_idx = blockIdx.x;
  const std::int64_t element_start = chunk_idx * chunk_elements;
  const std::int64_t remaining = element_count - element_start;
  const std::int64_t expected_symbols =
      remaining < chunk_elements ? remaining : chunk_elements;

  if (threadIdx.x == 0) {
    chunk_error = 0;
    const std::int64_t byte_offset = chunk_byte_offsets[chunk_idx];
    const std::int64_t bit_length = chunk_bit_lengths[chunk_idx];
    std::int64_t produced = 0;
    std::int64_t node = 0;
    for (std::int64_t bit_idx = 0; bit_idx < bit_length; ++bit_idx) {
      const std::uint8_t byte =
          exponent_payload[byte_offset + bit_idx / 8];
      const int bit = (byte >> (7 - (bit_idx % 8))) & 1;
      const std::int64_t child = bit == 0 ? trie_left[node] : trie_right[node];
      if (child < 0 || child >= trie_nodes) {
        chunk_error = 1;
        break;
      }
      node = child;
      const std::int16_t symbol = trie_symbol[node];
      if (symbol >= 0) {
        if (produced >= expected_symbols) {
          chunk_error = 2;
          break;
        }
        exponent_cache[produced++] = static_cast<std::uint8_t>(symbol);
        node = 0;
      }
    }
    if (chunk_error == 0 && node != 0) {
      chunk_error = 3;
    }
    if (chunk_error == 0 && produced != expected_symbols) {
      chunk_error = 4;
    }
    errors[chunk_idx] = chunk_error;
  }
  __syncthreads();

  if (chunk_error != 0) {
    return;
  }
  for (std::int64_t local_idx = threadIdx.x; local_idx < expected_symbols;
       local_idx += blockDim.x) {
    const std::int64_t global_idx = element_start + local_idx;
    const std::uint8_t packed = sign_mantissa[global_idx];
    const std::uint16_t mantissa = packed & 0x7fU;
    const std::uint16_t sign =
        static_cast<std::uint16_t>(packed & 0x80U) << 8U;
    const std::uint16_t exponent =
        static_cast<std::uint16_t>(exponent_cache[local_idx]) << 7U;
    destination[global_idx] = sign | exponent | mantissa;
  }
}

__global__ void decode_huffman_batched_kernel(
    const std::uint8_t* sign_mantissa, const std::uint8_t* exponent_payload,
    const std::int64_t* chunk_byte_offsets,
    const std::int64_t* chunk_bit_lengths,
    const std::int64_t* chunk_destination_offsets,
    const std::int64_t* chunk_element_counts,
    const std::int64_t* chunk_expert_indices,
    const int* segment_bit_offsets,
    const std::int64_t* expert_trie_offsets,
    const std::int64_t* expert_trie_node_counts,
    const std::int16_t* trie_left, const std::int16_t* trie_right,
    const std::int16_t* trie_symbol, std::int64_t expert_count,
    std::uint16_t* destination, int* errors, std::int64_t destination_elements) {
  const std::int64_t chunk_idx = blockIdx.x;
  const std::int64_t destination_start = chunk_destination_offsets[chunk_idx];
  const std::int64_t chunk_symbols = chunk_element_counts[chunk_idx];
  const std::int64_t expert_idx = chunk_expert_indices[chunk_idx];
  if (threadIdx.x == 0) {
    errors[chunk_idx] = 0;
  }
  __syncwarp();
  if (expert_idx < 0 || expert_idx >= expert_count || destination_start < 0 ||
      chunk_symbols <= 0 || chunk_symbols > kRequiredChunkElements ||
      destination_start + chunk_symbols > destination_elements) {
    if (threadIdx.x == 0) {
      errors[chunk_idx] = 5;
    }
    return;
  }
  const std::int64_t segment_idx = threadIdx.x;
  const std::int64_t segment_element_start = segment_idx * kSegmentElements;
  if (segment_element_start >= chunk_symbols) {
    return;
  }
  const std::int64_t segment_remaining =
      chunk_symbols - segment_element_start;
  const std::int64_t expected_symbols =
      segment_remaining < kSegmentElements ? segment_remaining
                                           : kSegmentElements;
  const std::int64_t metadata_idx = chunk_idx * kSegmentThreads + segment_idx;
  const std::int64_t bit_start = segment_bit_offsets[metadata_idx];
  const std::int64_t bit_end =
      segment_idx + 1 < kSegmentThreads &&
              segment_element_start + kSegmentElements < chunk_symbols
          ? segment_bit_offsets[metadata_idx + 1]
          : chunk_bit_lengths[chunk_idx];
  if (bit_start < 0 || bit_end <= bit_start ||
      bit_end > chunk_bit_lengths[chunk_idx]) {
    atomicCAS(errors + chunk_idx, 0, 6);
    return;
  }

  const std::int64_t trie_base = expert_trie_offsets[expert_idx];
  const std::int64_t trie_nodes = expert_trie_node_counts[expert_idx];
  const std::int64_t byte_offset = chunk_byte_offsets[chunk_idx];
  std::int64_t produced = 0;
  std::int64_t node = 0;
  for (std::int64_t bit_idx = bit_start; bit_idx < bit_end; ++bit_idx) {
    const std::uint8_t byte = exponent_payload[byte_offset + bit_idx / 8];
    const int bit = (byte >> (7 - (bit_idx % 8))) & 1;
    const std::int64_t child =
        bit == 0 ? trie_left[trie_base + node]
                 : trie_right[trie_base + node];
    if (child < 0 || child >= trie_nodes) {
      atomicCAS(errors + chunk_idx, 0, 1);
      return;
    }
    node = child;
    const std::int16_t symbol = trie_symbol[trie_base + node];
    if (symbol >= 0) {
      if (produced >= expected_symbols) {
        atomicCAS(errors + chunk_idx, 0, 2);
        return;
      }
      const std::int64_t global_idx =
          destination_start + segment_element_start + produced;
      const std::uint8_t packed = sign_mantissa[global_idx];
      const std::uint16_t mantissa = packed & 0x7fU;
      const std::uint16_t sign =
          static_cast<std::uint16_t>(packed & 0x80U) << 8U;
      const std::uint16_t exponent = static_cast<std::uint16_t>(symbol) << 7U;
      destination[global_idx] = sign | exponent | mantissa;
      ++produced;
      node = 0;
    }
  }
  if (node != 0) {
    atomicCAS(errors + chunk_idx, 0, 3);
  } else if (produced != expected_symbols) {
    atomicCAS(errors + chunk_idx, 0, 4);
  }
}

}  // namespace

void huffman_decode_cuda(
    const torch::Tensor& sign_mantissa,
    const torch::Tensor& exponent_payload,
    const torch::Tensor& chunk_byte_offsets,
    const torch::Tensor& chunk_bit_lengths, const torch::Tensor& trie_left,
    const torch::Tensor& trie_right, const torch::Tensor& trie_symbol,
    const torch::Tensor& destination, const torch::Tensor& errors,
    std::int64_t chunk_elements, std::uint64_t stream_handle) {
  TORCH_CHECK(destination.is_cuda(), "destination must be a CUDA tensor");
  TORCH_CHECK(destination.scalar_type() == at::kBFloat16,
              "destination must have BF16 dtype");
  TORCH_CHECK(destination.is_contiguous(), "destination must be contiguous");
  const int device = destination.get_device();
  c10::cuda::CUDAGuard device_guard(device);

  require_cuda_vector<std::uint8_t>(sign_mantissa, at::kByte,
                                    "sign_mantissa", device);
  require_cuda_vector<std::uint8_t>(exponent_payload, at::kByte,
                                    "exponent_payload", device);
  require_cuda_vector<std::int64_t>(chunk_byte_offsets, at::kLong,
                                    "chunk_byte_offsets", device);
  require_cuda_vector<std::int64_t>(chunk_bit_lengths, at::kLong,
                                    "chunk_bit_lengths", device);
  require_cuda_vector<std::int16_t>(trie_left, at::kShort, "trie_left",
                                    device);
  require_cuda_vector<std::int16_t>(trie_right, at::kShort, "trie_right",
                                    device);
  require_cuda_vector<std::int16_t>(trie_symbol, at::kShort, "trie_symbol",
                                    device);
  require_cuda_vector<int>(errors, at::kInt, "errors", device);

  TORCH_CHECK(chunk_elements == kRequiredChunkElements,
              "chunk_elements must be 4096");
  TORCH_CHECK(sign_mantissa.numel() == destination.numel(),
              "sign/mantissa element count does not match destination");
  TORCH_CHECK(chunk_byte_offsets.numel() == chunk_bit_lengths.numel(),
              "chunk metadata tensors have different lengths");
  TORCH_CHECK(errors.numel() == chunk_byte_offsets.numel(),
              "error tensor length does not match chunk count");
  TORCH_CHECK(trie_left.numel() == trie_right.numel() &&
                  trie_left.numel() == trie_symbol.numel(),
              "Huffman trie tensors have different lengths");
  TORCH_CHECK(trie_left.numel() > 1 && trie_left.numel() <= 511,
              "Huffman trie must contain 2 to 511 nodes");
  TORCH_CHECK(destination.numel() > 0, "destination must not be empty");
  TORCH_CHECK(chunk_byte_offsets.numel() > 0,
              "at least one chunk is required");
  TORCH_CHECK(chunk_byte_offsets.numel() <=
                  std::numeric_limits<unsigned int>::max(),
              "too many Huffman chunks for one launch");

  const auto chunk_count =
      static_cast<unsigned int>(chunk_byte_offsets.numel());
  decode_huffman_kernel<<<chunk_count, kThreads,
                          static_cast<std::size_t>(chunk_elements),
                          stream_from_handle(stream_handle)>>>(
      sign_mantissa.data_ptr<std::uint8_t>(),
      exponent_payload.data_ptr<std::uint8_t>(),
      chunk_byte_offsets.data_ptr<std::int64_t>(),
      chunk_bit_lengths.data_ptr<std::int64_t>(),
      trie_left.data_ptr<std::int16_t>(), trie_right.data_ptr<std::int16_t>(),
      trie_symbol.data_ptr<std::int16_t>(), trie_left.numel(),
      reinterpret_cast<std::uint16_t*>(destination.data_ptr<at::BFloat16>()),
      errors.data_ptr<int>(), destination.numel(), chunk_elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void huffman_decode_batched_cuda(
    const torch::Tensor& sign_mantissa,
    const torch::Tensor& exponent_payload,
    const torch::Tensor& chunk_byte_offsets,
    const torch::Tensor& chunk_bit_lengths,
    const torch::Tensor& chunk_destination_offsets,
    const torch::Tensor& chunk_element_counts,
    const torch::Tensor& chunk_expert_indices,
    const torch::Tensor& segment_bit_offsets,
    const torch::Tensor& expert_trie_offsets,
    const torch::Tensor& expert_trie_node_counts,
    const torch::Tensor& trie_left, const torch::Tensor& trie_right,
    const torch::Tensor& trie_symbol, const torch::Tensor& destination,
    const torch::Tensor& errors, std::int64_t chunk_elements,
    std::uint64_t stream_handle) {
  TORCH_CHECK(destination.is_cuda(), "destination must be a CUDA tensor");
  TORCH_CHECK(destination.scalar_type() == at::kBFloat16,
              "destination must have BF16 dtype");
  TORCH_CHECK(destination.is_contiguous(), "destination must be contiguous");
  const int device = destination.get_device();
  c10::cuda::CUDAGuard device_guard(device);

  require_cuda_vector<std::uint8_t>(sign_mantissa, at::kByte,
                                    "sign_mantissa", device);
  require_cuda_vector<std::uint8_t>(exponent_payload, at::kByte,
                                    "exponent_payload", device);
  require_cuda_vector<std::int64_t>(chunk_byte_offsets, at::kLong,
                                    "chunk_byte_offsets", device);
  require_cuda_vector<std::int64_t>(chunk_bit_lengths, at::kLong,
                                    "chunk_bit_lengths", device);
  require_cuda_vector<std::int64_t>(chunk_destination_offsets, at::kLong,
                                    "chunk_destination_offsets", device);
  require_cuda_vector<std::int64_t>(chunk_element_counts, at::kLong,
                                    "chunk_element_counts", device);
  require_cuda_vector<std::int64_t>(chunk_expert_indices, at::kLong,
                                    "chunk_expert_indices", device);
  require_cuda_vector<int>(segment_bit_offsets, at::kInt,
                           "segment_bit_offsets", device);
  require_cuda_vector<std::int64_t>(expert_trie_offsets, at::kLong,
                                    "expert_trie_offsets", device);
  require_cuda_vector<std::int64_t>(expert_trie_node_counts, at::kLong,
                                    "expert_trie_node_counts", device);
  require_cuda_vector<std::int16_t>(trie_left, at::kShort, "trie_left",
                                    device);
  require_cuda_vector<std::int16_t>(trie_right, at::kShort, "trie_right",
                                    device);
  require_cuda_vector<std::int16_t>(trie_symbol, at::kShort, "trie_symbol",
                                    device);
  require_cuda_vector<int>(errors, at::kInt, "errors", device);

  const auto chunk_count = chunk_byte_offsets.numel();
  TORCH_CHECK(chunk_elements == kRequiredChunkElements,
              "chunk_elements must be 4096");
  TORCH_CHECK(sign_mantissa.numel() == destination.numel(),
              "sign/mantissa element count does not match destination");
  TORCH_CHECK(chunk_count > 0 &&
                  chunk_count <= std::numeric_limits<unsigned int>::max(),
              "invalid batched Huffman chunk count");
  TORCH_CHECK(chunk_bit_lengths.numel() == chunk_count &&
                  chunk_destination_offsets.numel() == chunk_count &&
                  chunk_element_counts.numel() == chunk_count &&
                  chunk_expert_indices.numel() == chunk_count &&
                  errors.numel() == chunk_count,
              "batched chunk metadata tensors have different lengths");
  TORCH_CHECK(segment_bit_offsets.numel() == chunk_count * kSegmentThreads,
              "batched segment metadata has the wrong length");
  TORCH_CHECK(expert_trie_offsets.numel() > 0 &&
                  expert_trie_offsets.numel() ==
                      expert_trie_node_counts.numel(),
              "expert trie metadata tensors have different lengths");
  TORCH_CHECK(trie_left.numel() == trie_right.numel() &&
                  trie_left.numel() == trie_symbol.numel(),
              "Huffman trie tensors have different lengths");

  decode_huffman_batched_kernel<<<
      static_cast<unsigned int>(chunk_count), kSegmentThreads, 0,
      stream_from_handle(stream_handle)>>>(
      sign_mantissa.data_ptr<std::uint8_t>(),
      exponent_payload.data_ptr<std::uint8_t>(),
      chunk_byte_offsets.data_ptr<std::int64_t>(),
      chunk_bit_lengths.data_ptr<std::int64_t>(),
      chunk_destination_offsets.data_ptr<std::int64_t>(),
      chunk_element_counts.data_ptr<std::int64_t>(),
      chunk_expert_indices.data_ptr<std::int64_t>(),
      segment_bit_offsets.data_ptr<int>(),
      expert_trie_offsets.data_ptr<std::int64_t>(),
      expert_trie_node_counts.data_ptr<std::int64_t>(),
      trie_left.data_ptr<std::int16_t>(), trie_right.data_ptr<std::int16_t>(),
      trie_symbol.data_ptr<std::int16_t>(), expert_trie_offsets.numel(),
      reinterpret_cast<std::uint16_t*>(destination.data_ptr<at::BFloat16>()),
      errors.data_ptr<int>(), destination.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace flexmoe
