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

}  // namespace flexmoe
