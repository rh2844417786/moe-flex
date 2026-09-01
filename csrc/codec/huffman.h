#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace flexmoe {

void huffman_decode_cuda(
    const torch::Tensor& sign_mantissa,
    const torch::Tensor& exponent_payload,
    const torch::Tensor& chunk_byte_offsets,
    const torch::Tensor& chunk_bit_lengths, const torch::Tensor& trie_left,
    const torch::Tensor& trie_right, const torch::Tensor& trie_symbol,
    const torch::Tensor& destination, const torch::Tensor& errors,
    std::int64_t chunk_elements, std::uint64_t stream_handle);

}  // namespace flexmoe
