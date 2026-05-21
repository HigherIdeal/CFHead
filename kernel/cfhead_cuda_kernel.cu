#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

__global__ void cfhead_coarse_kernel(
    const float* __restrict__ lut,
    const uint8_t* __restrict__ index_t,
    float* __restrict__ out,
    int batch_size,
    int num_tiles,
    int cluster_size,
    int vocab_size) {
    const int vocab_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y;
    if (vocab_id >= vocab_size || batch_id >= batch_size) {
        return;
    }

    float acc = 0.0f;
    const int lut_batch_offset = batch_id * num_tiles * cluster_size;

    #pragma unroll 4
    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        const uint8_t cluster = index_t[tile_id * vocab_size + vocab_id];
        acc += lut[lut_batch_offset + tile_id * cluster_size + static_cast<int>(cluster)];
    }

    out[batch_id * vocab_size + vocab_id] = acc;
}

__inline__ __device__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename scalar_t>
__global__ void cfhead_fine_kernel(
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ candidate_idx,
    float* __restrict__ out,
    int batch_size,
    int top_k,
    int hidden_size) {
    const int candidate_pos = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int tid = threadIdx.x;

    if (candidate_pos >= top_k || batch_id >= batch_size) {
        return;
    }

    const int64_t token_id = candidate_idx[batch_id * top_k + candidate_pos];
    const scalar_t* hidden_row = hidden + batch_id * hidden_size;
    const scalar_t* weight_row = weight + token_id * hidden_size;

    float acc = 0.0f;
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        acc += static_cast<float>(hidden_row[h]) * static_cast<float>(weight_row[h]);
    }

    acc = warp_reduce_sum(acc);

    __shared__ float warp_sums[32];
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = acc;
    }
    __syncthreads();

    const int warp_count = blockDim.x >> 5;
    float block_sum = (tid < warp_count) ? warp_sums[tid] : 0.0f;
    if (tid < 32) {
        block_sum = warp_reduce_sum(block_sum);
    }
    if (tid == 0) {
        out[batch_id * top_k + candidate_pos] = block_sum;
    }
}

}  // namespace

void cfhead_coarse_forward_cuda(torch::Tensor lut, torch::Tensor index_t, torch::Tensor out) {
    const int batch_size = static_cast<int>(lut.size(0));
    const int num_tiles = static_cast<int>(lut.size(1));
    const int cluster_size = static_cast<int>(lut.size(2));
    const int vocab_size = static_cast<int>(index_t.size(1));

    const int threads = 256;
    const dim3 blocks((vocab_size + threads - 1) / threads, batch_size);
    auto stream = at::cuda::getCurrentCUDAStream();

    cfhead_coarse_kernel<<<blocks, threads, 0, stream>>>(
        lut.data_ptr<float>(),
        index_t.data_ptr<uint8_t>(),
        out.data_ptr<float>(),
        batch_size,
        num_tiles,
        cluster_size,
        vocab_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cfhead_fine_forward_cuda(torch::Tensor hidden, torch::Tensor weight, torch::Tensor candidate_idx, torch::Tensor out) {
    const int batch_size = static_cast<int>(hidden.size(0));
    const int hidden_size = static_cast<int>(hidden.size(1));
    const int top_k = static_cast<int>(candidate_idx.size(1));

    const int threads = 256;
    const dim3 blocks(top_k, batch_size);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        hidden.scalar_type(),
        "cfhead_fine_forward_cuda",
        [&] {
            cfhead_fine_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                hidden.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                candidate_idx.data_ptr<int64_t>(),
                out.data_ptr<float>(),
                batch_size,
                top_k,
                hidden_size);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
