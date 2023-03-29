// our code is based-off the official codes of ActNN and EXACT.
// ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training.
// Backprop with Approximate Activations for Memory-efficient Network Training.
// EXACT: scalable graph neural networks training via extreme activation compression

#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#include <c10/cuda/CUDAStream.h>

#define BLOCK_Y_DIM_MAX ((1l << 16) - 1) // 2^16 -1

using torch::IntArrayRef;
using torch::Tensor;
using at::cuda::CUDAStream;


// Pack float16/32 data into int8 bit stream (device function)
template<typename scalar_t, bool boundary_check>
__global__ void pack_single_precision_kernel(int32_t bits, const scalar_t* __restrict__ data, const scalar_t* __restrict__ scale, const scalar_t* __restrict__ min, int8_t* __restrict__ packed, std::pair<uint64_t, uint64_t> seeds, int64_t N, int64_t group_size, int64_t block_idx_y_base) {
  const int64_t no = blockIdx.x + block_idx_y_base;
  const int d = threadIdx.x;
  const int work_per_thread = 8 / bits;
  const int64_t global_thread_id = no * group_size + d; // idx in packed array (group_size is fixed, no is determined by work_per_thread)
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_thread_id, seeds.second, &state);
  uint8_t local_packed = 0;
  for (int ni = 0; ni < work_per_thread; ni++) {
    const int64_t n = no * work_per_thread + ni; // idx for retrieve quant params (mul worker_per_thread to recover original batch idx)
    if (boundary_check && n >= N) { break; }  // check boundary because there exists some threads which may exceed the batch size when their work is large than 1
    const int64_t id = n * group_size + d; // idx fir retrieve input data (batch idx * feat_dim + thread_idx)
    const float noise = curand_uniform(&state);
    const int32_t val = __float2int_rn(fmax((data[id] - min[n]) * scale[n] + noise - 0.5, 0.0f)); // stochastical quantization
    local_packed |= (val << (ni * bits)); // move cal to corresonding location 
  }
  packed[global_thread_id] = local_packed; // insert one uint8 data (maybe contain several original quantized data)
}

// Pack float16/32 data into int8 bit stream (host function)
Tensor pack_single_precision_cuda(Tensor data, Tensor min, Tensor max, Tensor scale, int bits, bool stochastic, CUDAStream stream) {
  int64_t N = data.size(0);  // batch size
  int64_t group_size = data.size(1);  // block_Size
  // Compute total bits
  int work_per_thread = 8 / bits;
  TORCH_CHECK(8 % bits == 0);
  int64_t N_round = N + (work_per_thread - N % work_per_thread) % work_per_thread;  // N_round must be the multiple of worke_per_thread
  int64_t total_bits = (int64_t)bits * (N_round * group_size);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(data.device());  // set dtype to char (torch::kint8)
  Tensor packed = torch::empty({(total_bits + 8) / 8,}, options);  // actual 8 bits stream shape (must be the multiple of 8)
  // Random number generator 
  auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(group_size * work_per_thread);
  }
  TORCH_CHECK(stochastic);
  // Call pack kernels
  int64_t logical_block_y_dim = (N + work_per_thread - 1) / work_per_thread;
  // use for loop in case that all data can not be dealed at one iteration
  for (int64_t block_idx_y_base = 0; block_idx_y_base < logical_block_y_dim; block_idx_y_base += BLOCK_Y_DIM_MAX) {
    dim3 block_dim(std::min(logical_block_y_dim - block_idx_y_base, BLOCK_Y_DIM_MAX));  // 1-d block-dim (if logical_block_y_dim is too large, the block_dim can only be set to BLOCK_Y_DIM_MAX)
    dim3 thread_dim(group_size);  // split the thread from feat_dim view
    if (N % work_per_thread == 0) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_single_precision", ([&] {
        pack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim, 0, stream>>>(
          bits,
          data.data_ptr<scalar_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          packed.data_ptr<int8_t>(),
          rng_engine_inputs,
          N, group_size, block_idx_y_base);
      }));
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_single_precision", ([&] {
        pack_single_precision_kernel<scalar_t, true><<<block_dim, thread_dim, 0, stream>>>(
          bits,
          data.data_ptr<scalar_t>(),
          scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
          packed.data_ptr<int8_t>(),
          rng_engine_inputs,
          N, group_size, block_idx_y_base);
      }));
    }
  }
  return packed;
}

// Unpack int32 bit stream to float16/32 data (device function)
template<typename scalar_t, bool boundary_check>
__global__ void unpack_single_precision_kernel(int32_t bits, const int8_t* __restrict__ data, const scalar_t* __restrict__ scale, const scalar_t* __restrict__ min, scalar_t* __restrict__ unpacked, int64_t N, int group_size, int64_t num_blocks) {
  for(int64_t no=blockIdx.x; no < num_blocks; no += gridDim.x){ // why need to loop all the blocks ?
    const int d = threadIdx.x;
    const int64_t global_thread_id = no * group_size + d;
    int work_per_thread = 8 / bits;
    uint8_t local_packed = data[global_thread_id];
    int mask = ((1 << bits) - 1);
    for (int ni = 0; ni < work_per_thread; ni++) {
      const int64_t n = no * work_per_thread + ni;
      if (boundary_check && n >= N) { break; }
      const int val = (local_packed >> (ni * bits)) & mask;
      const int64_t id = n * group_size + d;
      unpacked[id] = ((scalar_t)val) / scale[n] + min[n];
    }
  }
}

// Unpack int32 bit stream to float16/32 data (host function)
Tensor unpack_single_precision_cuda(Tensor data, int bits, Tensor scale, Tensor min, int64_t N, int group_size, CUDAStream stream) {
  auto options = torch::TensorOptions().dtype(scale.dtype()).device(data.device());
  Tensor unpacked = torch::empty({N, group_size}, options); // recover to default shape
  int work_per_thread = 8 / bits;
  TORCH_CHECK(8 % bits == 0);
  // Call unpack kernels
  int64_t num_blocks = (N + work_per_thread - 1) / work_per_thread;
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/group_size;
  dim3 dim_block(group_size); 
  dim3 grid(num_blocks); 
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
  if (N % work_per_thread == 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
      unpack_single_precision_kernel<scalar_t, false><<<grid, dim_block, 0, stream>>>(
        bits,
        data.data_ptr<int8_t>(),
        scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
        unpacked.data_ptr<scalar_t>(),
        N, group_size, num_blocks);
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
      unpack_single_precision_kernel<scalar_t, true><<<grid, dim_block, 0, stream>>>(
        bits,
        data.data_ptr<int8_t>(),
        scale.data_ptr<scalar_t>(), min.data_ptr<scalar_t>(),
        unpacked.data_ptr<scalar_t>(),
        N, group_size, num_blocks);
    }));
  }
  return unpacked;
}
