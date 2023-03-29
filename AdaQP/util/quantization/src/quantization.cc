/*
 * Cuda operators for quantization and packing
 */

#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "check.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;
using torch::IntArrayRef;
using at::cuda::CUDAStream;

Tensor pack_single_precision_cuda(
    Tensor data, Tensor min, Tensor max, Tensor scale, int bits, bool stochastic, CUDAStream stream);
Tensor unpack_single_precision_cuda(
    Tensor data, int bits, Tensor scale, Tensor min, int64_t N, int group_size, CUDAStream stream); // param N is used to recover tensor shape

Tensor pack_single_precision(Tensor data, Tensor min, Tensor max, Tensor scale, int bits, bool stochastic) {
  CHECK_CUDA_TENSOR_DIM_FLOAT(data, 2);
  CHECK_CUDA_TENSOR_DIM_FLOAT(min, 1);
  CHECK_CUDA_TENSOR_DIM_FLOAT(max, 1);
  CHECK_CUDA_TENSOR_DIM_FLOAT(scale, 1);
  CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
  {
    at::cuda::CUDAStreamGuard guard(current_stream);
    return pack_single_precision_cuda(data, min, max, scale, bits, stochastic, current_stream);
  }
}

Tensor unpack_single_precision(Tensor data, int bits, Tensor scale, Tensor min, int64_t N, int64_t group_size) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);
  CHECK_CUDA_TENSOR_DIM_FLOAT(scale, 1);
  CHECK_CUDA_TENSOR_DIM_FLOAT(min, 1);
  CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
  {
    at::cuda::CUDAStreamGuard guard(current_stream);
    return unpack_single_precision_cuda(data, bits, scale, min, N, group_size, current_stream);
  }

}

// bind to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_single_precision", &pack_single_precision);
  m.def("unpack_single_precision", &unpack_single_precision);
}