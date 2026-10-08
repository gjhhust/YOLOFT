#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
template <typename scalar_t>
std::vector<torch::Tensor> corr_cuda_forward_impl(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius, int stride);

std::vector<torch::Tensor> corr_cuda_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius, int stride);

// CUDA backward declarations
template <typename scalar_t>
std::vector<torch::Tensor> corr_cuda_backward_impl(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius, int stride);

std::vector<torch::Tensor> corr_cuda_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius, int stride);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> corr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius, int stride) {
  
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  
  TORCH_CHECK(fmap1.dtype() == torch::kFloat16 || fmap1.dtype() == torch::kFloat32,
              "Input must be FP16 or FP32");
  
  if (fmap1.dtype() == torch::kFloat16) {
    return corr_cuda_forward_impl<at::Half>(fmap1, fmap2, coords, radius, stride);
  } else {
    return corr_cuda_forward_impl<float>(fmap1, fmap2, coords, radius, stride);
  }
}

std::vector<torch::Tensor> corr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius, int stride) {
  
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);
  
  TORCH_CHECK(fmap1.dtype() == torch::kFloat16 || fmap1.dtype() == torch::kFloat32,
              "Input must be FP16 or FP32");
  
  if (fmap1.dtype() == torch::kFloat16) {
    return corr_cuda_backward_impl<at::Half>(fmap1, fmap2, coords, corr_grad, radius, stride);
  } else {
    return corr_cuda_backward_impl<float>(fmap1, fmap2, coords, corr_grad, radius, stride);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &corr_forward, "CORR forward");
  m.def("backward", &corr_backward, "CORR backward");
}