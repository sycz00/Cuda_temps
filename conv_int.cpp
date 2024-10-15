#include <torch/extension.h>



// accept stride and padding
torch::Tensor tensor_core_sp_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h,
        int32_t padding_h);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);



// stride, padding, group
torch::Tensor sp_conv(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h,
        int32_t padding_h){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv(input, weight, stride_h, padding_h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{ m.def("run", &sp_conv, "int8 convolution forward Nvidia GPU tensor core"); }
