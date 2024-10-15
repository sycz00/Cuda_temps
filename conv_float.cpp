#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>

// CUDA forward declarations
at::Tensor conv_layer_kernel(
    const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,
    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1},
    float alpha=1.f, float beta=0.f,
    std::string split_k_mode="serial", int split_k_slices=1);

// C++ interface
at::Tensor conv_layer(
    const at::Tensor& A, const at::Tensor& B, at::optional<const at::Tensor> C=at::nullopt,
    std::tuple<int, int> stride={1, 1}, std::tuple<int, int> padding={0, 0}, std::tuple<int, int> dilation={1, 1},
    float alpha=1.f, float beta=0.f,
    std::string split_k_mode="serial", int split_k_slices=1) {
    return conv_layer_kernel(A, B, C, stride, padding, dilation, alpha, beta, split_k_mode, split_k_slices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run",
  py::overload_cast<
    const at::Tensor&, const at::Tensor&, at::optional<const at::Tensor>,
    std::tuple<int, int>, std::tuple<int, int>, std::tuple<int, int>, float, float,  std::string, int>(
        &conv_layer), py::arg("A"), py::arg("B"), py::arg("C") = nullptr,
        py::arg("stride") = std::make_tuple(1, 1), py::arg("padding") = std::make_tuple(1, 1), py::arg("dilation") = std::make_tuple(1, 1),
        py::arg("alpha") = 1.f, py::arg("beta") = 0.f,
        py::arg("split_k_mode") = "serial", py::arg("split_k_slices") = 1);
}
