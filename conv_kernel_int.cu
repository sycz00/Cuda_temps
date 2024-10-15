#include <torch/extension.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cuda_runtime.h"




torch::Tensor tensor_core_sp_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h,
        int32_t padding_h){
    // The code section below describes datatype for input, output tensors and computation between
    // elements
    using ElementAccumulator = int32_t;                 // Data type of accumulator
    using ElementComputeEpilogue = float;               // Data type of epilogue computation (alpha, beta)

    using ElementInputA = int8_t;             // Data type of elements in input tensor
    using ElementInputB = int8_t;             // Data type of elements in input tensor
    using ElementOutput = int32_t;             // Data type of elements in output tensor

    using LayoutInputA = cutlass::layout::TensorNHWC;
    using LayoutInputB = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm75;

    // This code section describes the tile size a thread block will compute
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;  // Threadblock tile shape

    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;         // Warp tile shape

    // This code section describes the size of MMA op
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;    // TensorCore instruction shape

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // Number of pipelines you want to use
    constexpr int NumStages = 2;

    // This code section describes the epilogue part of the kernel, we use default value
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
        ElementOutput,                                     // Data type of output matrix.
        //128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
        4,
                                                           // memory access. This becomes the vector width of
                                                           // math instructions in the epilogue too.
        ElementAccumulator,                                // Data type of accumulator
        ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination


    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    // mode (kCrossCorrelation or kConvolution)
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    cutlass::Tensor4DCoord input_size;
    input_size.n() = input.size(0);
    input_size.h() = input.size(1);
    input_size.w() = input.size(2);
    input_size.c() = input.size(3);
    //std::cout<<input_size.n()<<" "<<input_size.h()<<" "<<input_size.w()<<" "<<input_size.c()<<" "<<std::endl;
    cutlass::Tensor4DCoord filter_size;
    filter_size.n() = weight.size(0);
    filter_size.h() = weight.size(1);
    filter_size.w() = weight.size(2);
    filter_size.c() = weight.size(3);

    
    cutlass::Tensor4DCoord padding;
    padding = {padding_h, padding_h, padding_h, padding_h};

    cutlass::MatrixCoord conv_stride;
    conv_stride = {stride_h, stride_h};

    cutlass::MatrixCoord dilation;
    dilation = {1, 1};

    cutlass::Tensor4DCoord output_size;
    output_size.n() = input.size(0);
    output_size.h() = (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1;
    output_size.w() = (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1;
    output_size.c() = weight.size(0);
    auto y = torch::empty({output_size.n(), output_size.h(), output_size.w(), output_size.c()}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    cutlass::conv::Conv2dProblemSize problem_size(      
            input_size,
            filter_size,
            padding,
            conv_stride,
            dilation,
            output_size,
            mode,
            split_k_slices);

    //TensorRef<ElementInputA, LayoutInputA> input_ref();
    cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(input.data_ptr<int8_t>(),LayoutInputA::packed(input_size));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(weight.data_ptr<int8_t>(), LayoutInputB::packed(filter_size));
    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(y.data_ptr<int32_t>(), LayoutOutput::packed(output_size));
    typename ImplicitGemm::Arguments arguments{problem_size,
        input_ref,
        weight_ref,
        output_ref,
        output_ref,
        {1, 0},
    };
    ImplicitGemm implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    // Allocate workspace memory
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::dtype(torch::kUInt8).device(torch::kCUDA, 0));

    cutlass::Status status;

    status = implicit_gemm_op.initialize(arguments, workspace.data_ptr<uint8_t>());
    if (status != cutlass::Status::kSuccess) {
                                                        
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(status) << " at: " << __LINE__ 
                << std::endl;                                                                    
      exit(EXIT_FAILURE);                                                                        
         
    }

    status = implicit_gemm_op();
    
    if (status != cutlass::Status::kSuccess) {
                                                        
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(status) << " at: " << __LINE__ 
                << std::endl;                                                                    
      exit(EXIT_FAILURE);                                                                        
         
    }

    return y;
}




