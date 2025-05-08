// vector_add.cpp
#include <torch/extension.h>

//**************************************//
//Declaration of CUDA function wrappers//            
//**************************************//

torch::Tensor vector_add_cuda(const torch::Tensor& a, const torch::Tensor& b);
torch::Tensor matrix_add_cuda(const torch::Tensor& a, const torch::Tensor& b);
torch::Tensor matrix_mul_cuda(const torch::Tensor& a, const torch::Tensor& b);
torch::Tensor matrix_mul_tiled_cuda(const torch::Tensor& a, const torch::Tensor& b);

//******************************************************************//
// Python-visible function that handles both CPU and CUDA tensors   //
//******************************************************************//

// vector addition
torch::Tensor vector_add(const torch::Tensor& a, const torch::Tensor& b) 
{
    // Ensure same device
    TORCH_CHECK(a.device() == b.device(), "Input tensors must be on the same device");
    
    // Move tensors to CUDA if they're not already there
    // auto a_cuda = a.cuda();
    // auto b_cuda = b.cuda();
    
    return vector_add_cuda(a, b);
}

// matrix addition
torch::Tensor matrix_add(const torch::Tensor& a, const torch::Tensor& b)
{
    // Ensure same device
    TORCH_CHECK(a.device() == b.device(), "Input matrices must be on same device");

    // move tensors to CUDA if they're not already there
    // auto a_cuda = a.cuda();
    // auto b_cuda = b.cuda();

    return matrix_add_cuda(a, b);
}

// matric multiplication
torch::Tensor matrix_mul(const torch::Tensor& a, const torch::Tensor& b)
{
    // Ensure same device
    TORCH_CHECK(a.device() == b.device(), "Input matrices must be on same device");
    
    // Move tensors to CUDA if they're not already there
    // auto a_cuda = a.cuda();
    // auto b_cuda = b.cuda();

    return matrix_mul_cuda(a, b);
}

// matric multiplication
torch::Tensor matrix_mul_tiled(const torch::Tensor& a, const torch::Tensor& b)
{
    // Ensure same device
    TORCH_CHECK(a.device() == b.device(), "Input matrices must be on same device");
    
    // Move tensors to CUDA if they're not already there
    // auto a_cuda = a.cuda();
    // auto b_cuda = b.cuda();

    return matrix_mul_tiled_cuda(a, b);
}



//**************************************//
//           PYBIND bindings            //
//**************************************//
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Vector addition (CUDA)");
    m.def("matrix_add", &matrix_add, "Matrix addition (CUDA)");
    m.def("matrix_mul", &matrix_mul, "Matrix multiplication (CUDA)");
    m.def("matrix_mul_tiled", &matrix_mul_tiled, "Matrix multiplication tiled (CUDA)");
}