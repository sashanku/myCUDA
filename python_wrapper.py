# vector_add.py
import torch
import cuda_ops

# VECTOR ADD
class VectorAdd(torch.nn.Module):
    def __init__(self):
        super(VectorAdd, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Add two PyTorch tensors using CUDA.
        
        Args:
            a (torch.Tensor): First input tensor
            b (torch.Tensor): Second input tensor
            
        Returns:
            torch.Tensor: Result of element-wise addition
        """
        return cuda_ops.vector_add(a, b)

# Create a function for easier use
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Convenient function to add two tensors using our CUDA kernel.
    
    Args:
        a (torch.Tensor): First input tensor
        b (torch.Tensor): Second input tensor
        
    Returns:
        torch.Tensor: Result of element-wise addition
    """
    return VectorAdd()(a, b)


# MATRIX ADD
class MatrixAdd(torch.nn.Module):
    def __init__(self):
        super(MatrixAdd, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Add two PyTorch matrices using CUDA.
        
        Args:
            a (torch.Tensor): First input matrix
            b (torch.Tensor): Second input matrix
            
        Returns:
            torch.Tensor: Result of element-wise matrix addition
        """
        return cuda_ops.matrix_add(a, b)

# Create a function for easier use
def matrix_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Convenient function to add two matrices using our CUDA kernel.
    
    Args:
        a (torch.Tensor): First input matrix
        b (torch.Tensor): Second input matrix
        
    Returns:
        torch.Tensor: Result of element-wise matrix addition
    """
    return MatrixAdd()(a, b)

# MATRIX MULTIPLY
class MatrixMultiply(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiply, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Multiply two PyTorch matrices using CUDA.
        
        Args:
            a (torch.Tensor): First input matrix
            b (torch.Tensor): Second input matrix
            
        Returns:
            torch.Tensor: Result of matrix multiplication
        """
        return cuda_ops.matrix_mul(a, b)

# Create a function for easier use
def matrix_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Convenient function to multiply two matrices using our CUDA kernel.
    
    Args:
        a (torch.Tensor): First input matrix
        b (torch.Tensor): Second input matrix
        
    Returns:
        torch.Tensor: Result of matrix multiplication
    """
    return MatrixMultiply()(a, b)

# MATRIX MULTIPLY
class MatrixMultiplyTiled(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplyTiled, self).__init__()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Multiply two PyTorch matrices using CUDA.
        
        Args:
            a (torch.Tensor): First input matrix
            b (torch.Tensor): Second input matrix
            
        Returns:
            torch.Tensor: Result of matrix multiplication
        """
        return cuda_ops.matrix_mul_tiled(a, b)

# Create a function for easier use
def matrix_mul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Convenient function to multiply two matrices using our CUDA kernel.
    
    Args:
        a (torch.Tensor): First input matrix
        b (torch.Tensor): Second input matrix
        
    Returns:
        torch.Tensor: Result of matrix multiplication
    """
    return MatrixMultiplyTiled()(a, b)