# myCUDA

## Overview
This repository serves as a scratch pad for my thoughts and implementations as I explore, implement, and optimize CUDA kernels. The implementations found here are experimental in nature and not intended for production use. The main purpose is to:

1. Speculate on various design choices for CUDA kernel implementations
2. Implement these design choices to test their viability
3. Benchmark the performance against industry-standard implementations (e.g., cuBLASS, CUDNN)
4. Document findings and optimizations

## Current Implementations

### MatMul
- **Performance**: Achieved 92% of cuBLASS performance on NVIDIA Ampere architecture
- **In Progress**: Developing a double/triple buffered approach with asynchronous loads and tensor core math
- **Note**: Some kernels have intentionally modified dimensions for experimental purposes. These will need to be re-tuned for optimal performance.

### Softmax
- **Performance**: Achieved 104% of cuDNN performance on Ampere for large matrices and 132% for moderate-sized ones
- **In Progress**: Implementing a block tiling based approach for further optimization

### TopK FA2
- **Current Work**: Implementing a topK FA2 kernel to reduce time complexity from O(n²) to O([TIME COMPLEXITY TO BE DETERMINED])
- This implementation aims to significantly improve performance for top-k operations

## Usage/Testing

```bash
# Clone the repository
git clone https://github.com/sashanku/myCUDA.git

# Navigate to the repository directory
cd myCUDA

# Build the project
./build.sh

# Run benchmarks with a specific kernel ID
cd build
./benchmark_softmax/matmul [kernel_id]

# Example:
./benchmark_softmax 05
```

## Disclaimer
These implementations are experimental and intended for learning and benchmarking purposes. Performance results may vary across different hardware configurations and CUDA versions.