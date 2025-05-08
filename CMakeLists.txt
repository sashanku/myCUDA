cmake_minimum_required(VERSION 3.10)
project(custom_kernels_lib LANGUAGES CXX CUDA)

# Set CUDA architectures if needed (optional)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find required packages
find_package(CUDA REQUIRED)

# Manual cuDNN detection (add these lines)
# Set the path to your cuDNN installation
set(CUDNN_ROOT_DIR "/usr" CACHE PATH "Path to cuDNN installation")
set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT_DIR}/include" CACHE PATH "Path to cuDNN include directory")
set(CUDNN_LIBRARY_DIR "${CUDNN_ROOT_DIR}/lib/x86_64-linux-gnu" CACHE PATH "Path to cuDNN library directory")

# Find cuDNN library
find_library(CUDNN_LIBRARY 
    NAMES cudnn
    HINTS ${CUDNN_LIBRARY_DIR}
    REQUIRED
)

# Set CUDNN includes
set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})

# Print CUDNN paths for debugging
message(STATUS "Using CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
message(STATUS "Using CUDNN_INCLUDE_DIRS: ${CUDNN_INCLUDE_DIRS}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS}  # Add cuDNN include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include #cutlass
)

# Collect all kernel source files
file(GLOB MATMUL_KERNEL_SOURCES "src/kernels/matmul/kernel*.cu") #MatMul kernels
file(GLOB SOFTMAX_KERNEL_SOURCES "src/kernels/softmax/kernel*.cu")  #Softmax kernels
file(GLOB FA_KERNEL_SOURCES "src/kernels/FA2/kernel*.cu")  #FA kernels


# Print collected kernel sources (for debugging)
message(STATUS "Found matmul kernel sources: ${MATMUL_KERNEL_SOURCES}")
message(STATUS "Found softmax kernel sources: ${SOFTMAX_KERNEL_SOURCES}")  # Add this if you have softmax kernels
message(STATUS "Found FA kernel sources: ${FA_KERNEL_SOURCES}")     
# Add the matmul benchmark executable
add_executable(benchmark_matmul 
    utils/benchmark.cu
    ${MATMUL_KERNEL_SOURCES}
)

# Add the softmax benchmark executable (if needed)
add_executable(benchmark_softmax
    utils/benchmark_softmax.cu  # Make sure this file exists
    ${SOFTMAX_KERNEL_SOURCES}
)
# Add the softmax benchmark executable (if needed)
# add_executable(benchmark_topK
#     utils/benchmark_topKFA2.cu  # Make sure this file exists
#     ${FA_KERNEL_SOURCES}
# )


# Link libraries for matmul benchmark
target_link_libraries(benchmark_matmul
    ${CUDA_LIBRARIES}
    cublas
)

# Link libraries for softmax benchmark (with cuDNN)
target_link_libraries(benchmark_softmax
    ${CUDA_LIBRARIES}
    cublas
    ${CUDNN_LIBRARY}  # Link against cuDNN
)

# # Link libraries for topK benchmark
# target_link_libraries(benchmark_topK
#     ${CUDA_LIBRARIES}
#     cublas
#     ${CUDNN_LIBRARY}  # Link against cuDNN
# )

# Set some compiler flags (optional)
set_target_properties(benchmark_matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(benchmark_softmax PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Installation rules
install(TARGETS benchmark_matmul benchmark_softmax 
    RUNTIME DESTINATION bin
)

# Optional: Create a directory structure if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/softmax)