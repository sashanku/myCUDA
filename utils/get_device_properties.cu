#include <stdio.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

// Helper function to convert bytes to GB
float bytesToGB(size_t bytes) {
    return static_cast<float>(bytes) / (1024*1024*1024);
}

// Helper function to print memory info
void printMemoryInfo() {
    size_t free, total;
    cudaError_t error = cudaMemGetInfo(&free, &total);
    checkCudaError(error, "Failed to get memory info");
    
    printf("Memory Information:\n");
    printf("Total GPU Memory: %.2f GB\n", bytesToGB(total));
    printf("Free GPU Memory: %.2f GB\n", bytesToGB(free));
    printf("Used GPU Memory: %.2f GB\n", bytesToGB(total - free));
}

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    checkCudaError(error, "Failed to get CUDA device count");

    printf("Found %d CUDA devices\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp props;
        error = cudaGetDeviceProperties(&props, i);
        checkCudaError(error, "Failed to get device properties");

        printf("Device %d: %s\n", i, props.name);
        printf("============================================\n");
        
        // Architecture Information
        printf("Architecture Information:\n");
        printf("Compute Capability: %d.%d\n", props.major, props.minor);
        printf("GPU Architecture: ");
        if (props.major == 7) printf("Volta/Turing");
        else if (props.major == 8) printf("Ampere");
        else if (props.major == 9) printf("Hopper/Ada Lovelace");
        else printf("Unknown/Other");
        printf("\n");
        
        // Tensor Core Support and Count
        bool hasTensorCores = (props.major >= 7);
        printf("Tensor Cores: %s\n", hasTensorCores ? "Yes" : "No");
        if (hasTensorCores) {
            int tensorsPerSM = 0;
            if (props.major == 7) {
                if (props.minor == 0) tensorsPerSM = 8;  // Volta (V100)
                else if (props.minor == 5) tensorsPerSM = 8;  // Turing
            }
            else if (props.major == 8) tensorsPerSM = 4;  // Ampere
            else if (props.major == 9) tensorsPerSM = 4;  // Hopper/Ada Lovelace
            
            int totalTensorCores = tensorsPerSM * props.multiProcessorCount;
            printf("Tensor Cores per SM: %d\n", tensorsPerSM);
            printf("Total Tensor Cores: %d\n", totalTensorCores);
            
            // Theoretical tensor core throughput (FP16 TFLOPS)
            float clockGHz = static_cast<float>(props.clockRate) / 1e6;
            float tensorOpsPerClock = 0.0f;
            
            // Operations per clock cycle per Tensor Core
            if (props.major == 7) tensorOpsPerClock = 64.0f;  // Volta/Turing
            else if (props.major == 8) tensorOpsPerClock = 256.0f;  // Ampere
            else if (props.major == 9) tensorOpsPerClock = 512.0f;  // Hopper
            
            float theoreticalTensorTFLOPS = (tensorOpsPerClock * totalTensorCores * clockGHz) / 1000.0f;
            printf("Theoretical Tensor Core Performance (FP16): %.2f TFLOPS\n", theoreticalTensorTFLOPS);
        }
        
        // Ray Tracing Support (RTX)
        bool hasRTX = (props.major >= 7 && props.minor >= 5);
        printf("RT Cores (RTX): %s\n\n", hasRTX ? "Yes" : "No");

        // Core Configuration
        printf("Core Configuration:\n");
        printf("Number of SMs: %d\n", props.multiProcessorCount);
        printf("Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
        printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
        printf("Warp size: %d\n", props.warpSize);
        printf("Maximum warps per SM: %d\n", 
            props.maxThreadsPerMultiProcessor / props.warpSize);
        printf("Maximum blocks per SM: %d\n\n", 
            props.maxBlocksPerMultiProcessor);

        // Memory Hierarchy
        printf("Memory Hierarchy:\n");
        printf("Total global memory: %.2f GB\n", bytesToGB(props.totalGlobalMem));
        printf("L2 cache size: %d KB\n", props.l2CacheSize / 1024);
        printf("Shared memory per SM: %zu KB\n", props.sharedMemPerMultiprocessor / 1024);
        printf("Shared memory per block: %zu KB\n", props.sharedMemPerBlock / 1024);
        printf("Total constant memory: %zu KB\n", props.totalConstMem / 1024);
        printf("Memory clock rate: %.2f GHz\n", static_cast<float>(props.memoryClockRate) / 1e6);
        printf("Memory bus width: %d bits\n", props.memoryBusWidth);
        printf("Peak memory bandwidth: %.2f GB/s\n\n",
            2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6);

        // Execution Configuration
        printf("Execution Configuration:\n");
        printf("Max block dimensions: [%d, %d, %d]\n", 
            props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
        printf("Max grid dimensions: [%d, %d, %d]\n", 
            props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
        printf("Clock rate: %.2f GHz\n\n", static_cast<float>(props.clockRate) / 1e6);

        // Advanced Features
        printf("Advanced Features:\n");
        printf("Concurrent kernels: %s\n", props.concurrentKernels ? "Yes" : "No");
        printf("Async engine count: %d\n", props.asyncEngineCount);
        printf("Unified addressing: %s\n", props.unifiedAddressing ? "Yes" : "No");
        printf("Compute mode: %d\n", props.computeMode);
        printf("ECC enabled: %s\n", props.ECCEnabled ? "Yes" : "No");
        // Get stream priority range if supported
        int priority_low, priority_high;
        if (props.streamPrioritiesSupported) {
            cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
            printf("Stream priority range: [%d, %d]\n", priority_low, priority_high);
        } else {
            printf("Stream priorities: Not supported\n");
        printf("Cooperative launch: %s\n", props.cooperativeLaunch ? "Yes" : "No");
        printf("Multi-GPU board: %s\n", props.isMultiGpuBoard ? "Yes" : "No");
        }
        printf("Integrated GPU: %s\n\n", props.integrated ? "Yes" : "No");

        // Current Memory Status
        printf("Current ");
        printMemoryInfo();
        printf("\n");
    }

    return 0;
}