#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include "kernels.cuh"  // Include your softmax kernel header

// Helper function for cuDNN error handling
#define checkCUDNN(expression)                               \
{                                                            \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
        fprintf(stderr, "Error on line %d: %s\n",            \
                __LINE__, cudnnGetErrorString(status));      \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
}

// cuDNN implementation of softmax for reference
void cudnn_softmax(cudnnHandle_t cudnn, float *input, float *output, int batch_size, int vector_size) {
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch_size, 1, 1, vector_size));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch_size, 1, 1, vector_size));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Use cuDNN's softmax implementation
    checkCUDNN(cudnnSoftmaxForward(cudnn,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_INSTANCE,
                                   &alpha,
                                   input_descriptor,
                                   input,
                                   &beta,
                                   output_descriptor,
                                   output));

    // Clean up
    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
}


void randomize_data(float *data, int size) {
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < size; i++) {
        // Generate values in a reasonable range for softmax (-10 to 10)
        data[i] = (float)(rand() % 2000) / 100.0f - 10.0f;
    }
}

bool verify_results(float *cudnn_result, float *gpu_result, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        float diff = fabsf(cudnn_result[i] - gpu_result[i]);
        if (diff > tolerance) {
            printf("Verification failed at index %d: cuDNN %f vs GPU %f (diff: %f)\n", 
                   i, cudnn_result[i], gpu_result[i], diff);
            return false;
        }
    }
    return true;
}

void test_kernel(int kernel_num, int batch_size, int vector_size, float *d_input, float *d_output) {
    switch (kernel_num) {
        case 1:
            launch_softmax_kernel01(d_input, d_output, vector_size, batch_size);
            break;
        case 2:
            launch_softmax_kernel02(d_input, d_output, batch_size, vector_size);
            break;
        case 3:
            launch_softmax_kernel03(d_input, d_output, batch_size, vector_size);
            break;
        case 4:
            launch_softmax_kernel04(d_input, d_output, batch_size, vector_size);
            break;
        case 5:
            launch_softmax_kernel05(d_input, d_output, batch_size, vector_size);
            break;
        // Add more cases as you develop more kernels
        default:
            printf("Invalid kernel number\n");
            exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <kernel_number>\n", argv[0]);
        printf("Kernel numbers: 1-N (custom kernels)\n");
        return 1;
    }

    int kernel_num = atoi(argv[1]);
    printf("Selected kernel: %d\n", kernel_num);

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // Setup timing for custom kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    
    // Setup timing for cuDNN
    cudaEvent_t cudnn_start, cudnn_stop;
    cudaEventCreate(&cudnn_start);
    cudaEventCreate(&cudnn_stop);
    float cudnn_elapsed_time;

    // Define test configurations (batch_size, vector_size)
    typedef struct {
        int batch_size;
        int vector_size;
    } TestConfig;

    // Define various test configurations to benchmark
    const int NUM_CONFIGS = 6;
    TestConfig configs[NUM_CONFIGS] = {
        {128, 1024},    // Small batch, medium vector
        {1024, 1024},   // Medium batch, medium vector
        {4096, 1024},   // Large batch, medium vector
        {128, 4096},    // Small batch, large vector
        {1024, 4096},   // Medium batch, large vector
        {4096, 4096}    // Large batch, large vector
    };

    // Parameters for benchmark
    const int WARMUP_RUNS = 5;
    const int BENCHMARK_RUNS = 10;
    const float TOLERANCE = 1e-4f;  // Tolerance for verification

    // Results table header
    printf("\nBenchmark Results:\n");
    printf("Batch Size\tVector Size\tCustom(ms)\tcuDNN(ms)\tCustom/cuDNN(%%)\tGB/s\tVerification\n");
    printf("--------------------------------------------------------------------------------------------\n");

    // For each configuration
    for (int config_idx = 0; config_idx < NUM_CONFIGS; config_idx++) {
        int batch_size = configs[config_idx].batch_size;
        int vector_size = configs[config_idx].vector_size;
        int total_elements = batch_size * vector_size;
        size_t data_size = total_elements * sizeof(float);

        // Allocate host memory
        float *h_input = (float*)malloc(data_size);
        float *h_output = (float*)malloc(data_size);
        float *h_cudnn_result = (float*)malloc(data_size);

        // Initialize input data
        randomize_data(h_input, total_elements);

        // Allocate device memory
        float *d_input, *d_output, *d_cudnn;
        cudaMalloc(&d_input, data_size);
        cudaMalloc(&d_output, data_size);
        cudaMalloc(&d_cudnn, data_size);

        // Copy data to device
        cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice);

        // Warmup runs for cuDNN
        for (int i = 0; i < WARMUP_RUNS; i++) {
            cudnn_softmax(cudnn, d_input, d_cudnn, batch_size, vector_size);
        }
        cudaDeviceSynchronize();
        
        // Benchmark cuDNN
        cudaEventRecord(cudnn_start);
        for (int run = 0; run < BENCHMARK_RUNS; run++) {
            cudnn_softmax(cudnn, d_input, d_cudnn, batch_size, vector_size);
        }
        cudaEventRecord(cudnn_stop);
        cudaEventSynchronize(cudnn_stop);
        cudaEventElapsedTime(&cudnn_elapsed_time, cudnn_start, cudnn_stop);
        
        // Calculate average time for cuDNN
        float avg_cudnn_time = cudnn_elapsed_time / BENCHMARK_RUNS;
        
        // Copy cuDNN results for verification
        cudaMemcpy(h_cudnn_result, d_cudnn, data_size, cudaMemcpyDeviceToHost);

        // Warmup runs for custom kernel
        for (int i = 0; i < WARMUP_RUNS; i++) {
            test_kernel(kernel_num, batch_size, vector_size, d_input, d_output);
        }
        cudaDeviceSynchronize();

        // Benchmark runs for custom kernel
        cudaEventRecord(start);
        for (int run = 0; run < BENCHMARK_RUNS; run++) {
            test_kernel(kernel_num, batch_size, vector_size, d_input, d_output);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);

        // Calculate average time and bandwidth
        float avg_time = elapsed_time / BENCHMARK_RUNS;
        
        // Calculate custom/cuDNN percentage
        float percentage = (avg_cudnn_time / avg_time) * 100.0f;
        
        // Copy results back for verification
        cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost);
        
        // Compare with cuDNN result
        bool passed = verify_results(h_cudnn_result, h_output, total_elements, TOLERANCE);
        if (!passed) {
            printf("Verification failed for batch_size=%d, vector_size=%d\n", 
                   batch_size, vector_size);
            continue;
        }

        // Calculate GB/s: 2 memory operations (read input, write output) per element
        float gb_per_s = (2.0f * data_size) / (avg_time * 1e6);

        // Print results
        printf("%d\t\t%d\t\t%.3f\t\t%.3f\t\t%.2f%%\t\t%.2f\tPASSED\n", 
               batch_size, vector_size, avg_time, avg_cudnn_time, percentage, gb_per_s);

        // Cleanup for this configuration
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_cudnn);
        free(h_input);
        free(h_output);
        free(h_cudnn_result);
    }

    // Final cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(cudnn_start);
    cudaEventDestroy(cudnn_stop);
    cudnnDestroy(cudnn);

    return 0;
}