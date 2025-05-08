#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels.cuh"  // Include the kernel header

void randomize_matrix(float *mat, int N) {
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double diff = 0.0;
    for (int i = 0; i < N; i++) {
        diff = fabs((double)mat1[i] - (double)mat2[i]);
        if (diff > 1e-2) {
            printf("Verification failed at index %d: %f vs %f\n", i, mat1[i], mat2[i]);
            return false;
        }
    }
    return true;
}

void test_kernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle) {
    switch (kernel_num) {
        case 0: // cuBLAS reference
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
            break;
        case 1:
            launch_kernel01(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            launch_kernel02(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            launch_kernel03(M, N, K, alpha, A, B, beta, C);
            break;
        case 4:
            launch_kernel04(M, N, K, alpha, A, B, beta, C);
            break;
        case 5:
            launch_kernel05(M, N, K, alpha, A, B, beta, C);
            break;
        // Add more cases as needed
        default:
            printf("Invalid kernel number\n");
            exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <kernel_number>\n", argv[0]);
        printf("Kernel numbers: 0 (cuBLAS), 1-N (custom kernels)\n");
        return 1;
    }

    int kernel_num = atoi(argv[1]);
    printf("Selected kernel: %d\n", kernel_num);

    // Initialize cuBLAS
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed\n");
        return 1;
    }

    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    // Matrix sizes to test
    // const int NUM_SIZES = 10;
    // int sizes[NUM_SIZES];
    // for (int i = 0; i < NUM_SIZES; i++) {
    //     sizes[i] = 512 + i * 512; // Test from 512 to 5120 in steps of 512
    // }
    const int NUM_SIZES = 1;
    int sizes[1];
    sizes[0] = 4092;
    // for (int i = 0; i < NUM_SIZES; i++) {
    //     sizes[i] = 512 + i * 512; // Test from 512 to 5120 in steps of 512
    // }


    // Parameters
    float alpha = 1.0f;
    float beta = 0.0f;
    const int WARMUP_RUNS = 1;
    const int BENCHMARK_RUNS = 10;

    // Results table header
    printf("\nBenchmark Results:\n");
    printf("Size\tTime(ms)\tGFLOPS\tVerification\n");
    printf("----------------------------------------\n");

    // For each matrix size
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        int N = sizes[size_idx];
        size_t matrix_size = N * N * sizeof(float);

        // Allocate host memory
        float *h_A = (float*)malloc(matrix_size);
        float *h_B = (float*)malloc(matrix_size);
        float *h_C = (float*)malloc(matrix_size);
        float *h_Reference = (float*)malloc(matrix_size);

        // Initialize matrices
        randomize_matrix(h_A, N * N);
        randomize_matrix(h_B, N * N);
        randomize_matrix(h_C, N * N);

        // Allocate device memory
        float *d_A, *d_B, *d_C, *d_Reference;
        cudaMalloc(&d_A, matrix_size);
        cudaMalloc(&d_B, matrix_size);
        cudaMalloc(&d_C, matrix_size);
        cudaMalloc(&d_Reference, matrix_size);

        // Copy data to device
        cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Reference, h_C, matrix_size, cudaMemcpyHostToDevice);

        // Warmup runs
        for (int i = 0; i < WARMUP_RUNS; i++) {
            test_kernel(kernel_num, N, N, N, alpha, d_A, d_B, beta, d_C, handle);
        }
        cudaDeviceSynchronize();

        // // Verification against cuBLAS
        if (kernel_num != 0) {
            test_kernel(0, N, N, N, alpha, d_A, d_B, beta, d_Reference, handle);
            test_kernel(kernel_num, N, N, N, alpha, d_A, d_B, beta, d_C, handle);
            cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Reference, d_Reference, matrix_size, cudaMemcpyDeviceToHost);
            bool passed = verify_matrix(h_Reference, h_C, N * N);
            if (!passed) {
                printf("Verification failed for size %d\n", N);
                continue;
            }
        }

        // Benchmark runs
        cudaEventRecord(start);
        for (int run = 0; run < BENCHMARK_RUNS; run++) {
            test_kernel(kernel_num, N, N, N, alpha, d_A, d_B, beta, d_C, handle);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);

        // Calculate average time and GFLOPS
        float avg_time = elapsed_time / BENCHMARK_RUNS;
        float gflops = (2.0f * N * N * N) / (avg_time * 1e6); // 2*N^3 FLOPs for matrix multiplication

        // Print results
        printf("%d\t%.2f\t\t%.2f\tPASSED\n", N, avg_time, gflops);

        // Cleanup for this size
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_Reference);
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_Reference);
    }

    // Final cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    return 0;
}