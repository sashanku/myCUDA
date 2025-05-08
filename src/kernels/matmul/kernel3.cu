#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>

// Your CUDA kernel definitions

// GMEM Coalesed
#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8


// Thread level 1d tiling
// __global__ void kernel03(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

//     // Allocate SMEM for A and B
//     __shared__ float As [BM * BK];
//     __shared__ float Bs [BK * BN];

//     // Threads inside C. Used to compute values in C
//     const uint thread_row = threadIdx.x / BN;
//     const uint thread_col = threadIdx.x % BN;
    
//     // Move pointers to start of current block
//     A += blockIdx.y * K * BM;
//     B += blockIdx.x * BN;
//     C += blockIdx.y * N * BM + blockIdx.x * BN;

//     // Threads to access A and B and move them into SMEM(warp-levl GMEM Coalesed memory access)
//     const uint rowA = threadIdx.x / BK; 
//     const uint colA = threadIdx.x % BK;
//     const uint rowB = threadIdx.x / BN; 
//     const uint colB = threadIdx.x % BN;

//     // allocate thread cache to store results in register files(here we are allocating TM registers)
//     float thread_results[TM] = {0.0};

//     // Calculate global indices for boundary checking
//     int global_row_start = blockIdx.y * BM + thread_row * TM;
//     int global_col = blockIdx.x * BN + thread_col;

//     for (int block = 0; block < K; block += BK)
//     {
//         // populate SMEM caches
//         As[rowA * BK + colA] = A[rowA * K + colA];
//         Bs[rowB * BN + colB] = B[rowB * N + colB];
//         __syncthreads();

//         // advance A and B pointers by BK
//         A += BK;
//         B += BK * N;

//         for(int dotidx = 0; dotidx < BK; ++dotidx)
//         {
//             float tempB = Bs[dotidx * BN + thread_col];
//             for(int residx = 0; residx < TM; ++residx)
//             {
//                 thread_results[residx] += As[(thread_row * TM + residx) * BK + dotidx] * tempB;
//             }

//         }
//     __syncthreads();
//     }

//     // write results
//     // for (uint resIdx = 0; resIdx < TM; ++resIdx) {
//     //     C[(thread_row * TM + resIdx) * N + thread_col] =
//     //         alpha * thread_results[resIdx] +
//     //         beta * C[(thread_row * TM + resIdx) * N + thread_col];
//     // }
//     for(int thread = 0; thread < TM; thread++){
//                 int global_row = global_row_start + thread;
//                 if(global_row < M && global_col < N) {
//                     C[(thread_row * TM + thread) * N + thread_col] = thread_results[thread];
//                 }
// }
// }

// 1D SMEM Block tiling
__global__ void kernel03(int M, int N, int K, float* A, float* B, float*C){
    //allocate shared memory blocks
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    //move pointers to current row/col blocks
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;


    //col and rows(converting 1D grid to rows and columns for both matrices)
    int rowA = threadIdx.x / BK;
    int colA = threadIdx.x % BK;
    int rowB = threadIdx.x / BN;
    int colB = threadIdx.x % BN;

    // global row and col
    int row = threadIdx.x / BN;
    int col = threadIdx.x % BN;
    //create array to store thread results(TM results per thread)
    float thread_results[TM] = {0.0};

    //outer loop to populate shared memory blocks
    for(int block = 0; block < K; block += BK){
        As[rowA * BK + colA] = A[rowA * K + colA];
        Bs[rowB * BN + colB] = B[rowB * N + colB];
        __syncthreads();
        // Advance pointers
        A += BK;
        B += BK * N;
    // 1st inner loop to process the product of this thread
        for(int dotIdx = 0; dotIdx < BK; dotIdx++){
            float temp = Bs[dotIdx * BN + col];
        // inner most loop to compute TM resulst per thread
            for(int thread = 0; thread < TM; thread ++){
                thread_results[thread] += As[(row * TM + thread) * BK + dotIdx] * temp;
            }
        }
        __syncthreads();  
    }
    // Calculate global indices for boundary checking
    int global_row_start = blockIdx.y * BM + row * TM;
    int global_col = blockIdx.x * BN + col;
    // write results
    for(int thread = 0; thread < TM; thread++){
        int global_row = global_row_start + thread;
        if(global_row < M && global_col < N) {
            C[(row * TM + thread) * N + col] = thread_results[thread];
        }
}
}

// Kernel launcher function
void launch_kernel03(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // Set your grid and block dimensions
    dim3 grid((N + BN-1) / BN, (M + BM-1) / BM, 1);
    dim3 block((BN * BM) / TM);
    
    // Launch the kernel
    // kernel03<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    kernel03<<<grid, block>>>(M, N, K, A, B, C);
}



