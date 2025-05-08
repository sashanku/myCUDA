#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>

// Your CUDA kernel definitions

// GMEM Coalesed
#define BLOCK_SIZE 32
#define BM 32
#define BN 32
#define BK 32


//SMEM
__global__ void kernel02(int M, int N, int K, float* A, float* B, float* C){
    // smem declaration
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // variables(rows, cols, temp etc)
    int rowC = threadIdx.x / BLOCK_SIZE;
    int colC = threadIdx.x % BLOCK_SIZE;
    float temp = 0.0;
    
    // Advance Pointers to current row/col for easier indexing 
    A += blockIdx.y * BLOCK_SIZE * K;
    B += blockIdx.x * BLOCK_SIZE;
    C += blockIdx.y * BLOCK_SIZE * N + blockIdx.x * BLOCK_SIZE;

    // outer loop to populate the SMEM;
    for(int blockIdx = 0; blockIdx < K; blockIdx += BLOCK_SIZE){
        As[rowC * BLOCK_SIZE + colC] = A[rowC * K + colC];
        Bs[rowC * BLOCK_SIZE + colC] = B[rowC * N + colC];

        // sync block to wait to complete populating of smem
        __syncthreads();
        // Advance pointers for next block
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // Inner loop to compute the dot product
        for(int dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++){
            temp += As[rowC * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + colC];
        }
        // This sync block will prevent faster threads from loading next block into smem
        // while slower threads are computing dot ptoduct
        __syncthreads();
    }
    // write out results
    // Calculate global indices for bounds checking
    int globalRowC = blockIdx.y * BLOCK_SIZE + rowC;
    int globalColC = blockIdx.x * BLOCK_SIZE + colC;
    if(globalRowC < M & globalColC < N){
        C[rowC * N + colC] = temp;
    }
}



// Kernel launcher function
void launch_kernel02(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // Set your grid and block dimensions
    const int block_size = 32;
    dim3 grid((N + block_size-1) / block_size, (M + block_size-1) / block_size, 1);
    dim3 block(block_size*block_size,1, 1);
    
    // Launch the kernel
    kernel02<<<grid, block>>>(M, N, K, A, B, C);
}




// __global__ void kernel02p(int M, int N, int K, float* A, float* B, float* C){
//     // declare shared memory
//     __shared__ float As[BM * BK];
//     __shared__ float Bs[BK * BN];

//     // Advance pointers
//     A += blockIdx.y * BM * K;
//     B += blockIdx.x * BK;
//     C += blockIdx.y * BM * N + blockIdx.x * BN;
    
//     // variables(rows/cols)
//     int rowA = threadIdx.x / BK;
//     int colA = threadIdx.x % BK; 
//     int rowB = threadIdx.x / BN;
//     int colB = threadIdx.x % BN; 
//     int rowC = threadIdx.x / BN;
//     int colC = threadIdx.x % BN; 

//     float temp = 0.;
//     // Outer loop to move the block
//     for(int bkIdx = 0; bkIdx < BK; bkIdx++){
//         As[rowA * BK + colA] = A[rowA * K + colA];
//         Bs[rowB * BN + colB] = B[rowB * N + colB];

//         // Advance pointers 
//         A += BK;
//         B += BK * N;
//         __syncthreads();
        
//         // inner loop to progress dot product
//         for(int dotIdx = 0; dotIdx < BK; dotIdx++){
//             temp += As[rowC * BK + dotIdx] * Bs[dotIdx * BN + colC];
//         }
//         __syncthreads();
//     }

//     // write results
//     C[rowC * BN + colC] = temp;


// }