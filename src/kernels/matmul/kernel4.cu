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


// 2D SMEM Block tiling
// __global__ void kernel04(int M, int N, int K, float* A, float* B, float*C){
//     // shared memory allocation
//     __shared__ float As[BM*BK];
//     __shared__ float Bs[BN*BK];

//     // computations
//     int num_computatations_per_block = BM * BN;
//     int num_computations_per_thread = TM * TN;
//     int num_threads_req = num_computatations_per_block / num_computations_per_thread;

//     // variable definitions and initializations
//     // number of required threads must be equal to number of threads each block has been launched with
//     assert(num_threads_req == blockDim.x);

//     // row/col variables
//     int rowA = threadIdx.x / BK;
//     int colA = threadIdx.x % BK;
//     int rowB = threadIdx.x / BN;
//     int colB = threadIdx.x % BN;
//     int rowC = threadIdx.x / (BN/TN);
//     int colC = threadIdx.x % (BN/TN);
    
//     // Stride computation to chech how many rows of A and B each thread block loads at once
//     int strideA = num_threads_req / BK;
//     int strideB = num_threads_req / BN;

//     // registers
//     float thread_results[TM * TN] = {0.0};
//     float regM[TM] = {0.0};
//     float regN[TN] = {0.0};

//     // initialze pointers to current row and col
//     A += blockIdx.y * BM * K;
//     B += blockIdx.x * BN;
//     C += blockIdx.y * BM * N + blockIdx.x * BN; 
    
//     // outer loop to populate SMEM
//     for(int block = 0; block < K; block += BK){
//         // Independent sub loops to populate A and B SMEMS
//         for(int loadoffset = 0; loadoffset < BM; loadoffset += strideA){
//             As[(rowA + loadoffset) * BK + colA] = A[(rowA + loadoffset) * K + colA];
//             // As[(colA) * BM + rowA + loadoffset] = A[(rowA + loadoffset) * K + colA];
        
//         }
//         for(int loadoffset = 0; loadoffset < BK; loadoffset += strideB){
//             Bs[(rowB + loadoffset) * BN + colB] = B[(rowB + loadoffset) * N + colB];
//         }
//         __syncthreads();

//         // Advance pointers
//         A += BK;
//         B += BK * N;
//         // Inner loop to compute per-thread results
//         for(int dotIdx = 0; dotIdx < BK; dotIdx++){
//             //independent sub loops to load TM chunk of A and TN chunk of B into registers
//             for(int i = 0; i < TM; i++){
//                 regM[i] = As[(rowC * TM + i) * BK + dotIdx];
//                 // regM[i] = As[dotIdx * BM + rowC * TM + i];

//             }
//             for(int i = 0; i < TN; i++){
//                 regN[i] = Bs[dotIdx * BN + colC * TN + i];
//             }
//             for(int resIdxM = 0; resIdxM < TM; resIdxM++){
//                 for(int resIdxN = 0; resIdxN < TN; resIdxN++){
//                     thread_results[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
//                 }
//             }
//         }
//         __syncthreads();
//     }
//     // Write results with bounds checking
//     int globalRowStart = blockIdx.y * BM + rowC * TM;
//     int globalColStart = blockIdx.x * BN + colC * TN;
    
//     for(int resIdxM = 0; resIdxM < TM; resIdxM++){
//         int globalRow = globalRowStart + resIdxM;
//         if(globalRow >= M) continue; // Skip if row is out of bounds
        
//         for(int resIdxN = 0; resIdxN < TN; resIdxN++){
//             int globalCol = globalColStart + resIdxN;
//             if(globalCol < N) { // Only write if column is within bounds
//                 C[(rowC * TM + resIdxM) * N + colC * TN + resIdxN] = thread_results[resIdxM * TN + resIdxN];
//             }
//         }
//     }
// }

// // 2d thread-tiling
__global__ void kernel04(int  M, int N, int K, float* A, float* B, float* C){
    // shared memory declarations
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Variables(row, col, registers)
    int rowA = threadIdx.x / BK;
    int colA = threadIdx.x % BK;
    int rowB = threadIdx.x / BN;
    int colB = threadIdx.x % BN;
    int rowC = threadIdx.x / (BN / TN);
    int colC = threadIdx.x % (BN / TN);
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};
    float thread_results[TM * TN] = {0.0};

    int num_elements_in_block = BM * BN;
    int num_computations_per_thread = TM*TN;

    int num_threads = num_elements_in_block / num_computations_per_thread;

    int strideA = num_threads / BK;
    int strideB = num_threads / BN;

    // Advance pointers to current row/col
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN; 

    // outer loop to populate smem
    for(int blockIdx = 0; blockIdx < K; blockIdx += BK){
        // Independent loops to populate A and B smems
        for(int offsetA = 0; offsetA < BM; offsetA += strideA){
            As[(rowA + offsetA) * BK + colA] = A[(rowA + offsetA) * K + colA];
        }
        for(int offsetB = 0; offsetB < BK; offsetB += strideB){
            Bs[(rowB + offsetB) * BN + colB] = B[(rowB + offsetB) * N + colB];
        }
        __syncthreads();
        // Advance pointers to net block
        A += BK;
        B += BK * N;

        // Inner loop to advance dot product
        for(int dotIdx = 0; dotIdx < BK; dotIdx++){
            // populate A and B registers from smem
            for(int regIdxA = 0; regIdxA < TM; regIdxA++){
                regA[regIdxA] = As[(rowC * TM + regIdxA) * BK + dotIdx];
            }
            for(int regIdxB = 0; regIdxB < TN; regIdxB++){
                regB[regIdxB] = Bs[dotIdx * BN + colC * TN + regIdxB];
            }

            // Compute the dot product
            for(int regIdxA = 0; regIdxA < TM; regIdxA++){
                for(int regIdxB = 0; regIdxB < TN; regIdxB++){
                    thread_results[regIdxA * TN + regIdxB] += regA[regIdxA] * regB[regIdxB];
                }
            }
        }
        __syncthreads();
    }

    // Output
    // Write results with bounds checking
    int globalRowStart = blockIdx.y * BM + rowC * TM;
    int globalColStart = blockIdx.x * BN + colC * TN;
    for(int regIdxM = 0; regIdxM < TM; regIdxM++){
        int global_row = globalRowStart + regIdxM;
        if(global_row >= M) continue;

        for(int regIdxN = 0; regIdxN < TN; regIdxN++){
            int global_col = globalColStart + regIdxN;
            if(global_col < N){
                C[(rowC * TM + regIdxM) * N + colC * TN + regIdxN] = thread_results[regIdxM * TN + regIdxN];
            }
        }
    }
}


// kernel launcher function
void launch_kernel04(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // Set your grid and block dimensions
    dim3 grid((N + BN-1) / BN, (M + BM-1) / BM, 1);
    dim3 block((BN * BM) / (TM * TN));
    
    // Launch the kernel
    // kernel03<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    kernel04<<<grid, block>>>(M, N, K, A, B, C);

}



