#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void kernel05(int M, int N, int K, float* A, float* B, float* C){
    // shared memory allocation
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // variables(rows,cols,registers)
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innercolA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    float thread_results[TM * TN] = {0.0};
    
    // move pointers to current block( for easier indexing later)
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // outer loop to move over block
    for(int block = 0; block < K; block += BK){
        // populate SMEM using float4(transpose A)
        // Populate A
        float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innercolA * 4])[0];
        // Transpose A
        As[(innercolA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innercolA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innercolA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innercolA * 4 + 3) * BM + innerRowA] = tmp.w;
        
        // populate B
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = 
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        // Advance pointers
        A += BK;
        B += BK * N;
        // inner loop to progress dot product
        for(int dotIdx = 0; dotIdx < BK; dotIdx++){
            // populate registers independently
            for(int i = 0; i < TM;  i++){
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for(int i = 0; i < TN; i++){
                regN[i] = Bs[dotIdx * BN +  threadCol * TN + i];
            }

            // inner most loop to do the product
            for(int resIdxM = 0; resIdxM < TM; resIdxM++){
                for(int resIdxN = 0; resIdxN < TN; resIdxN++){
                    thread_results[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
    __syncthreads();
    }
    // (INT VS UINT MAKE NOTES)
    // Write results with bounds checking
    int globalRowStart = blockIdx.y * BM + threadRow * TM;
    int globalColStart = blockIdx.x * BN + threadCol * TN;

    for(int resIdxM = 0; resIdxM < TM; resIdxM++){
        int globalRow = globalRowStart + resIdxM;
        if(globalRow >= M) continue; // Skip if row is out of bounds
        
        for(int resIdxN = 0; resIdxN < TN; resIdxN++){
            int globalCol = globalColStart + resIdxN;
            if(globalCol < N) { // Only write if column is within bounds
                C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = thread_results[resIdxM * TN + resIdxN];
            }
        }
    }

}



// __global__ void kernel05(int M, int N, int K, float *A,
//     float *B, float *C) {
//     const int cRow = blockIdx.y;
//     const int cCol = blockIdx.x;

//     // BN/TN are the number of threads to span a column
//     const int threadCol = threadIdx.x % (BN / TN);
//     const int threadRow = threadIdx.x / (BN / TN);

//     // allocate space for the current blocktile in smem
//     __shared__ float As[BM * BK];
//     __shared__ float Bs[BK * BN];

//     // Move blocktile to beginning of A's row and B's column
//     A += cRow * BM * K;
//     B += cCol * BN;
//     C += cRow * BM * N + cCol * BN;

//     // calculating the indices that this thread will load into SMEM
//     // we'll load 128bit / 32bit = 4 elements per thread at each step
//     const int innerRowA = threadIdx.x / (BK / 4);
//     const int innerColA = threadIdx.x % (BK / 4);
//     const int innerRowB = threadIdx.x / (BN / 4);
//     const int innerColB = threadIdx.x % (BN / 4);

//     // allocate thread-local cache for results in registerfile
//     float threadResults[TM * TN] = {0.0};
//     float regM[TM] = {0.0};
//     float regN[TN] = {0.0};

//     // outer-most loop over block tiles
//     for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
//         // populate the SMEM caches
//         // transpose A while loading it
//         float4 tmp =
//         reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
//         As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
//         As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
//         As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
//         As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

//         reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
//         reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
//         __syncthreads();

//         // advance blocktile
//         A += BK;     // move BK columns to right
//         B += BK * N; // move BK rows down

//         // calculate per-thread results
//         for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
//             // block into registers
//             for (int i = 0; i < TM; ++i) {
//                 regM[i] = As[dotIdx * BM + threadRow * TM + i];
//                 }
//             for (int i = 0; i < TN; ++i) {
//                 regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
//                 }
//             for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
//                 for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
//                     threadResults[resIdxM * TN + resIdxN] +=
//                     regM[resIdxM] * regN[resIdxN];
//                     }
//                 }   
//             }
//         __syncthreads();
//     }

//     // write out the results with bounds checking
//     const int globalRowStart = cRow * BM + threadRow * TM;
//     const int globalColStart = cCol * BN + threadCol * TN;
    
//     for (int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
//         // Check if this row is within bounds
//         if (globalRowStart + resIdxM < M) {
//             for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
//                 // Check if this column group is within bounds
//                 if (globalColStart + resIdxN + 3 < N) {
//                     // All four elements are within bounds, use float4
//                     float4 tmp = reinterpret_cast<float4 *>(
//                     &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
//                     // perform GEMM update in reg
//                     tmp.x = threadResults[resIdxM * TN + resIdxN];
//                     tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
//                     tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
//                     tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
//                     // write back
//                     reinterpret_cast<float4 *>(
//                     &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
//                     tmp;
//                 } else {
//                     // Handle partial float4 (some elements might be out of bounds)
//                     for (int i = 0; i < 4 && globalColStart + resIdxN + i < N; i++) {
//                         C[(globalRowStart + resIdxM) * N + globalColStart + resIdxN + i] = 
//                             threadResults[resIdxM * TN + resIdxN + i];
//                     }
//                 }
//             }
//         }
//     }
// }


// kernel launcher function
void launch_kernel05(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // Set your grid and block dimensions
    dim3 grid((N + BN-1) / BN, (M + BM-1) / BM, 1);
    dim3 block((BN * BM) / (TM * TN));
    
    // Launch the kernel
    // kernel03<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    kernel05<<<grid, block>>>(M, N, K, A, B, C);

}


