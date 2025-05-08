#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>

// Your CUDA kernel definitions

// GMEM Coalesed
#define BLOCK_SIZE 32
#define BLOCKSIZE 32
#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8


__global__ void kernel01(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    
    const int cy = blockIdx.y * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE; //iterates rows
    const int cx = blockIdx.x * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE; //iterates columns
       
    
    if(cy < M && cx < N)
    {
        float p = 0;
        for(int i = 0; i < K; i++)
        {
            p += A[cy*K + i] * B[i*N + cx]; //direct access to GMEM(expensive)
        }
        C[cy*N + cx] = p;
    }
}
//********ANALASYS*********//
/*
1. blockIdx.x is responsible to handle all elements of each row. blockIdx.x will determine 
which row each instance is processing. 
2. blockIdx.y is responsible to handle all elements of each col. blockIdx.y will determine 
which col each instance is processing.
3. Now, once the row and col being processed by each instance are fixed, we fix the element being processed
by this instace. This is done by using the threadIdx.x(if launching 1D blocks) or both threadIdx.x & threadIdx.y
(if launching 2D blocks). 
4. Similar to increments in blockIdx, the y coordinate is incremented after the x coordinate is exhausted
for the given y. Thus means threadIdx.x will go from 0 to BLOCK_SIZE before threadIdx.y is incremented
5. Since we want to process elements in a row wise manner(row major storage so row wise processing for 
coalsced access) we use the modulo operator (threadIdx.x % BLOCK_SIZE) for cx to iterate over columns
(hence processing rows). Hence consecutive threads will access consecutive elements in rows.
6. In kernel 1, each thread is responcible for calculating "ONE" element in matrix C. We set this element 
for each instance using thread and block indices as described above.acoshf32x
7. The next important part is our memory access patterns. How are we fetching the memory required to compute
the result of the current element?
8. In this kernel, we are doing the inefficient way. We are fetching the required A and B elements for each 
element in C directly from the GMEM. The only good thing about this is that it is Coalesced(A matrix only).
*/


// Kernel launcher function
void launch_kernel01(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // Set your grid and block dimensions
    const int block_size = 32;
    dim3 grid((N + block_size-1) / block_size, (M + block_size-1) / block_size, 1);
    dim3 block(block_size * block_size, 1, 1);
    
    // Launch the kernel
    kernel01<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}



// __global__ void kernel01p(int M, int N, int K, float* A, float* B, float* C){
//     // variables(rows,cols etc)
//     int rowC = blockIdx.y * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
//     int colC = blockIdx.x * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;
//     float temp = 0.;
//     for(int i = 0; i < N; i++){
//         temp += A[rowC * K + i] * B[i * N + colC];
//     }
//     C[rowC * N + colC] = temp;
// }
