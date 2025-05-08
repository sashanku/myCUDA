#ifndef _KERNELS_CUH_
#define _KERNELS_CUH_
#include <cuda_runtime.h>
#include <cuda_fp16.h>
// Declare kernel launcher functions
// MATMUL
void launch_kernel01(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void launch_kernel02(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void launch_kernel03(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void launch_kernel04(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void launch_kernel05(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

// SOFTMAX
void launch_softmax_kernel01(float* input, float* output, int vec_len, int batch_size);
void launch_softmax_kernel02(float* input, float* output, int vec_len, int batch_size);
void launch_softmax_kernel03(float* input, float* output, int vec_len, int batch_size);
void launch_softmax_kernel04(float* input, float* output, int vec_len, int batch_size);
void launch_softmax_kernel05(float* input, float* output, int vec_len, int batch_size);



// TOP-K FLASH ATTENTION
void launch_flash_attention_topk(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim, int top_k,
    float scale, cudaStream_t stream
);

// Add more kernel declarations as needed

#endif