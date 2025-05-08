#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>


__global__ void kernel02(float* input, float* output, int vec_len){

    // variables
    int batchIdx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float max = -INFINITY;
    float l = 0.;
    // compute maximum value to use of stability
    for(int i = 0; i < vec_len; i++){
        if(input[batchIdx * vec_len + i] > max){
            l *= expf(max - input[batchIdx * vec_len + i]);
            max = input[batchIdx * vec_len + i];
        }
        l += expf(input[batchIdx * vec_len + i] - max);
        }
    __syncthreads();


    // normalize by dividing each element by sum
    for(int i = tid; i < vec_len; i += stride){
        if(i  < vec_len){
        output[batchIdx * vec_len + i] = expf(input[batchIdx * vec_len + i] - max) / l;
        }
    }
}


void launch_softmax_kernel02(float* input, float* output, int vec_len, int batch_size){

    int threads_per_block = 128;
    
    kernel02<<< batch_size, threads_per_block>>>(input, output, vec_len);
}