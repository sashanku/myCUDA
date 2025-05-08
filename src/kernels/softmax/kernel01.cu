#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>


__global__ void kernel01(float* input, float* output, int vec_len){

    // variables
    int batchIdx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float max = -INFINITY;
    // compute maximum value to use of stability
    for(int i = 0; i < vec_len; i++){
        if(input[batchIdx * vec_len + i] > max){
            max = input[batchIdx * vec_len + i];
        }
    }
    __syncthreads();

    // compute exp(element - max) for all elemets
    for(int i = tid; i < vec_len; i += stride){
        output[batchIdx * vec_len + i] = expf(input[batchIdx * vec_len + i] - max);
    }
    


    float sum = 0.;
    // compute sum of exp for nomalization
    for(int i = 0; i < vec_len; i++){
        sum += output[batchIdx * vec_len + i];
    }
    __syncthreads();


    // normalize by dividing each element by sum
    for(int i = tid; i < vec_len; i += stride){
        if(i  < vec_len){
        output[batchIdx * vec_len + i] /= sum;
        }
    }
}


void launch_softmax_kernel01(float* input, float* output, int vec_len, int batch_size){

    int threads_per_block = 256;
    
    kernel01<<< batch_size, threads_per_block>>>(input, output, vec_len);
}