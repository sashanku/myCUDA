#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>


// SMEM
__global__ void kernel03(float* input, float* output, int vec_len){

    //shared memory declaration
    __shared__ float smem[1024];
    // variables
    float local_max = -INFINITY;
    float l = 0.0;
    int batchIdx = blockIdx.x;

    for(int i = threadIdx.x; i < vec_len; i += blockDim.x){
        if(input[batchIdx * vec_len + i] > local_max){
            l *= expf(input[batchIdx * vec_len + i] - local_max);
            local_max = input[batchIdx * vec_len + i];
        }
        l += expf(input[batchIdx * vec_len + i]);
    }

    // store local max in smem
    smem[threadIdx.x] = local_max;
    __syncthreads();

    // paralell block level reductions
    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if(threadIdx.x < stride){
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    

    float row_max = smem[0];
    // __syncthreads();

    // we now store corrected local norm in smem
    smem[threadIdx.x] = l * exp(local_max - row_max);
    
    // paralell block level reduction for global norm factor
    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if(threadIdx.x < stride){
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float global_norm = smem[0];
    // __syncthreads();

    // output
    for(int i = threadIdx.x; i < vec_len; i += blockDim.x){
        output[batchIdx * vec_len + i] = exp(input[batchIdx * vec_len + i] - row_max) / global_norm;
    }
}



void launch_softmax_kernel03(float* input, float* output, int vec_len, int batch_size){

    int threads_per_block = 1024;
    
    kernel03<<< batch_size, threads_per_block>>>(input, output, vec_len);
}