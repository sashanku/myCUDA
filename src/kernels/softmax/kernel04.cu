#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))


__global__ void kernel04(float* input, float* output, int vec_len){
    // sgared memory declaration
    __shared__ float smem[1024];

    // variables
    int batchIdx = blockIdx.x;
    float local_max = -INFINITY;
    float l = 0.0;
    int warp_size = 32;

    //COMPUTE ROW MAX and NORM FACTOR
    // compute local maximum & local norm factor
    // Update norm factor on the fly when a better max is found
    for(int i = threadIdx.x; i < vec_len; i += blockDim.x){
        if(input[batchIdx * vec_len + i] > local_max){
            l *= exp(input[batchIdx * vec_len + i] - local_max);
            local_max = input[batchIdx * vec_len + i];
        }
        l += exp(input[batchIdx * vec_len + i] - local_max);
    }
    __syncthreads();

    // warp reduction
    float val = local_max; //store in register for warp shuffle
    for(int offset = warp_size / 2; offset > 0; offset /= 2){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    // when we have more than 1 warp,
    // each warp will collect its local max as done above
    //we then use the 1st warp to collect max values from all warps into smem
    // then we do a block reduction on the smem to get global max

    // we will first collect all local warp-level maximums into smem
    if(blockDim.x > warp_size){ //enters only if num_warps > 1
        if(threadIdx.x % warp_size){//identifies thread 0 of each warp
            smem[threadIdx.x / warp_size] = val; //loads max value into smem
        }
    
        // Now we have smem populated with all local warp level maximums
        // We can just do another warp level reduction on the smem using the 1st warp(num of threads = 1024 hence warps = 32 hence number of loca max = 32 hence just one warp required)
        if(threadIdx.x < warp_size){
            val = (threadIdx.x < CEIL_DIV(blockDim.x, warp_size)) ? smem[threadIdx.x] : -INFINITY;
            for(int offset = warp_size / 2; offset > 0; offset /= 2){
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
        }
    }   

    __syncthreads();
    // write global max to register
    float row_max = val;
    // __syncthreads();

    // correct local norm factor
    val = l * expf(local_max - row_max);
    // reduce the corrected local norm factors
    for(int offset = warp_size / 2; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // when we have more than 1 warp,
    // each warp will collect its local norm factor as done above
    //we then use the 1st warp to collect factor values from all warps into smem
    // then we do a block reduction on the smem to get global factor
    smem[threadIdx.x] = val;
    // __syncthreads();

    if(blockDim.x > warp_size){
        if(threadIdx.x % warp_size){
            smem[threadIdx.x / warp_size] = val;
        }

        if(threadIdx.x < warp_size){
            for(int offset = warp_size / 2; offset > 0; offset /= 2){
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
        }
    }

    __syncthreads();
    float row_norm = val;

    // OUTPUT
    for(int i = threadIdx.x; i < vec_len; i += blockDim.x){
        output[batchIdx* vec_len + i] = expf(input[batchIdx * vec_len + i] - row_max) / row_norm;
    }
}


void launch_softmax_kernel04(float* input, float* output, int vec_len, int batch_size){

    int threads_per_block = 256;
    
    kernel04<<< batch_size, threads_per_block>>>(input, output, vec_len);
}