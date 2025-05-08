#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cassert>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

/*
This kernel implements an online softmax operation on batches of vectors.
The softmax operation is performed on each vector.

How this works:
Instead of accessing shared memory and having sync barrier overhead, we use warp-level primitives (then
block-level) for performing max and sum reductions. The benefit is: it is faster than shared
memory access and also does not need syncing since each warp executes
an instruction parallely on GPU so no chance of race conditions.

We also use vectorized loads and stores for better memory throughput.
*/
__global__ void kernel05(float* input, float* output, int vec_len) {
    // Remove the assert check for vec_len % 4 == 0
    
    // Use shared memory for reductions
    extern __shared__ float smem[];

    int batchIdx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_size = 32;

    float* input_row = input + batchIdx * vec_len;
    float* output_row = output + batchIdx * vec_len;
    float local_max = -INFINITY;
    float l = 0.0f;

    // Process aligned portion with float4
    int aligned_len = (vec_len / 4) * 4; // Floor to nearest multiple of 4
    int n_float4s = aligned_len / 4;
    
    // Cast aligned portion as float4 for vectorized access
    float4* input_row_vec = reinterpret_cast<float4*>(input_row);
    float4* output_row_vec = reinterpret_cast<float4*>(output_row);
    float maxval = -INFINITY;

    // Process aligned portion with vectorized float4 loads
    #pragma unroll
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];

        maxval = fmaxf(maxval, elem.x);
        maxval = fmaxf(maxval, elem.y);
        maxval = fmaxf(maxval, elem.z);
        maxval = fmaxf(maxval, elem.w);
        if (maxval > local_max) {
            l *= __expf(local_max - maxval);
            local_max = maxval;
        }
        l += __expf(elem.x - maxval);
        l += __expf(elem.y - maxval);
        l += __expf(elem.z - maxval);
        l += __expf(elem.w - maxval);
    }
    
    // Process remaining elements (0-3) individually
    for (int i = aligned_len + tid; i < vec_len; i += blockDim.x) {
        float elem = input_row[i];
        if (elem > local_max) {
            l *= __expf(local_max - elem);
            local_max = elem;
        }
        l += __expf(elem - local_max);
    }
    
    __syncthreads();

    // Warp level reduction for max
    float val = local_max;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    // Block level reduction for max
    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : -INFINITY;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    // Get global max
    float row_max = smem[0];
    __syncthreads();

    // Sum reduction for normalization factor
    val = l * expf(local_max - row_max);
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : 0.0f;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    float row_norm = smem[0];
    __syncthreads();

    // Compute softmax for aligned portion
    #pragma unroll
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];
        elem.x = __expf(elem.x - row_max) / row_norm;
        elem.y = __expf(elem.y - row_max) / row_norm;
        elem.z = __expf(elem.z - row_max) / row_norm;
        elem.w = __expf(elem.w - row_max) / row_norm;

        output_row_vec[i] = elem;
    }
    
    // Compute softmax for remaining elements
    for (int i = aligned_len + tid; i < vec_len; i += blockDim.x) {
        output_row[i] = __expf(input_row[i] - row_max) / row_norm;
    }
}

void launch_softmax_kernel05(float* input, float* output, int batch_size, int vec_len) {
    int threads_per_block = 128;
    
    // Calculate proper shared memory size for reductions
    int shared_mem_size = sizeof(float) * CEIL_DIV(threads_per_block, 32);
    
    kernel05<<< batch_size, threads_per_block, shared_mem_size>>>(input, output, vec_len);
}