#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Helper function for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel for updating KV cache with new tokens
template <typename T>
__global__ void update_kv_cache_kernel(
    // Input tensors
    const T* __restrict__ new_keys,         // [batch_size, num_heads, 1, head_dim]
    const T* __restrict__ new_values,       // [batch_size, num_heads, 1, head_dim]
    // Output cached tensors
    T* __restrict__ key_cache,              // [batch_size, num_heads, max_seq_len, head_dim]
    T* __restrict__ value_cache,            // [batch_size, num_heads, max_seq_len, head_dim]
    // Parameters
    const int batch_size,
    const int num_heads,
    const int head_dim,
    const int max_seq_len,
    const int current_len                   // Current sequence length before adding new token
) {
    // Calculate thread indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute coordinates in the tensors
    const int b = tid / (num_heads * head_dim);                // Batch index
    const int h = (tid / head_dim) % num_heads;                // Head index
    const int d = tid % head_dim;                              // Dimension index
    
    if (b < batch_size && h < num_heads && d < head_dim) {
        // Calculate source and destination offsets
        const int new_token_offset = b * num_heads * head_dim + h * head_dim + d;
        const int cache_offset = b * num_heads * max_seq_len * head_dim + 
                                h * max_seq_len * head_dim +
                                current_len * head_dim + d;
        
        // Update the cache with the new key and value
        key_cache[cache_offset] = new_keys[new_token_offset];
        value_cache[cache_offset] = new_values[new_token_offset];
    }
}

// Kernel for computing attention scores between new query and cached keys
template <typename T>
__global__ void compute_attention_scores_kernel(
    // Input tensors
    const T* __restrict__ query,            // [batch_size, num_heads, 1, head_dim]
    const T* __restrict__ key_cache,        // [batch_size, num_heads, current_len, head_dim]
    // Output tensor
    float* __restrict__ attn_scores,        // [batch_size, num_heads, current_len]
    // Parameters
    const int batch_size,
    const int num_heads,
    const int head_dim,
    const int current_len,                  // Current sequence length including new token
    const float scale                       // Scaling factor for attention scores (1/sqrt(head_dim))
) {
    // Shared memory for partial dot products
    extern __shared__ float s_mem[];
    
    // Calculate thread indices
    const int b = blockIdx.z;                                  // Batch index
    const int h = blockIdx.y;                                  // Head index
    const int seq_pos = blockIdx.x;                            // Sequence position
    const int tid = threadIdx.x;                               // Thread index within block
    
    if (b >= batch_size || h >= num_heads || seq_pos >= current_len) return;
    
    // Calculate offsets
    const int q_base = b * num_heads * head_dim + h * head_dim;
    const int k_base = b * num_heads * current_len * head_dim + 
                      h * current_len * head_dim +
                      seq_pos * head_dim;
    
    // Compute partial dot product
    float thread_sum = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        thread_sum += static_cast<float>(query[q_base + d]) * static_cast<float>(key_cache[k_base + d]);
    }
    
    // Store in shared memory
    s_mem[tid] = thread_sum;
    __syncthreads();
    
    // Perform parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Write final scaled attention score
    if (tid == 0) {
        const int score_offset = b * num_heads * current_len + h * current_len + seq_pos;
        attn_scores[score_offset] = s_mem[0] * scale;
    }
}

// Kernel for applying softmax to attention scores
template <typename T>
__global__ void apply_softmax_kernel(
    // Input/output tensor
    float* __restrict__ attn_scores,        // [batch_size, num_heads, current_len]
    // Parameters
    const int batch_size,
    const int num_heads,
    const int current_len
) {
    // Each thread processes one full attention distribution
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / num_heads;                             // Batch index
    const int h = idx % num_heads;                             // Head index
    
    if (b >= batch_size || h >= num_heads) return;
    
    // Calculate base offset for this batch and head
    const int base_offset = b * num_heads * current_len + h * current_len;
    
    // Find maximum score for numerical stability
    float max_score = -INFINITY;
    for (int i = 0; i < current_len; i++) {
        max_score = max(max_score, attn_scores[base_offset + i]);
    }
    
    // Apply exp and compute sum
    float sum_exp = 0.0f;
    for (int i = 0; i < current_len; i++) {
        attn_scores[base_offset + i] = expf(attn_scores[base_offset + i] - max_score);
        sum_exp += attn_scores[base_offset + i];
    }
    
    // Normalize
    for (int i = 0; i < current_len; i++) {
        attn_scores[base_offset + i] /= sum_exp;
    }
}

// Kernel for applying attention weights to cached values
template <typename T>
__global__ void apply_attention_weights_kernel(
    // Input tensors
    const float* __restrict__ attn_scores,  // [batch_size, num_heads, current_len]
    const T* __restrict__ value_cache,      // [batch_size, num_heads, current_len, head_dim]
    // Output tensor
    T* __restrict__ output,                 // [batch_size, num_heads, 1, head_dim]
    // Parameters
    const int batch_size,
    const int num_heads,
    const int head_dim,
    const int current_len
) {
    // Shared memory for attention scores
    extern __shared__ float s_attn_scores[];
    
    // Calculate thread indices
    const int b = blockIdx.z;                                  // Batch index
    const int h = blockIdx.y;                                  // Head index
    const int tid = threadIdx.x;                               // Thread ID within block
    
    if (b >= batch_size || h >= num_heads) return;
    
    // Load attention scores into shared memory
    for (int i = tid; i < current_len; i += blockDim.x) {
        const int score_offset = b * num_heads * current_len + h * current_len + i;
        s_attn_scores[i] = attn_scores[score_offset];
    }
    
    __syncthreads();
    
    // Each thread computes weighted sum for dimensions it's responsible for
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float weighted_sum = 0.0f;
        
        for (int seq_pos = 0; seq_pos < current_len; seq_pos++) {
            const int v_offset = b * num_heads * current_len * head_dim + 
                                h * current_len * head_dim +
                                seq_pos * head_dim + d;
            
            weighted_sum += s_attn_scores[seq_pos] * static_cast<float>(value_cache[v_offset]);
        }
        
        // Write output
        const int out_offset = b * num_heads * head_dim + h * head_dim + d;
        output[out_offset] = static_cast<T>(weighted_sum);
    }
}

// Main class for KV cache operations
template <typename T>
class KVCache {
public:
    KVCache(int batch_size, int num_heads, int head_dim, int max_seq_len) 
        : batch_size_(batch_size),
          num_heads_(num_heads),
          head_dim_(head_dim),
          max_seq_len_(max_seq_len),
          current_len_(0) {
        
        // Allocate memory for KV cache
        const size_t cache_size = batch_size * num_heads * max_seq_len * head_dim * sizeof(T);
        CUDA_CHECK(cudaMalloc(&key_cache_, cache_size));
        CUDA_CHECK(cudaMalloc(&value_cache_, cache_size));
        
        // Initialize cache to zeros
        CUDA_CHECK(cudaMemset(key_cache_, 0, cache_size));
        CUDA_CHECK(cudaMemset(value_cache_, 0, cache_size));
        
        // Calculate attention scaling factor
        scale_ = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    ~KVCache() {
        // Free allocated memory
        if (key_cache_) CUDA_CHECK(cudaFree(key_cache_));
        if (value_cache_) CUDA_CHECK(cudaFree(value_cache_));
    }
    
    // Update KV cache with new token
    void update(const T* new_keys, const T* new_values, cudaStream_t stream = 0) {
        if (current_len_ >= max_seq_len_) {
            fprintf(stderr, "Error: KV cache is full (max_seq_len=%d)\n", max_seq_len_);
            return;
        }
        
        const int total_threads = batch_size_ * num_heads_ * head_dim_;
        const int block_size = 256;
        const int grid_size = (total_threads + block_size - 1) / block_size;
        
        update_kv_cache_kernel<<<grid_size, block_size, 0, stream>>>(
            new_keys, new_values,
            key_cache_, value_cache_,
            batch_size_, num_heads_, head_dim_, max_seq_len_, current_len_
        );
        
        // Increment sequence length
        current_len_++;
    }
    
    // Compute attention for the current token
    void compute_attention(const T* query, T* output, cudaStream_t stream = 0) {
        if (current_len_ == 0) {
            fprintf(stderr, "Error: KV cache is empty\n");
            return;
        }
        
        // Allocate temporary memory for attention scores
        float* attn_scores;
        CUDA_CHECK(cudaMalloc(&attn_scores, batch_size_ * num_heads_ * current_len_ * sizeof(float)));
        
        // Step 1: Compute attention scores
        const int block_size_scores = 256;
        dim3 grid_scores(current_len_, num_heads_, batch_size_);
        const int shared_mem_size_scores = block_size_scores * sizeof(float);
        
        compute_attention_scores_kernel<<<grid_scores, block_size_scores, shared_mem_size_scores, stream>>>(
            query, key_cache_, attn_scores,
            batch_size_, num_heads_, head_dim_, current_len_, scale_
        );
        
        // Step 2: Apply softmax
        const int total_softmax_items = batch_size_ * num_heads_;
        const int block_size_softmax = 256;
        const int grid_size_softmax = (total_softmax_items + block_size_softmax - 1) / block_size_softmax;
        
        apply_softmax_kernel<<<grid_size_softmax, block_size_softmax, 0, stream>>>(
            attn_scores,
            batch_size_, num_heads_, current_len_
        );
        
        // Step 3: Apply attention weights to values
        const int block_size_weights = 256;
        dim3 grid_weights(1, num_heads_, batch_size_);
        const int shared_mem_size_weights = current_len_ * sizeof(float);
        
        apply_attention_weights_kernel<<<grid_weights, block_size_weights, shared_mem_size_weights, stream>>>(
            attn_scores, value_cache_, output,
            batch_size_, num_heads_, head_dim_, current_len_
        );
        
        // Free temporary memory
        CUDA_CHECK(cudaFree(attn_scores));
    }
    
    // Get current sequence length
    int get_current_length() const {
        return current_len_;
    }
    
    // Reset cache (start a new sequence)
    void reset() {
        current_len_ = 0;
    }
    
    // Get pointers to internal cache (for advanced use cases)
    T* get_key_cache() { return key_cache_; }
    T* get_value_cache() { return value_cache_; }
    
private:
    int batch_size_;
    int num_heads_;
    int head_dim_;
    int max_seq_len_;
    int current_len_;
    float scale_;
    
    T* key_cache_;
    T* value_cache_;
};

// Explicit template instantiations
template class KVCache<float>;
template class KVCache<half>;