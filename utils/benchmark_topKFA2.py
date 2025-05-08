import torch
import flash_attention_topk  # your compiled custom kernel
import torch.nn.functional as F
import time

def benchmark_fn(fn, warmup=10, runs=100):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timing
    start = time.time()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / runs

def topk_flash_attention_emulator(Q, K, V, top_k, scale):
    """
    Emulates CUDA Top-K Flash Attention:
    - Computes QK^T
    - Selects top-k per query
    - Applies softmax over top-k
    - Multiplies selected V
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape

    # [batch, heads, q_seq, k_seq]
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

    # Top-k selection
    topk_values, topk_indices = torch.topk(scores, k=top_k, dim=-1)

    # Softmax over top-k
    softmax_topk = torch.softmax(topk_values, dim=-1)

    # Gather V vectors
    V_selected = torch.gather(
        V.unsqueeze(2).expand(-1, -1, seq_len, -1, -1),  # [batch, heads, query, key, dim]
        dim=3,
        index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)  # [batch, heads, query, topk, dim]
    )

    # Weighted sum
    output = torch.sum(softmax_topk.unsqueeze(-1) * V_selected, dim=3)  # [batch, heads, query, dim]
    return output

# Parameters
batch_size = 1
num_heads = 8
seq_len = 1024
head_dim = 64
top_k = 32
scale = 1.0 / (head_dim ** 0.5)

# Random data
Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

# Custom kernel output
custom_output = flash_attention_topk.flash_attention_topk_forward(Q, K, V, top_k, scale)

# Emulator output (golden reference)
emulator_output = topk_flash_attention_emulator(Q, K, V, top_k, scale)

# Compare correctness
error = (custom_output - emulator_output).abs().max()
mean_error = (custom_output - emulator_output).abs().mean()
cos_sim = torch.nn.functional.cosine_similarity(
    custom_output.flatten(), emulator_output.flatten(), dim=0
)

print(f"✅ Correctness check:")
print(f"  Max error: {error.item()}")
print(f"  Mean error: {mean_error.item()}")
print(f"  Cosine similarity: {cos_sim.item()}")

# Timing
def run_custom():
    return flash_attention_topk.flash_attention_topk_forward(Q, K, V, top_k, scale)

def run_torch_flash():
    return F.scaled_dot_product_attention(Q, K, V, is_causal=False)

custom_time = benchmark_fn(run_custom)
torch_time = benchmark_fn(run_torch_flash)

print(f"\n⏱ Timing comparison:")
print(f"  Custom FlashAttention Top-K time: {custom_time*1e3:.3f} ms")
print(f"  PyTorch FlashAttention (full) time: {torch_time*1e3:.3f} ms")
