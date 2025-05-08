#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// forward-declare launcher (now takes stream)
void launch_fa2_topk_ampere(const half* Q,
                            const half* K,
                            const half* V,
                            half*       O,
                            int         B,
                            int         H,
                            int         S
                            // cudaStream_t stream
                        );

// ---------------- wrapper ----------------
torch::Tensor flash_attention_topk_forward(torch::Tensor Q,
                                           torch::Tensor K,
                                           torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == torch::kHalf &&
                K.scalar_type() == torch::kHalf &&
                V.scalar_type() == torch::kHalf,
                "tensors must be FP16");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);

    auto O = torch::empty_like(Q);

    launch_fa2_topk_ampere(
        reinterpret_cast<const half*>(Q.data_ptr<torch::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<torch::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(O.data_ptr<torch::Half>()),
        B, H, S
        // at::cuda::getCurrentCUDAStream()
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_topk_forward",
          &flash_attention_topk_forward,
          "FlashAttention-2 Top-K forward (Ampere, FP16)");
}
