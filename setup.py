from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention_topk',
    ext_modules=[
        CUDAExtension('flash_attention_topk', [
            'src/kernels/FA2/flash_attention_topk_binding.cpp',
            'src/kernels/FA2/kernel_topK_FA2.cu',  # This is your big file you posted
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
