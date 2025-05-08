import torch
import time
from typing import Callable, Dict, List, Tuple
import argparse
from python_wrapper import vector_add, matrix_add, matrix_mul, matrix_mul_tiled

class CUDABenchmark:
    def __init__(self, num_iterations: int = 1000, num_warmup: int = 100):
        self.num_iterations = num_iterations
        self.num_warmup = num_warmup
        self.operations = {
            'vector_add': self._setup_vector_add,
            'matrix_add': self._setup_matrix_add,
            'matrix_mul': self._setup_matrix_mul,
            'matrix_mul_tiled': self._setup_matrix_mul_tiled
        }

    def _setup_vector_add(self, size: int) -> Tuple[Callable, Callable, List[torch.Tensor]]:
        a = torch.rand(size, device='cuda')
        b = torch.rand(size, device='cuda')
        return (
            lambda x, y: vector_add(x, y),
            lambda x, y: x + y,
            [a, b]
        )

    def _setup_matrix_add(self, size: int) -> Tuple[Callable, Callable, List[torch.Tensor]]:
        a = torch.rand((size, size), device='cuda')
        b = torch.rand((size, size), device='cuda')
        return (
            lambda x, y: matrix_add(x, y),
            lambda x, y: x + y,
            [a, b]
        )
    
    def _setup_matrix_mul(self, size: int) -> Tuple[Callable, Callable, List[torch.Tensor]]:
        matrix_size = max(32, size - (size % 32))
        if matrix_size > 4096:
            matrix_size = 4096
        a = torch.rand((matrix_size, matrix_size), device='cuda')
        b = torch.rand((matrix_size, matrix_size), device='cuda')
        return (
            lambda x, y: matrix_mul(x, y),
            lambda x, y: torch.matmul(x, y),
            [a, b]
        )

    def _setup_matrix_mul_tiled(self, size: int) -> Tuple[Callable, Callable, List[torch.Tensor]]:
        matrix_size = max(32, size - (size % 32))
        if matrix_size > 4096:
            matrix_size = 4096
        a = torch.rand((matrix_size, matrix_size), device='cuda')
        b = torch.rand((matrix_size, matrix_size), device='cuda')
        return (
            lambda x, y: matrix_mul_tiled(x, y),
            lambda x, y: torch.matmul(x, y),
            [a, b]
        )

    def benchmark_operation(self, op_name: str, size: int) -> Dict:
        try:
            custom_op, pytorch_op, tensors = self.operations[op_name](size)
            
            # Verify correctness
            result_custom = custom_op(*tensors)
            result_pytorch = pytorch_op(*tensors)
            torch.cuda.synchronize()
            
            # Validate results
            if not torch.allclose(result_custom, result_pytorch, rtol=1e-3, atol=1e-3):
                print(f"Warning: Results for {op_name} don't match PyTorch implementation")

            # Warmup
            for _ in range(self.num_warmup):
                _ = custom_op(*tensors)
                # _ = pytorch_op(*tensors)
                torch.cuda.synchronize()

            # Benchmark custom implementation
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(self.num_iterations):
                _ = custom_op(*tensors)
            torch.cuda.synchronize()
            custom_time = time.perf_counter() - start_time

            # Benchmark PyTorch implementation
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(self.num_iterations):
                _ = pytorch_op(*tensors)
            torch.cuda.synchronize()
            pytorch_time = time.perf_counter() - start_time

            # Calculate averages and speedup
            custom_avg = custom_time / self.num_iterations
            pytorch_avg = pytorch_time / self.num_iterations
            speedup = pytorch_avg / custom_avg if custom_avg > 0 else 0

            return {
                'operation': op_name,
                'size': size,
                'custom_time': custom_avg,
                'pytorch_time': pytorch_avg,
                'speedup': speedup
            }
            
        except Exception as e:
            print(f"Error in {op_name} (size {size}): {str(e)}")
            return None

def print_results(results: List[Dict]):
    if not results:
        print("\nNo results to display")
        return
        
    print("\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Operation':>15} | {'Size':>8} | {'Custom (ms)':>12} | {'PyTorch (ms)':>12} | {'Speedup':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['operation']:>15} | {r['size']:8d} | {r['custom_time']*1000:12.3f} | "
              f"{r['pytorch_time']*1000:12.3f} | {r['speedup']:8.2f}x")
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description='Benchmark CUDA operations')
    parser.add_argument('--operations', nargs='+', 
                       choices=['vector_add', 'matrix_add', 'matrix_mul', 'matrix_mul_tiled'],
                       default=['matrix_mul', 'matrix_mul_tiled'])
    parser.add_argument('--sizes', nargs='+', type=int, default=[32, 64, 128, 256])
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=100)
    
    args = parser.parse_args()
    benchmark = CUDABenchmark(args.iterations, args.warmup)
    results = []

    for op in args.operations:
        for size in args.sizes:
            result = benchmark.benchmark_operation(op, size)
            if result is not None:
                results.append(result)

    print_results(results)

if __name__ == "__main__":
    main()
    # Add this to your benchmark.py
# benchmark.py
# if __name__ == "__main__":
#     # Correct argument names
#     benchmark = CUDABenchmark(num_iterations=1, num_warmup=0)
#     # a = torch.rand(1000, device='cuda')
#     # b = torch.rand(1000, device='cuda')
#     matrix_size = 4000
#     a = torch.rand((matrix_size, matrix_size), device='cuda')
#     b = torch.rand((matrix_size, matrix_size), device='cuda')
#     result = matrix_mul(a, b)
#     result = matrix_mul_tiled(a, b)
#     result = a * b
#     torch.cuda.synchronize()