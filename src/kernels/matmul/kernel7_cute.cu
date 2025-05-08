#include <cstdlib>
#include <cstdio>
#include <cassert>


/*
1. thrust/host_vector.h

    a. Provides thrust::host_vector<T>, a container similar to std::vector that stores data in host (CPU) memory
    b. Automatically manages memory allocation and deallocation on the host
    c. Provides STL-like interface with methods like push_back(), size(), begin(), end(), etc.

2. thrust/device_vector.h

    a. Provides thrust::device_vector<T>, a container that stores data in device (GPU) memory
    b. Automatically manages CUDA memory allocation and deallocation
    c. Seamlessly handles data transfer between host and device
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>



// KERNEL ENTRY POINT
template <class ProblemShape, class CtaTiler, 
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>

/*
static Keyword indicates that this function/variable has internal linkage. It means that it's only visible 
within the current translation unit/file. When applied to a CUDA kernel:

    a. Internal linkage: The kernel is only visible within the current compilation unit (typically the current .cu file)
    b. Name mangling avoidance: Prevents symbol conflicts if multiple files define kernels with the same name
    c. Optimization opportunities: Compiler can make more aggressive optimizations since it knows all uses of the function
    d. No external access: Other compilation units cannot call this kernel

*/

/*
decltype is a C++ keyword (introduced in C++11) that stands for "declared type". It's a compile-time operator that determines the type 
of an expression without evaluating that expression.

a. CThreadLayout{}: Creates a temporary instance of the CThreadLayout type using uniform initialization
b. size(CThreadLayout{}): Calls the size() function on that temporary object. In CuTe/CUTLASS, size() typically returns the total number of elements in a layout or shape as a compile-time constant
c. decltype(...): Gets the type of the expression without evaluating it at runtime. The result is typically a type like cute::Int<256> (a compile-time integer)
d. ::value: Accesses the static value member of that type, which contains the actual integer value

*/
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
/*
a. There are many template parameters, let's quickly review them and then go into more depth on their uses.
b. ProblemShape. The MxNxK problem shape of this matrix multiply.
c. CtaTiler. A CuTe tiler concept that determines how to extract a tile of data from the problem shape.
d. TA const* A, TB const* B, TC* C. The types and pointers to the A, B, and C data, respectively.
e. AStride, BStride, CStride. The layout strides corresponding to the ProblemShape for each A, B, and C.
f. ASmemLayout, BSmemLayout, CSmemLayout. The layouts, if needed, of shared memory to use for staging A-data, B-data, and C-data within each CTA.
g. AThreadLayout, BThreadLayout, CThreadLayout. The layouts of threads to be used in partitioning each stage.
c. Alpha alpha, Beta beta. The types and values of the scalar constants to compute GEMM: C = alpha * A * B + beta * C.
*/

using namespace cute;

//
// Preconditions
//
CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

static_assert(is_static<AThreadLayout>::value);
static_assert(is_static<BThreadLayout>::value);
static_assert(is_static<CThreadLayout>::value);

CUTE_STATIC_ASSERT_V(size(tA) == size(tB));                          // NumThreads
CUTE_STATIC_ASSERT_V(size(tC) == size(tA));                          // NumThreads

CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});  // BLK_M / THR_M
CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});  // BLK_K / THR_K
CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});  // BLK_N / THR_N
CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});  // BLK_K / THR_K
CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});  // BLK_M / THR_M
CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});  // BLK_N / THR_N

static_assert(is_static<ASmemLayout>::value);
static_assert(is_static<BSmemLayout>::value);
static_assert(is_static<CSmemLayout>::value);

CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN
// End Preconditions


//
// Full and Tiled Tensors
//

// Represent the full tensors
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

/*
-> the make_tensor function requires an engine and layout to be specified as args. hence: make_tensor(pointer, shape, stride)
Breaking down each component:

a. make_gmem_ptr(A): Creates a global memory pointer wrapper from the raw pointer A. This tells CuTe that this data resides in global memory.

b. select<0,2>(shape_MNK):
    shape_MNK is  a tuple containing three dimensions (M, N, K)
    select<0,2> extracts elements at indices 0 and 2 from this tuple
    For matrix A, this gives us shape (M, K)

c. dA: This is the stride/layout information for tensor A, which defines how to map 2D coordinates to linear memory addresses
d. make_tensor(...): Combines the pointer, shape, and stride information to create a tensor view

For each matrix:

Matrix A: Shape (M, K) - the left matrix in multiplication
Matrix B: Shape (N, K) - the right matrix in multiplication
Matrix C: Shape (M, N) - the result matrix

The comments // (M,K), // (N,K), and // (M,N) indicate the dimensions of each tensor.
This setup is creating tensor views for a matrix multiplication C = A × B where:
A is M×K
B is K×N (but stored as N×K, which is why select<1,2> is used)
C is M×N
*/

/*
For each of the (M,K), (N,K), and (M,N) tensors, the gemm_nt and gemm_tn construct the strides those tensors will use. In gemm_nt the strides are defined as

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);    // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);    // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);    // (dM, dN)

and in gemm_tn the strides are defined as

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});    // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});    // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);    // (dM, dN)
*/


// SIDE NOTE
/*
M-major, N-major, K-major
We've found that the BLAS convention of using "non-transposed" (N) and "transposed" (T) flags in conjunction with the mode conventions of MxK * KxN to confuse 
the core issue of "what layout does this matrix use" and "in which mode does my matrix have a stride-1?". Indeed, the answer to those questions can always be found 
by inspecting the CuTe Layout.

Instead of row-major or column-major (or Transposed and Not-Transposed), we have found it much more convenient to say that a matrix is "M-major" if it is stride-1 in the M-mode, 
"N-major" if it is stride-1 in the N-mode, or "K-major" if it is stride-1 in the K-mode. Furthermore, knowing that matrix multiply always performs a reduction in the K-mode, it is 
very convenient from a software perspective to always have the K-mode in the same place and adopt the mode convention MxK * NxK. Implementations will always reduce over the second mode 
(the K mode) of both input matrices and leads to cases where implementations can treat both input matrices the same way.

How do we translate this into the BLAS user's experience?

BLAS	A Majorness	 A Layout	 B Majorness	B Layout
NT	     M-major	(M,K):(1,ldA)	N-major	  (N,K):(1,ldB)
TN	     K-major	(M,K):(ldA,1)	K-major	  (N,K):(ldB,1)
NN	     M-major	(M,K):(1,ldA)	K-major	  (N,K):(ldB,1)
TT	     K-major	(M,K):(ldA,1)	N-major	  (N,K):(1,ldB)

BLAS Operation Types Explained

1. NT (Not transposed A, Transposed B):

  A is M-major (row-major): fastest access along rows
  B is N-major (row-major in transposed form): fastest access along rows
  This is the optimal CuTE convention where both matrices can use stride-1 access patterns

2. TN (Transposed A, Not transposed B):

  A is K-major (column-major): fastest access along columns
  B is K-major (column-major): fastest access along columns
  Both matrices have optimal access for column-oriented operations

3. NN (Not transposed A, Not transposed B):

  A is M-major (row-major): fastest access along rows
  B is K-major (column-major): fastest access along columns
  Mixed access pattern - typical in BLAS implementations

4. TT (Transposed A, Transposed B):

  A is K-major (column-major): fastest access along columns
  B is N-major (row-major in transposed form): fastest access along rows
  Mixed access pattern
*/

// Get the appropriate blocks for this thread block

auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)



    
}





