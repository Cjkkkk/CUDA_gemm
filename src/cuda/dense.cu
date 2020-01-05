#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "bcsr.hpp"
#include "csr.hpp"
#include "utils.hpp"

// gemm
#define A(i, j) A[i * K + j]
#define B(i, j) B[i * N + j]
#define C(i, j) C[i * N + j]
#define A_tile(i, j) A_tile[i * blockDim.x + j]
#define B_tile(i, j) B_tile[i * blockDim.y + j]

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

static const char *_cuBlasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}


#define checkCuBlasErrors(func)				\
{									\
    cublasStatus_t e = (func);			\
    if(e != CUBLAS_STATUS_SUCCESS)						                \
        printf ("%s %d CuBlas: %s", __FILE__,  __LINE__, _cuBlasGetErrorEnum(e));		\
}

 
template <int BLOCK_SIZE>
__global__ void MatMul(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float A_tile[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_tile[BLOCK_SIZE * BLOCK_SIZE];

    float accu = 0;

    for(int tileIdx = 0; tileIdx < K / blockDim.x; tileIdx ++) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        // Column j of matrix A
        int j = tileIdx * blockDim.x + threadIdx.x;
        // Load A(i,j) to shared mem
        A_tile(threadIdx.y, threadIdx.x) = A(i,j);
        // Load B(j,i) to shared mem
        B_tile(threadIdx.x, threadIdx.y) = B(j,i); // Global Mem Not coalesced
        // Synchronize before computation
        __syncthreads();
        #pragma unroll
        for(int k = 0 ; k < blockDim.x ; k ++ ) {
            accu = accu + A_tile(threadIdx.y, k) * B_tile(k, threadIdx.x);
        }
        __syncthreads(); 
    }

    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;   
    // Store accumulated value to C(i,j)
    C(i,j) = accu;       

}

template <int BLOCK_SIZE>
__global__ void MatMul1(float* A, float* B, float* C, int M, int K, int N) {

    float accu = 0;

    for(int tileIdx = 0; tileIdx < K / blockDim.x; tileIdx ++) {

        /* Load one tile of A and one tile of B into shared mem */

        // Row i of matrix A
        __shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float B_tile[BLOCK_SIZE][BLOCK_SIZE];

        int i = blockIdx.y * blockDim.y + threadIdx.y;
        // Column j of matrix A
        int j = tileIdx * blockDim.x + threadIdx.x;
        // Load A(i,j) to shared mem
        A_tile[threadIdx.y][threadIdx.x] = A(i,j);
        // Load B(j,i) to shared mem
        B_tile[threadIdx.y][threadIdx.x] = B(i,j);
        __syncthreads();

        /* Accumulate one tile of C from tiles of A and B in shared mem */
        #pragma unroll
        for(int k = 0 ; k < BLOCK_SIZE ; k ++ ) {
            accu = accu + A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        __syncthreads(); 
    }

    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;   
    // Store accumulated value to C(i,j)
    C(i,j) = accu;       

}

template <int BLOCK_SIZE> __global__ void MatrixMulCUDA( 
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C, 
    const int K,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = K * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + K - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * N;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + K * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub = fma(As[ty][k], Bs[k][tx], Csub);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[N * ( BLOCK_SIZE * by + ty ) + BLOCK_SIZE * bx + tx ] = Csub;
}


texture<float, 2, cudaReadModeElementType> tex_A;
texture<float, 2, cudaReadModeElementType> tex_B;

template <int BLOCK_SIZE> __global__ void MatrixMulCUDA1( 
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C, 
    const int K,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int aBegin = K * BLOCK_SIZE * by;
    int aEnd   = aBegin + K - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * N;
    float Csub = 0;
    int tile_idx = 0;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep, tile_idx +=1) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = tex2D<float>(tex_A, BLOCK_SIZE * tile_idx + tx, BLOCK_SIZE * by + ty);
        Bs[ty][tx] = tex2D<float>(tex_B, BLOCK_SIZE * bx + tx, BLOCK_SIZE * tile_idx + ty);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // Csub += As[ty][k] * Bs[k][tx];
            Csub = fma(As[ty][k], Bs[k][tx], Csub);
        }

        __syncthreads();
    }

    C[N * ( BLOCK_SIZE * by + ty ) + BLOCK_SIZE * bx + tx ] = Csub;
}

__global__ void read (int m, int n) 
{
    float val;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            val = tex2D (tex_B, col, row);
            printf ("%f  ", val);
        }
        printf ("\n");
    }
}


// block 16 * 16 grid M / 64 N / 64 => every thread compute 16 elements
// warp 数目减少 locality增加
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA2( 
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C, 
    const int K,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Csub[4] = {0, 0, 0, 0};
    for (int tile_idx = 0 ; tile_idx < K / BLOCK_SIZE ;  tile_idx +=1) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // ((float4*)(&As[ty][tx * 4]))[0] = ((float4*)(&A[BLOCK_SIZE * tile_idx + tx * 4 + (BLOCK_SIZE * by + ty) * K]))[0];
        // ((float4*)(&Bs[ty][tx * 4]))[0] = ((float4*)(&B[BLOCK_SIZE * bx + tx * 4 + (BLOCK_SIZE * tile_idx + ty) * N]))[0];
        As[ty][tx * 4] = A[(BLOCK_SIZE * by + ty) * K + BLOCK_SIZE * tile_idx + tx * 4];
        Bs[ty][tx * 4] = B[(BLOCK_SIZE * tile_idx + ty) * N + BLOCK_SIZE * bx + tx * 4];
        As[ty][tx * 4 + 1] = A[(BLOCK_SIZE * by + ty) * K + BLOCK_SIZE * tile_idx + tx * 4 + 1];
        Bs[ty][tx * 4 + 1] = B[(BLOCK_SIZE * tile_idx + ty) * N + BLOCK_SIZE * bx + tx * 4 + 1];
        As[ty][tx * 4 + 2] = A[(BLOCK_SIZE * by + ty) * K + BLOCK_SIZE * tile_idx + tx * 4 + 2];
        Bs[ty][tx * 4 + 2] = B[(BLOCK_SIZE * tile_idx + ty) * N + BLOCK_SIZE * bx + tx * 4 + 2];
        As[ty][tx * 4 + 3] = A[(BLOCK_SIZE * by + ty) * K + BLOCK_SIZE * tile_idx + tx * 4 + 3];
        Bs[ty][tx * 4 + 3] = B[(BLOCK_SIZE * tile_idx + ty) * N + BLOCK_SIZE * bx + tx * 4 + 3];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub[0] = fma(As[ty][k], Bs[k][tx * 4], Csub[0]);
            Csub[1] = fma(As[ty][k], Bs[k][tx * 4 + 1], Csub[1]);
            Csub[2] = fma(As[ty][k], Bs[k][tx * 4 + 2], Csub[2]);
            Csub[3] = fma(As[ty][k], Bs[k][tx * 4 + 3], Csub[3]);
        }

        __syncthreads();
    }

    C[N * ( BLOCK_SIZE * by + ty ) + BLOCK_SIZE * bx + tx * 4 ] = Csub[0];
    C[N * ( BLOCK_SIZE * by + ty ) + BLOCK_SIZE * bx + tx * 4 + 1 ] = Csub[1];
    C[N * ( BLOCK_SIZE * by + ty ) + BLOCK_SIZE * bx + tx * 4 + 2 ] = Csub[2];
    C[N * ( BLOCK_SIZE * by + ty ) + BLOCK_SIZE * bx + tx * 4 + 3 ] = Csub[3];
}

// float4
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA3( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int K,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float4 Csub = {0, 0, 0, 0};

    for (int tile_idx = 0 ; tile_idx < K / BLOCK_SIZE ;  tile_idx +=1) {
        __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

        reinterpret_cast<float4*>(As + ty * BLOCK_SIZE + tx * 4)[0] = reinterpret_cast<float4*>(A + (BLOCK_SIZE * by + ty) * K + BLOCK_SIZE * tile_idx + tx * 4 )[0];
        reinterpret_cast<float4*>(Bs + ty * BLOCK_SIZE + tx * 4)[0] = reinterpret_cast<float4*>(B + (BLOCK_SIZE * tile_idx + ty) * N + BLOCK_SIZE * bx + tx * 4 )[0];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub.x = fma(As[ty * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4], Csub.x);
            Csub.y = fma(As[ty * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 1], Csub.y);
            Csub.z = fma(As[ty * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 2], Csub.z);
            Csub.w = fma(As[ty * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 3], Csub.w);
        }

        __syncthreads();
    }

    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty ) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub;
}

// two level block
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA4( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int K,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float4 Csub[4] = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}};

    for (int tile_idx = 0 ; tile_idx < K / BLOCK_SIZE ;  tile_idx +=1) {
        __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

        #pragma unroll
        for ( int i = 0 ; i < 4 ; i ++ ) {
            reinterpret_cast<float4*>(As + BLOCK_SIZE * (ty * 4 + i) + tx * 4)[0] 
                = reinterpret_cast<float4*>(A + (BLOCK_SIZE * by + ty * 4 + i ) * K + BLOCK_SIZE * tile_idx + tx * 4 )[0];
            reinterpret_cast<float4*>(Bs + BLOCK_SIZE * (ty * 4 + i) + tx * 4)[0] 
                = reinterpret_cast<float4*>(B + (BLOCK_SIZE * tile_idx + ty * 4 + i ) * N + BLOCK_SIZE * bx + tx * 4 )[0];
        }
    
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            
            Csub[0].x = fma(As[ty * 4 * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4], Csub[0].x);
            Csub[0].y = fma(As[ty * 4 * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 1], Csub[0].y);
            Csub[0].z = fma(As[ty * 4 * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 2], Csub[0].z);
            Csub[0].w = fma(As[ty * 4 * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 3], Csub[0].w);
            Csub[1].x = fma(As[(ty * 4 + 1) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4], Csub[1].x);
            Csub[1].y = fma(As[(ty * 4 + 1) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 1], Csub[1].y);
            Csub[1].z = fma(As[(ty * 4 + 1) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 2], Csub[1].z);
            Csub[1].w = fma(As[(ty * 4 + 1) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 3], Csub[1].w);
            Csub[2].x = fma(As[(ty * 4 + 2) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4], Csub[2].x);
            Csub[2].y = fma(As[(ty * 4 + 2) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 1], Csub[2].y);
            Csub[2].z = fma(As[(ty * 4 + 2) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 2], Csub[2].z);
            Csub[2].w = fma(As[(ty * 4 + 2) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 3], Csub[2].w);
            Csub[3].x = fma(As[(ty * 4 + 3) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4], Csub[3].x);
            Csub[3].y = fma(As[(ty * 4 + 3) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 1], Csub[3].y);
            Csub[3].z = fma(As[(ty * 4 + 3) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 2], Csub[3].z);
            Csub[3].w = fma(As[(ty * 4 + 3) * BLOCK_SIZE + k], Bs[k * BLOCK_SIZE + tx * 4 + 3], Csub[3].w);
            
        }

        __syncthreads();
    }

    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty * 4 ) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub[0];
    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty * 4 + 1) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub[1];
    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty * 4 + 2) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub[2];
    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty * 4 + 3) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub[3];
}

// global memory col
template <
    int BLOCK_SIZE_M, 
    int BLOCK_SIZE_K,
    int BLOCK_SIZE_N,
    int THREAD_SIZE_M,
    int THREAD_SIZE_N
    > 
__global__ void MatrixMulCUDA6( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int K,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // shared memory
    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];
    
    // registers for C
    float accum[THREAD_SIZE_M][THREAD_SIZE_N];
    for ( int i = 0 ; i < THREAD_SIZE_M ; i ++) {
        for ( int j = 0 ; j < THREAD_SIZE_N ; j ++) {
            accum[i][j] = 0;
        }
    }

    // registers for A and B
    float frag_a[THREAD_SIZE_M];
    float frag_b[THREAD_SIZE_N];
    
    int tid = ty * 8 + tx;
    for (int tile_idx = 0 ; tile_idx < K / BLOCK_SIZE_K ;  tile_idx +=1) {
        
        int A_row = (tid / ( BLOCK_SIZE_K / 4 ));
        int A_col = (tid % ( BLOCK_SIZE_K / 4 )) * 4;

        int B_row = (tid / ( BLOCK_SIZE_N / 4 ));
        int B_col = (tid % ( BLOCK_SIZE_N / 4 )) * 4;

        #pragma unroll
        for ( int i = 0 ; i < 4 ; i ++) {
            reinterpret_cast<float4*>(As + (A_row + i * 8) * BLOCK_SIZE_K + A_col)[0]
            = reinterpret_cast<float4*>(A + K * (BLOCK_SIZE_M * by + (A_row + i * 8)) + A_col + BLOCK_SIZE_K * tile_idx)[0];
            // reinterpret_cast<float4*>(Bs + (B_row + i * 8) * BLOCK_SIZE_N + B_col)[0]
            // = reinterpret_cast<float4*>(B + K * (BLOCK_SIZE_K * tile_idx + (B_row + i * 8)) + B_col + BLOCK_SIZE_N * bx)[0];
        }

        #pragma unroll
        for ( int i = 0 ; i < 4 ; i ++) {
            // reinterpret_cast<float4*>(As + (A_row + i * 8) * BLOCK_SIZE_K + A_col)[0]
            // = reinterpret_cast<float4*>(A + K * (BLOCK_SIZE_M * by + (A_row + i * 8)) + A_col + BLOCK_SIZE_K * tile_idx)[0];
            reinterpret_cast<float4*>(Bs + (B_row + i * 8) * BLOCK_SIZE_N + B_col)[0]
            = reinterpret_cast<float4*>(B + K * (BLOCK_SIZE_K * tile_idx + (B_row + i * 8)) + B_col + BLOCK_SIZE_N * bx)[0];
        }
        
        // #pragma unroll
        // for ( int i = 0 ; i < 4 ; i ++ ) {
        //     #pragma unroll
        //     for (int j = 0 ; j < 4 ; j ++ ) {
        //         As [BLOCK_SIZE_K * (ty * 4 + i) + tx + 8 * j]
        //         = A [(BLOCK_SIZE_K * by + ty * 4 + i ) * K + BLOCK_SIZE_K * tile_idx + tx + 8 * j];
        //         Bs [BLOCK_SIZE_K * (ty * 4 + i) + tx + 8 * j]
        //         = B[(BLOCK_SIZE_K * tile_idx + ty * 4 + i ) * N + BLOCK_SIZE_K * bx + tx + 8 * j];
        //     }
        // }
    
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
                frag_a[thread_y] = As[BLOCK_SIZE_K *  (ty * THREAD_SIZE_M + thread_y) + k];
            }

            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
                frag_b[thread_x] = Bs[BLOCK_SIZE_N *  k + THREAD_SIZE_N * tx + thread_x];
            }
            
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
                    accum[thread_y][thread_x ] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_M; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_N; ++thread_x) {
            C[N * ( BLOCK_SIZE_M * by + ty * THREAD_SIZE_M + thread_y) + BLOCK_SIZE_N * bx + tx * THREAD_SIZE_N + thread_x]
            = accum[thread_y][thread_x];
        }
    }
}

// TODO add shuffle to enable GPU write back col


int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    size_t bytes = sizeof(float) * M * K;
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* h_C1 = (float*)malloc(bytes);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes));
    checkCudaErrors(cudaMalloc(&d_B, bytes));
    checkCudaErrors(cudaMalloc(&d_C, bytes));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;
    int k_block = K / BLOCK_SIZE_K;
    int stride = 2;

    // 生成A的数据
    for( int i = 0; i < M * K; i++ ) {
        int row = (i / K);
        int col = (i % K);
        int row_block = row / BLOCK_SIZE_M;
        int col_block = col / BLOCK_SIZE_K;
        if ((row_block * k_block + col_block) % stride == 0) h_A[i] = 1;
        else {
            h_A[i] = 0;
        }
    }

    // 生成B的数据
    for( int i = 0; i < K * N; i++ ) {
        if ( i >= K * N / 2) h_B[i] = 2;
        else {
            h_B[i] = 0;
        }
    }

    // texture
    // size_t pitch, tex_ofs;
    // size_t pitch1, tex_ofs1;
    // checkCudaErrors(cudaMallocPitch((void**)&d_A, &pitch, K*sizeof(float), M));
    // checkCudaErrors(cudaMemcpy2D(d_A, pitch, h_A, K*sizeof(float), K*sizeof(float), M,cudaMemcpyHostToDevice));
    // tex_A.normalized = false;
    // checkCudaErrors (cudaBindTexture2D (&tex_ofs, &tex_A, d_A, &tex_A.channelDesc,
    //                                    K, M, pitch));
    
    // checkCudaErrors(cudaMallocPitch((void**)&d_B, &pitch1, N*sizeof(float), K));
    // checkCudaErrors(cudaMemcpy2D(d_B, pitch1, h_B, N*sizeof(float), N*sizeof(float), K,cudaMemcpyHostToDevice));
    // tex_B.normalized = false;
    // checkCudaErrors (cudaBindTexture2D (&tex_ofs1, &tex_B, d_B, &tex_B.channelDesc,
    //                                     N, K, pitch1));

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    // read<<<1, 1>>>(M, K);
    for (int run = 0 ; run < nIter; run ++ ) {
        //MatMul <BLOCK_SIZE> <<< dimGrid, dimBlock >>> (d_A, d_B, d_C, M, K, N);

        // dim3 dimBlock(BLOCK_SIZE / 4, BLOCK_SIZE);
        // MatrixMulCUDA3<BLOCK_SIZE> <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, K, N);

        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_N, BLOCK_SIZE_M / THREAD_SIZE_M);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        MatrixMulCUDA6<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_N> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, K, N);

    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle;  
    checkCuBlasErrors ( cublasCreate(&blas_handle) );
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        checkCuBlasErrors (
            cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
                M, N, K, &alpha, 
                d_A, M, d_B, K, &beta, d_C, K
            )
        );
        // dim3 dimBlock(BLOCK_SIZE / 4, BLOCK_SIZE / 4);
        // MatrixMulCUDA4<BLOCK_SIZE> <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, K, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 

    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        // h_C1 是转置
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}