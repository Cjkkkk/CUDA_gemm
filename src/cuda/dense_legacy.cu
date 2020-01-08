/*
This is legacy code for dense matrix multiplication
*/

#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "dense_help_func.hpp"

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