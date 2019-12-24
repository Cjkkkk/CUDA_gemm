#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

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
        printf ("%s %d CUDA: %s", __FILE__,  __LINE__, cudaGetErrorString(e));		\
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
        for(int k = 0 ; k < blockDim.x ; k ++ ) {
            accu = accu + A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        __syncthreads(); 
    }

    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;   
    // Store accumulated value to C(i,j)
    C(i,j) = accu;       

}


template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *A,
    float *B, float *C, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

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
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        #pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[wB * ( BLOCK_SIZE * by + ty ) + ( BLOCK_SIZE * bx + tx ) ] = Csub;
}

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

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes));
    checkCudaErrors(cudaMalloc(&d_B, bytes));
    checkCudaErrors(cudaMalloc(&d_C, bytes));

    float valB = 0.5;

    for( int i = 0; i < M * K; i++ ) {
        h_A[i] = 1.0;
    }

    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = valB;
    }
    const int BLOCK_SIZE = 32;
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, M / dimBlock.y);
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        MatMul1 <BLOCK_SIZE> <<< dimGrid, dimBlock >>> (d_A, d_B, d_C, M, K, N);
        //MatrixMulCUDA<BLOCK_SIZE> <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, K, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * M * N * K;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf( "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);


    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        double abs_err = fabs(h_C[i] - (M * valB));
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], M * valB, eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}