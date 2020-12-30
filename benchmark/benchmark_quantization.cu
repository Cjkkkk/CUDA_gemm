#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "dense_help_func.hpp"
#include "quantization_8bit.cu"
#include "dense.cu"

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    // for uint8
    size_t bytes = sizeof(uint32_t) * M * K / 4;
    uint32_t* h_A = (uint32_t*)malloc(bytes);
    uint32_t* h_B = (uint32_t*)malloc(bytes);
    uint32_t* h_C = (uint32_t*)malloc(bytes);

    uint32_t* d_A;
    uint32_t* d_B;
    uint32_t* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes));
    checkCudaErrors(cudaMalloc(&d_B, bytes));
    checkCudaErrors(cudaMalloc(&d_C, bytes));

    // for float
    size_t fbytes = sizeof(float) * M * K;
    float* fh_A = (float*)malloc(fbytes);
    float* fh_B = (float*)malloc(fbytes);
    float* fh_C = (float*)malloc(fbytes);

    float* fd_A;
    float* fd_B;
    float* fd_C;

    checkCudaErrors(cudaMalloc(&fd_A, fbytes));
    checkCudaErrors(cudaMalloc(&fd_B, fbytes));
    checkCudaErrors(cudaMalloc(&fd_C, fbytes));

    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_X = 4;
    const int THREAD_SIZE_Y = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;
    const int BIT_WIDTH = 8;
    int k_block = K / BLOCK_SIZE_K;
    int stride = 2;

    // 生成A的数据
    for( int i = 0; i < M * K; i++ ) {
        int row = (i / K);
        int col = (i % K);
        int row_block = row / BLOCK_SIZE_M;
        int col_block = col / BLOCK_SIZE_K;
        if ((row_block * k_block + col_block) % stride == 0) {
            h_A[i/4] = 0x01010101;
            fh_A[i] = 1;
        }
        else {
            h_A[i/4] = 0;
            fh_A[i] = 0;
        }
    }

    // 生成B的数据
    for( int i = 0; i < K * N; i++ ) {
        if ( i >= K * N / 2) {
            h_B[i/4] = 0x02020202;
            fh_B[i] = 2;
        }
        else {
            h_B[i/4] = 0;
            fh_B[i] = 0;
        }
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));

    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        MatrixMulCUDAQuantize8bit<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, BIT_WIDTH, ENABLE_DOUBLE_BUFFER> 
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

    // dense
    checkCudaErrors(cudaMemcpy( fd_A, fh_A, fbytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( fd_B, fh_B, fbytes, cudaMemcpyHostToDevice));
    

    checkCudaErrors(cudaMemcpy( fd_C, fh_C, fbytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));

    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        MatrixMulCUDA6<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(fd_A, fd_B, fd_C, K, N);

    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    checkCudaErrors(cudaMemcpy( fh_C, fd_C, fbytes, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "My gemm1 Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i/4] - fh_C[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i/4]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, (float)h_C[i], fh_C[col * M + row], eps);
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


    cudaFree(fd_A);
    cudaFree(fd_B);
    cudaFree(fd_C);
    free(fh_A);
    free(fh_B);
    free(fh_C);
}