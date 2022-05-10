#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "dense_help_func.hpp"
#include "dense.cu"
#include "utils.hpp"

#define ASIZE(type) (sizeof(type) * M * K)
#define BSIZE(type) (sizeof(type) * K * N)
#define CSIZE(type) (sizeof(type) * M * N)

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    float* h_A = (float*)malloc(ASIZE(float));
    float* h_B = (float*)malloc(BSIZE(float));
    float* h_C = (float*)malloc(CSIZE(float));
    float* h_C1 = (float*)malloc(CSIZE(float));

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, ASIZE(float)));
    checkCudaErrors(cudaMalloc(&d_B, BSIZE(float)));
    checkCudaErrors(cudaMalloc(&d_C, CSIZE(float)));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_X = 4;
    const int THREAD_SIZE_Y = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    float alpha = 2.0;
    float beta = 0.0;

    // 生成A的数据
    genRandomMatrix(h_A, M, K);
    genRandomMatrix(h_B, K, N);
    genRandomMatrix(h_C, M, N);
    copyMatrix(h_C1, h_C, M, N);


    checkCudaErrors(cudaMemcpy( d_A, h_A, ASIZE(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, BSIZE(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    checkCudaErrors(cudaMemcpy( d_C, h_C, CSIZE(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));

    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        MatrixMulCUDA6<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, K, N, alpha, beta);

    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    checkCudaErrors(cudaMemcpy( h_C, d_C, CSIZE(float), cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle;  
    checkCuBlasErrors ( cublasCreate(&blas_handle) );
    checkCudaErrors(cudaMemcpy( d_C, h_C1, CSIZE(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        checkCuBlasErrors (
            cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
                M, N, K, &alpha, 
                d_A, K, d_B, N, &beta, d_C, M
            )
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, CSIZE(float), cudaMemcpyDeviceToHost));

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