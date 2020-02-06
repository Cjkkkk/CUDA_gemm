#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "bcsr.hpp"
#include "csr.hpp"
#include "utils.hpp"
#include "sparse_help_func.hpp"
#include "sparse.cu"


int main(int argc, char** argv) {
    if (argc != 5) {
        printf("usage: ./main [M] [K] [N] [Sparsity]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);
    size_t Sparsity = atoi(argv[4]);

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
    const int THREAD_SIZE_X = 4;
    const int THREAD_SIZE_Y = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    float alpha = 1.0;
    float beta = 0;
    
    // 生成A的数据
    // worst case random 
    int nnz = M * K * (Sparsity / 100.0);
    int nnz_stride = M * K / nnz;
    for ( int i = 0; i < M * K; i++ ) {
            if (i % nnz_stride == 0) h_A[i] = 1;
            else {
                h_A[i] = 0;
            }
        }

    // 生成B的数据
    for( int i = 0 ; i < K; i ++ ) {
        for ( int j = 0 ; j < N ; j ++) {
            if ( i < K / 2 && j < N / 2) h_B[i * N + j] = 0;
            else if ( i < K / 2 && j >= N / 2) h_B[i * N + j] = 1;
            else if ( i >= K / 2 && j < N / 2) h_B[i * N + j] = 2;
            else {
                h_B[i * N + j] = 3;
            }
        }
    }
    
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    // bcsr
    // convert to bcsr mat
    bcsr bcsr_mat{(int)M, (int)K, BLOCK_SIZE_M, BLOCK_SIZE_K};
    cal_block(&bcsr_mat, h_A);

    bcsr_mat.row_ptr = (int*)malloc(sizeof(int) * ( bcsr_mat.m_block + 1 ));
    bcsr_mat.col_idx = (int*)malloc(sizeof(int) * bcsr_mat.nnz_block_num );
    bcsr_mat.val = (float*)malloc(sizeof(float) * bcsr_mat.nnz_block_num * bcsr_mat.m_block_sz * bcsr_mat.n_block_sz);
    
    generate_bcsr(&bcsr_mat, h_A);


    float* val;
    int* col_idx;
    int* row_ptr;

    checkCudaErrors(cudaMalloc(&val, sizeof(float) * bcsr_mat.nnz_block_num * bcsr_mat.m_block_sz * bcsr_mat.n_block_sz));
    checkCudaErrors(cudaMalloc(&col_idx, sizeof(int) * bcsr_mat.nnz_block_num));
    checkCudaErrors(cudaMalloc(&row_ptr, sizeof(int) * ( bcsr_mat.m_block + 1 )));
    
    checkCudaErrors(cudaMemcpy( val, bcsr_mat.val, sizeof(float) * bcsr_mat.nnz_block_num * bcsr_mat.m_block_sz * bcsr_mat.n_block_sz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( col_idx, bcsr_mat.col_idx, sizeof(int) * bcsr_mat.nnz_block_num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( row_ptr, bcsr_mat.row_ptr, sizeof(int) * ( bcsr_mat.m_block + 1), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        MatrixMulCUDA5<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
        <<< dimGrid, dimBlock >>>(val, col_idx, row_ptr, d_B, d_C, K, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My sparse block gemm Performance= %.0f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cusparse csr
    csr csr_mat{(int)M, (int)K}; 
    cal_nnz(&csr_mat, h_A);

    csr_mat.row_ptr = (int*)malloc(sizeof(int) * ( csr_mat.m + 1 ));
    csr_mat.col_idx = (int*)malloc(sizeof(int) * csr_mat.nnz_num );
    csr_mat.val = (float*)malloc(sizeof(float) * csr_mat.nnz_num );

    generate_csr(&csr_mat, h_A);
    
    float* csr_val;
    int* csr_col_idx;
    int* csr_row_ptr;

    checkCudaErrors(cudaMalloc(&csr_val, sizeof(float) * csr_mat.nnz_num));
    checkCudaErrors(cudaMalloc(&csr_col_idx, sizeof(int) * csr_mat.nnz_num));
    checkCudaErrors(cudaMalloc(&csr_row_ptr, sizeof(int) * ( csr_mat.m + 1 )));
    
    checkCudaErrors(cudaMemcpy( csr_val, csr_mat.val, sizeof(float) * csr_mat.nnz_num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( csr_col_idx, csr_mat.col_idx, sizeof(int) * csr_mat.nnz_num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( csr_row_ptr, csr_mat.row_ptr, sizeof(int) * ( csr_mat.m + 1 ), cudaMemcpyHostToDevice));

    cusparseHandle_t cusparse_handle;
    cusparseSpMatDescr_t descrA;
    cusparseDnMatDescr_t descrB, descrC;

    checkCuSparseErrors(cusparseCreate(&cusparse_handle));

    checkCuSparseErrors(
        cusparseCreateDnMat(
            &descrB,
            K,
            M,
            K,
            d_B,
            CUDA_R_32F,
            CUSPARSE_ORDER_COL));
    
    checkCuSparseErrors(
        cusparseCreateDnMat(
            &descrC,
            M,
            N,
            M,
            d_C,
            CUDA_R_32F,
            CUSPARSE_ORDER_COL));
    
    checkCuSparseErrors (
        cusparseCreateCsr(&descrA,
            M,
            K,
            csr_mat.nnz_num,
            csr_row_ptr,
            csr_col_idx,
            csr_val,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));

    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    size_t buffer_size;
    checkCuSparseErrors(
        cusparseSpMM_bufferSize(
                cusparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_TRANSPOSE,
                &alpha,
                descrA,
                descrB,
                &beta,
                descrC,
                CUDA_R_32F,
                CUSPARSE_CSRMM_ALG1,
                &buffer_size
            ));
    float* externalBuffer;
    checkCudaErrors(cudaMalloc(&externalBuffer, buffer_size));
    
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        checkCuSparseErrors(
            cusparseSpMM(cusparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_TRANSPOSE,
                &alpha,
                descrA,
                descrB,
                &beta,
                descrC,
                CUDA_R_32F,
                CUSPARSE_CSRMM_ALG1,
                externalBuffer
                )
            );
        
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuSparse Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);
    
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