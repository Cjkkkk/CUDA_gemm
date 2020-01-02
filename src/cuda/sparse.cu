#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "bcsr.hpp"
#include "csr.hpp"
#include "utils.hpp"

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}


static const char *_cuSparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "<unknown>";
}


#define checkCuSparseErrors(func)				\
{									\
    cusparseStatus_t e = (func);			\
    if(e != CUSPARSE_STATUS_SUCCESS)						                \
        printf ("%s %d CuSparse: %s", __FILE__,  __LINE__, _cuSparseGetErrorEnum(e));		\
}

template <int BLOCK_SIZE> __global__ void MatrixMulCUDA5( 
    float * __restrict__ A_Val,
    int* __restrict__ A_col_idx,
    int* __restrict__ A_row_ptr,
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
    
    int row_ptr_start = A_row_ptr[by];
    int row_ptr_end = A_row_ptr[by + 1];
    
    for (int row_ptr = row_ptr_start ; row_ptr < row_ptr_end ; row_ptr = row_ptr + 1) {
        int tile_idx = A_col_idx[row_ptr];
        __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];
        float* A = A_Val + BLOCK_SIZE * BLOCK_SIZE * row_ptr;
        #pragma unroll
        for ( int i = 0 ; i < 4 ; i ++ ) {
            reinterpret_cast<float4*>(As + BLOCK_SIZE * (ty * 4 + i) + tx * 4)[0] 
                = reinterpret_cast<float4*>( A + BLOCK_SIZE * (ty * 4 + i) + tx * 4 )[0];
            
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
        // wait threads to finish , otherwise next tile will overwrite the shared memory
        __syncthreads();
    }

    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty * 4 ) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub[0];
    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty * 4 + 1) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub[1];
    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty * 4 + 2) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub[2];
    reinterpret_cast<float4*> (C + N * ( BLOCK_SIZE * by + ty * 4 + 3) + BLOCK_SIZE * bx + tx * 4 )[0] = Csub[3];
}

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

    const int BLOCK_SIZE = 32;
    int m_block = M / 32;
    int k_block = K / 32;
    int nnz_block = m_block * k_block * (Sparsity / 100.0);
    int stride = m_block * k_block / nnz_block;
    float alpha = 1.0;
    float beta = 0;
    
    // 生成A的数据
    for( int i = 0; i < M * K; i++ ) {
        int row = (i / K);
        int col = (i % K);
        int row_block = row / 32;
        int col_block = col / 32;
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
    
    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, M / dimBlock.y);
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 100;

    // bcsr
    // convert to bcsr mat
    bcsr bcsr_mat{(int)M, (int)K, 32, 32};
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
        dim3 dimBlock(BLOCK_SIZE / 4, BLOCK_SIZE / 4);
        dim3 dimGrid(N / 32, M / 32);
        
        MatrixMulCUDA5<BLOCK_SIZE> <<< dimGrid, dimBlock >>>(val, col_idx, row_ptr, d_B, d_C, K, N);
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
    

    // sort four methods
    // int idx[4] = {0, 1, 2, 3};
    // for ( int i = 0 ; i < 4 ; i ++) {
    //     for ( int j = i + 1 ; j < 4 ; j ++ ) {
    //         if (msecPerMatrixMul[j] <= msecPerMatrixMul[i]) {
    //             int temp_idx = idx[i];
    //             idx[i] = idx[j];
    //             idx[j] = temp_idx;
                
    //             float temp = msecPerMatrixMul[j];
    //             msecPerMatrixMul[j] = msecPerMatrixMul[i];
    //             msecPerMatrixMul[i] = temp;
    //         }
    //     }
    // }

    // printf("\u001b[31m\n");
    // for ( int i = 0 ; i < 4 ; i ++ ) {
    //     if (idx[i] == 0 ) printf("my gemm: %.3f msec\n", msecPerMatrixMul[i]);
    //     else if (idx[i] == 1 ) printf("cublas: %.3f msec\n", msecPerMatrixMul[i]);
    //     else if (idx[i] == 2 ) printf("my block sparse: %.3f msec\n", msecPerMatrixMul[i]);
    //     else if (idx[i] == 3 ) printf("cusparse(csr): %.3f msec\n", msecPerMatrixMul[i]);
    //     else {

    //     }
    // }
    // printf("\u001b[0m\n");
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}