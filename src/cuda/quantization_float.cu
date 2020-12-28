#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "dense_help_func.hpp"


// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer uint
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_UINT(pointer) (reinterpret_cast<uint32_t*>(&(pointer))[0])
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const int BIT_WIDTH,    // real datatype
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void MatrixMulCUDAQuantize( 
    uint32_t * __restrict__ A,
    uint32_t * __restrict__ B,
    uint32_t * __restrict__ C, 
    const int K,
    const int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = ty * bszx + tx;

    // shared memory

    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];
    
    // threads needed to load one row of tile
    const int per_load_element = 16; // int4 = 16 byte
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / per_load_element; // 2
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / per_load_element; // 2
    
    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW; 
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * per_load_element;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * per_load_element;
    
    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW; // 64 / 2 = 32
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW; // 64 / 2 = 32
    uint8_t data_a[16];
    uint8_t data_b[16];

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {
        // load A from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            FETCH_UINT4(data_a) = FETCH_UINT4(A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL + tile_idx, // col
                K ) / 4]);
            // unpack
            #pragma unroll
            for ( int j = 0 ; j < 16 ; j += 1) {
                As[A_TILE_ROW_START + i][A_TILE_COL + j] = __uint2float_rd(data_a[j]) * 1.0; 
            }
        }

        // load B from global memory to shared memory
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            FETCH_UINT4(data_b) = FETCH_UINT4(B[OFFSET(
                tile_idx + B_TILE_ROW_START + i, // row
                B_TILE_COL + BLOCK_SIZE_N * bx, // col
                K ) / 4 ]);
            
            // unpack
            #pragma unroll
            for ( int j = 0 ; j < 16 ; j += 1) {
                Bs[B_TILE_ROW_START + i][B_TILE_COL + j] = __uint2float_rd(data_b[j]) * 1.0; 
            }
        }
    
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k][THREAD_SIZE_X * tx + thread_x]);
            }
            
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        // pack
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            data_a[thread_x] = __float2uint_rd(accum[thread_y][thread_x] * 1.0);
        }
        FETCH_UINT(C[OFFSET(
            BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
            BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + 0,
            N) / 4 ]) = FETCH_UINT(data_a[0]);
    }
}

// TODO add shuffle to enable GPU write back col