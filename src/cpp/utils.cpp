#include "utils.hpp"
#include <stdlib.h>
#include <stdio.h>
void cal_block(float* mat, int* is_block_present, int* nnz_block_num, size_t m, size_t n, size_t m_block_sz, size_t n_block_sz) {
    int m_block = m / m_block_sz;
    int n_block = n / n_block_sz;
    
    *nnz_block_num = 0;
    for ( int i = 0 ; i < m_block * n_block ; i ++ ) {
        is_block_present[i] = 0;
    }
    for ( int i = 0 ; i < m * n ; i ++ ) {
        if (mat[i] != 0) {
            // 计算属于哪一个block
            int m_block_idx = i / n / m_block_sz;
            int n_block_idx = i % n / n_block_sz;
            if (is_block_present[m_block_idx * n_block + n_block_idx] == 0) {
                is_block_present[m_block_idx * n_block + n_block_idx] = 1;
                *nnz_block_num += 1;
            }
        }
    }
}

void generate_bcsr(float* mat, float* val, int* col_idx, int* row_ptr, size_t m, size_t n, size_t m_block_sz, size_t n_block_sz) {
    int nnz_block_num;
    int m_block = m / m_block_sz;
    int n_block = n / n_block_sz;
    int* is_block_present = (int*)malloc(sizeof( int ) * m_block * n_block );
    cal_block(mat, is_block_present, &nnz_block_num, m, n, m_block_sz, n_block_sz);

    row_ptr = (int*)malloc(sizeof(int) * ( m_block + 1 ));
    col_idx = (int*)malloc(sizeof(int) * nnz_block_num );
    val = (float*)malloc(sizeof(float) * nnz_block_num * m_block_sz * n_block_sz);

    int ptr = 0;
    for ( int i = 0 ; i < m_block ; i += 1) {
        for ( int j = 0 ; j < n_block ; j += 1) {
            if ( is_block_present[i * n_block + j] == 1) {
                // copy whole block into val
                for (int i_block = 0 ; i < m_block_sz ; i ++ ) {
                    for ( int j_block = 0 ; j < n_block_sz ; j ++) {
                        val[ptr] = mat[ (i * m_block_sz + i_block) * n + (j * n_block_sz + j_block)];
                        ptr ++;
                    }
                }
            }
        }
    }
    free(is_block_present);
}