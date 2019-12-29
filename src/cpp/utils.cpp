#include "utils.hpp"
#include "bcsr.hpp"
#include <stdlib.h>
#include <stdio.h>

void cal_block(bcsr* mat, float* data) {
    for ( int i = 0 ; i < mat->m_block * mat->n_block ; i ++ ) {
        mat->is_block_present[i] = 0;
    }
    for ( int i = 0 ; i < mat->m * mat->n ; i ++ ) {
        if (data[i] != 0) {
            // 计算属于哪一个block
            int m_block_idx = i / mat->n / mat->m_block_sz;
            int n_block_idx = i % mat->n / mat->n_block_sz;
            if (mat->is_block_present[m_block_idx * mat->n_block + n_block_idx] == 0) {
                mat->is_block_present[m_block_idx * mat->n_block + n_block_idx] = 1;
                mat->nnz_block_num += 1;
            }
        }
    }
}

void generate_bcsr(bcsr* mat, float* data) {
    int ptr = 0;
    int block_ptr = 0;
    int row_ptr = 0;
    mat->row_ptr[row_ptr ++ ] = block_ptr;
    for ( int i = 0 ; i < mat->m_block ; i += 1) {
        for ( int j = 0 ; j < mat->n_block ; j += 1) {
            if ( mat->is_block_present[i * mat->n_block + j] == 1) {
                mat->col_idx[block_ptr ++ ] = j;
                // copy whole block into val
                for (int i_block = 0 ; i_block < mat->m_block_sz ; i_block ++ ) {
                    for ( int j_block = 0 ; j_block < mat->n_block_sz ; j_block ++) {
                        mat->val[ptr ++ ] = data[ (i * mat->m_block_sz + i_block) * mat->n + (j * mat->n_block_sz + j_block)];
                    }
                }
            }
        }
        // 记录row_ptr
        mat->row_ptr[row_ptr ++ ] = block_ptr;
    }
}

void cal_nnz(csr* mat, float* data) {
    for ( int i = 0 ; i < mat->m * mat->n ; i ++ ) {
        if (data[i] != 0) {
            mat->nnz_num += 1;
        }
    }
}

void generate_csr(csr* mat, float* data) {
    int ptr = 0;
    int row_ptr = 0;
    mat->row_ptr[row_ptr++] = ptr;
    for ( int i = 0 ; i < mat->m ; i += 1) {
        for ( int j = 0 ; j < mat->n; j += 1) {
            if ( data[i * mat->n + j] != 0) {
                mat->col_idx[ptr] = j;
                mat->val[ptr] = data[ i * mat->n + j];
                ptr ++;
            }
        }
        // 记录row_ptr
        mat->row_ptr[row_ptr++] = ptr;
    }
}