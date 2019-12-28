#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>

float arr[8*8] = {
    1, 2, 0, 0, 3, 4, 0, 5,
    0, 6, 0, 0, 7, 0, 0, 0,
    0, 0, 3, 0, 0, 0, 8, 1,
    0, 0, 1, 0, 0, 0, 1, 0,
    1, 6, 0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 0, 3, 1,
    1, 4, 0, 5, 4, 1, 0, 0,
    1, 4, 0, 1, 0, 1, 0, 0,
};

void test_cal_block() {
    int m = 8;
    int n = 8;
    int m_block_sz = 2;
    int n_block_sz = 2;

    int m_block = m / m_block_sz;
    int n_block = n / n_block_sz;
    int* is_block_present = (int*)malloc(sizeof( int ) * m_block * n_block );

    int nnz_block_num;

    int expected_is_block_present[4*4] = {
        1, 0, 1, 1,
        0, 1, 0, 1,
        1, 0, 0, 1,
        1, 1, 1, 0
    };

    cal_block(arr, is_block_present, &nnz_block_num, m, n, m_block_sz, n_block_sz);
    // compare
    for ( int i = 0 ; i < 4 ; i ++ ) {
        for ( int j = 0 ; j < 4 ; j ++ ) {
            if (expected_is_block_present[i * 4 + j] != is_block_present[i * 4 + j]) {
                printf("error in block position: (%d, %d) with val: (%d, %d)\n", i, j, expected_is_block_present[i * 4 + j], is_block_present[i * 4 + j]);
            }
        }
    }
    free(is_block_present);
}

void test_generate_bcsr() {
    int m = 8;
    int n = 8;
    int m_block_sz = 2;
    int n_block_sz = 2;
    int m_block = m / m_block_sz;
    int n_block = n / n_block_sz;
    
    float* val;
    int* col_idx;
    int* row_ptr;

    float expected_arr[64] = {
        1, 2, 0, 0, 3, 4, 0, 5,
        0, 6, 0, 0, 7, 0, 0, 0,
        0, 0, 3, 0, 0, 0, 8, 1,
        0, 0, 1, 0, 0, 0, 1, 0,
        1, 6, 0, 0, 0, 0, 2, 0,
        0, 0, 0, 0, 0, 0, 3, 1,
        1, 4, 0, 5, 4, 1, 0, 0,
        1, 4, 0, 1, 0, 1, 0, 0,
    };

    float expected_arr[40] = {
        1, 2, 0, 6,
        3, 4, 7, 0,
        0, 5, 0, 0,
        3, 0, 1, 0,
        8, 1, 1, 0,
        1, 6, 0, 0,
        2, 0, 3, 1,
        1, 4, 1, 4, 
        0, 5, 0, 1,
        4, 1, 0, 1,
    };

    float expected_col_idx[6] = {
        1,
        0, 2,
        1, 2,
        3
    };

    float expected_row_ptr[5] = {
        0, 1, 3, 5, 6
    };

    generate_bcsr(arr, val, col_idx, row_ptr, m, n, m_block_sz, n_block_sz);
    // compare
    for ( int i = 0 ; i < 40 ; i ++ ) {
        if (expected_val[i] != val[i]) {
            
        }
    }

    for ( int i = 0 ; i < 6 ; i ++ ) {
        if (expected_col_idx[i] != col_idx[i]) {
    
        }
    }

    for ( int i = 0 ; i < 5 ; i ++ ) {
        if (expected_row_ptr[i] != row_ptr[i]) {
    
        }
    }

    free(val);
    free(col_idx);
    free(row_ptr);
}

int main() {
    test_cal_block();
    test_generate_bcsr();
}
