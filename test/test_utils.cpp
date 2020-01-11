#include "utils.hpp"
#include "bcsr.hpp"
#include "csr.hpp"
#include <stdio.h>
#include <stdlib.h>

bool test_cal_block(bcsr* mat, float* data) {
    bool res = true;

    int expected_is_block_present[16] = {
        1, 0, 1, 1,
        0, 1, 0, 1,
        1, 0, 0, 1,
        1, 1, 1, 0
    };

    cal_block(mat, data);
    // compare
    for ( int i = 0 ; i < 4 ; i ++ ) {
        for ( int j = 0 ; j < 4 ; j ++ ) {
            if (expected_is_block_present[i * 4 + j] != mat->is_block_present[i * 4 + j]) {
                printf("error in block position: (%d, %d) with val: (%d, %d)\n", i, j, expected_is_block_present[i * 4 + j], mat->is_block_present[i * 4 + j]);
                res = false;
            }
        }
    }
    return res;
}

bool test_generate_bcsr(bcsr* mat, float* data) {
    bool res = true;

    cal_block(mat, data);
    mat->row_ptr = (int*)malloc(sizeof(int) * ( mat->m_block + 1 ));
    mat->col_idx = (int*)malloc(sizeof(int) * mat->nnz_block_num );
    mat->val = (float*)malloc(sizeof(float) * mat->nnz_block_num * mat->m_block_sz * mat->n_block_sz);
    
    float expected_val[40] = {
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

    int expected_col_idx[10] = {
        0, 2, 3,
        1, 3,
        0, 3,
        0, 1, 2,
    };

    int expected_row_ptr[5] = {
        0, 3, 5, 7, 10
    };

    generate_bcsr(mat, data);

    // compare
    for ( int i = 0 ; i < 40 ; i ++ ) {
        if (expected_val[i] != mat->val[i]) {
            printf("val error: %f %f %d\n", mat->val[i], expected_val[i], i);
            res = false;
        }
    }

    for ( int i = 0 ; i < 10 ; i ++ ) {
        if (expected_col_idx[i] != mat->col_idx[i]) {
            printf("col_idx error: %d %d\n", mat->col_idx[i], i);
            res = false;
        }
    }

    for ( int i = 0 ; i < 5 ; i ++ ) {
        if (expected_row_ptr[i] != mat->row_ptr[i]) {
            printf("row_ptr error: %d\n", mat->row_ptr[i]);
            res = false;
        }
    }

    return res;
}


bool test_generate_csr(csr* mat, float* data) {
    bool res = true;

    cal_nnz(mat, data);
    mat->row_ptr = (int*)malloc(sizeof(int) * ( mat->m + 1 ));
    mat->col_idx = (int*)malloc(sizeof(int) * mat->nnz_num );
    mat->val = (float*)malloc(sizeof(float) * mat->nnz_num );

    float expected_val[26] = {
        1, 2, 3, 4, 5,
        6, 7,
        3, 8, 1,
        1, 1,
        1, 6, 2,
        3, 1,
        1, 4, 5, 4, 1,
        1, 4, 1, 1, 
    };

    
    int expected_col_idx[26] = {
        0, 1, 4, 5, 7,
        1, 4,
        2, 6, 7,
        2, 6,
        0, 1, 6,
        6, 7,
        0, 1, 3, 4, 5,
        0, 1, 3, 5
    };

    int expected_row_ptr[9] = {
        0, 5, 7, 10, 12, 15, 17, 22, 26
    };

    generate_csr(mat, data);

    // compare
    for ( int i = 0 ; i < 26 ; i ++ ) {
        if (expected_val[i] != mat->val[i]) {
            printf("val error: %f %f %d\n", mat->val[i], expected_val[i], i);
            res = false;
        }
    }

    for ( int i = 0 ; i < 26 ; i ++ ) {
        if (expected_col_idx[i] != mat->col_idx[i]) {
            printf("col_idx error: %d %d\n", mat->col_idx[i], i);
            res = false;
        }
    }

    for ( int i = 0 ; i < 9 ; i ++ ) {
        if (expected_row_ptr[i] != mat->row_ptr[i]) {
            printf("row_ptr error: %d\n", mat->row_ptr[i]);
            res = false;
        }
    }

    return res;
}

int main() {
    printf("----- begin 3 tests -----\n");
    float data[64] = {
        1, 2, 0, 0, 3, 4, 0, 5,
        0, 6, 0, 0, 7, 0, 0, 0,
        0, 0, 3, 0, 0, 0, 8, 1,
        0, 0, 1, 0, 0, 0, 1, 0,
        1, 6, 0, 0, 0, 0, 2, 0,
        0, 0, 0, 0, 0, 0, 3, 1,
        1, 4, 0, 5, 4, 1, 0, 0,
        1, 4, 0, 1, 0, 1, 0, 0,
    };

    {
        bcsr mat{8, 8, 2, 2};
        test_cal_block(&mat, data);
    }
    {
        bcsr mat{8, 8, 2, 2};
        test_generate_bcsr(&mat, data);
    }

    {
        csr mat{8, 8};
        test_generate_csr(&mat, data);
    }
    printf("----- pass 3 tests ----- \n");
}
