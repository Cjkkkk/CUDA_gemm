#ifndef BCSR_H
#define BCSR_H

#include <stdlib.h>

class bcsr {
public:
    float* val;
    int* is_block_present;
    int* col_idx;
    int* row_ptr;
    int m, n, m_block_sz, n_block_sz, m_block, n_block, nnz_block_num;

    bcsr(int m, int n, int m_block_sz, int n_block_sz): m(m), n(n), m_block_sz(m_block_sz), n_block_sz(n_block_sz) {
        m_block = m / m_block_sz;
        n_block = n / n_block_sz;
        nnz_block_num = 0;
        is_block_present = (int*) malloc(sizeof(int) * m_block * n_block);
        val = NULL;
        col_idx = NULL;
        row_ptr = NULL;
    }
    
    ~bcsr() {
        free(is_block_present);
        if (val != NULL) free(val);
        if (col_idx != NULL) free(col_idx);
        if (row_ptr != NULL) free(row_ptr);
    }

    void print() {
        printf("is block present: \n");
        for ( int i = 0 ; i < m_block ; i ++ ) {
            for ( int j = 0 ; j < n_block ; j ++ ) {
                printf("%d ", is_block_present[i * n_block + j]);
            }
            printf("\n");
        }

        printf("row_ptr: \n");
        for ( int i = 0 ; i < m_block + 1 ; i ++ ) {
            printf("%d ", row_ptr[i]);
        }
        printf("\n");
        printf("col_idx: \n");
        for ( int i = 0 ; i < nnz_block_num ; i ++ ) {
            printf("%d ", col_idx[i]);
        }
    }
};
#endif