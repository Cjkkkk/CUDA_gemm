#ifndef CSR_H
#define CSR_H

#include <stdlib.h>

class csr {
public:
    float* val;
    int* col_idx;
    int* row_ptr;
    int m, n, nnz_num;

    csr(int m, int n): m(m), n(n) {
        nnz_num = 0;
        val = NULL;
        col_idx = NULL;
        row_ptr = NULL;
    }
    
    ~csr() {
        if (val != NULL) free(val);
        if (col_idx != NULL) free(col_idx);
        if (row_ptr != NULL) free(row_ptr);
    }

    void print() {
        printf("row_ptr: \n");
        for ( int i = 0 ; i < m + 1 ; i ++ ) {
            printf("%d ", row_ptr[i]);
        }
        printf("\n");
        printf("col_idx: \n");
        for ( int i = 0 ; i < nnz_num ; i ++ ) {
            printf("%d ", col_idx[i]);
        }
    }
};

#endif