#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

void cal_block(float* mat, int* is_block_present, int* nnz_block_num, size_t m, size_t n, size_t m_block_sz, size_t n_block_sz);
void generate_bcsr(float* mat, float* val, int* col_idx, int* row_idx, size_t m, size_t n, size_t m_block_sz, size_t n_block_sz);

#endif