#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "bcsr.hpp"
#include "csr.hpp"

void cal_block(bcsr*, float* );
void generate_bcsr(bcsr*, float* );

void cal_nnz(csr*, float* );
void generate_csr(csr*, float* );


void genRandomMatrix(float* A, int M, int N);
void FillMatrix(float* A, float num, int M, int N);
void genFixedMatrix(float* A, int M, int N);
void genSparseMatrix(float* A, int M, int N, int sparsity);
void copyMatrix(float* des, float* src, int M, int N);
#endif