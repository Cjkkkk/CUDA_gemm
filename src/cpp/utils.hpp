#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "bcsr.hpp"
#include "csr.hpp"

void cal_block(bcsr*, float* );
void generate_bcsr(bcsr*, float* );

void cal_nnz(csr*, float* );
void generate_csr(csr*, float* );
#endif