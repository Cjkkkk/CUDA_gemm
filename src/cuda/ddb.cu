void group_block_spmm_kernel_32x8x64(const int M,
                                     const int K,
                                     const int N,
                                     const int nnz,
                                     const float* __restrict__ val_src_a,
                                     const int* __restrict__ row_src_a,
                                     const int* __restrict__ col_src_a,
                                     const float* __restrict__ val_src_b,
                                     float* val_dst)
{
    //const int A_ld = M / 4;
    const int B_ld = K / 4;
    const int C_ld = M;

    float4* A_pointer = (float4*)val_src_a;
    float4* B_pointer = (float4*)val_src_b;
    float* C_pointer = val_dst;

    // register file
    float rC[16] = { 0 };
    float4 rA[2];
    float4 rB[2];

    // double buffering
    __shared__ float lA[2 * 256];
    __shared__ float lB[2 * 512];

    int lAstart = 0;
    int lBstart = 0;

    int gidx = blockIdx.x;
    int gidy = blockIdx.y;
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int idt = 8 * idy + idx;
    int idxT = idt & 15; // idt%16
    int idyT = idt >> 4; // idt/16

    int row_st = row_src_a[gidx];
    int row_ed = row_src_a[gidx + 1];


    int rxg = row_src_a[gidx];
    int ryg = col_src_a[row_src_a[gidx]];
    A_pointer = (float4*)val_src_a + (rxg*SPMM_GROUP_M + idyT*SPMM_GROUP_M) / 4 + idxT;
    B_pointer = (float4*)val_src_b + (gidy * 16 + idxT) + idyT*B_ld + ryg*B_ld;


    float* lAstore = lA + idyT * 64 + idxT * 4;
    float* lBstore = lB + idyT * 64 + idxT * 4;

    if (idyT < 4)
        reinterpret_cast<float4*>(lAstore + lAstart + 0)[0] = A_pointer[0];
    reinterpret_cast<float4*>(lBstore + lBstart + 0)[0] = B_pointer[0];


    for (int block_k = row_st + 1; block_k < row_ed; block_k++)
    {
        __syncthreads();
        float* lAread = lA + lAstart + 4 * idx;
        float* lBread = lB + lBstart + 4 * idy;


        #pragma unroll
        for (unsigned int k = 0; k < 8; k += 1)
        {

            //Fetch A to registers
            rA[0] = reinterpret_cast<float4*>(lAread + k * 32)[0];
            //Fetch B to registers
            rB[0] = reinterpret_cast<float4*>(lBread + k * 64)[0];

            //FMA computations
            rC[0] = fma(rA[0].x, rB[0].x, rC[0]);
            rC[4] = fma(rA[0].y, rB[0].x, rC[4]);
            rC[8] = fma(rA[0].z, rB[0].x, rC[8]);
            rC[12] = fma(rA[0].w, rB[0].x, rC[12]);
            rC[1] = fma(rA[0].x, rB[0].y, rC[1]);
            rC[5] = fma(rA[0].y, rB[0].y, rC[5]);
            rC[9] = fma(rA[0].z, rB[0].y, rC[9]);
            rC[13] = fma(rA[0].w, rB[0].y, rC[13]);
            rC[2] = fma(rA[0].x, rB[0].z, rC[2]);
            rC[6] = fma(rA[0].y, rB[0].z, rC[6]);
            rC[10] = fma(rA[0].z, rB[0].z, rC[10]);
            rC[14] = fma(rA[0].w, rB[0].z, rC[14]);
            rC[3] = fma(rA[0].x, rB[0].w, rC[3]);
            rC[7] = fma(rA[0].y, rB[0].w, rC[7]);
            rC[11] = fma(rA[0].z, rB[0].w, rC[11]);
            rC[15] = fma(rA[0].w, rB[0].w, rC[15]);
        }

        lAstart ^= 256;
        lBstart ^= 512;
        rxg = block_k;
        ryg = col_src_a[block_k];

        A_pointer = (float4*)val_src_a + (rxg*SPMM_GROUP_M + idyT*SPMM_GROUP_M + idxT) / 4;
        B_pointer = (float4*)val_src_b + (gidy * 16 + idxT) + idyT*B_ld + ryg*B_ld;

        //Fetch A to shared memory
        if (idyT < 4)
            reinterpret_cast<float4*>(lAstore + lAstart + 0)[0] = A_pointer[0];
        //Fetch B to shared memory
        reinterpret_cast<float4*>(lBstore + lBstart + 0)[0] = B_pointer[0];
        //reinterpret_cast<float4*>(lBstore + lBstart + 64)[0] = B_pointer[0 * B_ld + 16];

    }

    {
        __syncthreads();
        float* lAread = lA + lAstart + 4 * idx;
        float* lBread = lB + lBstart + 4 * idy;

        #pragma unroll
        for (unsigned int k = 0; k < 8; k += 2)
        {
            //Fetch A to registers
            rA[0] = reinterpret_cast<float4*>(lAread + k * 32 + 0 * 32 + 0 * 32)[0];

            //Fetch B to registers
            rB[0] = reinterpret_cast<float4*>(lBread + k * 64 + 0 * 32 + 0 * 64)[0];

            //FMA computations
            rC[0] = fma(rA[0].x, rB[0].x, rC[0]);
            rC[4] = fma(rA[0].y, rB[0].x, rC[4]);
            rC[8] = fma(rA[0].z, rB[0].x, rC[8]);
            rC[12] = fma(rA[0].w, rB[0].x, rC[12]);
            rC[1] = fma(rA[0].x, rB[0].y, rC[1]);
            rC[5] = fma(rA[0].y, rB[0].y, rC[5]);
            rC[9] = fma(rA[0].z, rB[0].y, rC[9]);
            rC[13] = fma(rA[0].w, rB[0].y, rC[13]);
            rC[2] = fma(rA[0].x, rB[0].z, rC[2]);
            rC[6] = fma(rA[0].y, rB[0].z, rC[6]);
            rC[10] = fma(rA[0].z, rB[0].z, rC[10]);
            rC[14] = fma(rA[0].w, rB[0].z, rC[14]);
            rC[3] = fma(rA[0].x, rB[0].w, rC[3]);
            rC[7] = fma(rA[0].y, rB[0].w, rC[7]);
            rC[11] = fma(rA[0].z, rB[0].w, rC[11]);
            rC[15] = fma(rA[0].w, rB[0].w, rC[15]);

            //Fetch A to registers
            rA[1] = reinterpret_cast<float4*>(lAread + k * 32 + 0 * 32 + 1 * 32)[0];

            //Fetch B to registers

            rB[1] = reinterpret_cast<float4*>(lBread + k * 64 + 0 * 32 + 1 * 64)[0];

            rC[0] = fma(rA[1].x, rB[1].x, rC[0]);
            rC[4] = fma(rA[1].y, rB[1].x, rC[4]);
            rC[8] = fma(rA[1].z, rB[1].x, rC[8]);
            rC[12] = fma(rA[1].w, rB[1].x, rC[12]);
            rC[1] = fma(rA[1].x, rB[1].y, rC[1]);
            rC[5] = fma(rA[1].y, rB[1].y, rC[5]);
            rC[9] = fma(rA[1].z, rB[1].y, rC[9]);
            rC[13] = fma(rA[1].w, rB[1].y, rC[13]);
            rC[2] = fma(rA[1].x, rB[1].z, rC[2]);
            rC[6] = fma(rA[1].y, rB[1].z, rC[6]);
            rC[10] = fma(rA[1].z, rB[1].z, rC[10]);
            rC[14] = fma(rA[1].w, rB[1].z, rC[14]);
            rC[3] = fma(rA[1].x, rB[1].w, rC[3]);
            rC[7] = fma(rA[1].y, rB[1].w, rC[7]);
            rC[11] = fma(rA[1].z, rB[1].w, rC[11]);
            rC[15] = fma(rA[1].w, rB[1].w, rC[15]);
        }

    }


    C_pointer += (gidx * 32 + idx * 4) + (gidy * 64 + idy * 4)*C_ld;
    C_pointer[0 + 0 * C_ld] = rC[0];
    C_pointer[0 + 1 * C_ld] = rC[1];
    C_pointer[0 + 2 * C_ld] = rC[2];
    C_pointer[0 + 3 * C_ld] = rC[3];
    C_pointer[1 + 0 * C_ld] = rC[4];
    C_pointer[1 + 1 * C_ld] = rC[5];
    C_pointer[1 + 2 * C_ld] = rC[6];
    C_pointer[1 + 3 * C_ld] = rC[7];
    C_pointer[2 + 0 * C_ld] = rC[8];
    C_pointer[2 + 1 * C_ld] = rC[9];
    C_pointer[2 + 2 * C_ld] = rC[10];
    C_pointer[2 + 3 * C_ld] = rC[11];
    C_pointer[3 + 0 * C_ld] = rC[12];
    C_pointer[3 + 1 * C_ld] = rC[13];
    C_pointer[3 + 2 * C_ld] = rC[14];
    C_pointer[3 + 3 * C_ld] = rC[15];
}