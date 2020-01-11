#include <stdio.h>

__device__ void swap(int* arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
    return;
}

__global__ void bitonic_sort(float* mat, int N, int* idx) {
    __shared__ int nnz[8]; 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float* start = mat + tid * N;

    // cal nnz
    nnz[tid] = 0;
    for ( int i = 0 ; i < N ; i ++ ) {
        if ( start[i] != 0 ) nnz[tid] += 1;
    }

    __syncthreads();
    for (unsigned int k = 2 ; k <= 8 ; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2){ 
            unsigned int ixj = tid ^ j; // determine which element to compare, every j changes if ixj is tid + j or tid - j
            if (ixj > tid){
                if ((tid & k) == 0) { // determine arrow direction, every k changes if tid & k is = 1 or = 0
                    if (nnz[tid] > nnz[ixj]) {
                        // swap both nnz and tid
                        swap(nnz, tid, ixj);
                        swap(idx, tid, ixj);
                    } 
                }
                else {
                    if (nnz[tid] < nnz[ixj]) {
                        swap(nnz, tid, ixj); 
                        swap(idx, tid, ixj);
                    }
                }
            } 
            __syncthreads(); 
        } 
    }
}

int main() {
    float arr[64] = {
        0, 1, 1, 1, 1, 0, 1, 0, // 5
        1, 1, 1, 1, 1, 0, 1, 0, // 6
        0, 1, 1, 1, 0, 0, 1, 0, // 4
        0, 1, 0, 0, 1, 0, 1, 0, // 3
        0, 1, 1, 1, 1, 1, 1, 1, // 7
        0, 0, 0, 0, 0, 0, 0, 0, // 0
        0, 1, 0, 0, 0, 0, 1, 0, // 2
        0, 0, 0, 0, 0, 0, 1, 0, // 1
    };
    // 5 7 6 3 2 0 1 4
    int res[8] = {
        0, 1, 2, 3, 4, 5, 6, 7
    };
    float* d_arr;
    int* d_res;
    
    cudaMalloc(&d_arr, sizeof(float) * 64); 
    cudaMalloc(&d_res, sizeof(int) * 8);
    cudaMemcpy(d_arr, arr, sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, sizeof(int) * 8, cudaMemcpyHostToDevice);
    bitonic_sort<<<1, 8>>>(d_arr, 8, d_res);
    cudaDeviceSynchronize();
    cudaMemcpy(res, d_res, sizeof(int) * 8, cudaMemcpyDeviceToHost);
    for ( int i = 0 ; i < 8 ; i ++ ) {
        printf("%d ", res[i]);
    }
    printf("\n");
}