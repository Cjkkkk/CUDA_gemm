#include <stdio.h>
__global__ void bitonic_sort(float* mat, int N, int* idx);

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