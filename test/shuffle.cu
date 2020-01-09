#include <stdio.h>
#include <cuda_runtime.h>
__global__ void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value;
    if (laneId == 0)        // Note unused variable for
        value = arg;        // all threads except lane 0
    else {
        value = arg + 3;
    }
    value = __shfl_sync(0xffffffff, value, 4);   // Synchronize all threads in warp, and get "value" from lane 0
    // if (value != arg)
    //     printf("Thread %d failed.\n", threadIdx.x);
    // else {
    printf("%d %d\n", threadIdx.x, value);
    //}
}

__global__ void reduce() {
    int laneId = threadIdx.x & 0x1f;
    int value = laneId;
    for ( int offset = 32 / 2 ; offset > 0 ; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);   // Synchronize all threads in warp, and get "value" from lane 0
    }
    printf("%d %d\n", threadIdx.x, value);
}

__global__ void reduce1() {
    int laneId = threadIdx.x & 0x1f;
    int value = laneId;
    value = __shfl_down_sync(0xffffffff, value, 16);
    printf("%d %d\n", threadIdx.x, value);
}

__global__ void reduce2() {
    int laneId = threadIdx.x & 0x1f;
    int value = laneId;
    for ( int offset = 32 / 2 ; offset > 0 ; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);   // Synchronize all threads in warp, and get "value" from lane 0
    }
    printf("%d %d\n", threadIdx.x, value);
}
int main() {
    reduce2<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}