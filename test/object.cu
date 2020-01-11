#include <stdio.h>

class A {
public:
    int num;
    __host__ __device__ A(int n) {
        num = n;
    }

    __host__ __device__ int get_num() {
        return num;
    }
};

__global__
void kernel(A* a) {
    printf("%d\n", a->get_num());
}

int main() {
    A* a = new A(3);
    printf("%d\n", a->get_num());
    A* d_a;
    cudaMalloc(&d_a, sizeof(A));
    cudaMemcpy(d_a, a, sizeof(A), cudaMemcpyHostToDevice);
    kernel<<< 1, 1 >>>(d_a);
    cudaDeviceSynchronize();
    return 0;
}