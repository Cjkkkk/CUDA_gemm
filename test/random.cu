#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#define MIN 2
#define MAX 7
#define ITER 10000000

__global__ void setup_kernel(curandState *state){

    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ void generate_kernel(curandState *my_curandstate, const unsigned int n, const unsigned *max_rand_int, const unsigned *min_rand_int,  unsigned int *result){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    int count = 0;
    while (count < n){
        float myrandf = curand_uniform(my_curandstate+idx);
        myrandf *= (max_rand_int[idx] - min_rand_int[idx]+0.999999);
        myrandf += min_rand_int[idx];
        int myrand = (int)truncf(myrandf);

        assert(myrand <= max_rand_int[idx]);
        assert(myrand >= min_rand_int[idx]);
        result[myrand-min_rand_int[idx]]++;
        count++;
    }
}

int main(){

    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    unsigned *d_result, *h_result;
    unsigned *d_max_rand_int, *h_max_rand_int, *d_min_rand_int, *h_min_rand_int;
    cudaMalloc(&d_result, (MAX-MIN+1) * sizeof(unsigned));
    h_result = (unsigned *)malloc((MAX-MIN+1)*sizeof(unsigned));
    cudaMalloc(&d_max_rand_int, sizeof(unsigned));
    h_max_rand_int = (unsigned *)malloc(sizeof(unsigned));
    cudaMalloc(&d_min_rand_int, sizeof(unsigned));
    h_min_rand_int = (unsigned *)malloc(sizeof(unsigned));
    cudaMemset(d_result, 0, (MAX-MIN+1)*sizeof(unsigned));
    setup_kernel<<<1,1>>>(d_state);

    *h_max_rand_int = MAX;
    *h_min_rand_int = MIN;
    cudaMemcpy(d_max_rand_int, h_max_rand_int, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_rand_int, h_min_rand_int, sizeof(unsigned), cudaMemcpyHostToDevice);
    generate_kernel<<<1,1>>>(d_state, ITER, d_max_rand_int, d_min_rand_int, d_result);
    cudaMemcpy(h_result, d_result, (MAX-MIN+1) * sizeof(unsigned), cudaMemcpyDeviceToHost);
    printf("Bin:    Count: \n");
    for (int i = MIN; i <= MAX; i++)
        printf("%d    %d\n", i, h_result[i-MIN]);

    return 0;
}