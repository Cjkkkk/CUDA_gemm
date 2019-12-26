#include <stdlib.h>
#include <stdio.h>

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

texture<float, 2, cudaReadModeElementType> tex;

__global__ void kernel (int m, int n, float* a) 
{
    float val;
    int i = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            val = tex2D (tex, col, row);
            printf ("%f  %f\n", val, a[i++]);
        }
        printf ("\n");
    }
}

int main (void)
{
    int m = 4; // height = #rows
    int n = 3; // width  = #columns
    size_t pitch, tex_ofs;
    float arr[4][3]= {{10.1, 11, 12},
                    {20, 21, 22},
                    {30, 31, 32},
                    {40, 41, 42}};
    float *arr_d;

    CUDA_SAFE_CALL(cudaMallocPitch((void**)&arr_d,&pitch,n*sizeof(*arr_d),m));
    CUDA_SAFE_CALL(cudaMemcpy2D(arr_d, pitch, arr, n*sizeof(arr[0][0]), n*sizeof(arr[0][0]), m, cudaMemcpyHostToDevice));
    tex.normalized = false;
    CUDA_SAFE_CALL (cudaBindTexture2D (&tex_ofs, &tex, arr_d, &tex.channelDesc,
                                       n, m, pitch));
    if (tex_ofs !=0) {
        printf ("tex_ofs = %zu\n", tex_ofs);
        return EXIT_FAILURE;
    }
    printf ("reading texture:\n");
    kernel<<<1,1>>>(m, n, arr_d);
    CHECK_LAUNCH_ERROR();
    CUDA_SAFE_CALL (cudaDeviceSynchronize());
    return EXIT_SUCCESS;
}