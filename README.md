## introduction
A simple high performance CUDA GEMM, Block Sparse GEMM and Non-uniform Quantized GEMM implementation.
```
C = alpha * A * B + beta * C
```
## algorithm
**located in src/cuda/**

* MatrixMulCUDA
    * one element of C is assigned one thread
    * global memory coalesce of B
* MatrixMulCUDA1
    * texture load
* MatrixMulCUDA2
    * one 4 * 4 grid of C is assigned one thread
* MatrixMulCUDA3
    * vectorized A B load
* MatrixMulCUDA4
    * vectorized C store
* MatrixMulCUDA5
    * block sparse version
* MatrixMulCUDA6
    * vectorized A B load coalesce
* MatrixMulCUDA7
    * warp shuffle to enable C store coalesce
* MatrixMulCUDAQuantize8bit
    * 8 bit non-uniform quantized matmul

## experiments
**located in benchmark/**
* benchmark_dense
    * Compare My Gemm with Cublas
* benchmark_sparse
    * Compare My block sparse Gemm with Cusparse
* benchmark_quantization_8bit
    * Compare My Gemm with Cublas
* benchmark_quantization
    * Compare My Gemm with My quantized non-uniform 8 bit Gemm

## TODO
* (MatrixMulCUDA7) write back to C matrix, warp shuffle to enable global memory coalesce
* (MatrixMulCUDA8) double buffering

## run
```
make benchmark_[experiment name]
bash scripts/benchmark_[experiment name].sh
```

## Note
* sparsity约为1%的时候, cusparse的性能可以超越cublas
* 合理分配寄存器 尽可能让参数在编译器确定节省计算资源和寄存器数目
