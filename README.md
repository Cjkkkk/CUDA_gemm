## introduction
A CUDA GEMM implementation

## algorithm
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

## experiments
My gemm : MatrixMulCUDA6
My sparse block gemm : MatrixMulCUDA5

----- benchmark size: 256 -----
My gemm Performance= 1998.01 GFlop/s, Time= 0.017 msec, Size= 33554432 Ops,
CuBlas Performance= 2283.19 GFlop/s, Time= 0.015 msec, Size= 33554432 Ops,
Result= PASS
ratio= 0.875098
My sparse block gemm Performance= 1469.23 GFlop/s, Time= 0.023 msec, Size= 33554432 Ops,
CuSparse Performance= 75.15 GFlop/s, Time= 0.446 msec, Size= 33554432 Ops,
Result= PASS
ratio= 19.550183

cublas: 0.015 msec
my gemm: 0.017 msec
my block sparse: 0.023 msec
cusparse(csr): 0.446 msec

----- benchmark size: 512 -----
My gemm Performance= 4711.56 GFlop/s, Time= 0.057 msec, Size= 268435456 Ops,
CuBlas Performance= 5132.69 GFlop/s, Time= 0.052 msec, Size= 268435456 Ops,
Result= PASS
ratio= 0.917952
My sparse block gemm Performance= 4235.39 GFlop/s, Time= 0.063 msec, Size= 268435456 Ops,
CuSparse Performance= 106.74 GFlop/s, Time= 2.515 msec, Size= 268435456 Ops,
Result= PASS
ratio= 39.678296

cublas: 0.052 msec
my gemm: 0.057 msec
my block sparse: 0.063 msec
cusparse(csr): 2.515 msec

----- benchmark size: 1024 -----
My gemm Performance= 6357.88 GFlop/s, Time= 0.338 msec, Size= 2147483648 Ops,
CuBlas Performance= 5957.73 GFlop/s, Time= 0.360 msec, Size= 2147483648 Ops,
Result= PASS
ratio= 1.067165
My sparse block gemm Performance= 6046.08 GFlop/s, Time= 0.355 msec, Size= 2147483648 Ops,
CuSparse Performance= 117.01 GFlop/s, Time= 18.353 msec, Size= 2147483648 Ops,
Result= PASS
ratio= 51.671461

my gemm: 0.338 msec
my block sparse: 0.355 msec
cublas: 0.360 msec
cusparse(csr): 18.353 msec

----- benchmark size: 1536 -----
My gemm Performance= 6781.69 GFlop/s, Time= 1.069 msec, Size= 7247757312 Ops,
CuBlas Performance= 6825.36 GFlop/s, Time= 1.062 msec, Size= 7247757312 Ops,
Result= PASS
ratio= 0.993602
My sparse block gemm Performance= 6387.92 GFlop/s, Time= 1.135 msec, Size= 7247757312 Ops,
CuSparse Performance= 56.87 GFlop/s, Time= 127.438 msec, Size= 7247757312 Ops,
Result= PASS
ratio= 112.319292

cublas: 1.062 msec
my gemm: 1.069 msec
my block sparse: 1.135 msec
cusparse(csr): 127.438 msec

----- benchmark size: 2048 -----
My gemm Performance= 6933.26 GFlop/s, Time= 2.478 msec, Size= 17179869184 Ops,
CuBlas Performance= 7585.98 GFlop/s, Time= 2.265 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 0.913956
My sparse block gemm Performance= 6571.80 GFlop/s, Time= 2.614 msec, Size= 17179869184 Ops,
CuSparse Performance= 61.28 GFlop/s, Time= 280.363 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 107.246920

cublas: 2.265 msec
my gemm: 2.478 msec
my block sparse: 2.614 msec
cusparse(csr): 280.363 msec

----- benchmark sparsity: 30% -----
My gemm Performance= 6930.43 GFlop/s, Time= 2.479 msec, Size= 17179869184 Ops,
CuBlas Performance= 7656.10 GFlop/s, Time= 2.244 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 0.905217
My sparse block gemm Performance= 18988.82 GFlop/s, Time= 0.905 msec, Size= 17179869184 Ops,
CuSparse Performance= 195.89 GFlop/s, Time= 87.703 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 96.938215

my block sparse: 0.905 msec
cublas: 2.244 msec
my gemm: 2.479 msec
cusparse(csr): 87.703 msec

----- benchmark sparsity: 10% -----
My gemm Performance= 6931.65 GFlop/s, Time= 2.478 msec, Size= 17179869184 Ops,
CuBlas Performance= 7655.65 GFlop/s, Time= 2.244 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 0.905430
My sparse block gemm Performance= 58011.25 GFlop/s, Time= 0.296 msec, Size= 17179869184 Ops,
CuSparse Performance= 869.95 GFlop/s, Time= 19.748 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 66.683149

my block sparse: 0.296 msec
cublas: 2.244 msec
my gemm: 2.478 msec
cusparse(csr): 19.748 msec

----- benchmark sparsity: 5% -----
My gemm Performance= 6928.78 GFlop/s, Time= 2.479 msec, Size= 17179869184 Ops,
CuBlas Performance= 7655.26 GFlop/s, Time= 2.244 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 0.905100
My sparse block gemm Performance= 106346.44 GFlop/s, Time= 0.162 msec, Size= 17179869184 Ops,
CuSparse Performance= 1726.94 GFlop/s, Time= 9.948 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 61.580704

my block sparse: 0.162 msec
cublas: 2.244 msec
my gemm: 2.479 msec
cusparse(csr): 9.948 msec

----- benchmark sparsity: 1% -----
My gemm Performance= 6934.62 GFlop/s, Time= 2.477 msec, Size= 17179869184 Ops,
CuBlas Performance= 7655.82 GFlop/s, Time= 2.244 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 0.905797
My sparse block gemm Performance= 289402.68 GFlop/s, Time= 0.059 msec, Size= 17179869184 Ops,
CuSparse Performance= 14005.05 GFlop/s, Time= 1.227 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 20.664161

my block sparse: 0.059 msec
cusparse(csr): 1.227 msec
cublas: 2.244 msec
my gemm: 2.477 msec

## TODO
* MatrixMulCUDA7
    * write back to C matrix, warp shuffle to enable global memory coalesce

## Note
* sparsity约为1%的时候, cusparse的性能可以超越cublas