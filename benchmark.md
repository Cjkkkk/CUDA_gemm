## experiments
**located in benchmark/**

My gemm : MatrixMulCUDA6
My sparse block gemm : MatrixMulCUDA5

----- benchmark size: 256 -----
```
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
```
----- benchmark size: 512 -----
```
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
```
----- benchmark size: 1024 -----
```
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
```
----- benchmark size: 1536 -----
```
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
```
----- benchmark size: 2048 -----
```
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
```
----- benchmark sparsity: 30% -----
```
My gemm Performance= 6930.43 GFlop/s, Time= 2.479 msec, Size= 17179869184 Ops,
CuBlas Performance= 7656.10 GFlop/s, Time= 2.244 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 0.905217
My sparse block gemm Performance= 18988.82 GFlop/s, Time= 0.905 msec, Size= 17179869184 Ops,CuSparse Performance= 195.89 GFlop/s, Time= 87.703 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 96.938215

my block sparse: 0.905 msec
cublas: 2.244 msec
my gemm: 2.479 msec
cusparse(csr): 87.703 msec
```
----- benchmark sparsity: 10% -----
```
My gemm Performance= 6931.65 GFlop/s, Time= 2.478 msec, Size= 17179869184 OpsCuBlas Performance= 7655.65 GFlop/s, Time= 2.244 msec, Size= 17179869184 Ops,
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
```
----- benchmark sparsity: 5% -----
```
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
```
----- benchmark sparsity: 1% -----
```
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
```

My gemm : MatrixMulCUDA6
My non-uniform quantized 8 bit gemm : MatrixMulCUDAQuantize8bit
```
----- benchmark size: 256 -----
My gemm Performance= 2066.61 GFlop/s, Time= 0.016 msec, Size= 33554432 Ops,
My gemm1 Performance= 1777.07 GFlop/s, Time= 0.019 msec, Size= 33554432 Ops,
Result= PASS
ratio= 1.162932
----- benchmark size: 512 -----
My gemm Performance= 4851.65 GFlop/s, Time= 0.055 msec, Size= 268435456 Ops,
My gemm1 Performance= 4595.69 GFlop/s, Time= 0.058 msec, Size= 268435456 Ops,
Error! Matrix[00000]=0.00000000, ref=256.00000000 error term is > 1.000000E-06
Result= FAIL
ratio= 1.055696
----- benchmark size: 1024 -----
My gemm Performance= 4806.85 GFlop/s, Time= 0.447 msec, Size= 2147483648 Ops,
My gemm1 Performance= 5792.35 GFlop/s, Time= 0.371 msec, Size= 2147483648 Ops,
Error! Matrix[00000]=0.00000000, ref=512.00000000 error term is > 1.000000E-06
Result= FAIL
ratio= 0.829862
----- benchmark size: 1536 -----
My gemm Performance= 6305.01 GFlop/s, Time= 1.150 msec, Size= 7247757312 Ops,
My gemm1 Performance= 6878.28 GFlop/s, Time= 1.054 msec, Size= 7247757312 Ops,
Error! Matrix[00000]=0.00000000, ref=768.00000000 error term is > 1.000000E-06
Result= FAIL
ratio= 0.916654
----- benchmark size: 2048 -----
My gemm Performance= 6446.45 GFlop/s, Time= 2.665 msec, Size= 17179869184 Ops,
My gemm1 Performance= 7064.28 GFlop/s, Time= 2.432 msec, Size= 17179869184 Ops,
Error! Matrix[00000]=0.00000000, ref=1024.00000000 error term is > 1.000000E-06
Result= FAIL
ratio= 0.912542
----- benchmark size: 2560 -----
My gemm Performance= 6569.60 GFlop/s, Time= 5.108 msec, Size= 33554432000 Ops,
My gemm1 Performance= 7133.01 GFlop/s, Time= 4.704 msec, Size= 33554432000 Ops,
Error! Matrix[00000]=0.00000000, ref=1280.00000000 error term is > 1.000000E-06
Result= FAIL
ratio= 0.921014
----- benchmark size: 3072 -----
My gemm Performance= 6681.76 GFlop/s, Time= 8.678 msec, Size= 57982058496 Ops,
My gemm1 Performance= 7213.02 GFlop/s, Time= 8.039 msec, Size= 57982058496 Ops,
Error! Matrix[00000]=0.00000000, ref=1536.00000000 error term is > 1.000000E-06
Result= FAIL
ratio= 0.926348
----- benchmark size: 3584 -----
My gemm Performance= 6788.71 GFlop/s, Time= 13.563 msec, Size= 92073361408 Ops,
My gemm1 Performance= 7202.32 GFlop/s, Time= 12.784 msec, Size= 92073361408 Ops,
Error! Matrix[00000]=0.00000000, ref=1792.00000000 error term is > 1.000000E-06
Result= FAIL
ratio= 0.942572
----- benchmark size: 4096 -----
My gemm Performance= 6722.88 GFlop/s, Time= 20.443 msec, Size= 137438953472 Ops,
My gemm1 Performance= 7246.33 GFlop/s, Time= 18.967 msec, Size= 137438953472 Ops,
Error! Matrix[00000]=0.00000000, ref=2048.00000000 error term is > 1.000000E-06
Result= FAIL
ratio= 0.927763
rm builds/benchmark_quantization.o
```