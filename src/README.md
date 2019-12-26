## introduction
A CUDA GEMM implementation

## experiments
./builds/main 256 256 256
My gemm Performance= 1090.08 GFlop/s, Time= 0.031 msec, Size= 33554432 Ops,
CuBlas Performance= 2382.36 GFlop/s, Time= 0.014 msec, Size= 33554432 Ops,
Result= PASS
ratio= 0.457561

./builds/main 512 512 512
My gemm Performance= 2980.63 GFlop/s, Time= 0.090 msec, Size= 268435456 Ops,
CuBlas Performance= 2804.28 GFlop/s, Time= 0.096 msec, Size= 268435456 Ops,
Result= PASS
ratio= 1.062884

./builds/main 1024 1024 1024
My gemm Performance= 3352.44 GFlop/s, Time= 0.641 msec, Size= 2147483648 Ops,
CuBlas Performance= 1402.04 GFlop/s, Time= 1.532 msec, Size= 2147483648 Ops,
Result= PASS
ratio= 2.391115

./builds/main 1536 1536 1536
My gemm Performance= 3767.84 GFlop/s, Time= 1.924 msec, Size= 7247757312 Ops,
CuBlas Performance= 3586.85 GFlop/s, Time= 2.021 msec, Size= 7247757312 Ops,
Result= PASS
ratio= 1.050458

./builds/main 1792 1792 1792
My gemm Performance= 4274.72 GFlop/s, Time= 2.692 msec, Size= 11509170176 Ops,
CuBlas Performance= 4817.64 GFlop/s, Time= 2.389 msec, Size= 11509170176 Ops,
Result= PASS
ratio= 0.887304

./builds/main 2048 2048 2048
My gemm Performance= 4708.21 GFlop/s, Time= 3.649 msec, Size= 17179869184 Ops,
CuBlas Performance= 5258.44 GFlop/s, Time= 3.267 msec, Size= 17179869184 Ops,
Result= PASS
ratio= 0.895364