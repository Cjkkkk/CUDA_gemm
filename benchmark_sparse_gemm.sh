for sparsity in 30 10 5 1
do
    echo "----- benchmark sparsity: ${sparsity}% -----"
    ./builds/sparse_gemm 2048 2048 2048 ${sparsity}
done