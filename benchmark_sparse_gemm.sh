for sparsity in 1 5 10 30
do
    echo "----- benchmark sparsity: ${sparsity}% -----"
    ./builds/sparse_gemm 2048 2048 2048 ${sparsity}
done