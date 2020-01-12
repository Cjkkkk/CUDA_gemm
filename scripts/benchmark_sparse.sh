for sparsity in 1 5 10 30
do
    echo "----- benchmark sparsity: ${sparsity}% -----"
    ./builds/benchmark_sparse 2048 2048 2048 ${sparsity}
done