for size in 256 512 1024 1536 2048
do
    echo "----- benchmark size: ${size} -----"
    ./builds/gemm ${size} ${size} ${size} 100
done


for sparsity in 30 10 5 1
do
    echo "----- benchmark sparsity: ${sparsity}% -----"
    ./builds/gemm 2048 2048 2048 ${sparsity}
done