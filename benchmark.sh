for size in 256 512 1024 1536 2048
do
    echo "----- benchmark ${size} -----"
    ./builds/gemm ${size} ${size} ${size}
done