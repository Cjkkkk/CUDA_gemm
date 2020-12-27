for size in 256 512 1024 1536 2048 2560 3072 3584 4096
do
    echo "----- benchmark size: ${size} -----"
    ./builds/benchmark_quantization_int ${size} ${size} ${size}
done