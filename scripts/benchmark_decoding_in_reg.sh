for size in 256 512 1024 2048 4096
do
    echo "----- benchmark benchmark_encoding -----"
    ./builds/benchmark_decoding_in_reg ${size} ${size} ${size}
done
