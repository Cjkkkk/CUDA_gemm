for size in 256 512 1024 2048
do
    echo "----- benchmark benchmark_encoding -----"
    ../builds/benchmark_decoding ${size} ${size} ${size}
done