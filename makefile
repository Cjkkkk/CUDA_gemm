CU=/usr/local/cuda-10.1/bin/nvcc
CC=g++
# FLAGS=-gencode arch=compute_61,code=sm_61
builds=./builds
std=c++11
link=-lcublas  -lcusparse

all: gemm

gemm:
	$(CU) ./src/main.cu  -o $(builds)/main $(link)

run: gemm
	./builds/main 256 256 256
	./builds/main 512 512 512
	./builds/main 1024 1024 1024
	./builds/main 1536 1536 1536
	./builds/main 1792 1792 1792
	./builds/main 2048 2048 2048

test: gemm
	./builds/main 64 64 64

.PHONY: prof
prof:
	nvprof ./builds/main 1024 1024 1024