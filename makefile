CU=/usr/local/cuda-10.1/bin/nvcc
CC=g++
# FLAGS=-gencode arch=compute_61,code=sm_61
builds=./builds
std=c++11
link=-lcublas -lcusparse

all: gemm

gemm:
	$(CU) ./src/main.cu  -o $(builds)/main

run: gemm
	./builds/main 1024 1024 1024

.PHONY: prof
prof:
	nvprof ./builds/main 1024 1024 1024