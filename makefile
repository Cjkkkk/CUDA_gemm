CU=/usr/local/cuda-10.2/bin/nvcc
CC=g++
LIBS=-lcublas -lcusparse
CPP_SOURCE=./src/cpp
CUDA_SOURCE=./src/cuda
TEST_SOURCE=./test

BUILD=./builds
STD=c++11

all: gemm

$(BUILD)/%.o: $(CPP_SOURCE)/%.cpp 
	$(CC) -std=$(STD) -c $< -o $@

$(BUILD)/%.o: $(TEST_SOURCE)/%.cpp
	$(CC) -std=$(STD) -I $(CPP_SOURCE) -c $< -o $@

$(BUILD)/%.o: $(CUDA_SOURCE)/%.cu
	$(CU) -std=$(STD) -I $(CPP_SOURCE) -c $< -o $@

gemm: $(BUILD)/utils.o $(BUILD)/dense.o
	$(CU) $(BUILD)/utils.o $(BUILD)/dense.o -std=$(STD) -o $(BUILD)/gemm $(LIBS)

sparse_gemm: $(BUILD)/utils.o $(BUILD)/sparse.o
	$(CU) $(BUILD)/utils.o $(BUILD)/sparse.o -std=$(STD) -o $(BUILD)/sparse_gemm $(LIBS)

test: $(BUILD)/test.o $(BUILD)/utils.o
	$(CC) $(BUILD)/utils.o $(BUILD)/test.o -std=$(STD) -o $(BUILD)/test -g
	./builds/test

benchmark_gemm: gemm
	sh benchmark_gemm.sh

benchmark_sparse_gemm: sparse_gemm
	sh benchmark_sparse_gemm.sh

gemm_test: gemm
	$(BUILD)/gemm 64 64 64 100