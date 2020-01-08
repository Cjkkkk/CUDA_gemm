CU=/usr/local/cuda-10.2/bin/nvcc
CC=g++
LIBS=-lcublas -lcusparse
CPP_SOURCE=./src/cpp
CUDA_SOURCE=./src/cuda
TEST_SOURCE=./test

BUILD=./builds
STD=c++11
FLAGS=-gencode=arch=compute_35,code=sm_35 \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_52,code=sm_52 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_60,code=compute_60 \
	--ptxas-options=-v 
# FLAGS=--ptxas-options=-v 

all: gemm

$(BUILD)/%.o: $(CPP_SOURCE)/%.cpp 
	$(CC) -std=$(STD) -c $< -o $@

$(BUILD)/%.o: $(TEST_SOURCE)/%.cpp
	$(CC) -std=$(STD) -I $(CPP_SOURCE) -c $< -o $@

$(BUILD)/%.o: $(CUDA_SOURCE)/%.cu
	$(CU) -std=$(STD) -I $(CPP_SOURCE) -c $< -o $@  $(FLAGS)

gemm: $(BUILD)/utils.o $(BUILD)/dense.o
	$(CU) $(BUILD)/utils.o $(BUILD)/dense.o -std=$(STD) -o $(BUILD)/gemm $(LIBS) $(FLAGS)

sparse_gemm: $(BUILD)/utils.o $(BUILD)/sparse.o
	$(CU) $(BUILD)/utils.o $(BUILD)/sparse.o -std=$(STD) -o $(BUILD)/sparse_gemm $(LIBS) $(FLAGS)

test: $(BUILD)/test.o $(BUILD)/utils.o
	$(CC) $(BUILD)/utils.o $(BUILD)/test.o -std=$(STD) -o $(BUILD)/test -g
	./builds/test

benchmark_gemm: gemm
	sh benchmark_gemm.sh

benchmark_sparse_gemm: sparse_gemm
	sh benchmark_sparse_gemm.sh

gemm_test: gemm
	$(BUILD)/gemm 64 64 64