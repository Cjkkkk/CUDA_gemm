CU=/usr/local/cuda-9.0/bin/nvcc
CC=g++
LIBS=-lcublas
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

gemm: $(BUILD)/utils.o $(BUILD)/main.o
	$(CU) $(BUILD)/utils.o $(BUILD)/main.o -std=$(STD) -o $(BUILD)/gemm $(LIBS)

test: $(BUILD)/test.o $(BUILD)/utils.o
	$(CC) $(BUILD)/utils.o $(BUILD)/test.o -std=$(STD) -o $(BUILD)/test -g
	./builds/test

benchmark: gemm
	$(BUILD)/gemm 256 256 256
	$(BUILD)/gemm 512 512 512
	$(BUILD)/gemm 1024 1024 1024
	$(BUILD)/gemm 1536 1536 1536
	$(BUILD)/gemm 2048 2048 2048
	$(BUILD)/gemm 4096 4096 4096