CU=/usr/local/cuda/bin/nvcc
CC=g++
LIBS=-lcublas -lcusparse
CPP_SOURCE=./src/cpp
CUDA_SOURCE=./src/cuda
TEST_SOURCE=./test
MAIN_SOURCE=./benchmark
SCRIPT_SOURCE=./scripts
INCLUDE_DIR=-I./src/cpp/include -I./src/cuda/include -I./src/cuda/
BUILD=./builds
STD=c++11
FLAGS=-gencode=arch=compute_35,code=sm_35 \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_52,code=sm_52 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_60,code=compute_60
DEP=$(CUDA_SOURCE)/encoding.cu \
	$(CUDA_SOURCE)/encoding_in_reg.cu \
	$(CUDA_SOURCE)/dense.cu \
	$(CUDA_SOURCE)/sparse.cu \
	$(CUDA_SOURCE)/quantization_int.cu \
	$(CUDA_SOURCE)/quantization_8bit.cu \
	$(CUDA_SOURCE)/quantization_4bit.cu \


$(BUILD)/%.o: $(CPP_SOURCE)/%.cpp 
	$(CC) -std=$(STD) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(TEST_SOURCE)/%.cpp
	$(CC) -std=$(STD) $(INCLUDE_DIR) -c $< -o $@

$(BUILD)/%.o: $(CUDA_SOURCE)/%.cu
	$(CU) -std=$(STD) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS)

$(BUILD)/%.o: $(MAIN_SOURCE)/%.cu $(DEP)
	$(CU) -std=$(STD) $(INCLUDE_DIR) -c $< -o $@  $(FLAGS)


benchmark_%: $(BUILD)/benchmark_%.o $(BUILD)/utils.o
	$(CU) $^ -std=$(STD) -o $(BUILD)/$@ $(LIBS) $(FLAGS)
	sh ${SCRIPT_SOURCE}/$@.sh

test: $(BUILD)/test.o $(BUILD)/utils.o
	$(CC) $^ -std=$(STD) -o $(BUILD)/$@ -g
	./builds/test

dense_ptx: $(MAIN_SOURCE)/benchmark_dense.cu
	$(CU) $^ $(INCLUDE_DIR) -ptx -src-in-ptx -lineinfo -gencode=arch=compute_60,code=compute_60 --ptxas-options=-v 

.PHONY: clean
clean:
	rm $(BUILD)/*
