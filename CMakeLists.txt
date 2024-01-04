cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

enable_language(CUDA)
project(cuda_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(src/cpp/include 
                    src/cuda/include
                    src/cuda)

function(setup_cuda_target TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE} src/cpp/utils.cpp)
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)
    target_link_libraries(${TARGET_NAME} cublas cusparse)
endfunction(setup_cuda_target)

setup_cuda_target(benchmark_dense benchmark/benchmark_dense.cu)
setup_cuda_target(benchmark_quantization benchmark/benchmark_quantization.cu)
setup_cuda_target(benchmark_quantization_int benchmark/benchmark_quantization_int.cu)
setup_cuda_target(benchmark_quantization_8bit benchmark/benchmark_quantization_8bit.cu)
setup_cuda_target(benchmark_sparse_worst_case benchmark/benchmark_sparse_worst_case.cu)
setup_cuda_target(benchmark_sparse benchmark/benchmark_sparse.cu)
