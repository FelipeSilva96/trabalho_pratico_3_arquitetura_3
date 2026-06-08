CC=gcc
NVCC=nvcc
CFLAGS=-O3 -march=native -fopenmp -fopt-info-vec-optimized
CUDAFLAGS=-O3
BUILD_DIR=build
RESULTS_TABLES=results/tables

.PHONY: all cpu cuda clean dirs

all: cpu cuda

dirs:
	mkdir -p $(BUILD_DIR) $(RESULTS_TABLES)

cpu: dirs
	$(CC) src/cpu/gemm_cpu.c $(CFLAGS) -o $(BUILD_DIR)/gemm_cpu 2> $(RESULTS_TABLES)/vec_report.txt

cuda: dirs
	$(NVCC) src/cuda/gemm_cuda.cu $(CUDAFLAGS) -o $(BUILD_DIR)/gemm_cuda

clean:
	rm -rf $(BUILD_DIR)/gemm_cpu $(BUILD_DIR)/gemm_cuda $(RESULTS_TABLES)/vec_report.txt
