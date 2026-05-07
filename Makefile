# Detect OS and set executable extension
ifeq ($(OS),Windows_NT)
	EXE_EXT = .exe
else
	EXE_EXT =
endif

CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER = g++
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
NVCC_DBG = -g -G -m64
NVCC_RELEASE = -O3 -lineinfo -m64

# NVCCFLAGS = $(NVCC_DBG)
NVCCFLAGS = $(NVCC_RELEASE)

GENCODE_FLAGS = -gencode arch=compute_89,code=sm_89

REPORT_DIR = reports
NSYS = $(CUDA_PATH)/bin/nsys
NCU = $(CUDA_PATH)/bin/ncu

TAG ?= run

cpuart: main.cpp
	$(HOST_COMPILER) -o cpuart$(EXE_EXT) main.cpp

cudart: main.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart$(EXE_EXT) main.cu

$(REPORT_DIR):
	mkdir -p $(REPORT_DIR)

profile_basic: cudart | $(REPORT_DIR)
	$(NSYS) profile \
		-t cuda \
		--stats=true \
		-o $(REPORT_DIR)/nsys_$(TAG) \
		./cudart$(EXE_EXT) > out.ppm

profile_compute: cudart | $(REPORT_DIR)
	$(NCU) \
		-o $(REPORT_DIR)/ncu_$(TAG) \
		./cudart$(EXE_EXT) > out.ppm

profile_metrics: cudart | $(REPORT_DIR)
	$(NCU) \
		--metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer \
		-o $(REPORT_DIR)/ncu_metrics_$(TAG) \
		./cudart$(EXE_EXT) > out.ppm

clean:
	rm -f cudart$(EXE_EXT) cpuart$(EXE_EXT) out.ppm out.jpg
