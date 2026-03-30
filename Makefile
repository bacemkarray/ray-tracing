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

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

cudart.o: main.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c main.cu

$(REPORT_DIR):
	mkdir -p $(REPORT_DIR)

profile_basic: cudart | $(REPORT_DIR)
	$(NSYS) profile \
		-t cuda \
		--stats=true \
		-o $(REPORT_DIR)/nsys_$(TAG) \
		./cudart > out.ppm

profile_compute: cudart | $(REPORT_DIR)
	$(NCU) \
		-o $(REPORT_DIR)/ncu_$(TAG) \
		./cudart > out.ppm

profile_metrics: cudart | $(REPORT_DIR)
	$(NCU) \
		--metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer \
		-o $(REPORT_DIR)/ncu_metrics_$(TAG) \
		./cudart > out.ppm

clean:
	rm -f cudart cudart.o out.ppm out.jpg