GCC     = gcc
CFLAGS  = -Wall -std=gnu99
NVCC    = nvcc
INCLUDE = -I ~/NVIDIA_CUDA-5.5_Samples/common/inc
TARGETS = helloWorld \
          cpuMatrixMultiple cpuMatrixSum \
		  gpuMatrixMultiple

helloWorld: helloWorld.cu
	$(NVCC) $(INCLUDE) $< -o $@

cpuMatrixMultiple: cpuMatrixMultiple.c
	$(GCC) $(CFLAGS) $< -o $@
gpuMatrixMultiple: gpuMatrixMultiple.cu
	$(NVCC) $(INCLUDE) $< -o $@

cpuMatrixSum: cpuMatrixSum.c
	$(GCC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(TARGETS) a.out *~