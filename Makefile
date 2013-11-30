GCC     = gcc
CFLAGS  = -Wall -std=gnu99
NVCC    = nvcc
INCLUDE = -I ~/NVIDIA_CUDA-5.5_Samples/common/inc
TARGETS = helloWorld cpuMatrixMultiple

helloWorld: helloWorld.cu
	$(NVCC) $(INCLUDE) $< -o $@

cpuMatrixMultiple: cpuMatrixMultiple.c
	$(GCC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(TARGETS) a.out *~