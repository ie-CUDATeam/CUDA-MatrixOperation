NVCC    = nvcc
INCLUDE = -I ~/NVIDIA_CUDA-5.5_Samples/common/inc
TARGETS = helloWorld

helloWorld: helloWorld.cu
	$(NVCC) $(INCLUDE) $< -o $@

clean:
	$(RM) $(TARGETS) a.out *~