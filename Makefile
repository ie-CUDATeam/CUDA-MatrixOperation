NVCC    = nvcc
INCLUDE = -I ~/NVIDIA_CUDA-5.5_Samples/common/inc
TARGETS = hello_world

hello_world: hello_world.cu
	$(NVCC) $(INCLUDE) $< -o $@

clean:
	$(RM) $(TARGETS) a.out *~