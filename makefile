CUDA_ARCH = -arch=native

main: main.cu native_cpu.cpp gemm_gpu.cu
	nvcc $(CUDA_ARCH) $< -o $@
clean:
	rm main

run: main
	./main