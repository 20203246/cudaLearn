CUDA_ARCH = -arch=native

main: main.cu
	nvcc $(CUDA_ARCH) $< -o $@
clean:
	rm main

run: main
	./main