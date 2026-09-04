CUDA_ARCH = -arch=native

main: main.cu *.cu *.cpp *.h
	nvcc $(CUDA_ARCH) $< -o $@
clean:
	rm main

run: main
	./main