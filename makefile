CUDA_ARCH = -arch=native

main: main.cu *.cu *.cpp *.h
	nvcc -g -G $(CUDA_ARCH) $< -o $@
clean:
	rm main
test: main
	nvprof ./main

run: main
	clear
	./main