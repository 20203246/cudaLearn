#pragma once

#include <iostream>
using namespace std;

__global__ void MDeviceLaunch() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d %d %d %d\n", blockDim.x, blockDim.y, idx);
    __syncthreads();
}

class MDevice {
public:
    void test() {
        dim3 blockSize(12, 32);
        dim3 gridSize(3,3);
        cout << "begin MDevice" << endl;
        MDeviceLaunch<<<gridSize,blockSize>>>();
        cudaDeviceSynchronize();
        cout << "end" << endl;
    }
};