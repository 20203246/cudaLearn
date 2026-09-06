#include <bits/stdc++.h>
#include "native_cpu.cpp"
#include "gemm_gpu.cu"
#include "gemm_advance.cu"
#include "test_device.cu"
#include "add_gpu.cu"
#include "reduce.cu"

using namespace std;

void displayDeviceInformation() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Wrap size: " << prop.warpSize << endl;
    cout << "max Thread Per Streaming Multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,gemmGpuLaunch,0,0);
    cout << "minGridSize: " << minGridSize << " blockSize: " << blockSize << endl;
}

int main()
{
    // displayDeviceInformation();
    // SgemmNativeCpu sgemmNativeCpu;
    // sgemmNativeCpu.test();
    // SgemmNative sgemmNative;
    // sgemmNative.test();
    // AddNative addNative;
    // addNative.test();
    MGemmAdvance mGemmAdvance;
    mGemmAdvance.test();
    // MDevice mDevice;
    // mDevice.test();
    // ReduceKernel reduceKernel;
    // reduceKernel.test();
}