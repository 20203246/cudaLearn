#include <bits/stdc++.h>
#include "native_cpu.cpp"
#include "gemm_gpu.cu"
#include "gemm_advance.cu"
#include "test_device.cu"

using namespace std;

void displayDeviceInformation() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Wrap size: " << prop.warpSize << endl;
    cout << "max Thread Per Streaming Multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
}

int main()
{
    displayDeviceInformation();
    int M = 3, N = 3, K = 3;
    SgemmNativeCpu sgemmNativeCpu;
    sgemmNativeCpu.test();
    SgemmNative sgemmNative;
    sgemmNative.test();
    MGemmAdvance mGemmAdvance;
    mGemmAdvance.test();
}