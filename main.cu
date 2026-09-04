#include <bits/stdc++.h>
#include "native_cpu.cpp"
#include "gemm_gpu.cu"

using namespace std;

int main()
{
    int M = 3, N = 3, K = 3;
    SgemmNativeCpu sgemmNativeCpu;
    sgemmNativeCpu.test();
    SgemmNative sgemmNative;
    sgemmNative.test();
}