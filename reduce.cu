#pragma once
#include "mmatmul.h"


template<int BUFFER_SIZE>
__global__ void reduceGpuBaselineLaunch(int N, float *A, float *out) {
    __shared__ float sdata[BUFFER_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int idx = bid * bdim + tid;
    if (idx < N) {
        sdata[tid] = A[idx];
    }
    __syncthreads();
    for(int i = 1; i < bdim; i <<= 1) {
        if (tid % (2 * i) == 0 && i + idx < N) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[bid] = sdata[0];
    }
}

template<int BUFFER_SIZE>
__global__ void reduceGpuInterleavedLaunch(int N, float *A, float *out) {
    __shared__ float sa[BUFFER_SIZE];
    const uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) sa[threadIdx.x] = A[idx];
    else sa[threadIdx.x] = 0;
    __syncthreads();
    // 0 ... threadDim.x-1
    for(int h = 1; h < blockDim.x; h <<= 1) {
        int from = 2 * h * threadIdx.x;
        if ((from + h < blockDim.x) && (idx + h <= N)) {
            sa[from] += sa[from + h];
        }
        __syncthreads();
    }
    out[blockIdx.x] = sa[0];
}

class ReduceKernel: public MMatmul {
public:
    void test() override {
        int M = 300;
        Tensor<float> ha(M, 1, 2);
        constexpr int BUFFER_SIZE = 256;
        Tensor<float> hb((M+BUFFER_SIZE-1)/BUFFER_SIZE,1,0);
        ha.fill(2);
        float *da, *db;
        cudaMalloc(&da, M * sizeof(float));
        cudaMalloc(&db, hb.M * sizeof(float));
        cudaMemcpy(da, ha.p.get(), M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.p.get(), hb.M * sizeof(float), cudaMemcpyHostToDevice);
        dim3 gridSize((M+BUFFER_SIZE-1)/BUFFER_SIZE);
        dim3 blockSize(BUFFER_SIZE);
        for(int lp = 0; lp < 1; lp++){
            // reduceGpuBaselineLaunch<BUFFER_SIZE><<<gridSize,blockSize>>>(M,da,db);
            reduceGpuInterleavedLaunch<BUFFER_SIZE><<<gridSize,blockSize>>>(M,da,db);
        }
        cudaDeviceSynchronize();
        cudaMemcpy(ha.p.get(), da, M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.p.get(), db, hb.M * sizeof(float), cudaMemcpyDeviceToHost);
        cout << "Reduce:" << endl;
        for(int i = 0; i < min(10, hb.M); i++) {
            cout << hb[i] << " ";
        }
        cout << "\n=======test:" << do_test(accumulate(hb.p.get(),hb.p.get()+hb.M,0) == M*2) << "========\n";
        float *_free[] = {da, db};
        for(int i = 0; i < sizeof(_free) / sizeof(float*); i++)
            cudaFree(_free[i]);
    }
};