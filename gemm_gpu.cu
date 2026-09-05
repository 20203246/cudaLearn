#pragma once

static __global__ void gemmGpuLaunch(int M, int N, int K, 
    const float *A,const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // __syncthreads();
    if (x < M && y < N) {
        float tmp = 0;
        for(int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}

class SgemmNative: public MMatmul {
public:
    void test() override {
        int M = 300, N = 300, K = 300;
        Tensor<float> ha(M, K, 2), hb(K, N, 2), hc(M, N, 0);
        float *da, *db, *dc;
        cudaMalloc(&da, M * K * sizeof(float));
        cudaMalloc(&db, K * N * sizeof(float));
        cudaMalloc(&dc, M * N * sizeof(float));
        cudaMemcpy(da, ha.p.get(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.p.get(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc.p.get(), M * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(32, 32);
        dim3 gridSize((M+31)/32,(N+31)/32);
        for(int lp = 0; lp < 100; lp++)
            gemmGpuLaunch<<<gridSize,blockSize>>>(M,N,K,da,db,dc);
        cudaDeviceSynchronize();
        cudaMemcpy(hc.p.get(), dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cout << "SgemmNative:" << endl;
        for(int i = 0; i < min(10, M * N); i++) {
            cout << hc[i] << " ";
        }
        cout << "\n===============\n";
        float *_free[] = {da, db, dc};
        for(int i = 0; i < sizeof(_free) / sizeof(float*); i++)
            cudaFree(_free[i]);
    }
};