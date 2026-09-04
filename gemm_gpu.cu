#pragma once

static __global__ void gemmGpuLaunch(int M, int N, int K, 
    const float *A,const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
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
        vector<float> ha = {2,0,0,0,2,0,0,0,2};
        vector<float> hb = {2,0,0,0,2,0,0,0,2};
        int M = 3, N = 3, K = 3;
        vector<float> hc(M * N, 0.0);
        float *da, *db, *dc;
        cudaMalloc(&da, M * K * sizeof(float));
        cudaMalloc(&db, K * N * sizeof(float));
        cudaMalloc(&dc, M * N * sizeof(float));
        cudaMemcpy(da, ha.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(32, 32);
        dim3 gridSize((M+31)/32,(N+31)/32);
        gemmGpuLaunch<<<gridSize,blockSize>>>(M,N,K,da,db,dc);
        cudaDeviceSynchronize();
        cudaMemcpy(hc.data(), dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cout << "SgemmNative:" << endl;
        for(int i = 0; i < M * N; i++) {
            cout << hc[i] << " ";
        }
        cout << "\n===============\n";
        float *_free[] = {da, db, dc};
        for(int i = 0; i < sizeof(_free) / sizeof(float*); i++)
            cudaFree(_free[i]);
    }
};