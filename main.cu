#include <bits/stdc++.h>
#include "native_cpu.cpp"
#include "gemm_gpu.cu"
using namespace std;

int main()
{
    int M = 3, N = 3, K = 3;
    vector<float> ha = {1,0,0,0,1,0,0,0,1};
    vector<float> hb = {1,0,0,0,1,0,0,0,1};
    vector<float> hc(M * N, 0.0);
    // sgemm_native_cpu(ha.data(), hb.data(), hc.data(), 3,3,3);
    for(auto &t: hc) cout << t << " "; cout << endl;
    float *da, *db, *dc;
    cudaMalloc(&da, M * K * sizeof(float));
    cudaMalloc(&db, K * N * sizeof(float));
    cudaMalloc(&dc, M * N * sizeof(float));
    cudaMemcpy(da, ha.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((M+31)/32,(N+31)/32);
    sgemm_native<<<gridSize,blockSize>>>(M,N,K,da,db,dc);
    cudaDeviceSynchronize();
    cudaMemcpy(hc.data(), dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < M * N; i++) {
        cout << hc[i] << " ";
    }
    float *_free[] = {da, db, dc};
    for(int i = 0; i < sizeof(_free) / sizeof(float*); i++)
        cudaFree(_free[i]);
}