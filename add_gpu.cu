#pragma once

static __global__ void addGpuLaunch(int N, const float *A,const float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint en = 256 * x + 256;
    if (en > N) en = N;
    __shared__ float sa[256], sb[256];
    for(int i = 256 * x; i < en; i++) {
        sa[i - 256 * x] = A[i];
        sb[i - 256 * x] = B[i];
    }
    __syncthreads();
    for(int i = 256 * x; i < en; i++){
        for (int lp = 0; lp < 10; lp++){
            float tmp = sa[i - 256 * x] + sb[i - 256 * x];
            C[i] = tmp;
            // C[i] = sa[i] + sb[i];
            // C[i] = A[i] + B[i];
        }
    }
}

class AddNative: public MMatmul {
public:
    void test() override {
        int M = 300, N = 300, K = 300;
        Tensor<float> ha(M, K, 2), hb(K, N, 2), hc(M, N, 0);
        ha.fill(2), hb.fill(2);
        float *da, *db, *dc;
        cudaMalloc(&da, M * K * sizeof(float));
        cudaMalloc(&db, K * N * sizeof(float));
        cudaMalloc(&dc, M * N * sizeof(float));
        cudaMemcpy(da, ha.p, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.p, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc.p, M * N * sizeof(float), cudaMemcpyHostToDevice);

        for(int lp = 0; lp < 1000; lp++)
            addGpuLaunch<<<2, N*M/256>>>(M*N,da,db,dc);
        cudaDeviceSynchronize();
        cudaMemcpy(hc.p, dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cout << "AddNative:" << endl;
        for(int i = 0; i < min(10, M * N); i++) {
            cout << hc[i] << " ";
        }
        cout << "\n===============\n";
        float *_free[] = {da, db, dc};
        for(int i = 0; i < sizeof(_free) / sizeof(float*); i++)
            cudaFree(_free[i]);
    }
};