#pragma once
/**
 * 之前的例子，一个thread计算输出一个元素
 * 本例子，一个thread计算输出TM个元素，提高计算密度 
 * 
 *  C矩阵 (8×8):
 *          col0  col1  col2  col3  col4  col5  col6  col7
 *      ┌────────────────────────────────────────────────┐
 * row0 │ T0   T1   T2   T3   T4   T5   T6   T7          │
 * row1 │ T0   T1   T2   T3   T4   T5   T6   T7          │
 * row2 │ T0   T1   T2   T3   T4   T5   T6   T7          │
 * row3 │ T0   T1   T2   T3   T4   T5   T6   T7          │ ← 每个线程负责4行
 * row4 │ T8   T9   T10  T11  T12  T13  T14  T15         │
 * row5 │ T8   T9   T10  T11  T12  T13  T14  T15         │
 * row6 │ T8   T9   T10  T11  T12  T13  T14  T15         │
 * row7 │ T8   T9   T10  T11  T12  T13  T14  T15         │
 *      └────────────────────────────────────────────────┘
 */
static constexpr int BLOCK_SIZE = 32;
__global__ void mGemmAdvanceLaunch(int M, int N, int K, 
    const float *A,const float *B, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    A += cRow * K * BLOCK_SIZE;
    B += cCol * BLOCK_SIZE;
    C += (cRow * N + cCol) * BLOCK_SIZE;
    float tmp = 0;
    __shared__ float sa[BLOCK_SIZE][BLOCK_SIZE], sb[BLOCK_SIZE][BLOCK_SIZE];
    for(int i = 0; i < K; i += BLOCK_SIZE) {
        sa[threadIdx.y][threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
        sb[threadIdx.y][threadIdx.x] = B[threadIdx.y * N + threadIdx.x];
        
        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; j++) {
            tmp += sa[threadIdx.y][j] * sb[j][threadIdx.x];
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
    }
    C[threadIdx.y * N + threadIdx.x] = tmp;

}

class MGemmAdvance : public MMatmul {
public:
    void test() {
        int M = 128, N = 128, K = 128;
        Tensor<float> ha(M, K, 3), hb(K, N, 3), hc(M, N, 0);
        ha.fill(3);
        float *da, *db, *dc;
        cudaMalloc(&da, M * K * sizeof(float));
        cudaMalloc(&db, K * N * sizeof(float));
        cudaMalloc(&dc, M * N * sizeof(float));
        cudaMemcpy(da, ha.p.get(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.p.get(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc.p.get(), M * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((M+BLOCK_SIZE-1)/BLOCK_SIZE,(N+BLOCK_SIZE-1)/BLOCK_SIZE);
        mGemmAdvanceLaunch<<<gridSize,blockSize>>>(M,N,K,da,db,dc);
        cudaDeviceSynchronize();
        cudaMemcpy(hc.p.get(), dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cout << "GemmAdvance:" << endl;
        cout << hc << endl;
        Tensor<float> truth(M, N);
        truth.fill(3);
        cout << "======= test:" << (hc == truth) << "========\n";
        float *_free[] = {da, db, dc};
        for(int i = 0; i < sizeof(_free) / sizeof(float*); i++)
            cudaFree(_free[i]);
    }
};