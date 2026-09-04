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
template<const int BM, const int BN, const int BK, const int TM>
__global__ void mGemmAdvanceLaunch(int M, int N, int K, 
    const float *A,const float *B, float *C) {

    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;
    __shared__ float A_shared[BM * BK], B_shared[BK * BN];
    const uint thread_row = threadIdx.x / BN;
    const uint thread_col = threadIdx.x % BN;


}

class MGemmAdvance : public MMatmul {
public:
    void test() {
        int M = 64, N = 64, K = 64;
        Tensor<float> ha(M, K, 3), hb(K, N, 3), hc(M, N, 0);
        float *da, *db, *dc;
        cudaMalloc(&da, M * K * sizeof(float));
        cudaMalloc(&db, K * N * sizeof(float));
        cudaMalloc(&dc, M * N * sizeof(float));
        cudaMemcpy(da, ha.p, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.p, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc.p, M * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(32, 32);
        dim3 gridSize((M+31)/32,(N+31)/32);
        // mGemmAdvanceLaunch<<<gridSize,blockSize>>>(M,N,K,da,db,dc);
        cudaDeviceSynchronize();
        cudaMemcpy(hc.p, dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cout << "GemmAdvance:" << endl;
        for(int i = 0; i < M * N; i++) {
            cout << hc[i] << " ";
        }
        cout << "\n===============\n";
        float *_free[] = {da, db, dc};
        for(int i = 0; i < sizeof(_free) / sizeof(float*); i++)
            cudaFree(_free[i]);
    }
};