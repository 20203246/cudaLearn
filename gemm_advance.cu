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
__device__ void print(const char* s) {
    printf("%s", s);
}

template<typename T, typename... Args>
__device__ void print(const char* s, T value, Args... args) {
    printf(s, value, args...);
}
template<int BM, int BN, int BK, int TM>
__global__ void mOnedimThreadTileGemmLaunch(int M, int N, int K,
    const float* A, const float *B, float *C)
{
    const uint cRow = blockIdx.y, cCol = blockIdx.x;
    const uint tRow = threadIdx.y, tCol = threadIdx.x;
    A += cRow * K * BM;
    B += cCol * BN;
    C += cRow * N * BM + cCol * BN;
    __shared__ float sa[BM][BK], sb[BK][BN];
    float tmp[TM] = {0};
    for (int i = 0; i < K; i += BK) {
        if (tRow < BM && tCol < BK)
            sa[tRow][tCol] = A[tRow * K + tCol];
        if (tRow < BK && tCol < BN) 
            sb[tRow][tCol] = B[tRow * N + tCol];
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int resIdx = 0; resIdx < TM; resIdx++) {
                int row = tRow * TM + resIdx;
                if (row < BM) {
                    tmp[resIdx] += sa[row][dotIdx] * sb[dotIdx][tCol];
                }
            }
        }       
        __syncthreads();

        A += BK;
        B += N * BK;
    }
    for (int i = 0; i < TM; i++) {
        int row = tRow * TM + i;
        if (row < BM && tCol < BN) {
            C[row * N + tCol] = tmp[i];
        }
    }
}
/**
 * TM = TN = 8
 * BK * iteratorNum = N
 */
template<int BM, int BN, int BK, int TM, int TN>
__global__ void mTwodimThreadTileGemmLaunch(int M, int N, int K,
    const float* A, const float *B, float *C)
{
    const uint tRow = threadIdx.y;
    const uint tCol = threadIdx.x;
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    A += K * cRow * BM;
    B += cCol * BN;
    C += N * cRow * BM + cCol * BN;

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("=== Debug Info ===\n");
    //     printf("M=%d, N=%d, K=%d\n", M, N, K);
    //     printf("BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\n", BM, BN, BK, TM, TN);
    //     printf("cRow=%d, cCol=%d\n", cRow, cCol);
    //     printf("A offset=%d, B offset=%d, C offset=%d\n", 
    //            cRow * BM * K, cCol * BN, cRow * BM * N + cCol * BN);
    // }
    __shared__ float sa[BM][BK], sb[BK][BN];
    float tmp[TM][TN] = {0};
    for(int i = 0; i < K; i += BK) {
        // CopyIn 
        for (int x = 0; x < TM; x++) {
            int row = tRow * TM + x;
            if (row < BM && tCol < BK) {
                sa[row][tCol] = A[row * K + tCol];
            }
        }
        for (int y = 0; y < TN; y++) {
            int col = tCol * TN + y;
            if (tRow < BK && col < BN) {
                sb[tRow][col] = B[tRow * N + col];
            }
        }
        __syncthreads();
        // if (tCol == 0 && tRow == 0 && cCol == 1 && cRow == 0) {
        //     printf("BM = %d, BK = %d, i = %d\n", BM, BK, i);
        //     DEBUG<float>(&sa[0][0], BM, BK, "from LEFT");
        //     DEBUG<float>(&sb[0][0], BK, BN, "from RIGHT");
        // }
        // Compute
        #pragma unroll
        for (int x = 0; x < TM; x++) {
            #pragma unroll
            for (int y = 0; y < TN; y++) {
                #pragma unroll
                for (int j = 0; j < BK; j++) {
                    int row = tRow * TM + x;
                    int col = tCol * TN + y;
                    if (row < BM && col < BN) {
                        tmp[x][y] += sa[row][j] * sb[j][col];
                    }
                }
            }
        }
        // if (cRow == 0 && cCol == 1 && tRow == 0 && tCol == 0) {
        //     printf("Generated Matrix");
        //     for (int x = 0; x < TM; x++) {
        //         for (int y = 0; y < TN; y++) {
        //             printf("%.0f ", tmp[x][y]);
        //         }
        //         printf("\n");
        //     }
        // }
        __syncthreads();

        A += BK;
        B += N * BK;
    }
    //CopyOut
    #pragma unroll
    for (int x = 0; x < TM; x++) {
        #pragma unroll
        for (int y = 0; y < TN; y++) {
            int row = tRow * TM + x;
            int col = tCol * TN + y;
            if (row < M && col < N) {
                C[row * N + col] = tmp[x][y];
            }
        }
    }
}

class MGemmAdvance : public MMatmul {
public:
    void test() {
        int M = 128, N = 128, K = 128;
        Tensor<float> ha(M, K, 3), hb(K, N, 3), hc(M, N, 0);
        ha.fill(3);
        hb.fill(1);
        // ha.arange(0, M*K);
        float *da, *db, *dc;
        cudaMalloc(&da, M * K * sizeof(float));
        cudaMalloc(&db, K * N * sizeof(float));
        cudaMalloc(&dc, M * N * sizeof(float));
        cudaMemcpy(da, ha.p.get(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.p.get(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc.p.get(), M * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((N+BLOCK_SIZE-1)/BLOCK_SIZE,(M+BLOCK_SIZE-1)/BLOCK_SIZE);
        for(int _ = 0; _ < 100; _++){
            mGemmAdvanceLaunch<<<gridSize,blockSize>>>(M,N,K,da,db,dc);
            mOnedimThreadTileGemmLaunch<32,32,8,4><<<gridSize, blockSize>>>(M,N,K,da,db,dc);
            mTwodimThreadTileGemmLaunch<32,32,8,2,2><<<dim3(N/32,M/32),dim3(32/2,32/2) >>>(M,N,K,da,db,dc);
        }
        cudaDeviceSynchronize();
        cudaMemcpy(hc.p.get(), dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cout << "GemmAdvance:" << endl;
        cout << hc << endl;
        Tensor<float> truth(M, N);
        truth.fill(384);
        cout << "======= test:" << do_test(hc == truth) << "========\n";
        float *_free[] = {da, db, dc};
        for(int i = 0; i < sizeof(_free) / sizeof(float*); i++)
            cudaFree(_free[i]);
    }
};