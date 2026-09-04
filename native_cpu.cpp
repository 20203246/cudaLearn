#pragma once
#include "mmatmul.h"
#include <vector>
using namespace std;

class SgemmNativeCpu: public MMatmul {
public:
    int M, N, K;
    void launch(float *A, float *B, float *C, int M, int N, int K) override {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                float sum = 0;
                for(int x = 0; x < K; x++) {
                    sum += A[i * K + x] * B[x * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    void test() override {
        M = N = K = 3;
        vector<float> ha = {2,0,0,0,2,0,0,0,2};
        vector<float> hb = {2,0,0,0,2,0,0,0,2};
        vector<float> hc(M * N, 0.0);
        launch(ha.data(), hb.data(), hc.data(), M, N, K);
        cout << "SegmmNativeCpu: \n";
        for(auto &t: hc) {
            cout << t << " ";
        }
        cout << "\n===============" << endl;
    }
};
