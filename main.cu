#include <bits/stdc++.h>
using namespace std;

void sgemm_native_cpu(float *A, float *B, float *C, int M, int N, int K) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0;
            for(int x = 0; x < K; x++) {
                sum += A[i * N + x] * B[x * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    vector<float> a = {1,0,0,0,1,0,0,0,1};
    vector<float> b = {1,0,0,0,1,0,0,0,1};
    vector<float> c(a.size(), 0.0);
    sgemm_native_cpu(a.data(), b.data(), c.data(), 3,3,3);
    for(auto &t: c) cout << t << " ";
}