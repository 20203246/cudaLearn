#pragma once

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