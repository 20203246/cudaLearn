#pragma once
#include <cassert>
#include <vector>
template<typename T>
class Tensor {
// 易发生释放以释放的指针
public:
    T *p = nullptr;
    int M = -1, N = -1;
    Tensor() = delete;
    Tensor(int M, int N, T defValue=0): M(M), N(N) {
        p = new T[M * N];
        ones(defValue);
    }
    void ones(T x = 1) {
        assert(M == N);
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                p[i * N + j] = (i == j ? x : 0);
            }
        }
    }

    ~Tensor() {
        if(p != nullptr) {
            delete p;
            p = nullptr;
        }
    }
    T operator[] (int idx) const {
        assert(idx >= 0 && idx < M * N);
        return p[idx];
    }
    void operator *= (const int scale) {
        for(int i = 0; i < M * N; i++) {
            p[i] *= scale;
        }
    }
};

class MMatmul {
public:
    virtual void launch(float *A, float *B, float *C, int M, int N, int K) { };
    virtual void test() { }
};