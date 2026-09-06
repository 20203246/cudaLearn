#pragma once
#include <cassert>
#include <vector>
#include <iostream>
#include <memory>
#include <cstring>
using namespace std;

template<typename T>
__device__ void DEBUG(T *ptr, int M, int N, const char *s) {
    printf("%s\n", s);
    for(int i = 0; i < M; i++) {
        printf("i=%d | ", i);
        for (int j = 0; j < N; j++) {
            if constexpr (std::is_same_v<T, int>) {
                printf("%d ", ptr[i * N + j]);
            } else if constexpr (std::is_same_v<T, float>) {
                printf("%.0f ", ptr[i * N + j]);
            }
        }
        printf("\n");
    }
}

template<typename T>
class Tensor {
public:
    shared_ptr<T[]> p;
    int M = -1, N = -1;
    Tensor() = delete;
    Tensor(int M, int N, int defValue=0): M(M), N(N) {
        p = shared_ptr<T[]>(new T[M*N]);
        ones(defValue);
    }
    Tensor(const Tensor& t) {
        M = t.M;
        N = t.N;
        p = shared_ptr<T[]>(new T[M*N]);
        for (int i = 0; i < M * N; i++) {
            p[i] = t.p[i];
        }
        cout << "copy constructor" << endl;
    }
    void ones(T x = 1) {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                p.get()[i * N + j] = (i == j ? x : 0);
            }
        }
    }

    void fill(T x=1) {
        for(int i = 0; i < M * N; i++) {
            p.get()[i] = x;
        }
    }

    void arange(T from, T to) {
        for(int i = 0; from < to; from++) {
            p.get()[i++] = from;
        }
    }

    void upper(T x=1){
        assert(M == N);
        for(int i = 0; i < M; i++) {
            for(int j = i; j < N; j++) {
                p[i * N + j] = x;
            }
        }
    }
    T operator[] (int idx) const {
        assert(idx >= 0 && idx < M * N);
        return p.get()[idx];
    }
    void operator *= (const int scale) {
        for(int i = 0; i < M * N; i++) {
            p.get()[i] *= scale;
        }
    }

    Tensor operator * (const Tensor &other) {
        assert(N == other.M);
        Tensor tmp(M, other.N, 0);
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < other.N; j++) {
                for (int k = 0; k < N; k++) {
                    tmp.p[i * other.N + j] = p[i * N + k] * other.p[k * other.N + j];
                }
            }
        }
        return tmp;
    }

    template<typename H>
    bool operator == (const Tensor<H> &other) const {
        if (!std::is_same_v<T, H>) {
            return false;
        }
        if (M != other.M || N != other.N) return false;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (p.get()[i * N + j] != other.p.get()[i * N + j]) {
                    return false;
                }
            }
        }
        return true;
    }

    friend ostream& operator << (ostream&out, const Tensor<T>& t) {
        int _END = min(t.N, t.M);
        _END = min(_END, 20);
        cout << "M=" << t.M << " " << "N=" << t.N << endl;
        T* data = t.p.get();
        assert(data != nullptr);
        for(int i = 0; i < _END; i++) {
            cout << "i: " << i << "| ";
            for(int j = 0; j < _END; j++) {
                out << data[i * t.N + j] << " \n"[j == _END - 1];
            }
        }
        return out;
    }
};

class MMatmul {
public:
    virtual void launch(float *A, float *B, float *C, int M, int N, int K) { };
    virtual void test() { }
    string do_test(bool flag) {
        return flag ? "pass" : "fail";
    }
};