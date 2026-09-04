#pragma once

class MMatmul {
public:
    virtual void launch(float *A, float *B, float *C, int M, int N, int K) { };
    virtual void test() { }
};