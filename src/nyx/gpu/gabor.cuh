#include <cufftXt.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#define CUFFT_MAX_SIZE 1 << 27

typedef float2 Complex;
typedef cufftComplex CuComplex;

namespace CuGabor{
    
    void conv_dud_gpu(
        double* C,  // must be zeroed before call
        const unsigned int* A,
        double* B,
        int na, int ma, int nb, int mb);

    void conv_dud_gpu_fft(
        double* C,  // must be zeroed before call
        const unsigned int* A,
        double* B,
        int na, int ma, int nb, int mb);

    __global__ void multiply(
        CuComplex* A, 
        int row_size, 
        int col_size, 
        CuComplex* B, 
        CuComplex* result, 
        int batch_size);

    void cmat_mult(
        CuComplex* A, 
        int row_size, 
        int col_size, 
        CuComplex* B, 
        CuComplex* result, 
        int batch_size);
    
    void cmat_mult(
        cufftDoubleComplex* A, 
        int row_size, 
        int col_size, 
        cufftDoubleComplex* B, 
        cufftDoubleComplex* result, 
        int batch_size);


    void conv_dud_gpu_fft_multi_filter(double* out, 
        const unsigned int* image, 
        double* kernel, 
        int image_n, int image_m, int kernel_n, int kernel_m, int batch_size);
}