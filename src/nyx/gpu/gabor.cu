#include "gabor.cuh"

using namespace std;

namespace CuGabor {
    __global__ void multiply(CuComplex* A, int row_size, int col_size, CuComplex* B, CuComplex* result, int batch_size) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        int index = i*col_size*batch_size+j;
        double a,b,c,d;
        if (i < row_size && j < batch_size*col_size) {
            a = A[index].x;
            b = A[index].y;
            c = B[index].x;
            d = B[index].y;

            result[index].x = a*c - b*d;
            result[index].y = a*d + b*c;
        }
    }

    void cmat_mult(CuComplex* A, int row_size, int col_size, CuComplex* B, CuComplex* result, int batch_size){
        int block = 16;
        dim3 threadsPerBlock(block, block);
        dim3 blocksPerGrid(ceil(row_size/block)+1, ceil(col_size*batch_size/block)+1);

        multiply<<<blocksPerGrid, threadsPerBlock>>>(A, row_size, col_size, B, result, batch_size);

        // Wait for device to finish all operation
        cudaDeviceSynchronize();

        // Check if kernel execution generated and error
        //getLastCudaError("Kernel execution failed [ solvePoisson ]");
        cudaError_t err = cudaGetLastError();   
        if ( err != cudaSuccess ){
                //fprintf(stderr, "Kernel execution failed [ solvePoisson ]\n");
                printf("CUDA Error: %s\n", cudaGetErrorString(err));   
                return;	
        }
    }

    __global__ void multiply(cufftDoubleComplex* A, int row_size, int col_size, cufftDoubleComplex* B, cufftDoubleComplex* result, int batch_size) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        int index = i*col_size*batch_size+j;
        double a,b,c,d;
        if (i < row_size && j < batch_size*col_size) {
            a = A[index].x;
            b = A[index].y;
            c = B[index].x;
            d = B[index].y;

            result[index].x = a*c - b*d;
            result[index].y = a*d + b*c;
        }
    }

    void cmat_mult(cufftDoubleComplex* A, int row_size, int col_size, cufftDoubleComplex* B, cufftDoubleComplex* result, int batch_size){
        int block = 16;
        dim3 threadsPerBlock(block, block);
        dim3 blocksPerGrid(ceil(row_size/block)+1, ceil(col_size*batch_size/block)+1);

        multiply<<<blocksPerGrid, threadsPerBlock>>>(A, row_size, col_size, B, result, batch_size);

        // Wait for device to finish all operation
        cudaDeviceSynchronize();

        // Check if kernel execution generated and error
        //getLastCudaError("Kernel execution failed [ solvePoisson ]");
        cudaError_t err = cudaGetLastError();   
        if ( err != cudaSuccess ){
                //fprintf(stderr, "Kernel execution failed [ solvePoisson ]\n");
                printf("CUDA Error: %s\n", cudaGetErrorString(err));   
                return;	
        }
    }

    void conv_dud_gpu_fft(double* out, 
                            const unsigned int* image, 
                            double* kernel, 
                            int image_n, int image_m, int kernel_n, int kernel_m){

        
        
        int batch_size = 1;

        // calculate new size of image based on padding size
        int row_size = image_m + kernel_m - 1;
        int col_size = image_n + kernel_n - 1;
        int size = row_size * col_size;

        if ((2 * size * batch_size) >= CUFFT_MAX_SIZE) {
            throw invalid_argument("Batch of images is too large. The maximumum number of values in cuFFT is 2^27.");
        }

        // allocate space for linear indexed arrays
        Complex* linear_image = (Complex*)malloc(size * batch_size * sizeof(Complex));
        Complex* result = (Complex*)malloc(size * batch_size * sizeof(Complex));
        Complex* linear_kernel = (Complex*)malloc(size * batch_size * sizeof(Complex));

        int index, index2;
        
        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                index = (i*col_size + j);
                linear_image[index].y = 0.f;
                if (i < image_m && j < image_n) { 
                    index2 = (i*image_n + j) ;
                    linear_image[index].x = image[index2];
                } else {
                    linear_image[index].x = 0.f; // add padding
                }
            }
        }

        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                index = (i*col_size + j);
                index2 = (i*kernel_n + j);
                if (i < kernel_m && j < kernel_n) {
                    linear_kernel[index].x = kernel[2*index2];
                    //linear_kernel[index].y = kernel[2*index2+1];
                } else {
                    linear_kernel[index].x = 0.f;
                    linear_kernel[index].y = 0.f;
                }
            }
        }

        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                index = (i*col_size + j);
            }
        }


        CuComplex* d_image;
        CuComplex* d_result;
        CuComplex* d_kernel;

        int n[2] = {row_size, col_size};

        cudaMalloc((void**)&d_image, sizeof(CuComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }

        cudaMalloc((void**)&d_result, sizeof(CuComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }
        
        cudaMalloc((void**)&d_kernel, sizeof(CuComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }
        
        // copy data to GPU
        cudaMemcpy(d_image, linear_image, batch_size*size*sizeof(CuComplex), cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }
        cudaMemcpy(d_kernel, linear_kernel, batch_size*size*sizeof(CuComplex), cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }

        cufftHandle plan;
        cufftHandle plan_k;
        int idist = size;
        int odist = size;
        
        int inembed[] = {row_size, col_size};
        int onembed[] = {row_size, col_size};

        int istride = 1;
        int ostride = 1;

        if (cufftPlanMany(&plan, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_size) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            //return;	
        }
        if (cufftPlanMany(&plan_k, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_size) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            //return;	
        }

        if (cufftExecC2C(plan, d_image, d_image, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        if (cufftExecC2C(plan_k, d_kernel, d_kernel, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        if (cudaDeviceSynchronize() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to synchronize\n");
            //return;
        }

        // element-wise multiplication of the image and kernel
        cmat_mult(d_image, row_size, col_size, d_kernel, d_result, batch_size);

        // transform out of fourier space
        if (cufftExecC2C(plan, d_result, d_result, CUFFT_INVERSE) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        // copy results from device to host
        cudaMemcpy(result, d_result, batch_size*size*sizeof(CuComplex), cudaMemcpyDeviceToHost); 

        // transfer to output array 
        for(int i = 0; i < size; ++i) {
            out[2*i] = (result[i].x/(size));
            out[2*i + 1] = (result[i].y/(size));
        }
        
        // free host memory
        free(linear_image);
        free(result);
        free(linear_kernel);

        // free device memory
        cufftDestroy(plan);
        cufftDestroy(plan_k);
        cudaFree(d_image);
        cudaFree(d_result);
        cudaFree(d_kernel);
                            
    }

     void conv_dud_gpu_fft_multi_filter(double* out, 
                            const unsigned int* image, 
                            double* kernel, 
                            int image_n, int image_m, int kernel_n, int kernel_m, int batch_size){
        
        // calculate new size of image based on padding size
        int row_size = image_m + kernel_m - 1;
        int col_size = image_n + kernel_n - 1;
        int size = row_size * col_size;

        if ((2 * size * batch_size) >= CUFFT_MAX_SIZE) {
            throw invalid_argument("Batch of images is too large. The maximumum number of values in cuFFT is 2^27.");
        }

        // allocate space for linear indexed arrays
        Complex* linear_image = (Complex*)malloc(size * batch_size * sizeof(Complex));
        Complex* result = (Complex*)malloc(size * batch_size * sizeof(Complex));
        Complex* linear_kernel = (Complex*)malloc(size * batch_size * sizeof(Complex));

        int index, index2;
        
        int batch_idx, batch_idx2;
        for (int batch = 0; batch < batch_size; ++batch) {
            batch_idx = batch * size;
            //batch_idx2 = batch * image_m * image_n;
            for (int i = 0; i < row_size; ++i) {
                for (int j = 0; j < col_size; ++j) {
                    index = batch_idx + (i*col_size + j);
                    linear_image[index].y = 0.f;
                    if (i < image_m && j < image_n) { 
                        //index2 = batch_idx2 + (i*image_n + j) ;
                        index2 = (i*image_n + j);
                        linear_image[index].x = image[index2];
                    } else {
                        linear_image[index].x = 0; // add padding
                    }
                }
            }
        }

        for (int batch = 0; batch < batch_size; ++batch) {
            batch_idx = batch * size;
            batch_idx2 = batch * kernel_m * kernel_n;
            for (int i = 0; i < row_size; ++i) {
                for (int j = 0; j < col_size; ++j) {
                    index = batch_idx + (i*col_size + j);
                    index2 = batch_idx2 + (i*kernel_n + j);
                    if (i < kernel_m && j < kernel_n) {
                        linear_kernel[index].x = kernel[2*index2];
                        linear_kernel[index].y = kernel[2*index2+1];
                    } else {
                        linear_kernel[index].x = 0;
                        linear_kernel[index].y = 0;
                    }
                }
            }
        }

        CuComplex* d_image;
        CuComplex* d_result;
        CuComplex* d_kernel;

        int n[2] = {row_size, col_size};

        cudaMalloc((void**)&d_image, sizeof(CuComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }

        cudaMalloc((void**)&d_result, sizeof(CuComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }
        
        cudaMalloc((void**)&d_kernel, sizeof(CuComplex)*size*batch_size);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }
        
        // copy data to GPU
        cudaMemcpy(d_image, linear_image, batch_size*size*sizeof(CuComplex), cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }
        cudaMemcpy(d_kernel, linear_kernel, batch_size*size*sizeof(CuComplex), cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");	
        }

        cufftHandle plan;
        cufftHandle plan_k;
        int idist = size;
        int odist = size;
        
        int inembed[] = {row_size, col_size};
        int onembed[] = {row_size, col_size};

        int istride = 1;
        int ostride = 1;

        if (cufftPlanMany(&plan, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_size) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            //return;	
        }
        if (cufftPlanMany(&plan_k, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_size) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            //return;	
        }

        if (cufftExecC2C(plan, d_image, d_image, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        if (cufftExecC2C(plan_k, d_kernel, d_kernel, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        if (cudaDeviceSynchronize() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to synchronize\n");
            //return;
        }

        // element-wise multiplication of the image and kernel
        cmat_mult(d_image, row_size, col_size, d_kernel, d_result, batch_size);

        // transform out of fourier space
        if (cufftExecC2C(plan, d_result, d_result, CUFFT_INVERSE) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            //return;		
        }

        // copy results from device to host
        cudaMemcpy(result, d_result, batch_size*size*sizeof(CuComplex), cudaMemcpyDeviceToHost); 

        
        // transfer to output array 
        for(int batch = 0; batch < batch_size; ++batch){
            batch_idx = batch*size;
            for(int i = 0; i < size; ++i) {
                out[2*batch_idx + 2*i] = (result[batch_idx + i].x/((double)size));
                out[2*batch_idx + 2*i + 1] = (result[batch_idx + i].y/((double)size));
            }
        }
        
        
        // free host memory
        free(linear_image);
        free(result);
        free(linear_kernel);

        // free device memory
        cufftDestroy(plan);
        cufftDestroy(plan_k);
        cudaFree(d_image);
        cudaFree(d_result);
        cudaFree(d_kernel);
                            
    }


    __global__ void conv_dud_gpu_helper(double* result, const unsigned int* image, CuComplex* kernel, int col_size, int row_size, int kernel_n, int kernel_m, int kernel_offset){

        // calculate row and column positions
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        // check bounds
        if(row >= row_size || col >= col_size) return;

        int iFlip, jFlip; // flipped kernel indices
        int ii, jj;
        float temp_x = 0, temp_y=0;

        for(int i = 0; i < kernel_m; ++i){

            iFlip = kernel_m - 1 - i;

            for(int j = 0; j < kernel_n; ++j){

                jFlip = kernel_n - 1 - j;

                ii = row + (kernel_offset - iFlip);
                jj = col + (kernel_offset - jFlip);

                if(ii >= 0 && ii < row_size && jj >= 0 && jj < col_size) {
                    temp_x += image[ii * col_size + jj] * kernel[iFlip * kernel_n + jFlip].x;
                    temp_y += image[ii * col_size + jj] * kernel[iFlip * kernel_n + jFlip].y;
                }
            }
        }

        result[2*(row * col_size + col)] = temp_x;
        result[2*(row * col_size + col)+1] = temp_y;
    }

    void conv_dud_gpu(double* out, 
            const unsigned int* image, 
            double* kernel, 
            int image_n, int image_m, int kernel_n, int kernel_m){
        
        int batch_size = 1;

        cudaError_t err;

        // calculate new size of image based on padding size
        int row_size = image_m + kernel_m - 1;
        int col_size = image_n + kernel_n - 1;
        int size = row_size * col_size;
        
        // allocate space for linear indexed arrays
        unsigned int* linear_image = (unsigned int*)malloc(size * sizeof(unsigned int));
        Complex* linear_kernel = (Complex*)malloc(kernel_n*kernel_m * sizeof(Complex));
        
        int index, index2;
        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                index = (i*col_size + j);
                if (i < image_m && j < image_n) { 
                    index2 = (i*image_n + j) ;
                    linear_image[index] = image[index2];
                } else {
                    linear_image[index] = 0.f; // add padding
                }
            }
        }
        
        for (int i = 0; i < kernel_m; ++i) {
            for (int j = 0; j < kernel_n; ++j) {
                index = (i*kernel_n + j);
                
                linear_kernel[index].x = kernel[2*index];
                linear_kernel[index].y = kernel[2*index+1];
            }
        }

        double* d_result;
        CuComplex* d_kernel;
        unsigned int* d_image;

        // allocate vectors for GPU
        cudaMalloc(&d_image, 2*size*sizeof(unsigned int));
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }

        cudaMalloc(&d_result, 2*size*sizeof(double));
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }

        cudaMalloc(&d_kernel, kernel_n*kernel_m*sizeof(CuComplex));
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }
        
        // copy data to GPU
        cudaMemcpy(d_image, linear_image, 2*size*sizeof(unsigned int), cudaMemcpyHostToDevice);
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }

        cudaMemcpy(d_kernel, linear_kernel, kernel_n*kernel_m*sizeof(CuComplex), cudaMemcpyHostToDevice);
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }

        //cout << "threads: " << threads << endl;
        int X_THREADS = 16;
        int Y_THREADS = X_THREADS;
        int X_BLOCKS = (col_size + X_THREADS - 1) / X_THREADS;
        int Y_BLOCKS = (row_size + Y_THREADS - 1) / Y_THREADS;

        //cout << X_BLOCKS << endl;
        //cout << Y_BLOCKS << endl;

        string str;

        //cin >> str;

        dim3 block_dim(X_THREADS, Y_THREADS);
        dim3 grid_dim(X_BLOCKS, Y_BLOCKS);

        int offset = kernel_n / 2.; // center of filter
        
        // call kernel
        //conv_dud_gpu_helper(double* result, const unsigned int* image, Complex* kernel, int col_size, int row_size, int kernel_n, int kernel_m, int kernel_offset){
        conv_dud_gpu_helper<<<grid_dim, block_dim>>>(d_result, d_image, d_kernel, col_size, row_size, kernel_n, kernel_m, offset);

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error at sync: %s\n", cudaGetErrorString(err));       
        }

        // copy results from device to host
        cudaMemcpy(out, d_result, 2*size*sizeof(double), cudaMemcpyDeviceToHost);
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error at memcpy: %s\n", cudaGetErrorString(err));       
        }

        // free host memory
        free(linear_image);
        
        free(linear_kernel);

        // free device memory
        cudaFree(d_image);
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }

        cudaFree(d_result);
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }

        cudaFree(d_kernel);
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }
        
    }
}


