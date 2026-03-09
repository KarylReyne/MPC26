#include "Tools.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>
// header for NVIDIA's Tensor Core programming API
#include <mma.h>

using namespace std;
using namespace nvcuda; 

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

//#define VERBOSE // Prints input matrix and Ds. Only uncomment for small matrix sizes!
#define RUN_CPU // Runs CPU code for reference (slow!!!)
// !!!!! n = m = k = N in our case !!!!!
#define N 1024 // Must be a multiple of WARP_SIZE
#define WARP_SIZE 32
#define S 16 // size of one dimension of one fragment -> we work on 16x16x16 fragments


//CPU
void multiplyMatrix(float* D, const __half* A, const __half* B, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            float result = 0.;
            for (unsigned int k = 0; k < n; k++)
            {
                result += __half2float(A[i * n + k]) * __half2float(B[k * n + j]);
            }
            D[i * n + j] = result;
        }
    }   
}

void dumpMatrix(const float* m, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cout << setw(3) << setprecision(3) << m[i * n + j] << " ";
        }
        cout << endl;
    }
}

void dumpMatrix(const __half* m, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cout << setw(3) << setprecision(3) << __half2float(m[i * n + j]) << " ";
        }
        cout << endl;
    }
}

float randF(const float min = 0.0f, const float max = 1.0f)
{
    int randI = rand();
    float randF = (float)randI / (float)RAND_MAX;
    float D = min + randF * (max - min);

    return D;
}


__global__ void TensorCoreMatrixMultiplication(__half const* A, __half const* B, float* C, uint32_t n)
{
    // ToDo: Your code goes here: implement a square matrix multiplication using tensor cores via warp matrix functions (wmma namespace)
    // compute a 16x16 tile per warp
    
    // Tile using a 2D grid.
    // Determine the warp 2D index.
    uint32_t const warpIdx_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    uint32_t const warpIdx_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Declare the fragments. 
    wmma::fragment<wmma::matrix_a, S, S, S, __half, wmma::row_major> a_frag; 
    wmma::fragment<wmma::matrix_b, S, S, S, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, S, S, S, float> acc_frag; 
    //wmma::fragment<wmma::accumulator, S, S, S, float> d_frag; // probably unnecessary? 

    // Make sure the accumulator starts from 0.
    wmma::fill_fragment(acc_frag, 0.0f); 

    for (int k = 0; k < n; k += S)
    {
        // Load the matrices into the fragments 

        // base + warpIdx_y * tile height * row length + k 
        wmma::load_matrix_sync(a_frag, A + warpIdx_y * S * n + k , n); 
        // base + k * row length + warpIdx_x * tile width
        wmma::load_matrix_sync(b_frag, B + k*n + warpIdx_x * S, n); 
    
        // Perform the multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    //store the output
    wmma::store_matrix_sync(C+ warpIdx_y * S * n + warpIdx_x * S, acc_frag, n, wmma::mem_row_major);  
}

int main(int argc, char** argv)
{
    __int64_t startTime;
    __int64_t endTime;

    // Allocate all memory: we now work with half precision matrices as the tensor cores work on half precision and output full precision
    __half* h_matrix_a = new __half[N * N];
    __half* h_matrix_b = new __half[N * N];
    float* h_matrix_d = new float[N * N];

    __half* d_matrix_a;
    __half* d_matrix_b;
    float* d_matrix_d;

    cudaMalloc(&d_matrix_a, sizeof(__half) * N * N);
    cudaMalloc(&d_matrix_b, sizeof(__half) * N * N);
    cudaMalloc(&d_matrix_d, sizeof(float) * N * N);

    // Initialize matrices and upload to CUDA
    for (unsigned int n = 0; n < N * N; n++)
    {
        float AValue = randF(-1.0, 1.0);
        float BValue = randF(-1.0, 1.0);

        h_matrix_a[n] = __float2half(AValue);
        h_matrix_b[n] = __float2half(BValue);
    }
    cudaMemcpy(d_matrix_a, h_matrix_a, sizeof(__half) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, h_matrix_b, sizeof(__half) * N * N, cudaMemcpyHostToDevice);

#ifdef VERBOSE
    cout << "Input Matrices:" << endl;
    dumpMatrix(h_matrix_a, N);
    cout << endl;
    dumpMatrix(h_matrix_b, N);
    cout << endl << endl;
#endif

#ifdef RUN_CPU
    // Calculations on CPU
    startTime = continuousTimeNs();
    multiplyMatrix(h_matrix_d, h_matrix_a, h_matrix_b, N);
    endTime = continuousTimeNs();
#ifdef VERBOSE
    cout << "CPU:" << endl;
    dumpMatrix(h_matrix_d, N);
    cout << endl;
#endif
    cout << "CPU time: " << (endTime - startTime) << "ns" << endl;
#endif

    dim3 gridDim;
    dim3 blockDim;


    // Tensor Cores are used per warp
    // each warp computes a 16x16 output tile
    // we cofigure blockDim in such a way, that each block is responsible for #4x#4 output tiles
    // -> each warp does a 16x16 output tile
    // -> each block does a 64x64 output tile
    // then we can determine how large the grid dim needs to be
    int const num_warps_x = 4;
    int const num_warps_y = 4;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    // Round up
    gridDim.x = (N + (S * num_warps_x - 1)) / (S * num_warps_x);
    gridDim.y = (N + S * num_warps_y - 1) / (S * num_warps_y);
    // Calculations on GPU: Compute matrix multiplication using CUDA WMMA.
    startTime = continuousTimeNs();
    TensorCoreMatrixMultiplication<<<gridDim, blockDim>>>(d_matrix_a, d_matrix_b, d_matrix_d, N);
    cudaDeviceSynchronize();

    endTime = continuousTimeNs();
    cudaMemcpy(h_matrix_d, d_matrix_d, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU Tensor Cores:" << endl;
    dumpMatrix(h_matrix_d, N);
    cout << endl;
#endif
    cout << "GPU Tensor Cores time: " << (endTime - startTime) << "ns" << endl;


    // Free all memory
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_d);
    delete[] h_matrix_a;
    delete[] h_matrix_b;
    delete[] h_matrix_d;

    checkCUDAError("end of program");
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
