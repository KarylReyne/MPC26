// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009-2010
//
//   Ulm University
//
// Creator: Hendrik Lensch
// Email:   {hendrik.lensch,johannes.hanika}@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "PPM.hh"

using namespace std;
using namespace ppm;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_THREADS 128

//-------------------------------------------------------------------------------

__device__ float getDistance(const float& _a, const float& _b)
{
    return abs(_a-_b);
}

/* compute gamma correction on the float image _src of resolution dim,
 outputs the gamma corrected image should be stored in_dst[blockIdx.x *
 blockDim.x + threadIdx.x]. Each thread computes on pixel element.
 */
__global__ void differenceKernel(float* _dst, const float* _src_a, const float* _src_b, int _w)
{
    int x = blockIdx.x * MAX_THREADS + threadIdx.x;
    int y = blockIdx.y;
    int pos = y * _w + x;

    if (x < _w)
    {
        _dst[pos] = getDistance(_src_a[pos], _src_b[pos]);
    }
}

//-------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    int acount = 1; // parse command line

    if (argc < 4)
    {
        printf("usage: %s <inImgA> <inImgB> <outImg>\n", argv[0]);
        exit(1);
    }

    float* img_a;
    float* img_b;

    bool success = true;
    int w, h;
    success &= readPPM(argv[acount++], w, h, &img_a);
    success &= readPPM(argv[acount++], w, h, &img_b);
    if (!success) {
        exit(1);
    }

    int nPix = w * h;

    float* gpuImgA;
    float* gpuImgB;
    float* gpuDiffImg;

    //-------------------------------------------------------------------------------
    printf("Executing the GPU Version\n");
    // copy the image to the device
    cudaMalloc((void**)&gpuImgA, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuImgB, nPix * 3 * sizeof(float));
    cudaMalloc((void**)&gpuDiffImg, nPix * 3 * sizeof(float));
    cudaMemcpy(gpuImgA, img_a, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuImgB, img_b, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // calculate the block dimensions
    dim3 threadBlock(MAX_THREADS);
    // select the number of blocks vertically (*3 because of RGB)
    dim3 blockGrid((w * 3) / MAX_THREADS + 1, h, 1);
    printf("bl/thr: %d  %d %d\n", blockGrid.x, blockGrid.y, threadBlock.x);

    differenceKernel<<<blockGrid, threadBlock>>>(gpuDiffImg, gpuImgA, gpuImgB, w * 3);

    // download result
    cudaMemcpy(img_b, gpuDiffImg, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpuImgA);
    cudaFree(gpuImgB);
    cudaFree(gpuDiffImg);

    writePPM(argv[acount++], w, h, (float*)img_b);

    delete[] img_a;
    delete[] img_b;

    checkCUDAError("end of program");

    printf("  done\n");
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
