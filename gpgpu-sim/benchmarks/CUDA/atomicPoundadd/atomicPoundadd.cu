/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A simple program demonstrating trivial use of global memory atomic 
   device functions (atomic*() functions).
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cutil_inline.h"

// includes, kernels
#include "atomicPoundadd_kernel.cu"

static char *sSDKsample = "[atomicPoundadd]\0";

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
int computeGold( int* gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( dev = cutGetMaxGflopsDeviceId() );

    cutilSafeCall( cudaChooseDevice(&dev, &deviceProp) );
    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, dev) );

    if(deviceProp.major > 1 || deviceProp.minor > 0)
    {
        printf("Using Device %d: \"%s\"\n", dev, deviceProp.name);
    }
    else
    {
        printf("There is no device supporting CUDA compute capability 1.1.\n");
        printf("TEST PASSED");
        cudaThreadExit();
        cutilExit(argc, argv);
    }

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    int numThreads = 256;
    unsigned int numBlocks = 64;
    unsigned int numData = 1;
    unsigned int memSize = sizeof(int) * numData;

    if (cutCheckCmdLineFlag(argc, (const char **) argv, "n")) {
	cutGetCmdLineArgumenti( argc, (const char**) argv, "n", &numThreads); }

    //allocate mem for the result on host side
    int *h_odata = (int *)malloc(memSize);

    //initalize the memory
    for(unsigned int i = 0; i < numData; i++)
        h_odata[i] = 0;

    //To make the AND and XOR tests generate something other than 0...
    //h_odata[16] = h_odata[20] = 0xff; 

    // allocate device memory for result
    int *d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, memSize));
    // copy host memory to device to initialize to zers
    cutilSafeCall( cudaMemcpy( d_odata, h_odata, memSize, cudaMemcpyHostToDevice) );

    // //create high-precision timer for each operation
    // double gpuTime;
    // unsigned int timer = 0;
    // cutilCheckError( cutCreateTimer( &timer));

    // Create events to time kernel execution
    float exec_time;
    cudaEvent_t start, stop;
    int eventflags = cudaEventBlockingSync;

    cutilSafeCall( cudaEventCreateWithFlags(&start, eventflags) );
    cutilSafeCall( cudaEventCreateWithFlags(&stop, eventflags) );

    // // Atomic Add kernel
    // cutilCheckError( cutResetTimer( timer));
    // cutilCheckError( cutStartTimer( timer));
    cudaEventRecord(start, 0);
    atomicAddKernel<<<numBlocks, numThreads>>>(d_odata);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cutilSafeCall(cudaEventElapsedTime(&exec_time, start, stop));
    cutilCheckMsg("Kernel execution failed");
    // gpuTime = cutGetTimerValue(timer);
    printf( "Atomic Add time: %f (ms)\n", exec_time);
    //Copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, memSize, cudaMemcpyDeviceToHost) );

    // cutilCheckError( cutStopTimer( timer));
    // cutilCheckError( cutDeleteTimer( timer));

    	printf("\n%s - Test Summary\n", sSDKsample);

    //compute reference solution
    if(computeGold(h_odata, numThreads * numBlocks))
        printf("PASSED\n");
    else
        printf("FAILED\n");

    // cleanup memory
    free(h_odata);
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();
}
