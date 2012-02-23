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

/* Simple kernel demonstrating atomic functions in device code. */

#ifndef _ATOMICPOUNDADD_KERNEL_H_
#define _ATOMICPOUNDADD_KERNEL_H_

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for atomic instructions
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
atomicAddKernel(int* g_odata) 
{
  // access thread id
  //    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  
  atomicAdd(g_odata, 10);

  // Atomic addition
  //  for(unsigned int i = 0; i < 100; i++) {
    // atomicAdd(&g_odata[0], 10);
    // atomicAdd(&g_odata[1], 10);
    // atomicAdd(&g_odata[2], 10);
    // atomicAdd(&g_odata[3], 10);
    // atomicAdd(&g_odata[4], 10);
    // atomicAdd(&g_odata[5], 10);
    // atomicAdd(&g_odata[6], 10);
    // atomicAdd(&g_odata[7], 10);
    // atomicAdd(&g_odata[8], 10);
    // atomicAdd(&g_odata[9], 10);
    //  }
}

#endif // #ifndef _ATOMICPOUNDADD_KERNEL_H_
