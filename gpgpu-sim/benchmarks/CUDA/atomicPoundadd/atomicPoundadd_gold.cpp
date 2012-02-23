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

#include <math.h>
#include <stdio.h>

#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
int computeGold( int* gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
int
computeGold(int* gpuData, const int len) 
{
    int val = 0;
    for( int i = 0; i < len; ++i) 
    {
        val += 10;
    }
    if (val != gpuData[0])
    {
        printf("atomicAdd failed\n");
        return false;
    }

    return true;
}

