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

#ifndef _STDPOUNDADD_KERNEL_H_
#define _STDPOUNDADD_KERNEL_H_

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for atomic instructions
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
addKernel(int* g_odata) 
{
  int old;

  old = *g_odata;
  old += 10;
  *g_odata = old;

  //for(unsigned int i = 0; i < 100; i++) {
  // old = g_odata[0]; // does compiler optimize this away?
  // old += 10;
  // g_odata[0] = old;

  // old = g_odata[1]; // does compiler optimize this away?
  // old += 10;
  // g_odata[1] = old;

  // old = g_odata[2]; // does compiler optimize this away?
  // old += 10;
  // g_odata[2] = old;

  // old = g_odata[3]; // does compiler optimize this away?
  // old += 10;
  // g_odata[3] = old;

  // old = g_odata[4]; // does compiler optimize this away?
  // old += 10;
  // g_odata[4] = old;

  // old = g_odata[5]; // does compiler optimize this away?
  // old += 10;
  // g_odata[5] = old;

  // old = g_odata[6]; // does compiler optimize this away?
  // old += 10;
  // g_odata[6] = old;

  // old = g_odata[7]; // does compiler optimize this away?
  // old += 10;
  // g_odata[7] = old;

  // old = g_odata[8]; // does compiler optimize this away?
  // old += 10;
  // g_odata[8] = old;

  // old = g_odata[9]; // does compiler optimize this away?
  // old += 10;
  // g_odata[9] = old;
  //}
}

#endif // #ifndef _STDPOUNDADD_KERNEL_H_
