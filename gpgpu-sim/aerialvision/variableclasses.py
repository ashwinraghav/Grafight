#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel
# and the University of British Columbia, Vancouver, 
# BC V6T 1Z4, All Rights Reserved.
# 
# THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
# TERMS AND CONDITIONS.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 
# NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
# are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
# (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
# benchmarks/template/ are derived from the CUDA SDK available from 
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
# src/intersim/ are derived from Booksim (a simulator provided with the 
# textbook "Principles and Practices of Interconnection Networks" available 
# from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA 
# which is distributed seperately by NVIDIA under separate terms and 
# conditions.  To use this version of GPGPU-Sim with OpenCL requires a
# recent version of NVIDIA's drivers which support OpenCL.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the University of British Columbia nor the names of
# its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
#  
# 5. No nonprofit user may place any restrictions on the use of this software,
# including as modified by the user, by any other authorized user.
# 
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
# Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
# Vancouver, BC V6T 1Z4


class variable:
    
    def __init__(self, type, bool):
        self.data = []
        self.type = type
        self.bool = bool
        

class bookmark:

    def __init__(self):
        self.title = ""
        self.fileChosen = []
        self.dataChosenX = []
        self.dataChosenY = []
        self.graphChosen = []
        self.dydx = []
        self.description = ""
        
lineStatName = ['count', 'latency', 'dram_traffic', 'smem_bk_conflicts', 'smem_warp', 
                'gmem_access_generated', 'gmem_warp', 'exposed_latency', 'warp_divergence']
    
class cudaLineNo:

    debug = 0
    
    def __init__(self, ptxLines, ptxStats):
        self.stats = {}
        self.ptxLines = ptxLines
        for statName in lineStatName:
            self.stats[statName] = []

        #Filling up count appropriately
        for iter in ptxStats:
            for statID in range(0, len(iter)):
                if (iter[statID] != "Null"):
                    self.stats[lineStatName[statID]].append(int(iter[statID]))
               
    def sum(self,key):
        sum = 0
        for iter in self.stats[key]:
            sum += int(iter)    
        return sum
    
    def takeMax(self,key):
        try:
            tmp = max(self.stats[key])
        except:
            tmp = 0
            if cudaLineNo.debug:
                print 'Exception in cudaLineNo.takeMax()', self.stats[key]
        return tmp
        
    def takeRatioSums(self, key1,key2):
        tmp1 = float(self.sum(key1))
        tmp2 = float(self.sum(key2))

        try:
            return tmp1/tmp2
        except:
            if cudaLineNo.debug:
                print tmp1, tmp2
            if tmp2 == 0 and cudaLineNo.debug:
                print 'infinite'
            return 0
    
        

class ptxLineNo:

    debug = 0

    def __init__(self, ptxStats):
        self.stats = {}

        for statID in range(0, len(ptxStats)):
            self.stats[lineStatName[statID]] = int(ptxStats[statID])

    def returnStat(self, key):
        return self.stats[key]
        
    def returnRatio(self, key1, key2):
        tmp1 = float(self.stats[key1])
        tmp2 = float(self.stats[key2])
        try:
            return tmp1/tmp2
        except:
            if tmp2 == 0 and ptxLineNo.debug:
                print 'infinite'
            return 0
            
    


    
    
    
    
    
    
