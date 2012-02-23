#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Tor M. Aamodt, Wilson W. L. Fung
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


import lexyacctexteditor
import variableclasses as vc

global convertCFLog2CUDAsrc
global skipCFLog

convertCFLog2CUDAsrc = 0
skipCFLog = 1 

def organizedata(fileVars):

    print "Organizing data into internal format...";
    
    #Section 2.1 for adding a variable
    if fileVars.has_key('shaderInsn'):
        fileVars['shaderInsn'].data = nullOrganizedShader(fileVars['shaderInsn'].data)
    
    if fileVars.has_key('shdrctacount'):
        fileVars['shdrctacount'].data = nullOrganizedShader(fileVars['shdrctacount'].data)
    
    if fileVars.has_key('cacheMissRate_globalL1_all'):
        fileVars['cacheMissRate_globalL1_all'].data = nullOrganizedShaderv2(fileVars['cacheMissRate_globalL1_all'].data)
    
    if fileVars.has_key('cacheMissRate_textureL1_all'):
        fileVars['cacheMissRate_textureL1_all'].data = nullOrganizedShaderv2(fileVars['cacheMissRate_textureL1_all'].data)
    
    if fileVars.has_key('cacheMissRate_constL1_all'):
        fileVars['cacheMissRate_constL1_all'].data = nullOrganizedShaderv2(fileVars['cacheMissRate_constL1_all'].data)
    
    if fileVars.has_key('cacheMissRate_globalL1_noMgHt'):
        fileVars['cacheMissRate_globalL1_noMgHt'].data = nullOrganizedShaderv2(fileVars['cacheMissRate_globalL1_noMgHt'].data)
    
    if fileVars.has_key('cacheMissRate_textureL1_noMgHt'):
        fileVars['cacheMissRate_textureL1_noMgHt'].data = nullOrganizedShaderv2(fileVars['cacheMissRate_textureL1_noMgHt'].data)
    
    if fileVars.has_key('cacheMissRate_constL1_noMgHt'):
        fileVars['cacheMissRate_constL1_noMgHt'].data = nullOrganizedShaderv2(fileVars['cacheMissRate_constL1_noMgHt'].data)

    if fileVars.has_key('dram_writes_per_cycle'):
        fileVars['dram_writes_per_cycle'].data = [0] + [float(x) for x in fileVars['dram_writes_per_cycle'].data]

    if fileVars.has_key('dram_reads_per_cycle'):
        fileVars['dram_reads_per_cycle'].data = [0] + [float(x) for x in fileVars['dram_reads_per_cycle'].data]

    if fileVars.has_key('globalInsn'):
        fileVars['globalInsn'].data = [0] + [int(x) for x in fileVars['globalInsn'].data]
    
    if fileVars.has_key('globalCycle'):            
        fileVars['globalCycle'].data = [0] + [int(x) for x in fileVars['globalCycle'].data]
  
    if fileVars.has_key('L1ReadMiss'):
        fileVars['L1ReadMiss'].data = [0] + [int(x) for x in fileVars['L1ReadMiss'].data]

    if fileVars.has_key('L1TextMiss'):
        fileVars['L1TextMiss'].data = [0] + fileVars['L1TextMiss'].data
    
    if fileVars.has_key('L1ConstMiss'):
        fileVars['L1ConstMiss'].data = [0] + fileVars['L1ConstMiss'].data
    
    if fileVars.has_key('shaderWarpDiv'):
        fileVars['shaderWarpDiv'].data = nullOrganizedShader(fileVars['shaderWarpDiv'].data)
    
    if fileVars.has_key('globalTotInsn'):
        fileVars['globalTotInsn'].data = fileVars['globalTotInsn'].data

    if fileVars.has_key('STmemlatdist'):
        fileVars['STmemlatdist'].data = nullOrganizedShader(fileVars['STmemlatdist'].data)
        
    if fileVars.has_key('LDmemlatdist'):
        fileVars['LDmemlatdist'].data = nullOrganizedShader(fileVars['LDmemlatdist'].data)    

    if fileVars.has_key('WarpDivergenceBreakdown'):
        fileVars['WarpDivergenceBreakdown'].data = nullOrganizedShader(fileVars['WarpDivergenceBreakdown'].data)    

    if fileVars.has_key('dramCMD'):
        fileVars['dramCMD'].data = nullOrganizedDram(fileVars['dramCMD'].data)
    
    if fileVars.has_key('dramNOP'):
        fileVars['dramNOP'].data = nullOrganizedDram(fileVars['dramNOP'].data)
    
    if fileVars.has_key('dramNACT'):
        fileVars['dramNACT'].data = nullOrganizedDram(fileVars['dramNACT'].data)

    if fileVars.has_key('dramNPRE'):            
        fileVars['dramNPRE'].data = nullOrganizedDram(fileVars['dramNPRE'].data)
    
    if fileVars.has_key('dramNREQ'):
        fileVars['dramNREQ'].data = nullOrganizedDram(fileVars['dramNREQ'].data)
    
    if fileVars.has_key('dramAveMRQS'):
        fileVars['dramAveMRQS'].data = nullOrganizedDram(fileVars['dramAveMRQS'].data)

    if fileVars.has_key('dramUtil'):
        fileVars['dramUtil'].data = nullOrganizedDram(fileVars['dramUtil'].data)
    
    if fileVars.has_key('dramEff'):
        fileVars['dramEff'].data = nullOrganizedDram(fileVars['dramEff'].data)
    
    if fileVars.has_key('dramglobal_acc_r'):
        fileVars['dramglobal_acc_r'].data  = nullOrganizedDramV2(fileVars['dramglobal_acc_r'].data)
    
    if fileVars.has_key('dramglobal_acc_w'):
        fileVars['dramglobal_acc_w'].data  = nullOrganizedDramV2(fileVars['dramglobal_acc_w'].data)
    
    if fileVars.has_key('dramlocal_acc_r'):
        fileVars['dramlocal_acc_r'].data  = nullOrganizedDramV2(fileVars['dramlocal_acc_r'].data)
    
    if fileVars.has_key('dramlocal_acc_w'):
        fileVars['dramlocal_acc_w'].data  = nullOrganizedDramV2(fileVars['dramlocal_acc_w'].data)
    
    if fileVars.has_key('dramconst_acc_r'):
        fileVars['dramconst_acc_r'].data  = nullOrganizedDramV2(fileVars['dramconst_acc_r'].data)
    
    if fileVars.has_key('dramtexture_acc_r'):
        fileVars['dramtexture_acc_r'].data  = nullOrganizedDramV2(fileVars['dramtexture_acc_r'].data)
            
    if fileVars.has_key('globalCompletedThreads'):
        fileVars['globalCompletedThreads'].data = [0] + fileVars['globalCompletedThreads'].data
    
    if fileVars.has_key('globalSentWrites'):
        fileVars['globalSentWrites'].data = [0] + fileVars['globalSentWrites'].data
    
    if fileVars.has_key('globalProcessedWrites'):
        fileVars['globalProcessedWrites'].data = [0] + fileVars['globalProcessedWrites'].data
    
    if fileVars.has_key('averagemflatency'):
        zeros = []
        for count in range(len(fileVars['averagemflatency'].data),len(fileVars['globalCycle'].data)):
            zeros.append(0)
        fileVars['averagemflatency'].data = zeros + fileVars['averagemflatency'].data
    if fileVars.has_key('gpu_stall_by_MSHRwb'):
        fileVars['gpu_stall_by_MSHRwb'].data = [0] + fileVars['gpu_stall_by_MSHRwb'].data
    
    if (skipCFLog == 0) and fileVars.has_key('CFLOG'):
        loadfile = open('recentfiles.txt', 'r')
        bool = 0
        while loadfile:
            line = loadfile.readline()
            if not line: break
            if '.ptx' in line:
                ptxFile = line
                bool += 1
            if 'gpgpu_inst_stats' in line:
                statFile = line
                bool += 1
            if bool == 2:
                break
        
        print "PC Histogram to CUDA Src = %d" % convertCFLog2CUDAsrc
        parseCFLOGCUDA = convertCFLog2CUDAsrc

        if parseCFLOGCUDA == 1:
            map = lexyacctexteditor.ptxToCudaMapping(ptxFile.rstrip())
            maxStats = max(lexyacctexteditor.textEditorParseMe(statFile.rstrip()).keys())

        if parseCFLOGCUDA == 1:
            newMap = {}
            for lines in map:
                for ptxLines in map[lines]:
                    newMap[ptxLines] = lines
            
            markForDel = []
            for ptxLines in newMap:
                if ptxLines > maxStats:
                    markForDel.append(ptxLines)
            for lines in markForDel:
                del newMap[lines]
    

        
        fileVars['CFLOGglobalPTX'] = vc.variable(2,0)
        fileVars['CFLOGglobalCUDA'] = vc.variable(2,0)
        
        count = 0
        for iter in fileVars['CFLOG']:

            print "Organizing data for %s" % iter

            fileVars[iter + 'PTX'] = fileVars['CFLOG'][iter]
            fileVars[iter + 'PTX'].data = CFLOGOrganizePTX(fileVars['CFLOG'][iter].data, fileVars['CFLOG'][iter].maxPC)
            if parseCFLOGCUDA == 1:
                fileVars[iter + 'CUDA'] = vc.variable(2,0)
                fileVars[iter + 'CUDA'].data = CFLOGOrganizeCuda(fileVars[iter + 'PTX'].data, newMap)

            try:
                if count == 0:
                    fileVars['globalPTX'] = fileVars[iter + 'PTX']
                    if parseCFLOGCUDA == 1:
                        fileVars['globalCUDA'] = fileVars[iter + 'CUDA']
                else:
                    for rows in range(0, len(fileVars[iter + 'PTX'].data)):
                        for columns in range(0, len(fileVars[iter + 'PTX'].data[rows])):
                            fileVars['globalPTX'].data[rows][columns] += fileVars[iter + 'PTX'].data[rows][columns]
                    if parseCFLOGCUDA == 1:
                        for rows in range(0, len(fileVars[iter + 'CUDA'].data)):
                            for columns in range(0, len(fileVars[iter + 'CUDA'].data[rows])): 
                                fileVars['globalCUDA'].data[rows][columns] += fileVars[iter + 'CUDA'].data[rows][columns]
            except:
                print "Error in generating globalCFLog data"

            count += 1
        del fileVars['CFLOG']


    return fileVars


def nullOrganizedShader(nullVar):
    #need to organize this array into usable information
    count = 0
    organized = []
    
    #determining how many shader cores are present
    for x in nullVar:
        if x != 'NULL':
            count += 1
        else:
            numPlots = count
            break
    count = 0
    
    #initializing 2D list
    for x in range(0, numPlots):
        organized.append([])
    
    #filling up list appropriately
    for x in range(0,(len(nullVar))):
        if nullVar[x] == 'NULL':
            count=0
        else:
            organized[count].append(int(nullVar[x]))
            count +=  1

    for x in range(0,len(organized)):
        organized[x] = [0] + organized[x]
    
    return organized

def nullOrganizedShaderv2(nullVar):
    #need to organize this array into usable information
    count = 0
    organized = []
    
    #determining how many shader cores are present
    for x in nullVar:
        if x != 'NULL':
            count += 1
        else:
            numPlots = count
            break
    count = 0
    
    #initializing 2D list
    for x in range(0, numPlots):
        organized.append([])
    
    #filling up list appropriately
    for x in range(0,(len(nullVar))):
        if nullVar[x] == 'NULL':
            count=0
        else:
            organized[count].append(float(nullVar[x]))
            count +=  1

    for x in range(0,len(organized)):
        organized[x] = [0] + organized[x]
    
    return organized


def nullOrganizedDram(nullVar):
    organized = [[0]]
    mem = 1
    for iter in nullVar:
        if iter == 'NULL':
            mem = 1
            continue
        elif mem == 1:
            memNum = iter
            mem = 0
            continue
        else:
            try:
                organized[memNum].append(iter)
            except:
                organized.append([0])
                organized[memNum].append(iter)
    return organized

def nullOrganizedDramV2(nullVar):
    organized = {}
    mem = 1
    for iter in nullVar:
        if iter == 'NULL':
            mem = 1
            continue
        elif mem == 1:
            ChipNum = iter
            mem += 1
            continue
        elif mem == 2:
            BankNum = iter
            mem = 0
            continue
        else:
            try:
                key = str(ChipNum) + '.' + str(BankNum)
                organized[key].append(iter)
            except:
                organized[key] = [0]
                organized[key].append(iter)

    return organized

def CFLOGOrganizePTX(list, maxPC):
    count = 0
    
    organizedThreadCount = list[1]
    organizedPC = list[0]

    nCycles = len(organizedPC)
    final = [[0 for cycle in range(nCycles)] for pc in range(maxPC + 1)] # fill the 2D array with zeros

    for cycle in range(0, nCycles):
        pcList = organizedPC[cycle]
        threadCountList = organizedThreadCount[cycle] 
        for n in range(0, len(pcList)):
            final[pcList[n]][cycle] = threadCountList[n]
    
    return final
    
def CFLOGOrganizeCuda(list, map):
    #We need to aggregate lines of PTX together
    cudaMaxLineNo = max(map.keys())
    tmp = {}
    #need to fill up the final matrix appropriately
    

    for lines in map:
        if tmp.has_key(map[lines]):
            pass
        else:
            tmp[map[lines]] = []
            for lengthData in range(0, len(list[0])):
                tmp[map[lines]].append(0)
                



    for lines in tmp:
        for lengthData in range(0, len(list[0])):
            for ptxLines in map:
                if map[ptxLines] == lines:
                    tmp[lines][lengthData] += list[ptxLines][lengthData]
    
    final = []           
    for iter in range(0,max(tmp.keys())):
        if tmp.has_key(iter):
            final.append(tmp[iter])            
        else:
            final.append([])
            for lengthData in range(0, len(list[0])):
                final[-1].append(0)

    return final
                

                #final[lines][lengthData] += 0
                #list[ptxLines][lengthData] += 0

    #print final
            
        
        
    
        
                

#def stackedBar(nullVar):
#    #Need to initialize organize ar
#    organized = [[]]
#    for iter in nullVar:
#        if iter != 'NULL':
#            organized[-1].append(iter)
#        else:
#            organized.append([])
#    organized.remove([])
#    return organized


