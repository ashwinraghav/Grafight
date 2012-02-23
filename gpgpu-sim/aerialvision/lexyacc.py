#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Tor M. Aamodt, Andrew Turner, Wilson W. L.
# Fung, Ali Bakhoda and the University of British Columbia, Vancouver, 
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


import sys
sys.path.insert(0,"Lib/site-packages/ply-3.2/ply-3.2")
import ply.lex as lex
import ply.yacc as yacc
import gzip
import variableclasses as vc

def parseMe(filename):
    
    #The lexer
        
    # List of token names.   This is always required
    tokens = ['WORD', 
       'NUMBERSEQUENCE',
    ]
    
    # Regular expression rules for tokens
    
    
    def t_WORD(t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        return t
    
    def t_NUMBERSEQUENCE(t):
        r'([-]{0,1}[0-9]+([\.][0-9]+){0,1}[ ]*)+'
        return t
    
        
    t_ignore = '[\t: ]+'
    
    def t_newline(t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")
        
    def t_error(t):
        print "Illegal character '%s'" % t.value[0]
        t.lexer.skip(1) 
    
    lex.lex()    
    
    # Section 1.1 for adding a variable
    shaderInsn = vc.variable(2,0)
    globalInsn = vc.variable(1,1)
    globalCycle = vc.variable(1,1)
    shaderWarpDiv = vc.variable(2,0)
    L1ConstMiss = vc.variable(1,0)
    L1TextMiss = vc.variable(1,0)
    L1ReadMiss = vc.variable(1,0)
    L1WriteMiss = vc.variable(1,0)
    L2ReadMiss = vc.variable(1,0)
    L2WriteMiss = vc.variable(1,0)
    L2WriteHit = vc.variable(1,0)
    L2ReadHit = vc.variable(1,0)
    globalTotInsn = vc.variable(1,0)
    dramCMD = vc.variable(2,0)
    dramNOP = vc.variable(2,0)
    dramNACT = vc.variable(2,0)
    dramNPRE = vc.variable(2,0)
    dramNREQ = vc.variable(2,0)
    dramMaxMRQS = vc.variable(2,0)
    dramAveMRQS = vc.variable(2,0)
    dramUtil = vc.variable(2,0)
    dramEff = vc.variable(2,0)
    globalCompletedThreads = vc.variable(1,1)
    globalSentWrites = vc.variable(1,0)
    globalProcessedWrites = vc.variable(1,0)
    averagemflatency = vc.variable(1,0)
    STmemlatdist = vc.variable(3,0)
    LDmemlatdist = vc.variable(3,0)
    WarpDivergenceBreakdown = vc.variable(3,0)
    dram_writes_per_cycle = vc.variable(1,0)   
    dram_reads_per_cycle = vc.variable(1,0)   
    gpu_stall_by_MSHRwb = vc.variable(1,0)
    dramglobal_acc_r = vc.variable(4,0)
    dramglobal_acc_w = vc.variable(4,0)
    dramlocal_acc_r = vc.variable(4,0)
    dramlocal_acc_w = vc.variable(4,0)
    dramconst_acc_r = vc.variable(4,0)
    dramtexture_acc_r = vc.variable(4,0)
    cacheMissRate_globalL1_all = vc.variable(2,0)
    cacheMissRate_textureL1_all = vc.variable(2,0)
    cacheMissRate_constL1_all = vc.variable(2,0)
    cacheMissRate_globalL1_noMgHt = vc.variable(2,0)
    cacheMissRate_textureL1_noMgHt = vc.variable(2,0)
    cacheMissRate_constL1_noMgHt = vc.variable(2,0)
    shdrctacount = vc.variable(2,0)
    CFLOG = {}
    
    
    
    inputData = 'NULL'

        
    def p_sentence(p):
        '''sentence : WORD NUMBERSEQUENCE'''
        #print p[0], p[1],p[2]
        num = p[2].split(" ")  
        for x in num:
            try:
                float(x)
            except:
                num.remove(x)
                
        #Section 1.2 for adding a variable        
        if p[1].lower() == "shaderinsncount":
            for x in num:
                shaderInsn.data.append(int(x))
            shaderInsn.data.append("NULL")

        elif p[1].lower() == 'cachemissrate_globallocall1_all':
            for x in num:
                cacheMissRate_globalL1_all.data.append(float(x))
            cacheMissRate_globalL1_all.data.append("NULL")         
        elif p[1].lower() == 'cachemissrate_texturel1_all':
            for x in num:
                cacheMissRate_textureL1_all.data.append(float(x))
            cacheMissRate_textureL1_all.data.append("NULL")        
        elif p[1].lower() == 'cachemissrate_constl1_all':
            for x in num:
                cacheMissRate_constL1_all.data.append(float(x))
            cacheMissRate_constL1_all.data.append("NULL")
            
        elif p[1].lower() == 'cachemissrate_globallocall1_nomght':
            for x in num:
                cacheMissRate_globalL1_noMgHt.data.append(float(x))
            cacheMissRate_globalL1_noMgHt.data.append("NULL")
            
        elif p[1].lower() == 'cachemissrate_texturel1_nomght':
            for x in num:
                cacheMissRate_textureL1_noMgHt.data.append(float(x))
            cacheMissRate_textureL1_noMgHt.data.append("NULL")
            
        elif p[1].lower() == 'cachemissrate_constl1_nomght':
            for x in num:
                cacheMissRate_constL1_noMgHt.data.append(float(x))
            cacheMissRate_constL1_noMgHt.data.append("NULL")
            
        elif p[1].lower() == 'shdrctacount':
            for x in num:
                shdrctacount.data.append(int(x))
            shdrctacount.data.append("NULL")
            
        
        
        elif p[1].lower() == "globalinsncount":
            for x in num:
                globalInsn.data.append(int(x))
                #globalInsn.append("NULL")
                #print globalInsn

        elif p[1].lower() == "globalcyclecount":
            for x in num:
                globalCycle.data.append(int(x))
                if int(x) % 10000 == 0:
                    print "Processing cycle %s" % x 
                #globalCycle.append("NULL")
                #print globalCycle
        elif p[1].lower() == "shaderwarpdiv":
            for x in num:
                shaderWarpDiv.data.append(int(x))
            shaderWarpDiv.data.append("NULL")
        elif p[1].lower() == "loneconstmiss":
            for x in num:
                L1ConstMiss.data.append(int(x))
        elif p[1].lower() == 'lonetexturemiss':
            for x in num:
                L1TextMiss.data.append(int(x))
        elif p[1].lower() == 'lonereadmiss':
            for x in num:
                L1ReadMiss.data.append(int(x))
        elif p[1].lower() == 'lonewritemiss':
            for x in num:
                L1WriteMiss.data.append(int(x))
        elif p[1].lower() == 'ltwowritemiss':
            for x in num:
                L2WriteMiss.data.append(int(x))
        elif p[1].lower() == 'ltwowritehit':
            for x in num:
                L2WriteHit.data.append(int(x))
        elif p[1].lower() == 'ltworeadmiss':
            for x in num:
                L2ReadMiss.data.append(int(x))
        elif p[1].lower() == 'ltworeadhit':
            for x in num:
                L2ReadHit.data.append(int(x))       
        elif p[1].lower() == "globaltotinsncount":
            for x in num:
                globalTotInsn.data.append(int(x))
        elif p[1].lower() == "dramncmd":
            for x in num:
                dramCMD.data.append(int(x))
            dramCMD.data.append('NULL')
        elif p[1].lower() == "dramnop":
            for x in num:
                dramNOP.data.append(int(x))
            dramNOP.data.append('NULL')  
        elif p[1].lower() == "dramnact":
            for x in num:
                dramNACT.data.append(int(x))
            dramNACT.data.append('NULL')  
        elif p[1].lower() == "dramnpre":
            for x in num:
                dramNPRE.data.append(int(x))
            dramNPRE.data.append('NULL') 
        elif p[1].lower() == "dramnreq":
            for x in num:
                dramNREQ.data.append(int(x))
            dramNREQ.data.append('NULL')  
        elif p[1].lower() == "drammaxmrqs":
            for x in num:
                dramMaxMRQS.data.append(int(x))
            dramMaxMRQS.data.append('NULL')  
        elif p[1].lower() == "dramavemrqs":
            for x in num:
                dramAveMRQS.data.append(int(x))
            dramAveMRQS.data.append('NULL')
        elif p[1].lower() == "dramutil":
            for x in num:
                dramUtil.data.append(int(x))
            dramUtil.data.append('NULL')
        elif p[1].lower() == "drameff":
            for x in num:
                dramEff.data.append(int(x))
            dramEff.data.append('NULL')
        elif p[1].lower() == 'gpucompletedthreads':
            for x in num:
                globalCompletedThreads.data.append(int(x))
        elif p[1].lower() == 'gpgpunsentwrites':
            for x in num:
                globalSentWrites.data.append(int(x))
        elif p[1].lower() == 'gpgpunprocessedwrites':
            for x in num:
                globalProcessedWrites.data.append(int(x))
        elif p[1].lower() == 'averagemflatency':
            for x in num:
                averagemflatency.data.append(int(x))
        elif p[1].lower() == 'ldmemlatdist':
            for x in num:
                LDmemlatdist.data.append(int(x))
            LDmemlatdist.data.append('NULL')
        elif p[1].lower() == 'stmemlatdist':
            for x in num:
                STmemlatdist.data.append(int(x))
            STmemlatdist.data.append('NULL')
        elif p[1].lower() == 'warpdivergencebreakdown':
            for x in num:
                WarpDivergenceBreakdown.data.append(int(x))
            WarpDivergenceBreakdown.data.append('NULL') 
        elif p[1].lower() == "dram_writes_per_cycle":
            for x in num:
                dram_writes_per_cycle.data.append(float(x))
        elif p[1].lower() == "dram_reads_per_cycle":
            for x in num:
                dram_reads_per_cycle.data.append(float(x))
        elif p[1].lower() == "gpu_stall_by_mshrwb":
            for x in num:
                gpu_stall_by_MSHRwb.data.append(int(x))

        elif p[1].lower() == 'dramglobal_acc_r':
            for x in num:
                dramglobal_acc_r.data.append(int(x))
            dramglobal_acc_r.data.append('NULL')
        elif p[1].lower() == 'dramglobal_acc_w':
            for x in num:
                dramglobal_acc_w.data.append(int(x))
            dramglobal_acc_w.data.append('NULL')
        elif p[1].lower() == 'dramlocal_acc_r':
            for x in num:
                dramlocal_acc_r.data.append(int(x))
            dramlocal_acc_r.data.append('NULL')
        elif p[1].lower() == 'dramlocal_acc_w':
            for x in num:
                dramlocal_acc_w.data.append(int(x))
            dramlocal_acc_w.data.append('NULL')
        elif p[1].lower() == 'dramconst_acc_r':
            for x in num:
                dramconst_acc_r.data.append(int(x))
            dramconst_acc_r.data.append('NULL')
        elif p[1].lower() == 'dramtexture_acc_r':
            for x in num:
               dramtexture_acc_r.data.append(int(x))
            dramtexture_acc_r.data.append('NULL')
        elif p[1].lower()[0:5] == 'cflog':
            count = 0
            pc = []
            threadcount = []
            for x in num:
                if (count % 2) == 0:
                    pc.append(int(x))
                else:
                    threadcount.append(int(x))
                count += 1

            if (p[1] not in CFLOG):
                CFLOG[p[1]] = vc.variable(2,0)
                CFLOG[p[1]].data.append([]) # pc[]
                CFLOG[p[1]].data.append([]) # threadcount[]
                CFLOG[p[1]].maxPC = 0

            CFLOG[p[1]].data[0].append(pc)
            CFLOG[p[1]].data[1].append(threadcount)
            MaxPC = max(pc)
            CFLOG[p[1]].maxPC = max(MaxPC, CFLOG[p[1]].maxPC)
        
        else:
            pass
        


    def p_error(p):
        if p:
            print("Syntax error at '%s'" % p.value)
        else:
            print("Syntax error at EOF")
    
    yacc.yacc()
   
    # detect for gzip'ed log file and gunzip on the fly
    if (filename.endswith('.gz')):
        file = gzip.open(filename, 'r')
    else:
        file = open(filename, 'r')
    while file:
        line = file.readline()
        if not line : break
        yacc.parse(line[0:-1])
    file.close()    
    


    #Section 1.3 for adding a variable
    variables = {'shaderInsn':shaderInsn, 'globalInsn':globalInsn, 'globalCycle':globalCycle, 'shaderWarpDiv':shaderWarpDiv, 'L1TextMiss':L1TextMiss, 'L1ConstMiss':L1ConstMiss,
    'L1ReadMiss':L1ReadMiss,'L1WriteMiss':L1WriteMiss, 'L2ReadMiss':L2ReadMiss,'L2WriteMiss':L2WriteMiss,'L2WriteHit':L2WriteHit,'L2ReadHit':L2ReadHit,'globalTotInsn':globalTotInsn, 'dramCMD':dramCMD,
    'dramNOP':dramNOP,'dramNACT':dramNACT,'dramNPRE':dramNPRE,'dramNREQ':dramNREQ,'dramMaxMRQS':dramMaxMRQS,'dramAveMRQS':dramAveMRQS,'dramUtil':dramUtil,'dramEff':dramEff, 'globalCompletedThreads':globalCompletedThreads,
    'globalSentWrites':globalSentWrites, 'globalProcessedWrites':globalProcessedWrites, 'averagemflatency' : averagemflatency, 'LDmemlatdist': LDmemlatdist, 'STmemlatdist':STmemlatdist, 'WarpDivergenceBreakdown':WarpDivergenceBreakdown, 'dram_writes_per_cycle':dram_writes_per_cycle,
    'dram_reads_per_cycle':dram_reads_per_cycle,'gpu_stall_by_MSHRwb':gpu_stall_by_MSHRwb, 'dramglobal_acc_r' : dramglobal_acc_r, 'dramglobal_acc_w' : dramglobal_acc_w, 'dramlocal_acc_r' : dramlocal_acc_r,
    'dramlocal_acc_w' : dramlocal_acc_w, 'dramconst_acc_r':dramconst_acc_r, 'dramtexture_acc_r':dramtexture_acc_r, 'cacheMissRate_globalL1_all':cacheMissRate_globalL1_all,'cacheMissRate_textureL1_all': cacheMissRate_textureL1_all,
    'cacheMissRate_constL1_all':cacheMissRate_constL1_all,'cacheMissRate_globalL1_noMgHt':cacheMissRate_globalL1_noMgHt,'cacheMissRate_textureL1_noMgHt':cacheMissRate_textureL1_noMgHt,'cacheMissRate_constL1_noMgHt':cacheMissRate_constL1_noMgHt,
    'CFLOG' : CFLOG, 'shdrctacount': shdrctacount}
    
    
    return variables
  



