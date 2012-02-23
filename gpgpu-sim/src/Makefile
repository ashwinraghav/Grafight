# Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
# George L. Yuan, Ivan Sham and the 
# University of British Columbia
# Vancouver, BC  V6T 1Z4
# All Rights Reserved.
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
# (property of NVIDIA).  The files benchmarks/BlackScholes/* and 
# benchmarks/template/* are derived from the CUDA SDK available from 
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files 
# src/gpgpusim_entrypoint.c and src/simplesim-3.0/* are derived from the 
# SimpleScalar Toolset available from http://www.simplescalar.com/ 
# (property of SimpleScalar LLC) and the files src/intersim/* are derived 
# from Booksim (Simulator provided with the textbook "Principles and 
# Practices of Interconnection Networks" available from 
# http://cva.stanford.edu/books/ppin/).  As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA 
# version 1.1, which is distributed seperately by NVIDIA under separate terms 
# and conditions.
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

# GPGPU-Sim Makefile

DEBUG=0
OPTFLAGS = -g3 -fPIC
ifneq ($(DEBUG),1)
	OPTFLAGS += -O3
endif

CPP = g++
CC = gcc

OFLAGS =  $(OPTFLAGS) -Wall
MAKE = make
RM = rm -f
LN = ln -s
OEXT = o
CS = ;
X=/

GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')


CFLAGS = -DDEBUG $(OFLAGS) 
CXXFLAGS = $(CFLAGS)

ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS += -std=c++0x
endif

GPUOBJS = gpgpu-sim/shader.$(OEXT) gpgpu-sim/gpu-cache.$(OEXT) gpgpu-sim/delayqueue.$(OEXT) \
	  gpgpu-sim/gpu-sim.$(OEXT) gpgpu-sim/dram.$(OEXT) \
          gpgpu-sim/stack.$(OEXT) gpgpu-sim/addrdec.$(OEXT) \
	  gpgpu-sim/gpu-misc.$(OEXT)  \
	  gpgpu-sim/icnt_wrapper.$(OEXT) 

GPGPU_CPP_OBJS = gpgpu-sim/dram_sched.$(OEXT)  \
	gpgpu-sim/dwf.$(OEXT) gpgpu-sim/warp_tracker.$(OEXT) \
	gpgpu-sim/visualizer.$(OEXT) option_parser.$(OEXT) \
        gpgpu-sim/stat-tool.$(OEXT)
 
all: $(PROGS) libgpgpusim.a 
	make  "CREATELIBRARY=1" "DEBUG=$(DEBUG)" -C ./intersim
	make  -C ./cuda-sim
	make libgpgpusim.a
	@echo "my work is done here..."

libgpgpusim.a:	$(GPGPU_CPP_OBJS) gpgpusim_entrypoint.$(OEXT)  $(GPUOBJS) $(SS_OBJS) intersim/libintersim.a cuda-sim/libgpgpu_ptx_sim.a
	make  "CREATELIBRARY=1" "DEBUG=$(DEBUG)" -C ./intersim
	make    -C ./cuda-sim
	ar rcs  libgpgpusim.a $(GPGPU_CPP_OBJS) gpgpusim_entrypoint.$(OEXT)  $(GPUOBJS) $(SS_OBJS) intersim/libintersim.a cuda-sim/libgpgpu_ptx_sim.a

intersim/libintersim.a:
	make  "CREATELIBRARY=1" "DEBUG=$(DEBUG)" -C ./intersim 
cuda-sim/libgpgpu_ptx_sim.a:
	make   -C ./cuda-sim
.c.$(OEXT):
	$(CC) $(CFLAGS) -o $*.$(OEXT) -c $*.c

.cc.$(OEXT):
	$(CPP) $(CXXFLAGS) -o $*.$(OEXT) -c $*.cc

clean:
	rm -f *.o *.obj *.exe core *~ MAKE.log Makefile.bak $(PROGS) *.a *.so
	cd gpgpu-sim; rm *.o; cd ..
	cd intersim $(CS) $(MAKE) "RM=rm -f" "CS=$(CS)" clean $(CS) cd ..
	rm -rf simplesim-3.0/*.o
	make clean -C ./cuda-sim

option_parser.$(OEXT): option_parser.h

gpgpu-sim/gpu-cache.$(OEXT): gpgpu-sim/gpu-cache.h 
gpgpu-sim/shader.$(OEXT): gpgpu-sim/shader.h gpgpu-sim/gpu-sim.h gpgpu-sim/gpu-cache.h gpgpu-sim/stack.h gpgpu-sim/addrdec.h  gpgpu-sim/cflogger.h gpgpu-sim/histogram.h gpgpu-sim/dwf.h gpgpu-sim/warp_tracker.h util.h
gpgpu-sim/delayqueue.$(OEXT): gpgpu-sim/delayqueue.h
gpgpu-sim/gpu-sim.$(OEXT): gpgpu-sim/gpu-sim.h gpgpu-sim/shader.h gpgpu-sim/addrdec.h gpgpu-sim/gpu-misc.h gpgpu-sim/cflogger.h gpgpu-sim/histogram.h gpgpu-sim/dwf.h gpgpu-sim/warp_tracker.h gpgpu-sim/mem_latency_stat.h gpgpu-sim/icnt_wrapper.h gpgpu-sim/mem_fetch.h
gpgpu-sim/dram.$(OEXT): gpgpu-sim/dram.h util.h
gpgpu-sim/dram_sched.$(OEXT): gpgpu-sim/dram.h gpgpu-sim/dram_sched.h gpgpu-sim/gpu-misc.h gpgpu-sim/gpu-sim.h util.h 
gpgpu-sim/addrdec.$(OEXT): gpgpu-sim/addrdec.h util.h
gpgpu-sim/gpu-misc.$(OEXT): gpgpu-sim/gpu-misc.h
gpgpu-sim/stack.$(OEXT): gpgpu-sim/stack.h util.h
gpgpu-sim/warp_tracker.$(OEXT): gpgpu-sim/warp_tracker.h gpgpu-sim/shader.h 
gpgpu-sim/dwf.$(OEXT): gpgpu-sim/dwf.h gpgpu-sim/histogram.h util.h
gpgpu-sim/icnt_wrapper.$(OEXT): gpgpu-sim/icnt_wrapper.h intersim/interconnect_interface.h
gpgpu-sim/stat-tool.$(OEXT): gpgpu-sim/histogram.h gpgpu-sim/cflogger.h util.h
