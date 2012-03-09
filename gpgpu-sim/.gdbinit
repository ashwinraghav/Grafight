# Provides some useful debugging macros.  To use this file, copy to your home
# directory or to your simulation directory then run GPGPU-Sim in gdb.

printf "\n  ** loading GPGPU-Sim debugging macros... ** \n\n"

set print pretty
set print array-indexes
set unwindonsignal on

define dp
	# Display pipeline state, then continue to next breakpoint.
	# arg0 : index of shader core you would like to see the pipeline state of
	#
	# This function displays the state of the pipeline on a single shader core
	# (setting different values for the first argument of the call to 
	# dump_pipeline_impl will cause different information to be displayed--
	# see the source code for more details)
        call dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
end

define dpc
	# Display pipeline state, then continue to next breakpoint.
	# arg0 : index of shader core you would like to see the pipeline state of
	#
	# This version is useful if you set a breakpoint where gpu_sim_cycle is 
	# incremented in gpu_sim_loop() in src/gpgpu-sim/gpu-sim.c
	# repeatly hitting enter will advance to show the pipeline contents on
	# the next cycle.
        call dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        continue
end

define ptxdis
	# disassemble PTX instructions between PCs of first and second argument 
	set $pc=$arg0
	printf "disassemble instructions from 0x%x to 0x%x\n", $arg0, $arg1
	call fflush(stdout)
	while ( $pc <= $arg1 )
	      printf "0x%04x (%4u)  : ", $pc, $pc
	      call ptx_print_insn( $pc, stdout )
	      call fflush(stdout)
	      set $pc = $pc + 1
	end
end 	

define ptxdis_func
	# arg0 : shader core number
	# arg1 : thread ID
	set $ptx_tinfo = (ptx_thread_info*)sc[$arg0]->thread[$arg1].ptx_thd_info
	set $finfo = $ptx_tinfo->m_func_info
	set $minpc = $finfo->m_start_PC
	set $maxpc = $minpc + $finfo->m_instr_mem_size
	printf "disassembly of function %s (min pc = %u, max pc = %u):\n", $finfo->m_name.c_str(), $minpc, $maxpc
	ptxdis $minpc $maxpc $arg0 $arg1
end

define ptx_tids2pcs
	# arg0 : array of tids
	# arg1 : size of array
	# arg2 : shader core number
	set $i = 0
	while ( $i < $arg1 )
		set $tid =  $arg0[$i]
  		set $pc = ((ptx_thread_info*)sc[$arg2]->thread[$tid].ptx_thd_info)->m_PC
		printf "%2u : tid = %3u  => pc = %d\n", $i, $tid, $pc
		set $i = $i + 1
	end
end
