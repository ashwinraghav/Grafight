/* 
 * gpu-sim.c
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Ivan Sham, Henry Wong, Dan O'Connor and the 
 * University of British Columbia
 * Vancouver, BC  V6T 1Z4
 * All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include "gpu-sim.h"

#include <time.h>
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "delayqueue.h"
#include "shader.h"
#include "icnt_wrapper.h"
#include "dram.h"
#include "addrdec.h"
#include "dwf.h"
#include "warp_tracker.h"
#include "cflogger.h"

#include "../cuda-sim/ptx-stats.h"
#include "../intersim/statwraper.h"
#include <stdio.h>
#include <string.h>
#define MAX(a,b) (((a)>(b))?(a):(b))

//SEAN - to queue atomic requests
//create struct for queue
typedef struct atom_q_t {
  unsigned long long int tag;
  mem_fetch_t *mf; //memory fetch associated with entry

  struct atom_q_t *next;
} atom_q;

//create list of structs
atom_q *atom_q_head=NULL;
atom_q *atom_q_tail=NULL;

unsigned L2_write_miss = 0;
unsigned L2_write_hit = 0;
unsigned L2_read_hit = 0;
unsigned L2_read_miss = 0;
unsigned made_read_mfs = 0;
unsigned made_write_mfs = 0;
unsigned freed_read_mfs = 0;
unsigned freed_L1write_mfs = 0;
unsigned freed_L2write_mfs = 0;
unsigned freed_dummy_read_mfs = 0;
unsigned long long  gpu_sim_cycle = 0;
unsigned long long  gpu_sim_insn = 0;
unsigned long long  gpu_sim_insn_no_ld_const = 0;
unsigned long long  gpu_sim_prev_insn = 0;
unsigned long long  gpu_sim_insn_last_update = 0;
unsigned long long  gpu_tot_sim_cycle = 0;
unsigned long long  gpu_tot_sim_insn = 0;
unsigned long long  gpu_last_sim_cycle = 0;
unsigned long long  gpu_completed_thread = 0;
unsigned long long  gpu_tot_completed_thread = 0;

unsigned int **concurrent_row_access; //concurrent_row_access[dram chip id][bank id]
unsigned int **num_activates; //num_activates[dram chip id][bank id]
unsigned int **row_access; //row_access[dram chip id][bank id]
unsigned int **max_conc_access2samerow; //max_conc_access2samerow[dram chip id][bank id]
unsigned int **max_servicetime2samerow; //max_servicetime2samerow[dram chip id][bank id]
unsigned int mergemiss = 0;
unsigned int L1_read_miss = 0;
unsigned int L1_write_miss = 0;
unsigned int L1_texture_miss = 0;
unsigned int L1_const_miss = 0;
unsigned int gpgpu_n_sent_writes = 0;
unsigned int gpgpu_n_processed_writes = 0;
unsigned int *L2_cbtoL2length;
unsigned int *L2_cbtoL2writelength;
unsigned int *L2_L2tocblength;
unsigned int *L2_dramtoL2length;
unsigned int *L2_dramtoL2writelength;
unsigned int *L2_L2todramlength;
unsigned int *max_return_queue_length;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_shd_mem = 0;
unsigned int gpu_stall_wr_back = 0;
unsigned int gpu_stall_dramfull = 0; 
unsigned int gpu_stall_icnt2sh = 0;
unsigned int gpu_stall_by_MSHRwb = 0;

//shader cannot send to icnt because icnt buffer is full
//Note: it is accumulative for all shaders and is never reset
//so it might increase 8 times in a cycle if we have 8 shaders
unsigned int gpu_stall_sh2icnt = 0;        
// performance counters to account for instruction distribution
extern unsigned int gpgpu_n_load_insn;
extern unsigned int gpgpu_n_store_insn;
extern unsigned int gpgpu_n_shmem_insn;
extern unsigned int gpgpu_n_tex_insn;
extern unsigned int gpgpu_n_const_insn;
extern unsigned int gpgpu_multi_unq_fetches;
char *gpgpu_runtime_stat;
int gpu_stat_sample_freq = 10000;
int gpu_runtime_stat_flag = 0;
extern int gpgpu_warpdistro_shader;

// GPGPU options
unsigned long long  gpu_max_cycle = 0;
unsigned long long  gpu_max_insn = 0;
int gpu_max_cycle_opt = 0;
int gpu_max_insn_opt = 0;
int gpu_deadlock_detect = 0;
int gpu_deadlock = 0;
static unsigned long long  last_gpu_sim_insn = 0;
int gpgpu_dram_scheduler = DRAM_FIFO;
int g_save_embedded_ptx = 0;
int gpgpu_simd_model = 0;
int gpgpu_no_dl1 = 0;
char *gpgpu_cache_texl1_opt;
char *gpgpu_cache_constl1_opt;
char *gpgpu_cache_dl1_opt;
char *gpgpu_cache_dl2_opt;
char *gpgpu_L2_queue_config;
int gpgpu_l2_readoverwrite = 0;
int gpgpu_partial_write_mask = 0;

int gpgpu_perfect_mem = FALSE;
char *gpgpu_shader_core_pipeline_opt;
extern unsigned int *requests_by_warp;
unsigned int gpgpu_dram_buswidth = 4;
unsigned int gpgpu_dram_burst_length = 4;
int gpgpu_dram_sched_queue_size = 0; 
char * gpgpu_dram_timing_opt;
int gpgpu_flush_cache = 0;
int gpgpu_mem_address_mask = 0;
unsigned int recent_dram_util = 0;

//SEAN - pipetrace support
char* g_pipetrace_out;
int g_pipetrace=0;

int gpgpu_cflog_interval = 0;

unsigned int finished_trace = 0;

unsigned g_next_request_uid = 1;

extern struct regs_t regs;

extern long int gpu_reads;

extern void ptx_dump_regs( void *thd );

int g_nthreads_issued;
int g_total_cta_left;

extern unsigned ptx_kernel_program_size();
extern void visualizer_printstat();
extern void time_vector_create(int ld_size,int st_size);
extern void time_vector_print(void);
extern void time_vector_update(unsigned int uid,int slot ,long int cycle,int type);
extern void check_time_vector_update(unsigned int uid,int slot ,long int latency,int type);
extern void node_req_hist_clear(void *p);
extern void node_req_hist_dump(void *p);
extern void node_req_hist_update(void * p,int node, long long cycle);

/* functionally simulated memory */
extern struct mem_t *mem;

/* Defining Clock Domains
basically just the ratio is important */

#define  CORE  0x01
#define  L2    0x02
#define  DRAM  0x04
#define  ICNT  0x08  

double core_time=0;
double icnt_time=0;
double dram_time=0;
double l2_time=0;

#define MhZ *1000000
double core_freq=2 MhZ;
double icnt_freq=2 MhZ;
double dram_freq=2 MhZ;
double l2_freq=2 MhZ;

double core_period  = 1 /( 2 MhZ);
double icnt_period   = 1 /( 2 MhZ);
double dram_period = 1 /( 2 MhZ);
double l2_period = 1 / (2 MhZ);

char * gpgpu_clock_domains;

/* GPU uArch parameters */
unsigned int gpu_n_mem = 8;
unsigned int gpu_mem_n_bk = 4;
unsigned int gpu_n_mem_per_ctrlr = 1;
unsigned int gpu_n_shader = 8;
int gpu_concentration = 1;
int gpu_n_tpc = 8;
unsigned int gpu_n_thread_per_shader = 128;
unsigned int gpu_n_warp_per_shader;
unsigned int gpu_n_mshr_per_thread = 1;

extern int gpgpu_interwarp_mshr_merge ;

extern unsigned int gpgpu_shmem_size;
extern unsigned int gpgpu_shader_registers;
extern unsigned int gpgpu_shader_cta;
extern int gpgpu_shmem_bkconflict;
extern int gpgpu_cache_bkconflict;
extern int gpgpu_n_cache_bank;
extern unsigned int warp_size; 
extern int pipe_simd_width;
extern unsigned int gpgpu_dwf_heuristic;
extern unsigned int gpgpu_dwf_regbk;
int gpgpu_reg_bankconflict = FALSE;
extern int gpgpu_shmem_port_per_bank;
extern int gpgpu_cache_port_per_bank;
extern int gpgpu_const_port_per_bank;
extern int gpgpu_shmem_pipe_speedup;  

extern unsigned int gpu_max_cta_per_shader;
extern unsigned int gpu_padded_cta_size;
extern int gpgpu_local_mem_map;

unsigned int gpgpu_pre_mem_stages = 0;
unsigned int gpgpu_no_divg_load = 0;
char *gpgpu_dwf_hw_opt;
unsigned int gpgpu_thread_swizzling = 0;
unsigned int gpgpu_strict_simd_wrbk = 0;

int pdom_sched_type = 0;
int n_pdom_sc_orig_stat = 0; //the selected pdom schedular is used 
int n_pdom_sc_single_stat = 0; //only a single warp is ready to go in that cycle.  
int *num_warps_issuable;
int *num_warps_issuable_pershader;

// Thread Dispatching Unit option 
int gpgpu_cuda_sim = 1;
int gpgpu_spread_blocks_across_cores = 1;

/* GPU uArch structures */
shader_core_ctx_t **sc;
dram_t **dram;
mem_fetch_t **L2request; //request currently being serviced by the L2 Cache
unsigned int common_clock = 0;
unsigned int more_thread = 1;
extern unsigned int n_regconflict_stall;
unsigned int warp_conflict_at_writeback = 0;
unsigned int gpgpu_commit_pc_beyond_two = 0;
extern int g_network_mode;
int gpgpu_cache_wt_through = 0;


//memory access classification
int gpgpu_n_mem_read_local = 0;
int gpgpu_n_mem_write_local = 0;
int gpgpu_n_mem_texture = 0;
int gpgpu_n_mem_const = 0;
int gpgpu_n_mem_read_global = 0;
int gpgpu_n_mem_write_global = 0;

#define MEM_LATENCY_STAT_IMPL
#include "mem_latency_stat.h"

unsigned char fq_has_buffer(unsigned long long int *addr, int *bsize, 
                            int n_addr, int sid);
unsigned char fq_push(unsigned long long int addr, int bsize, unsigned char write, unsigned long long int partial_write_mask, 
                      int sid, int mshr_idx, mshr_entry* mshr, int cache_hits_waiting,
                      enum mem_access_type mem_acc, address_type pc);
int issue_mf_from_fq(mem_fetch_t *mf);
unsigned char single_check_icnt_has_buffer(int chip, int sid, unsigned char is_write );
unsigned char fq_pop(int tpc_id);
void fill_shd_L1_with_new_line(shader_core_ctx_t * sc, mem_fetch_t * mf);

extern void set_option_gpgpu_spread_blocks_across_cores(int option);
extern void set_param_gpgpu_num_shaders(int num_shaders);
extern unsigned ptx_sim_grid_size();
extern void icnt_init_grid();
extern void interconnect_stats();
extern void icnt_overal_stat();
extern unsigned ptx_sim_cta_size();
extern unsigned ptx_sim_init_thread( void** thread_info,int sid,unsigned tid,unsigned threads_left,unsigned num_threads);

void L2c_create ( dram_t* dram_p, const char* cache_opt, const char* queue_opt );
unsigned char L2c_write_back(unsigned long long int addr, int bsize, int dram_id ); 
void L2c_print_stat( );

void gpu_sim_loop( int grid_num );

extern void print_shader_cycle_distro( FILE *fout ) ;
extern void find_reconvergence_points();
extern void dwf_process_reconv_pts();

extern int gpgpu_ptx_instruction_classification ;
extern int g_ptx_sim_mode;

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333
void L2c_log(int task);
void dram_log(int task);

extern unsigned long long int addrdec_packbits(unsigned long long int mask, 
                                               unsigned long long int val,
                                               unsigned char high, unsigned char low);

extern void visualizer_options(option_parser_t opp);
void gpu_reg_options(option_parser_t opp)
{
   option_parser_register(opp, "-save_embedded_ptx", OPT_BOOL, &g_save_embedded_ptx, 
                "saves ptx files embedded in binary as <n>.ptx",
                "0");
   option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &gpgpu_simd_model, 
               "0 = no recombination, 1 = post-dominator, 2 = MIMD, 3 = dynamic warp formation", "0");
   option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32, &gpgpu_dram_scheduler, 
               "0 = fifo (default), 1 = fast ideal", "0");

   option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");

   option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR, &gpgpu_cache_texl1_opt, 
                  "per-shader L1 texture cache  (READ-ONLY) config, i.e., {<nsets>:<linesize>:<assoc>:<repl>|none}",
                  "512:64:2:L");

   option_parser_register(opp, "-gpgpu_const_cache:l1", OPT_CSTR, &gpgpu_cache_constl1_opt, 
                  "per-shader L1 constant memory cache  (READ-ONLY) config, i.e., {<nsets>:<linesize>:<assoc>:<repl>|none}",
                  "64:64:2:L");

   option_parser_register(opp, "-gpgpu_no_dl1", OPT_BOOL, &gpgpu_no_dl1, 
                "no dl1 cache (voids -gpgpu_cache:dl1 option)",
                "0");

   option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR, &gpgpu_cache_dl1_opt, 
                  "shader L1 data cache config, i.e., {<nsets>:<bsize>:<assoc>:<repl>|none}",
                  "256:128:1:L");

   option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR, &gpgpu_cache_dl2_opt, 
                  "unified banked L2 data cache config, i.e., {<nsets>:<bsize>:<assoc>:<repl>|none}; disabled by default",
                  NULL); 

   option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL, &gpgpu_perfect_mem, 
                "enable perfect memory mode (no cache miss)",
                "0");

   option_parser_register(opp, "-gpgpu_shader_core_pipeline", OPT_CSTR, &gpgpu_shader_core_pipeline_opt, 
                  "shader core pipeline config, i.e., {<nthread>:<warpsize>:<pipe_simd_width>}",
                  "256:32:32");

   option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32, &gpgpu_shader_registers, 
                "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
                "8192");

   option_parser_register(opp, "-gpgpu_shader_cta", OPT_UINT32, &gpgpu_shader_cta, 
                "Maximum number of concurrent CTAs in shader (default 8)",
                "8");

   option_parser_register(opp, "-gpgpu_n_shader", OPT_UINT32, &gpu_n_shader, 
                "number of shaders in gpu",
                "8");
   option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &gpu_n_mem, 
                "number of memory modules (e.g. memory controllers) in gpu",
                "8");
   option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32, &gpu_n_mem_per_ctrlr, 
                "number of memory chips per memory controller",
                "1");
   option_parser_register(opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat, 
                  "display runtime statistics such as dram utilization {<freq>:<flag>}",
                  "10000:0");

   option_parser_register(opp, "-gpgpu_dwf_heuristic", OPT_UINT32, &gpgpu_dwf_heuristic, 
                "DWF scheduling heuristic: 0 = majority, 1 = minority, 2 = timestamp, 3 = pdom priority, 4 = pc-based, 5 = max-heap",
                "0");

   option_parser_register(opp, "-gpgpu_reg_bankconflict", OPT_BOOL, &gpgpu_reg_bankconflict, 
                "Check for bank conflict in the pipeline",
                "0");

   option_parser_register(opp, "-gpgpu_dwf_regbk", OPT_BOOL, (int*)&gpgpu_dwf_regbk, 
                "Have dwf scheduler to avoid bank conflict",
                "1");

   option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32, &gpgpu_memlatency_stat, 
               "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
               "0");

   option_parser_register(opp, "-gpgpu_mshr_per_thread", OPT_UINT32, &gpu_n_mshr_per_thread, 
                "Number of MSHRs per thread",
                "1");

   option_parser_register(opp, "-gpgpu_interwarp_mshr_merge", OPT_INT32, &gpgpu_interwarp_mshr_merge, 
               "interwarp coalescing",
               "0");

   option_parser_register(opp, "-gpgpu_dram_sched_queue_size", OPT_INT32, &gpgpu_dram_sched_queue_size, 
               "0 = unlimited (default); # entries per chip",
               "0");

   option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &gpgpu_dram_buswidth, 
                "default = 4 bytes (8 bytes per cycle at DDR)",
                "4");

   option_parser_register(opp, "-gpgpu_dram_burst_length", OPT_UINT32, &gpgpu_dram_burst_length, 
                "Burst length of each DRAM request (default = 4 DDR cycle)",
                "4");

   option_parser_register(opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt, 
               "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tWTR}",
               "4:2:8:12:21:13:34:9:4:5");


   option_parser_register(opp, "-gpgpu_mem_address_mask", OPT_INT32, &gpgpu_mem_address_mask, 
               "0 = old addressing mask, 1 = new addressing mask, 2 = new add. mask + flipped bank sel and chip sel bits",
               "0");

   option_parser_register(opp, "-gpgpu_flush_cache", OPT_BOOL, &gpgpu_flush_cache, 
                "Flush cache at the end of each kernel call",
                "0");

   option_parser_register(opp, "-gpgpu_l2_readoverwrite", OPT_BOOL, &gpgpu_l2_readoverwrite, 
                "Prioritize read requests over write requests for L2",
                "0");

   option_parser_register(opp, "-gpgpu_pre_mem_stages", OPT_UINT32, &gpgpu_pre_mem_stages, 
                "default = 0 pre-memory pipeline stages",
                "0");

   option_parser_register(opp, "-gpgpu_no_divg_load", OPT_BOOL, (int*)&gpgpu_no_divg_load, 
                "Don't allow divergence on load",
                "0");

   option_parser_register(opp, "-gpgpu_dwf_hw", OPT_CSTR, &gpgpu_dwf_hw_opt, 
                  "dynamic warp formation hw config, i.e., {<#LUT_entries>:<associativity>|none}",
                  "32:2");

   option_parser_register(opp, "-gpgpu_thread_swizzling", OPT_BOOL, (int*)&gpgpu_thread_swizzling, 
                "Thread Swizzling (1=on, 0=off)",
                "0");

   option_parser_register(opp, "-gpgpu_strict_simd_wrbk", OPT_BOOL, (int*)&gpgpu_strict_simd_wrbk, 
                "Applying Strict SIMD WriteBack Stage (1=on, 0=off)",
                "0");

   option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size, 
                "Size of shared memory per shader core (default 16kB)",
                "16384");

   option_parser_register(opp, "-gpgpu_shmem_bkconflict", OPT_BOOL, &gpgpu_shmem_bkconflict,  
                "Turn on bank conflict check for shared memory",
                "0");

   option_parser_register(opp, "-gpgpu_shmem_pipe_speedup", OPT_BOOL, &gpgpu_shmem_pipe_speedup,  
                "Number of groups each warp is divided for shared memory bank conflict check",
                "2");

   option_parser_register(opp, "-gpgpu_L2_queue", OPT_CSTR, &gpgpu_L2_queue_config, 
                  "L2 data cache queue length and latency config",
                  "0:0:0:0:0:0:10:10");

   option_parser_register(opp, "-gpgpu_cache_wt_through", OPT_BOOL, &gpgpu_cache_wt_through, 
                "L1 cache become write through (1=on, 0=off)", 
                "0");

   option_parser_register(opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect, 
                "Stop the simulation at deadlock (1=on (default), 0=off)", 
                "1");

   option_parser_register(opp, "-gpgpu_cache_bkconflict", OPT_BOOL, &gpgpu_cache_bkconflict, 
                "Turn on bank conflict check for L1 cache access", 
                "0");

   option_parser_register(opp, "-gpgpu_n_cache_bank", OPT_INT32, &gpgpu_n_cache_bank, 
               "Number of banks in L1 cache, also for memory coalescing stall", 
               "1");

   option_parser_register(opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader, 
               "Specify which shader core to collect the warp size distribution from", 
               "-1");


   option_parser_register(opp, "-gpgpu_pdom_sched_type", OPT_INT32, &pdom_sched_type, 
               "0 = first ready warp found, 1 = random, 8 = loose round robin", 
               "8");

   option_parser_register(opp, "-gpgpu_spread_blocks_across_cores", OPT_BOOL, 
                &gpgpu_spread_blocks_across_cores, 
                "Spread block-issuing across all cores instead of filling up core by core", 
                "1");

   option_parser_register(opp, "-gpgpu_cuda_sim", OPT_BOOL, &gpgpu_cuda_sim, 
                "use PTX instruction set", 
                "1");
   option_parser_register(opp, "-gpgpu_ptx_instruction_classification", OPT_INT32, 
               &gpgpu_ptx_instruction_classification, 
               "if enabled will classify ptx instruction types per kernel (Max 255 kernels now)", 
               "0");
   option_parser_register(opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode, 
               "Select between Performance (default) or Functional simulation (1)", 
               "0");
   option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR, &gpgpu_clock_domains, 
                  "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
                  "500.0:2000.0:2000.0:2000.0");

   option_parser_register(opp, "-gpgpu_shmem_port_per_bank", OPT_INT32, &gpgpu_shmem_port_per_bank, 
               "Number of access processed by a shared memory bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_cache_port_per_bank", OPT_INT32, &gpgpu_cache_port_per_bank, 
               "Number of access processed by a cache bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_const_port_per_bank", OPT_INT32, &gpgpu_const_port_per_bank,
               "Number of access processed by a constant cache bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval, 
               "Interval between each snapshot in control flow logger", 
               "0");
   option_parser_register(opp, "-gpgpu_partial_write_mask", OPT_INT32, &gpgpu_partial_write_mask, 
               "use partial write mask to filter memory requests <1>No extra reads(use this!)<2>extra reads generated for partial chunks", 
               "0");
   option_parser_register(opp, "-gpu_concentration", OPT_INT32, &gpu_concentration, 
               "Number of shader cores per interconnection port (default = 1)", 
               "1");
   option_parser_register(opp, "-gpgpu_local_mem_map", OPT_INT32, &gpgpu_local_mem_map, 
               "Mapping from local memory space address to simulated GPU physical address space (default = 1)", 
               "1");
   //SEAN - pipetrace support
  option_parser_register(opp, "-pipetrace_out", OPT_CSTR, &g_pipetrace_out,
			 "specifies file to output pipetrace to",
			 "pipetrace.txt");
  option_parser_register(opp, "-pipetrace", OPT_BOOL, &g_pipetrace,
			 "Turns pipetrace on/off",
			 "0");

   addrdec_setoption(opp);
   visualizer_options(opp);
   ptx_file_line_stats_options(opp);
}

/////////////////////////////////////////////////////////////////////////////

inline int mem2device(int memid) {
   return memid + gpu_n_tpc;
}

/////////////////////////////////////////////////////////////////////////////


/* Allocate memory for uArch structures */
void init_gpu () 
{ 
   int i;

   gpu_max_cycle = gpu_max_cycle_opt;
   gpu_max_insn = gpu_max_insn_opt;

   i = sscanf(gpgpu_shader_core_pipeline_opt,"%d:%d:%d", 
              &gpu_n_thread_per_shader, &warp_size, &pipe_simd_width);
   gpu_n_warp_per_shader = gpu_n_thread_per_shader / warp_size;
   num_warps_issuable = (int*) calloc(gpu_n_warp_per_shader+1, sizeof(int));
   num_warps_issuable_pershader = (int*) calloc(gpu_n_shader, sizeof(int));
   if (i == 2) {
      pipe_simd_width = warp_size;
   } else if (i == 3) {
      assert(warp_size % pipe_simd_width == 0);
   }

   sscanf(gpgpu_runtime_stat, "%d:%x",
          &gpu_stat_sample_freq, &gpu_runtime_stat_flag);

   sc = (shader_core_ctx_t**) calloc(gpu_n_shader, sizeof(shader_core_ctx_t*));
   int mshr_que = gpu_n_mshr_per_thread;
   for (i=0;i<gpu_n_shader;i++) {
      sc[i] = shader_create("sh", i, /* shader id*/
                            gpu_n_thread_per_shader, /* number of threads */
                            mshr_que, /* number of MSHR per threads */
                            fq_push, fq_has_buffer, gpgpu_simd_model);
   }

   ptx_file_line_stats_create_exposed_latency_tracker(gpu_n_shader);

   // initialize dynamic warp formation scheduler
   int dwf_lut_size, dwf_lut_assoc;
   sscanf(gpgpu_dwf_hw_opt,"%d:%d", &dwf_lut_size, &dwf_lut_assoc);
   char *dwf_hw_policy_opt = strchr(gpgpu_dwf_hw_opt, ';');
   int insn_size = 1; // for cuda-sim
   create_dwf_schedulers(gpu_n_shader, dwf_lut_size, dwf_lut_assoc, 
                         warp_size, pipe_simd_width, 
                         gpu_n_thread_per_shader, insn_size, 
                         gpgpu_dwf_heuristic, dwf_hw_policy_opt );

   init_mshr_pool();
   gpgpu_no_divg_load = gpgpu_no_divg_load && (gpgpu_simd_model == DWF);
   // always use no diverge on load for PDOM and NAIVE
   gpgpu_no_divg_load = gpgpu_no_divg_load || (gpgpu_simd_model == POST_DOMINATOR || gpgpu_simd_model == NO_RECONVERGE);
   if (gpgpu_no_divg_load)
      init_warp_tracker();

   assert(gpu_n_shader % gpu_concentration == 0);
   gpu_n_tpc = gpu_n_shader / gpu_concentration;

   dram = (dram_t**) calloc(gpu_n_mem, sizeof(dram_t*));
   L2request = (mem_fetch_t**) calloc(gpu_n_mem, sizeof(mem_fetch_t*));
   addrdec_setnchip(gpu_n_mem);
   unsigned int nbk,tCCD,tRRD,tRCD,tRAS,tRP,tRC,CL,WL,tWTR;
   sscanf(gpgpu_dram_timing_opt,"%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",&nbk,&tCCD,&tRRD,&tRCD,&tRAS,&tRP,&tRC,&CL,&WL,&tWTR);
   gpu_mem_n_bk = nbk;
   for (i=0;i<gpu_n_mem;i++) {
      dram[i] = dram_create(i, nbk, tCCD, tRRD, tRCD, tRAS, tRP, tRC, 
                            CL, WL, gpgpu_dram_burst_length/*BL*/, tWTR, gpgpu_dram_buswidth/*busW*/, 
                            gpgpu_dram_sched_queue_size, gpgpu_dram_scheduler);
      if (gpgpu_cache_dl2_opt)
         L2c_create(dram[i], gpgpu_cache_dl2_opt, gpgpu_L2_queue_config);
   }
   dram_log(CREATELOG);
   if (gpgpu_cache_dl2_opt && 1) {
      L2c_log(CREATELOG);
   }
   concurrent_row_access = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   num_activates = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   row_access = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   max_conc_access2samerow = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   max_servicetime2samerow = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));

   for (i=0;i<gpu_n_mem ;i++ ) {
      concurrent_row_access[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      row_access[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      num_activates[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      max_conc_access2samerow[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      max_servicetime2samerow[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
   }

   memlatstat_init();

   L2_cbtoL2length = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_cbtoL2writelength = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_L2tocblength = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_dramtoL2length = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_dramtoL2writelength = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_L2todramlength = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   max_return_queue_length = (unsigned int*) calloc(gpu_n_shader, sizeof(unsigned int));
   icnt_init(gpu_n_tpc, gpu_n_mem);

   common_clock = 0;

   time_vector_create(NUM_MEM_REQ_STAT,MR_2SH_ICNT_INJECTED);
}



void gpu_print_stat();

void init_clock_domains(void ) {
   sscanf(gpgpu_clock_domains,"%lf:%lf:%lf:%lf", 
          &core_freq, &icnt_freq, &l2_freq, &dram_freq);
   core_freq = core_freq MhZ;
   icnt_freq = icnt_freq MhZ;
   l2_freq = l2_freq MhZ;
   dram_freq = dram_freq MhZ;        
   core_period = 1/core_freq;
   icnt_period = 1/icnt_freq;
   dram_period = 1/dram_freq;
   l2_period = 1/l2_freq;
   core_time = 0 ;
   dram_time = 0 ;
   icnt_time = 0;
   l2_time = 0;
   printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n",core_freq,icnt_freq,l2_freq,dram_freq);
   printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",core_period,icnt_period,l2_period,dram_period);
}

void reinit_clock_domains(void)
{
   core_time = 0 ;
   dram_time = 0 ;
   icnt_time = 0;
   l2_time = 0;
}

void init_once(void ) {
   init_clock_domains();
}

// return the number of cycle required to run all the trace on the gpu 
unsigned int run_gpu_sim(int grid_num) 
{

   int not_completed;
   int mem_busy;
   int icnt2mem_busy;
   int i;

   gpu_sim_cycle = 0;
   not_completed = 1;
   mem_busy = 1;
   icnt2mem_busy = 1;
   finished_trace = 0;
   g_next_request_uid = 1;
   more_thread = 1;
   gpu_sim_insn = 0;
   gpu_sim_insn_no_ld_const = 0;

   gpu_completed_thread = 0;

   g_nthreads_issued = 0;

   static int one_time_inits_done = 0 ;
   if (!one_time_inits_done ) {
      init_once();
   }
   reinit_clock_domains();
   set_option_gpgpu_spread_blocks_across_cores(gpgpu_spread_blocks_across_cores);
   set_param_gpgpu_num_shaders(gpu_n_shader);
   if (gpgpu_spread_blocks_across_cores) {
      for (i=0;i<gpu_n_shader;i++) {
         shader_reinit(sc[i],0,sc[i]->n_threads);
      }
      g_total_cta_left =  ptx_sim_grid_size() ;
   }



   if (gpu_max_cycle && (gpu_tot_sim_cycle + gpu_sim_cycle) >= gpu_max_cycle) {
      return gpu_sim_cycle;
   }
   if (gpu_max_insn && (gpu_tot_sim_insn + gpu_sim_insn) >= gpu_max_insn) {
      return gpu_sim_cycle;
   }

   //refind the diverg/reconvergence pairs
   dwf_reset_reconv_pt();
   find_reconvergence_points();

   dwf_process_reconv_pts();
   dwf_reinit_schedulers(gpu_n_shader);

   // initialize the control-flow, memory access, memory latency logger
   create_thread_CFlogger( gpu_n_shader, gpu_n_thread_per_shader, ptx_kernel_program_size(), 0, gpgpu_cflog_interval );
   shader_CTA_count_create( gpu_n_shader, gpgpu_cflog_interval);
   if (gpgpu_cflog_interval != 0) {
      insn_warp_occ_create( gpu_n_shader, warp_size, ptx_kernel_program_size() );
      shader_warp_occ_create( gpu_n_shader, warp_size, gpgpu_cflog_interval);
      shader_mem_acc_create( gpu_n_shader, gpu_n_mem, 4, gpgpu_cflog_interval);
      shader_mem_lat_create( gpu_n_shader, gpgpu_cflog_interval);
      shader_cache_access_create( gpu_n_shader, 3, gpgpu_cflog_interval);
      set_spill_interval (gpgpu_cflog_interval * 40);
   }

   // calcaulte the max cta count and cta size for local memory address mapping
   gpu_max_cta_per_shader = max_cta_per_shader(sc[0]);
   //gpu_max_cta_per_shader is limited by number of CTAs if not enough    
   if (ptx_sim_grid_size() < gpu_max_cta_per_shader*gpu_n_shader) { 
      gpu_max_cta_per_shader = (ptx_sim_grid_size() / gpu_n_shader);
      if (ptx_sim_grid_size() % gpu_n_shader)
         gpu_max_cta_per_shader++;
   }
   unsigned int gpu_cta_size = ptx_sim_cta_size();
   gpu_padded_cta_size = (gpu_cta_size%32) ? 32*((gpu_cta_size/32)+1) : gpu_cta_size;

   if (g_network_mode) {
      icnt_init_grid(); 
   }
   last_gpu_sim_insn = 0;
   // add this condition as well? (gpgpu_n_processed_writes < gpgpu_n_sent_writes)
   while (not_completed || mem_busy || icnt2mem_busy) {
      gpu_sim_loop(grid_num);

      not_completed = 0;
      for (i=0;i<gpu_n_shader;i++) {
         not_completed += sc[i]->not_completed;
      }
      // dram_busy just check the request queue length into the dram 
      // to make sure all the memory requests (esp the writes) are done
      mem_busy = 0; 
      for (i=0;i<gpu_n_mem;i++) {
         mem_busy += dram_busy(dram[i]);
      }
     // icnt to the memory should clean of any pending tranfers as well
      icnt2mem_busy = icnt_busy( );

      if (gpu_max_cycle && (gpu_tot_sim_cycle + gpu_sim_cycle) >= gpu_max_cycle) {
         break;
      }
      if (gpu_max_insn && (gpu_tot_sim_insn + gpu_sim_insn) >= gpu_max_insn) {
         break;
      }
      if (gpu_deadlock_detect && gpu_deadlock) {
         break;
      }

   }
   memlatstat_lat_pw();
   gpu_tot_sim_cycle += gpu_sim_cycle;
   gpu_tot_sim_insn += gpu_sim_insn;
   gpu_tot_completed_thread += gpu_completed_thread;
   
   ptx_file_line_stats_write_file();
   //SEAN
   if(g_pipetrace) {
     pipe_stat_write_file();
     //     g_pipetrace = 0;
   }

   printf("stats for grid: %d\n", grid_num);
   gpu_print_stat();
   if (g_network_mode) {
      interconnect_stats();
      printf("----------------------------Interconnect-DETAILS---------------------------------" );
      icnt_overal_stat();
      printf("----------------------------END-of-Interconnect-DETAILS-------------------------" );
   }
   if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_QUEUELOGS ) {
      dramqueue_latency_log_dump();
      dram_log(DUMPLOG);
      if (gpgpu_cache_dl2_opt) {
         L2c_log(DUMPLOG);
         L2c_latency_log_dump();
      }
   }

#define DEADLOCK 0
   if (gpu_deadlock_detect && gpu_deadlock) {
      fflush(stdout);
      printf("ERROR ** deadlock detected: last writeback @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n", 
             (unsigned) gpu_sim_insn_last_update, (unsigned) (gpu_tot_sim_cycle-gpu_sim_cycle),
             (unsigned) (gpu_sim_cycle - gpu_sim_insn_last_update )); 
      fflush(stdout);
      assert(DEADLOCK);
   }
   return gpu_sim_cycle;
}

extern void ** g_inst_classification_stat;
extern void ** g_inst_op_classification_stat;
extern int g_ptx_kernel_count; // used for classification stat collection purposes 

void gpu_print_stat()
{  
   int i,j,k;

   printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
   printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
   printf("gpu_sim_no_ld_const_insn = %lld\n", gpu_sim_insn_no_ld_const);
   printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
   printf("gpu_completed_thread = %lld\n", gpu_completed_thread);
   printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle);
   printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn);
   printf("gpu_tot_ipc = %12.4f\n", (float)gpu_tot_sim_insn / gpu_tot_sim_cycle);
   printf("gpu_tot_completed_thread = %lld\n", gpu_tot_completed_thread);
   printf("gpgpu_n_sent_writes = %d\n", gpgpu_n_sent_writes);
   printf("gpgpu_n_processed_writes = %d\n", gpgpu_n_processed_writes);

   // performance counter for stalls due to congestion.
   printf("gpu_stall_by_MSHRwb= %d\n", gpu_stall_by_MSHRwb);
   printf("gpu_stall_shd_mem  = %d\n", gpu_stall_shd_mem );
   printf("gpu_stall_wr_back  = %d\n", gpu_stall_wr_back );
   printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
   printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh   );
   printf("gpu_stall_sh2icnt    = %d\n", gpu_stall_sh2icnt   );
   // performance counter that are not local to one shader
   shader_print_accstats(stdout);

   memlatstat_print();
   printf("max return queue length = ");
   for (i=0;i<gpu_n_shader;i++) {
      printf("%d ", max_return_queue_length[i]);
   }
   printf("\n");
   // merge misses
   printf("merge misses = %d\n", mergemiss);
   printf("L1 read misses = %d\n", L1_read_miss);
   printf("L1 write misses = %d\n", L1_write_miss);
   printf("L1 texture misses = %d\n", L1_texture_miss);
   printf("L1 const misses = %d\n", L1_const_miss);
   printf("L2_write_miss = %d\n", L2_write_miss);
   printf("L2_write_hit = %d\n", L2_write_hit);
   printf("L2_read_miss = %d\n", L2_read_miss);
   printf("L2_read_hit = %d\n", L2_read_hit);
   printf("made_read_mfs = %d\n", made_read_mfs);
   printf("made_write_mfs = %d\n", made_write_mfs);
   printf("freed_read_mfs = %d\n", freed_read_mfs);
   printf("freed_L1write_mfs = %d\n", freed_L1write_mfs);
   printf("freed_L2write_mfs = %d\n", freed_L2write_mfs);
   printf("freed_dummy_read_mfs = %d\n", freed_dummy_read_mfs);

   printf("gpgpu_n_mem_read_local = %d\n", gpgpu_n_mem_read_local);
   printf("gpgpu_n_mem_write_local = %d\n", gpgpu_n_mem_write_local);
   printf("gpgpu_n_mem_read_global = %d\n", gpgpu_n_mem_read_global);
   printf("gpgpu_n_mem_write_global = %d\n", gpgpu_n_mem_write_global);
   printf("gpgpu_n_mem_texture = %d\n", gpgpu_n_mem_texture);
   printf("gpgpu_n_mem_const = %d\n", gpgpu_n_mem_const);

   printf("max_n_mshr_used = ");
   for (i=0; i< gpu_n_shader; i++) printf("%d ", sc[i]->max_n_mshr_used);
   printf("\n");

   if (gpgpu_cache_dl2_opt) {
      L2c_print_stat( );
   }
   for (i=0;i<gpu_n_mem;i++) {
      dram_print(dram[i],stdout);
   }

   for (i=0, j=0, k=0;i<gpu_n_shader;i++) {
      shd_cache_print(sc[i]->L1cache,stdout);
      j+=sc[i]->L1cache->miss;
      k+=sc[i]->L1cache->access;
   }
   printf("L1 Data Cache Total Miss Rate = %0.3f\n", (float)j/k);

   for (i=0,j=0,k=0;i<gpu_n_shader;i++) {
      shd_cache_print(sc[i]->L1texcache,stdout);
      j+=sc[i]->L1texcache->miss;
      k+=sc[i]->L1texcache->access;
   }
   printf("L1 Texture Cache Total Miss Rate = %0.3f\n", (float)j/k);

   for (i=0,j=0,k=0;i<gpu_n_shader;i++) {
      shd_cache_print(sc[i]->L1constcache,stdout);
      j+=sc[i]->L1constcache->miss;
      k+=sc[i]->L1constcache->access;
   }
   printf("L1 Const Cache Total Miss Rate = %0.3f\n", (float)j/k);

   if (gpgpu_cache_dl2_opt) {
      for (i=0,j=0,k=0;i<gpu_n_mem;i++) {
         shd_cache_print(dram[i]->L2cache,stdout);
         j+=dram[i]->L2cache->miss;
         k+=dram[i]->L2cache->access;
      }
      printf("L2 Cache Total Miss Rate = %0.3f\n", (float)j/k);
   }
   printf("n_regconflict_stall = %d\n", n_regconflict_stall);

   if (gpgpu_simd_model == DWF) {
      dwf_print_stat(stdout);
   }

   if (gpgpu_simd_model == POST_DOMINATOR) {
      printf("num_warps_issuable:");
      for (i=0;i<(gpu_n_warp_per_shader+1);i++) {
         printf("%d ", num_warps_issuable[i]);
      }
      printf("\n");
   }
   if (gpgpu_strict_simd_wrbk) {
      printf("warp_conflict_at_writeback = %d\n", warp_conflict_at_writeback);
   }

   printf("gpgpu_commit_pc_beyond_two = %d\n", gpgpu_commit_pc_beyond_two);

   print_shader_cycle_distro( stdout );

   print_thread_pc_histogram( stdout );

   if (gpgpu_cflog_interval != 0) {
      spill_log_to_file (stdout, 1, gpu_sim_cycle);
      insn_warp_occ_print(stdout);
   }
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }
   time_vector_print();

   fflush(stdout);
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper function for shader cores' memory system: 
////////////////////////////////////////////////////////////////////////////////////

// a hack to make the size of a packet discrete multiples of the interconnect's flit_size.
static inline
unsigned int fill_to_next_flit(unsigned int size) 
{
   assert (g_network_mode == INTERSIM);
   return size;
}



unsigned char check_icnt_has_buffer(unsigned long long int *addr, int *bsize, 
                                    int n_addr, int sid )
{
   int i;
   addrdec_t tlx;
   static unsigned int *req_buffer = NULL;
   //the req_buf size can be equal to gpu_n_mem ; gpu_n_shader is added to make it compatible
   //with the case where a mem controller is sending to shd 
   if (!req_buffer) req_buffer = (unsigned int*)malloc((gpu_n_mem+gpu_n_tpc)*sizeof(unsigned int));
   memset(req_buffer, 0, (gpu_n_mem+gpu_n_tpc)*sizeof(unsigned int));

   // aggregate all buffer requirement of all memory accesses by dram chips
   for (i=0; i< n_addr; i++) {
      addrdec_tlx(addr[i],&tlx);
      req_buffer[tlx.chip] += fill_to_next_flit(bsize[i]);
   }

   // if it is using L1 writeback cache, reserves in the interconnectino buffer a cache line for each thread
   if ( !gpgpu_cache_wt_through && !gpgpu_no_dl1 ) {
      assert( g_network_mode == INTERSIM );
      req_buffer[0] += sc[sid]->L1cache->line_sz * gpu_n_thread_per_shader;
   }

   int tpc_id = sid / gpu_concentration;

   return icnt_has_buffer(tpc_id, req_buffer);
}

unsigned char single_check_icnt_has_buffer(int chip, int sid, unsigned char is_write )
{
   static unsigned int *req_buffer = NULL;
   //the req_buf size can be equal to gpu_n_mem ; gpu_n_shader is added to make it compatible
   //with the case where a mem controller is sending to shd 
   if (!req_buffer) req_buffer = (unsigned int*)malloc((gpu_n_mem+gpu_n_tpc)*sizeof(unsigned int));
   memset(req_buffer, 0, (gpu_n_mem+gpu_n_tpc)*sizeof(unsigned int));

   // aggregate all buffer requirement of all memory accesses by dram chips

   int b_size;
   if (is_write)
      b_size = sc[sid]->L1cache->line_sz;
   else
      b_size = READ_PACKET_SIZE;
   req_buffer[chip] += fill_to_next_flit(b_size);

   if ( !gpgpu_cache_wt_through && !gpgpu_no_dl1 ) {
      req_buffer[0] += sc[sid]->L1cache->line_sz * gpu_n_thread_per_shader;
   }

   int tpc_id = sid / gpu_concentration;

   return icnt_has_buffer(tpc_id, req_buffer);
}

int max_n_addr = 0;

// Check the memory system for buffer availability
unsigned char fq_has_buffer(unsigned long long int *addr, int *bsize, 
                            int n_addr, int sid )
{
   return check_icnt_has_buffer(addr, bsize, n_addr, sid);
}

// Takes in memory address and their parameters and pushes to the fetch queue 
unsigned char fq_push(unsigned long long int addr, int bsize, unsigned char write, unsigned long long int partial_write_mask, 
                      int sid, int mshr_idx, mshr_entry* mshr, int cache_hits_waiting,
                      enum mem_access_type mem_acc, address_type pc) 
{
   mem_fetch_t *mf;

   mf = (mem_fetch_t*) calloc(1,sizeof(mem_fetch_t));
   mf->request_uid = g_next_request_uid++;
   mf->addr = addr;
   mf->nbytes_L1 = bsize;
   mf->sid = sid;
   mf->source_node = sid / gpu_concentration;
   mf->wid = mshr_idx/warp_size;
   mf->cache_hits_waiting = cache_hits_waiting;
   mf->txbytes_L1 = 0;
   mf->rxbytes_L1 = 0;  
   mf->mshr_idx = mshr_idx;
   mf->mshr = mshr;
   if (mshr) mshr->mf = (void*)mf; // for debugging
   mf->write = write;

   //SEAN
   if(mshr != NULL)
     mf->is_atom = mshr->is_atom;

   if (write)
      made_write_mfs++;
   else
      made_read_mfs++;
   memlatstat_start(mf);
   addrdec_tlx(addr,&mf->tlx);
   mf->bank = mf->tlx.bk;
   mf->chip = mf->tlx.chip;
   if (gpgpu_cache_dl2_opt)
      mf->nbytes_L2 = dram[mf->tlx.chip]->L2cache->line_sz;
   else
      mf->nbytes_L2 = 0;
   mf->txbytes_L2 = 0;
   mf->rxbytes_L2 = 0;  

   mf->write_mask = partial_write_mask;
   if (!write) assert(partial_write_mask == NO_PARTIAL_WRITE);

   // stat collection codes 
   mf->mem_acc = mem_acc;
   mf->pc = pc;
   if (mf->mshr != NULL) {
      if (mf->mshr->islocal) {
         gpgpu_n_mem_read_local  += (write)? 0 : 1;
         // gpgpu_n_mem_write_local += (write)? 1 : 0; // migrated to other code because mshr is not available for writes!
      } else if (mf->mshr->istexture) {
         gpgpu_n_mem_texture++; // read only
      } else if (mf->mshr->isconst) {
         gpgpu_n_mem_const++;   // read only
      } else { // global
         gpgpu_n_mem_read_global  += (write)? 0 : 1;
         // gpgpu_n_mem_write_global += (write)? 1 : 0;
      }
   }

   return(issue_mf_from_fq(mf));

}

int issue_mf_from_fq(mem_fetch_t *mf){
   int destination; // where is the next level of memory?
   destination = mf->tlx.chip;
   int tpc_id = mf->sid / gpu_concentration;

   if (mf->mshr) mshr_update_status(mf->mshr, IN_ICNT2MEM);
   if (!mf->write) {
      mf->type = RD_REQ;
      assert( mf->timestamp == (gpu_sim_cycle+gpu_tot_sim_cycle) );
      time_vector_update(mf->mshr->inst_uid, MR_ICNT_PUSHED, gpu_sim_cycle+gpu_tot_sim_cycle, mf->type );
      icnt_push(tpc_id, mem2device(destination), (void*)mf, READ_PACKET_SIZE);
   } else {
      mf->type = WT_REQ;
      icnt_push(tpc_id, mem2device(destination), (void*)mf, mf->nbytes_L1);
      gpgpu_n_sent_writes++;
      assert( mf->timestamp == (gpu_sim_cycle+gpu_tot_sim_cycle) );
      time_vector_update(mf->request_uid, MR_ICNT_PUSHED, gpu_sim_cycle+gpu_tot_sim_cycle, mf->type ) ;
   }

   return 0;
}


inline void fill_shd_L1_with_new_line(shader_core_ctx_t * sc, mem_fetch_t * mf) {
   unsigned long long int repl_addr = -1;
   // When the data arrives, it flags all the appropriate MSHR
   // entries accordingly (by checking the address in each entry ) 
   memlatstat_read_done(mf);

   mshr_update_status(mf->mshr, FETCHED);
   if (mf->mshr->isconst) {
      shader_update_mshr(sc,mf->addr,mf->mshr_idx, CONSTC);
   } else if (mf->mshr->istexture) {
      shader_update_mshr(sc,mf->addr,mf->mshr_idx, TEXTC);
   } else
      shader_update_mshr(sc,mf->addr,mf->mshr_idx, DCACHE);


   if (mf->mshr->istexture) {
      if (!shd_cache_probe(sc->L1texcache, mf->addr))
         shd_cache_fill(sc->L1texcache,mf->addr,sc->gpu_cycle);
      repl_addr = -1;
   } else if (mf->mshr->isconst) {
      if (!shd_cache_probe(sc->L1constcache, mf->addr))
         shd_cache_fill(sc->L1constcache,mf->addr,sc->gpu_cycle);
      repl_addr = -1;
   } else {
     if (!gpgpu_no_dl1) {
       if (!shd_cache_probe(sc->L1cache, mf->addr)) {
	 repl_addr = shd_cache_fill(sc->L1cache,mf->addr,sc->gpu_cycle);
       }
     }
   }

   // only perform a write on cache eviction (write-back policy)
   if ((repl_addr != -1) && !(gpgpu_cache_wt_through || gpgpu_no_dl1)) { // Always false for no_dl1 
      L1_write_miss++;
      fq_push(repl_addr, sc->L1cache->line_sz, 1, NO_PARTIAL_WRITE, sc->sid, mf->mshr_idx, NULL, 0, 
              GLOBAL_ACC_W, mf->pc);

      if (mf->mshr->islocal) {
         gpgpu_n_mem_write_local++;
      } else {
         gpgpu_n_mem_write_global++;
      }
   }
   freed_read_mfs++;
   free(mf);
}

unsigned char fq_pop(int tpc_id) 
{
   mem_fetch_t *mf;

   static unsigned int *wb_size = NULL;
   if ( !wb_size ) {
      wb_size = (unsigned int*) calloc(gpu_n_mem+gpu_n_tpc, sizeof(unsigned int)); 
      //because wb_size is going to be passed to icnt_has_buffer its size must be n_mem + n_shd
      wb_size[0] = sc[0]->L1cache->line_sz;
   }

   // check resource for writeback before popping from interconnect (conservative)
   if ( !(gpgpu_cache_wt_through || gpgpu_no_dl1) && !icnt_has_buffer(tpc_id, wb_size) ) {
      gpu_stall_wr_back++;
      return 0;
   }

   mf = (mem_fetch_t*) icnt_pop(tpc_id);

   // if there is a memory fetch request coming back, forward it to the proper shader core
   if (mf) {
      assert(mf->type == REPLY_DATA);
      time_vector_update(mf->mshr->inst_uid ,MR_2SH_FQ_POP,gpu_sim_cycle+gpu_tot_sim_cycle, mf->type ) ;
      //TEST
      printf("SEAN:  Filling shader %i with %llu data\n", mf->sid, mf->addr);
      //TEST*/
      fill_shd_L1_with_new_line(sc[mf->sid], mf);
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////

int issue_block2core( shader_core_ctx_t *shdr, int grid_num  ) 
{
   int i;
   int tid, nthreads_2beissued, more_threads;
   int  nthreads_in_block= 0;
   int start_thread = 0;
   int end_thread = shdr->n_threads;
   int cta_id=-1;
   int cta_size=0;
   int padded_cta_size;

   cta_size = ptx_sim_cta_size();
   padded_cta_size = cta_size;

   if (gpgpu_spread_blocks_across_cores) {//should be if  muliple CTA per shader supported
      for ( i=0;i<max_cta_per_shader(shdr);i++ ) { //try to find next empty cta slot
         if (shdr->cta_status[i]==0) { //
            cta_id=i;
            break;
         }
      }
      assert( cta_id!=-1);//must have found a CTA to run
      if (padded_cta_size%warp_size) {
         padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);
      }
      start_thread = cta_id * padded_cta_size;
      end_thread  = start_thread +  cta_size;
      shader_reinit(shdr,start_thread, end_thread); 
   } else {
      abort();
   }
   // issue threads in blocks (if it is specified)
   if (gpgpu_spread_blocks_across_cores) {
      for (i = start_thread; i<end_thread; i++) {  //setup the block
         nthreads_in_block +=  ptx_sim_init_thread(&shdr->thread[i].ptx_thd_info,shdr->sid,i,cta_size-(i-start_thread),shdr->n_threads/*cta_size*/);
      }
      shader_init_CTA(shdr, start_thread, end_thread);
      nthreads_2beissued =  nthreads_in_block;
      shdr->cta_status[cta_id]+=nthreads_2beissued;
      assert( nthreads_2beissued ); //we should have not reached this point if there is no more thread to -
   } else {
      nthreads_2beissued = nthreads_in_block;
   }
   assert(nthreads_2beissued <= shdr->n_threads); //confirm threads to be issued is less than or equal to number of threads supported by microarchitecture

   int n_cta_issued= nthreads_2beissued/cta_size ;//+ nthreads_2beissued%cta_size; 
   shdr->n_active_cta +=  n_cta_issued;
   shader_CTA_count_log(shdr->sid, n_cta_issued);
   g_total_cta_left-= n_cta_issued;

   more_threads = 1;
   if (gpgpu_spread_blocks_across_cores) {
      nthreads_2beissued += start_thread;
   }
   printf("Shader %d initializing CTA #%d with hw tids from %d to %d @(%lld,%lld)", 
          shdr->sid, cta_id, start_thread, nthreads_2beissued, gpu_sim_cycle, gpu_tot_sim_cycle );
   printf(" shdr->not_completed = %d\n", shdr->not_completed);

   for (tid=start_thread;tid<nthreads_2beissued;tid++) {

      // reset complete flag for stream
      shdr->not_completed += 1;
      assert( shdr->thread[(tid - (tid % warp_size))].n_completed > 0 );
      assert( shdr->thread[(tid - (tid % warp_size))].n_completed <= warp_size);
      shdr->thread[(tid - (tid % warp_size))].n_completed--;

      // set avail4fetch flag to ready
      shdr->thread[tid].avail4fetch = 1;
      assert( shdr->thread[tid - (tid%warp_size)].n_avail4fetch < warp_size );
      shdr->thread[tid - (tid%warp_size)].n_avail4fetch++;

      g_nthreads_issued++;
   }

   if (!nthreads_in_block) more_threads = 0;
   return more_threads; //if there are no more threads to be issued, return 0
}

///////////////////////////////////////////////////////////////////////////////////////////
// wrapper code to to create an illusion of a memory controller with L2 cache.
// 
int mem_ctrl_full( int mc_id ) 
{
   if (gpgpu_cache_dl2_opt) {
      return(dq_full(dram[mc_id]->cbtoL2queue) || dq_full(dram[mc_id]->cbtoL2writequeue));
   } else {
      return( gpgpu_dram_sched_queue_size && dram_full(dram[mc_id]) );
   }
}

//#define DEBUG_PARTIAL_WRITES
void mem_ctrl_push( int mc_id, mem_fetch_t* mf ) 
{
   int debug_partial_writes = 0;
   if (gpgpu_cache_dl2_opt) {
      if (gpgpu_l2_readoverwrite && mf->write)
         dq_push(dram[mc_id]->cbtoL2writequeue, mf);
      else
         dq_push(dram[mc_id]->cbtoL2queue, mf);
      if (mf->mshr) mshr_update_status(mf->mshr, IN_CBTOL2QUEUE);
   } else {
      addrdec_t tlx;
      addrdec_tlx(mf->addr, &tlx);
      if (gpgpu_partial_write_mask && mf->write) {
         assert( gpgpu_no_dl1 ); // gpgpu_partial_write_mask is not supported with caches for now
#define NO_RW 0
#define ONLY_W 1
#define BOTH_RW 2
         int i;
         unsigned int shift = 0;
         unsigned long long int current_chunk=0;
         int n_chunks = 4 ; //Generalize me!
         int bytes_per_chunk = 16;
         int chunk[32];
         int chunk_size[32];
         int n_real_chunk=0; //counts total number of chunks that will be sent to mem
#ifdef DEBUG_PARTIAL_WRITES
         if (mf->write_mask != 0xFFFFFFFFFFFFFFFF) {
            debug_partial_writes=1;
            printf("writemask = %016llx\n", mf->write_mask);
         }
#endif
         if (debug_partial_writes) printf("chunk[]:");
         for (i=0;i<n_chunks ;i++) {
            chunk[i]=chunk_size[i]=-1;//initialize
            current_chunk = (mf->write_mask >> shift) & 0xFFFF; 
            if (current_chunk == 0) {
               chunk[i]=NO_RW;
            } else if ((current_chunk ^ 0xFFFF) != 0) {
               if (gpgpu_partial_write_mask==2) {
                  chunk[i]=BOTH_RW;
               } else {
                  chunk[i]=ONLY_W;
               }
            } else if ((current_chunk & 0xFFFF)==0xFFFF) {
               chunk[i]=ONLY_W;
            } else {
               assert(0);
            }
            shift+= 16;
            if (debug_partial_writes) printf("%d ", chunk[i]);
         }
         if (debug_partial_writes) printf("\n");
         //Now find the continuous portions
         int first_chunk=0;
         int current_size = 1;
         chunk[n_chunks]=-1; //ensure the boundary case does not mess up things
         for (i=0;i<n_chunks;i++) {
            if (chunk[i]!=chunk[i+1]) {
               chunk_size[first_chunk]=current_size;
               first_chunk=i+1;
               current_size=1;
               if (chunk[i]!=NO_RW) {
                  n_real_chunk++; 
               }
            } else {
               current_size++;
            }
         }
         i=0;
         int nbytes=0;
         int used_up_original_mf = 0;
         mem_fetch_t* new_mf=NULL;
         while (i<n_chunks) {
            nbytes = chunk_size[i]*bytes_per_chunk;
            if (chunk[i]!=NO_RW) {
               if (chunk[i]==BOTH_RW) {
                  new_mf = (mem_fetch_t*) malloc(sizeof(mem_fetch_t));
                  memcpy(new_mf,mf,sizeof(mem_fetch_t));
                  new_mf->nbytes_L1 = nbytes;
                  new_mf->write = 0; 
                  new_mf->addr += i*bytes_per_chunk;
                  new_mf->type=DUMMY_READ;
                  if (debug_partial_writes) printf("%d byte read\n", new_mf->nbytes_L1);
                  dram_push(dram[mc_id], 
                            tlx.bk, tlx.row, tlx.col, 
                            new_mf->nbytes_L1, //size 
                            new_mf->write,//read 
                            new_mf->wid, new_mf->sid, new_mf->cache_hits_waiting,
                            new_mf->addr, new_mf);
                  memlatstat_dram_access(new_mf, mc_id, tlx.bk);
               }
               if (!used_up_original_mf) {
                  used_up_original_mf=1;
                  new_mf = mf;               
               } else {
                  new_mf = (mem_fetch_t*) malloc(sizeof(mem_fetch_t));
                  memcpy(new_mf,mf,sizeof(mem_fetch_t));
               }
               new_mf->nbytes_L1 = nbytes;
               assert( new_mf->write ==1); 
               new_mf->addr += i*bytes_per_chunk;
               if (debug_partial_writes) printf("%d byte write\n", new_mf->nbytes_L1);
               dram_push(dram[mc_id], 
                         tlx.bk, tlx.row, tlx.col, 
                         new_mf->nbytes_L1, //size 
                         new_mf->write,//write 
                         new_mf->wid, new_mf->sid, new_mf->cache_hits_waiting,
                         new_mf->addr, new_mf);
               memlatstat_dram_access(new_mf, mc_id, tlx.bk);
            }
            i+=chunk_size[i];
         }
      } else {
         dram_push(dram[mc_id], 
                   tlx.bk, tlx.row, tlx.col, 
                   mf->nbytes_L1, mf->write, 
                   mf->wid, mf->sid, mf->cache_hits_waiting, mf->addr, mf);
         memlatstat_dram_access(mf, mc_id, tlx.bk);
         if (mf->mshr) mshr_update_status(mf->mshr, IN_DRAM_REQ_QUEUE);
      }
   }
}

void* mem_ctrl_pop( int mc_id ) 
{
   mem_fetch_t* mf;
   if (gpgpu_cache_dl2_opt) {
      mf = dq_pop(dram[mc_id]->L2tocbqueue);
      if (mf && mf->mshr && mf->mshr->inst.callback.function) {
         dram_callback_t* cb = &(mf->mshr->inst.callback);
	 //TEST
	 printf("SEAN:  Calling atom_callback function.  Time:  %llu\n", gpu_sim_cycle);
	 //TEST*/
         cb->function(cb->instruction, cb->thread);
      }
      return mf;
   } else {
      mf = dq_pop(dram[mc_id]->returnq); //dram_pop(dram[mc_id]);
      if (mf) mf->type = REPLY_DATA;
      if (mf && mf->mshr && mf->mshr->inst.callback.function) {
         dram_callback_t* cb = &(mf->mshr->inst.callback);
         cb->function(cb->instruction, cb->thread);
      }
      return mf;
   }
}

void* mem_ctrl_top( int mc_id ) 
{
   mem_fetch_t* mf;
   if (gpgpu_cache_dl2_opt) {
      return dq_top(dram[mc_id]->L2tocbqueue);
   } else {
      mf = dq_top(dram[mc_id]->returnq);//dram_top(dram[mc_id]);
      if (mf) mf->type = REPLY_DATA;
      return mf ;//dram_top(dram[mc_id]);
   }
}
//////////////////////////////////////////////// 
// L2 access functions

extern unsigned long long int addrdec_mask[5];

// L2 Cache Creation 
void L2c_create ( dram_t* dram_p, const char* cache_opt, const char* queue_opt )
{
   unsigned int shd_n_set;
   unsigned int shd_linesize;
   unsigned int shd_n_assoc;
   unsigned char shd_policy;

   unsigned int L2c_cb_L2_length;
   unsigned int L2c_cb_L2w_length;
   unsigned int L2c_L2_dm_length;
   unsigned int L2c_dm_L2_length;
   unsigned int L2c_dm_L2w_length;
   unsigned int L2c_L2_cb_length;
   unsigned int L2c_L2_cb_minlength;
   unsigned int L2c_L2_dm_minlength;

   sscanf(cache_opt,"%d:%d:%d:%c", 
          &shd_n_set, &shd_linesize, &shd_n_assoc, &shd_policy);
   char L2c_name[32];
   snprintf(L2c_name, 32, "L2c_%03d", dram_p->id);
   dram_p->L2cache = shd_cache_create(L2c_name, 
                                      shd_n_set, shd_n_assoc, shd_linesize, 
                                      shd_policy, 16, ~addrdec_mask[CHIP]);

   sscanf(queue_opt,"%d:%d:%d:%d:%d:%d:%d:%d", 
          &L2c_cb_L2_length, &L2c_cb_L2w_length, &L2c_L2_dm_length, 
          &L2c_dm_L2_length, &L2c_dm_L2w_length, &L2c_L2_cb_length,
          &L2c_L2_cb_minlength, &L2c_L2_dm_minlength );
   //(<name>,<latency>,<min_length>,<max_length>)
   dram_p->cbtoL2queue        = dq_create("cbtoL2queue",       0,0,L2c_cb_L2_length); 
   dram_p->cbtoL2writequeue   = dq_create("cbtoL2writequeue",  0,0,L2c_cb_L2w_length); 
   dram_p->L2todramqueue      = dq_create("L2todramqueue",     0,L2c_L2_dm_minlength,L2c_L2_dm_length);
   dram_p->dramtoL2queue      = dq_create("dramtoL2queue",     0,0,L2c_dm_L2_length);
   dram_p->dramtoL2writequeue = dq_create("dramtoL2writequeue",0,0,L2c_dm_L2w_length);
   dram_p->L2tocbqueue        = dq_create("L2tocbqueue",       0,L2c_L2_cb_minlength,L2c_L2_cb_length);

   dram_p->L2todram_wbqueue   = dq_create("L2todram_wbqueue",  0,L2c_L2_dm_minlength,
                                          L2c_L2_dm_minlength + gpgpu_dram_sched_queue_size + L2c_dm_L2_length);
}

void L2c_qlen ( dram_t *dram_p )
{
   printf("\n");
   printf("cb->L2{%d}\tcb->L2w{%d}\tL2->cb{%d}\n", 
          dram_p->cbtoL2queue->length, 
          dram_p->cbtoL2writequeue->length, 
          dram_p->L2tocbqueue->length);
   printf("dm->L2{%d}\tdm->L2w{%d}\tL2->dm{%d}\tL2->wb_dm{%d}\n", 
          dram_p->dramtoL2queue->length, 
          dram_p->dramtoL2writequeue->length, 
          dram_p->L2todramqueue->length,
          dram_p->L2todram_wbqueue->length);
}

// service memory request in icnt-to-L2 queue, writing to L2 as necessary
// (if L2 writeback miss, writeback to memory) 
void L2c_service_mem_req ( dram_t* dram_p, int dm_id )
{
   mem_fetch_t* mf;
   //SEAN
   unsigned int first=0; //indicate if this is first cycle associated with request

   if (!L2request[dm_id]) {
     //SEAN
     first = 1;
     //if not servicing L2 cache request..
     L2request[dm_id] = (mem_fetch_t*) dq_pop(dram_p->cbtoL2queue); //..then get one
     if (!L2request[dm_id]) {
       L2request[dm_id] = (mem_fetch_t*) dq_pop(dram_p->cbtoL2writequeue);
     }
   }

   mf = L2request[dm_id];

   if (!mf) return;

   //SEAN
   //Is this associated with an atomic operation? (Either a load or store)
   //need to find a better way than to have every write go through this (i.e. mark stores with is_atom).  If there's not a corresponding atomic load for this, it will be caught below
   if((first) && (mf->is_atom)) { 
     //Determine the cache line it's associated with
     unsigned long long int packed_addr;
     unsigned long long int tag; 
     if (dram_p->L2cache->bank_mask)
       packed_addr = addrdec_packbits(dram_p->L2cache->bank_mask, mf->addr, 64, 0);
     else
       packed_addr = mf->addr;

     tag = packed_addr >> (dram_p->L2cache->line_sz_log2 + dram_p->L2cache->nset_log2);

     //     if(!mf->write) { //load request
     if(mf->type == RD_REQ) {
       //Allocate entry into atom_q
       atom_q *add;
       add = (atom_q *) malloc(sizeof(atom_q));
       if(atom_q_head == NULL) {
	 atom_q_head = add;
       } else {
	 atom_q_tail->next = add;
       }
       atom_q_tail = add;
       add->next = NULL;
       add->tag = tag;
       add->mf = mf;

       //TEST
       printf("SEAN (%llu):  allocated entry for tag %llu.\n", gpu_sim_cycle, tag);
       //TEST*/

       //if one or more entries in atom_q exist for same line, stop processing 'mf' for now
       atom_q *curr = atom_q_head;
       atom_q *prev = curr;
       while(curr != add) {
	 if(curr->tag == tag) break;
	 prev = curr;
	 curr = curr->next;
       }
       if(curr != add) {
	 //you're either a completely separate request (for same address) => stall me
	 if(curr->mf != add->mf) {
	   //TEST
	   printf("SEAN (%llu):  not processing because %llu has an outstanding request\n", gpu_sim_cycle, tag);
	   //TEST*/
	   //Update mshr entry to indicate stalled
	   mshr_update_status(mf->mshr, STALLED_IN_ATOM_Q);
	   mf = NULL; //stop processing for now
	   L2request[dm_id] = NULL;
	 } else {
	 //or you're a previous entry of the same fetch I'm responsible for => issue you and delete me
	   //Unnecessary?
	   mf = curr->mf;
	   L2request[dm_id] = mf;

	   //find entry in Q prior to 'add'
	   while(curr->next != add) {
	     curr = curr->next;
	   }
	   atom_q_tail = curr;
	   curr->next = NULL;
	   free(add);

	   /*if(curr == atom_q_head) {
	     atom_q_head = atom_q_head->next;
	     if(atom_q_head == NULL) atom_q_tail = NULL;
	     free(add); //delete entry for load corresponding to this store
	     //TEST
	     printf("SEAN (%llu):  found corresponding load at head of Queue.\n", gpu_sim_cycle);
	     //TEST/
	   } else {
	     prev->next = curr->next;
	     free(add); //delete entry for load corresponding to this store
	   }*/
	 }
       }
     } else { //store request
       //Find corresponding load in atom_q & remove - HAVE TO ASSUME THAT THIS STORE *MUST* CORRESPOND TO THE FIRST LOAD FOR THE LINE ENCOUNTERED IN THE LIST (not sure of a good way to assert that this is true)
       //TEST
       printf("SEAN (%llu):  Write request received for tag %llu.\n", gpu_sim_cycle, tag);
       //TEST*/
       atom_q *curr = atom_q_head;
       atom_q *prev = curr;
       while(curr != NULL) {
	 if(curr->tag == tag) break;
	 prev = curr;
	 curr = curr->next;
       }

       assert(curr != NULL); //the only way the loop should exit is through the 'break' (otherwise, there's no corresponding atomic load for this store)

       if(curr == atom_q_head) {
	 atom_q_head = atom_q_head->next;
	 if(atom_q_head == NULL) atom_q_tail = NULL;
	 prev = atom_q_head;
	 free(curr); //delete entry for load corresponding to this store
	 //TEST
	 printf("SEAN (%llu):  found corresponding load at head of Queue.\n", gpu_sim_cycle);
	 //TEST*/
       } else {
	 prev->next = curr->next;
	 free(curr); //delete entry for load corresponding to this store
       }

       //Re-issue next waiting request (ideally should be pushed to the front of the queue to be handled next)
       //i.e. push to front of "retry" (dram_p->cbtoL2queue) queue
       //can I create dq_push_front function (in delayqueue.c) without breaking functionality of the delay queue (e.g. violating assumptions about not having more than one entry with 0 time_elapsed, or otherwise having a minimum time_elapsed difference between two consecutive entries?).  Need to evaluate this before modifying this to allow a push_front
       curr = prev;
       while (curr != NULL) {
	 if(curr->tag == tag) break;
	 curr = curr->next;
       }
       
       if(curr != NULL) {
	 assert(curr->tag == tag); //the only way curr could not be NULL is if it's pointing to an entry that matches 'tag'
	 mem_fetch_t *new_mf = curr->mf;
	 dq_push(dram_p->cbtoL2queue, new_mf);
	 //TEST
	 printf("SEAN (%llu):  found another request for the tag (%llu) the atomic op this write corresponds to was servicing\n", gpu_sim_cycle, tag);
	 //TEST*/
       } 
       //TEST
       else {
	 printf("SEAN (%llu):  Didn't find another request for tag %llu\n", gpu_sim_cycle, tag);
       }
       //TEST*/
     }
   }
   //SEAN*/

   if(mf != NULL) { //atomic handling above make this possible
     switch (mf->type) {
     case RD_REQ:
     case WT_REQ: {
       shd_cache_line_t *hit_cacheline = shd_cache_access(dram_p->L2cache,
							  mf->addr,
							  4, mf->write,
							  gpu_sim_cycle);
       
       if (hit_cacheline) { //L2 Cache Hit; reads are sent as a single command and need to be stored
	 if (!mf->write) { //L2 Cache Read
	   if ( dq_full(dram_p->L2tocbqueue) ) {
	     dram_p->L2cache->access--;
	   } else {
	     mf->type = REPLY_DATA;
	     dq_push(dram_p->L2tocbqueue, mf);
	     // at this point, should first check if earlier L2 miss is ready to be serviced
	     // if so, service earlier L2 miss first
	     L2request[dm_id] = NULL; //finished servicing
	     L2_read_hit++;
	     memlatstat_icnt2sh_push(mf);
	     if (mf->mshr) mshr_update_status(mf->mshr, IN_L2TOCBQUEUE_HIT);
	   }
	 } else { //L2 Cache Write aka servicing L1 Writeback
	   L2request[dm_id] = NULL;    
	   L2_write_hit++;
	   freed_L1write_mfs++;
	   /*THIS ISN'T WORKING TO COMPLETE THE WRITE INSN
	   mshr_update_status(mf->mshr, FETCHED); //to allow insn to be retired
	   dq_push(sc[mf->sid]->return_queue, mf->mshr);
	   //THIS ISN'T WORKING TO COMPLETE THE WRITE INSN*/
	   shader_update_mshr(sc[mf->sid],mf->addr,mf->mshr_idx, DCACHE);
	   free(mf); //writeback from L1 successful
	   gpgpu_n_processed_writes++;
	 }
       } else {
	 // L2 Cache Miss; issue commands accordingly
	 if ( dq_full(dram_p->L2todramqueue) ) {
	   dram_p->L2cache->miss--;
	   dram_p->L2cache->access--;
	 } else {
	   if (!mf->write) {
	     dq_push(dram_p->L2todramqueue, mf);
	   } else {
	     // if request is writeback from L1 and misses, 
	     // then redirect mf writes to dram (no write allocate)
	     mf->nbytes_L2 = mf->nbytes_L1 - READ_PACKET_SIZE;
	     dq_push(dram_p->L2todramqueue, mf);
	   }
	   if (mf->mshr) mshr_update_status(mf->mshr, IN_L2TODRAMQUEUE);
	   L2request[dm_id] = NULL;
	 }
       }
     }
       break;
     default: assert(0);
     }
   }
}

// service memory request in L2todramqueue, pushing to dram 
void L2c_push_miss_to_dram ( dram_t* dram_p )
{
   mem_fetch_t* mf;

   if ( gpgpu_dram_sched_queue_size && dram_full(dram_p) ) return;

   mf = (mem_fetch_t*) dq_pop(dram_p->L2todram_wbqueue); //prioritize writeback
   if (!mf) mf = (mem_fetch_t*) dq_pop(dram_p->L2todramqueue);
   if (mf) {
      if (mf->write) {
         L2_write_miss++;
      } else {
         L2_read_miss++;
      }
      memlatstat_dram_access(mf, dram_p->id, mf->tlx.bk);
      dram_push(dram_p,
                mf->tlx.bk, mf->tlx.row, mf->tlx.col,
                mf->nbytes_L2, mf->write,
                mf->wid, mf->sid, mf->cache_hits_waiting, mf->addr, mf);
      if (mf->mshr) mshr_update_status(mf->mshr, IN_DRAM_REQ_QUEUE);
   }
}

//Service writes that are finished in Dram 
//only updates the stats and frees the mf
void dramtoL2_service_write(mem_fetch_t * mf) {
   freed_L2write_mfs++;
   free(mf);
   gpgpu_n_processed_writes++;
}

// pop completed memory request from dram and push it to dram-to-L2 queue 
void L2c_get_dram_output ( dram_t* dram_p ) 
{
   mem_fetch_t* mf;
   mem_fetch_t* mf_top;
   if ( dq_full(dram_p->dramtoL2queue) || dq_full(dram_p->dramtoL2writequeue) ) return;
   mf_top = (mem_fetch_t*) dram_top(dram_p); //test
   mf = (mem_fetch_t*) dram_pop(dram_p);
   assert (mf_top==mf );
   if (mf) {
      if (gpgpu_l2_readoverwrite && mf->write)
         dq_push(dram_p->dramtoL2writequeue, mf);
      else
         dq_push(dram_p->dramtoL2queue, mf);
      if (mf->mshr) mshr_update_status(mf->mshr, IN_DRAMTOL2QUEUE);
   }
}

void get_dram_output ( dram_t* dram_p ) 
{
   mem_fetch_t* mf;
   mem_fetch_t* mf_top;
   mf_top = (mem_fetch_t*) dram_top(dram_p); //test
   if (mf_top) {
      if (mf_top->type == DUMMY_READ) {
         dram_pop(dram_p);
         free(mf_top);
         freed_dummy_read_mfs++;
         return;
      }
   }
   if (gpgpu_cache_dl2_opt) {
      L2c_get_dram_output( dram_p );
   } else {
      if ( dq_full(dram_p->returnq) ) return;
      mf = (mem_fetch_t*) dram_pop(dram_p);
      assert (mf_top==mf );
      if (mf) {
         dq_push(dram_p->returnq, mf);
         if (mf->mshr) mshr_update_status(mf->mshr, IN_DRAMRETURN_Q);
      }
   }
}

// service memory request in dramtoL2queue, writing to L2 as necessary
// (may cause cache eviction and subsequent writeback) 
void L2c_process_dram_output ( dram_t* dram_p, int dm_id ) 
{
   static mem_fetch_t **L2dramout = NULL; 
   static unsigned long long int *wb_addr = NULL;
   if (!L2dramout) L2dramout = (mem_fetch_t**)calloc(gpu_n_mem, sizeof(mem_fetch_t*));
   if (!wb_addr) {
      int i;
      wb_addr = (unsigned long long int*)calloc(gpu_n_mem, sizeof(unsigned long long int));
      for (i=0;i<gpu_n_mem; i++) wb_addr[i] = -1;
   }

   if (!L2dramout[dm_id]) {
      if (!L2dramout[dm_id]) L2dramout[dm_id] = (mem_fetch_t*) dq_pop(dram_p->dramtoL2queue);
      if (!L2dramout[dm_id]) L2dramout[dm_id] = (mem_fetch_t*) dq_pop(dram_p->dramtoL2writequeue);
   }

   mem_fetch_t* mf = L2dramout[dm_id];
   if (mf) {
      if (!mf->write) { //service L2 read miss

         // it is a pre-fill dramout mf
         if (wb_addr[dm_id] == -1) {
            if ( dq_full(dram_p->L2tocbqueue) ) goto RETURN;

            if (mf->mshr) mshr_update_status(mf->mshr, IN_L2TOCBQUEUE_MISS);

            //only transfer across icnt once the whole line has been received by L2 cache
            mf->type = REPLY_DATA;
            dq_push(dram_p->L2tocbqueue, mf);

            assert(mf->sid <= gpu_n_shader);           
            wb_addr[dm_id] = L2_shd_cache_fill(dram_p->L2cache, mf->addr, gpu_sim_cycle );
         }
         // only perform a write on cache eviction (write-back policy)
         // it is the 1st or nth time trial to writeback
         if (wb_addr[dm_id] != -1) {
            // performing L2 writeback (no false sharing for memory-side cache)
            int wb_succeed = L2c_write_back(wb_addr[dm_id], dram_p->L2cache->line_sz, dm_id ); 
            if (!wb_succeed) goto RETURN; //try again next cycle
         }

         L2dramout[dm_id] = NULL;
         wb_addr[dm_id] = -1;
      } else { //service L2 write miss
         dramtoL2_service_write(mf);
         L2dramout[dm_id] = NULL;
         wb_addr[dm_id] = -1;
      }
   }
   RETURN:   
   assert (L2dramout[dm_id] || wb_addr[dm_id]==-1);
}

// Writeback from L2 to DRAM: 
// - Takes in memory address and their parameters and pushes to dram request queue
// - This is used only for L2 writeback 
unsigned char L2c_write_back(unsigned long long int addr, int bsize, int dram_id ) 
{
   addrdec_t tlx;
   addrdec_tlx(addr,&tlx);

   if ( dq_full(dram[dram_id]->L2todram_wbqueue) ) return 0;

   mem_fetch_t *mf;

   mf = (mem_fetch_t*) malloc(sizeof(mem_fetch_t));
   made_write_mfs++;
   mf->request_uid = g_next_request_uid++;
   mf->addr = addr;
   mf->nbytes_L1 = bsize + READ_PACKET_SIZE;
   mf->txbytes_L1 = 0;
   mf->rxbytes_L1 = 0;  
   mf->nbytes_L2 = bsize;
   mf->sid = gpu_n_shader; // (gpu_n_shader+1);
   mf->wid = 0;
   mf->txbytes_L2 = 0;
   mf->rxbytes_L2 = 0;  
   mf->mshr_idx = -1;
   mf->mshr = NULL;
   mf->write = 1; // it is writeback
   memlatstat_start(mf);
   mf->tlx = tlx;
   mf->bank = mf->tlx.bk;
   mf->chip = mf->tlx.chip;


   //writeback
   mf->type = L2_WTBK_DATA;
   if (!dq_push(dram[dram_id]->L2todram_wbqueue, mf)) assert(0);
   gpgpu_n_sent_writes++;
   return 1;
}

unsigned int L2c_cache_flush ( dram_t* dram_p) {
   shd_cache_t *cp = dram_p->L2cache; 
   int i;
   int dirty_lines_flushed = 0 ;
   for (i=0; i < cp->nset * cp->assoc ; i++) {
      if ( (cp->lines[i].status & (DIRTY|VALID)) == (DIRTY|VALID) ) {
         dirty_lines_flushed++;
      }
      cp->lines[i].status &= ~VALID;
      cp->lines[i].status &= ~DIRTY;
   }
   return dirty_lines_flushed;
}

void L2c_print_stat( )
{
   int i;

   printf("                                     ");
   for (i=0;i<gpu_n_mem;i++) {
      printf(" dram[%d]", i);
   }
   printf("\n");

   printf("cbtoL2 queue maximum length         ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_cbtoL2length[i]);
   }
   printf("\n");

   printf("cbtoL2 write queue maximum length   ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_cbtoL2writelength[i]);
   }
   printf("\n");

   printf("L2tocb queue maximum length         =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_L2tocblength[i]);
   }
   printf("\n");

   printf("dramtoL2 queue maximum length       =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_dramtoL2length[i]);
   }
   printf("\n");

   printf("dramtoL2 write queue maximum length ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_dramtoL2writelength[i]);
   }
   printf("\n");

   printf("L2todram queue maximum length       =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_L2todramlength[i]);
   }
   printf("\n");
}

void L2c_print_debug( )
{
   int i;

   printf("                                     ");
   for (i=0;i<gpu_n_mem;i++) {
      printf(" dram[%d]", i);
   }
   printf("\n");

   printf("cbtoL2 queue length         ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", dram[i]->cbtoL2queue->length);
   }
   printf("\n");

   printf("cbtoL2 write queue length   ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", dram[i]->cbtoL2writequeue->length);
   }
   printf("\n");

   printf("L2tocb queue length         =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", dram[i]->L2tocbqueue->length);
   }
   printf("\n");

   printf("dramtoL2 queue length       =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", dram[i]->dramtoL2queue->length);
   }
   printf("\n");

   printf("dramtoL2 write queue length ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", dram[i]->dramtoL2writequeue->length);
   }
   printf("\n");

   printf("L2todram queue length       =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", dram[i]->L2todramqueue->length);
   }
   printf("\n");

   printf("L2todram writeback queue length       =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", dram[i]->L2todram_wbqueue->length);
   }
   printf("\n");
}

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

void L2c_log(int task)
{
   int i;
   static void ** cbtol2_Dist   ;  
   static void ** cbtoL2wr_Dist  ;  
   static void ** L2tocb_Dist     ; 
   static void ** dramtoL2_Dist   ;
   static void ** dramtoL2wr_Dist  ;
   static void ** L2todram_Dist    ;
   static void ** L2todram_wb_Dist ;
   if (task == CREATELOG) {
      cbtol2_Dist = (void **)     calloc(gpu_n_mem,sizeof(void*));
      cbtoL2wr_Dist = (void **)    calloc(gpu_n_mem,sizeof(void*));
      L2tocb_Dist =   (void **)   calloc(gpu_n_mem,sizeof(void*));
      dramtoL2_Dist =   (void **)calloc(gpu_n_mem,sizeof(void*));
      dramtoL2wr_Dist  =  (void **)calloc(gpu_n_mem,sizeof(void*));
      L2todram_Dist    = (void **)calloc(gpu_n_mem,sizeof(void*));
      L2todram_wb_Dist = (void **)calloc(gpu_n_mem,sizeof(void*));

      for (i=0;i<gpu_n_mem;i++) {
         cbtol2_Dist[i]      = StatCreate("cbtoL2",1,dram[i]->cbtoL2queue->max_len);
         cbtoL2wr_Dist[i]    = StatCreate("cbtoL2write",1,dram[i]->cbtoL2writequeue->max_len);
         L2tocb_Dist[i]      = StatCreate("L2tocb",1,dram[i]->L2tocbqueue->max_len);
         dramtoL2_Dist[i]    = StatCreate("dramtoL2",1,dram[i]->dramtoL2queue->max_len);
         dramtoL2wr_Dist[i]  = StatCreate("dramtoL2write",1,dram[i]->dramtoL2writequeue->max_len);
         L2todram_Dist[i]    = StatCreate("L2todram",1,dram[i]->L2todramqueue->max_len);
         L2todram_wb_Dist[i] = StatCreate("L2todram_wb",1,dram[i]->L2todram_wbqueue->max_len);
      }
   } else if (task == SAMPLELOG) {
      for (i=0;i<gpu_n_mem;i++) {
         StatAddSample(cbtol2_Dist[i],dram[i]->cbtoL2queue->length);
         StatAddSample(cbtoL2wr_Dist[i]  ,dram[i]->cbtoL2writequeue->length);
         StatAddSample(L2tocb_Dist[i]       ,dram[i]->L2tocbqueue->length);
         StatAddSample(dramtoL2_Dist[i]     ,dram[i]->dramtoL2queue->length);
         StatAddSample(dramtoL2wr_Dist[i],dram[i]->dramtoL2writequeue->length);
         StatAddSample(L2todram_Dist[i]     ,dram[i]->L2todramqueue->length);
         StatAddSample(L2todram_wb_Dist[i]  ,dram[i]->L2todram_wbqueue->length);
      }
   } else if (task == DUMPLOG) {
      for (i=0;i<gpu_n_mem;i++) {
         printf ("Queue Length DRAM[%d] ",i);StatDisp(cbtol2_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i);StatDisp(cbtoL2wr_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i);StatDisp(L2tocb_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i);StatDisp(dramtoL2_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i);StatDisp(dramtoL2wr_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i);StatDisp(L2todram_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i);StatDisp(L2todram_wb_Dist[i]);
      } 
   }
}


void dram_log (int task ) {
   int i;
   static void ** mrqq_Dist; //memory request queue inside DRAM  
   if (task == CREATELOG) {
      mrqq_Dist = (void **)     calloc(gpu_n_mem,sizeof(void*));
      for (i=0;i<gpu_n_mem;i++) {
         if (dram[i]->queue_limit)
            mrqq_Dist[i]      = StatCreate("mrqq_length",1,dram[i]->queue_limit);
         else //queue length is unlimited; 
            mrqq_Dist[i]      = StatCreate("mrqq_length",1,64); //track up to 64 entries
      }
   } else if (task == SAMPLELOG) {
      for (i=0;i<gpu_n_mem;i++) {
         StatAddSample(mrqq_Dist[i], dram_que_length(dram[i]));   
      }
   } else if (task == DUMPLOG) {
      for (i=0;i<gpu_n_mem;i++) {
         printf ("Queue Length DRAM[%d] ",i);StatDisp(mrqq_Dist[i]);
      } 
   }
}

void L2c_latency_log_dump()
{
   int i;
   for (i=0;i<gpu_n_mem;i++) {
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->cbtoL2queue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->cbtoL2writequeue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->L2tocbqueue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->dramtoL2queue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->dramtoL2writequeue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->L2todramqueue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->L2todram_wbqueue->lat_stat);
   }
}

void dramqueue_latency_log_dump()
{
   int i;
   for (i=0;i<gpu_n_mem;i++) {
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->mrqq->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->rwq->lat_stat);   
   }
}

//Find next clock domain and increment its time
inline int next_clock_domain(void) 
{
   double smallest = min3(core_time,icnt_time,dram_time);
   int mask = 0x00;
   if (gpgpu_cache_dl2_opt  //when no-L2 it will never be L2's turn
       && ( l2_time <= smallest) ) {
      smallest = l2_time;
      mask |= L2 ;
      l2_time += l2_period;
   }
   if ( icnt_time <= smallest ) {
      mask |= ICNT;
      icnt_time += icnt_period;
   }
   if ( dram_time <= smallest ) {
      mask |= DRAM;
      dram_time += dram_period;
   }
   if ( core_time <= smallest ) {
      mask |= CORE;
      core_time += core_period;
   }
   return mask;
}

extern time_t simulation_starttime;

void gpu_sim_loop( int grid_num ) 
{
   int i;
   int clock_mask = next_clock_domain();

   // shader core loading (pop from ICNT into shader core) follows CORE clock
   if (clock_mask & CORE ) {
      for (i=0;i<gpu_n_tpc;i++) {
	//SEAN:  Insert check for response here
         fq_pop(i); 
	 //TEST
	 printf("Cycle %llu:\n", gpu_sim_cycle);
	 mshr_print(stdout, sc[i]);
	 //TEST*/
      }
   }

   if (clock_mask & ICNT) {
      // pop from memory controller to interconnect
      static unsigned int *rt_size = NULL;
      if (!rt_size) rt_size = (unsigned int*) malloc ((gpu_n_tpc+gpu_n_mem)*sizeof(unsigned int));
      memset(rt_size, 0, (gpu_n_tpc+gpu_n_mem)*sizeof(unsigned int));

      for (i=0;i<gpu_n_mem;i++) {

         mem_fetch_t* mf;

         mf = (mem_fetch_t*) mem_ctrl_top(i); //(returns L2_top or DRAM returnq top)

         if (mf) {
            mf->source_node = mem2device(i);
            assert( mf->type != RD_REQ && mf->type != WT_REQ ); // never should a request come out from L2 or dram
            if (!mf->write) {
               int return_dev = -1;
               return_dev = mf->sid / gpu_concentration;
               assert(return_dev != -1);
               // check icnt resource for READ data return
               rt_size[return_dev] = mf->nbytes_L1;
               if ( icnt_has_buffer( mem2device(i), rt_size) ) {
                  if (mf->mshr) mshr_update_status(mf->mshr, IN_ICNT2SHADER);
                  memlatstat_icnt2sh_push(mf);
                  time_vector_update(mf->mshr->inst_uid ,MR_2SH_ICNT_PUSHED,gpu_sim_cycle+gpu_tot_sim_cycle,RD_REQ);
                  icnt_push( mem2device(i), return_dev, mf, mf->nbytes_L1);
                  mem_ctrl_pop(i);
		  //TEST
		  printf("SEAN (%llu): %llu data returned from DRAM, injected to ICNT back to core.  Time %llu\n", mf->addr, gpu_sim_cycle);
		  //TEST*/
               } else {
                  gpu_stall_icnt2sh++;
               }
               rt_size[return_dev] = 0; // clean up for the next dram_pop
            } else {
               time_vector_update(mf->request_uid ,MR_2SH_ICNT_PUSHED,gpu_sim_cycle+gpu_tot_sim_cycle,WT_REQ ) ;
               mem_ctrl_pop(i);
               free(mf);
               freed_L1write_mfs++;
               gpgpu_n_processed_writes++;
            }
         }
      }
   }

   if (clock_mask & DRAM) {
      for (i=0;i<gpu_n_mem;i++) {
         get_dram_output ( dram[i] ); 
      }    
      // Issue the dram command (scheduler + delay model) 
      for (i=0;i<gpu_n_mem;i++) {
         dram_issueCMD(dram[i]);
      }
      dram_log(SAMPLELOG);  
   }

   // L2 operations follow L2 clock domain
   if (clock_mask & L2) {
      for (i=0;i<gpu_n_mem;i++) {
         L2c_process_dram_output ( dram[i], i ); // pop from dram
         L2c_push_miss_to_dram ( dram[i] );  //push to dram
         L2c_service_mem_req ( dram[i], i );   // pop(push) from(to)  icnt2l2(l2toicnt) queues; service l2 requests 
      }
      if (gpgpu_cache_dl2_opt) { // L2 cache enabled
         for (i=0;i<gpu_n_mem;i++) {
            if (dram[i]->cbtoL2queue->length > L2_cbtoL2length[i])
               L2_cbtoL2length[i] = dram[i]->cbtoL2queue->length;
            if (dram[i]->cbtoL2writequeue->length > L2_cbtoL2writelength[i])
               L2_cbtoL2writelength[i] = dram[i]->cbtoL2writequeue->length;
            if (dram[i]->L2tocbqueue->length > L2_L2tocblength[i])
               L2_L2tocblength[i] = dram[i]->L2tocbqueue->length;
            if (dram[i]->dramtoL2queue->length > L2_dramtoL2length[i])
               L2_dramtoL2length[i] = dram[i]->dramtoL2queue->length;
            if (dram[i]->dramtoL2writequeue->length > L2_dramtoL2writelength[i])
               L2_dramtoL2writelength[i] = dram[i]->dramtoL2writequeue->length;
            if (dram[i]->L2todramqueue->length > L2_L2todramlength[i])
               L2_L2todramlength[i] = dram[i]->L2todramqueue->length;
         }
      }
      if (gpgpu_cache_dl2_opt) { //take a sample of l2c queue lengths
         L2c_log(SAMPLELOG); 
      }
   }

   if (clock_mask & ICNT) {
      // pop memory request from ICNT and 
      // push it to the proper memory controller (L2 or DRAM controller)
      for (i=0;i<gpu_n_mem;i++) {

         if ( mem_ctrl_full(i) ) {
            gpu_stall_dramfull++;
            continue;
         }

         mem_fetch_t* mf;    
         mf = (mem_fetch_t*) icnt_pop( mem2device(i) );

         if (mf) {
            if (mf->type==RD_REQ) {
               time_vector_update(mf->mshr->inst_uid ,MR_DRAMQ,gpu_sim_cycle+gpu_tot_sim_cycle,mf->type ) ;             
            } else {
               time_vector_update(mf->request_uid ,MR_DRAMQ,gpu_sim_cycle+gpu_tot_sim_cycle,mf->type ) ;
            }
            memlatstat_icnt2mem_pop(mf);
            mem_ctrl_push( i, mf );
         }
      }
      icnt_transfer( );
   }

   if (clock_mask & CORE) {
      // L1 cache + shader core pipeline stages 
      for (i=0;i<gpu_n_shader;i++) {
	if (sc[i]->not_completed || more_thread) {
	  shader_cycle(sc[i], i, grid_num);
	}
         sc[i]->gpu_cycle++;
      }
      gpu_sim_cycle++;

      for (i=0;i<gpu_n_shader && more_thread;i++) {
         if (gpgpu_spread_blocks_across_cores) {
            int cta_issue_count = 1;
            if ( ( (sc[i]->n_active_cta + cta_issue_count) <= max_cta_per_shader(sc[i]) )
                 && g_total_cta_left ) {
               int j;
               for (j=0;j<cta_issue_count;j++) {
                  issue_block2core(sc[i], grid_num);
               }
               if (!g_total_cta_left) {
                  more_thread = 0;
               }
               assert( g_total_cta_left > -1 );
            }
         } else {
            if (!(sc[i]->not_completed))
               more_thread = issue_block2core(sc[i], grid_num);
         }
      }


      // Flush the caches once all of threads are completed.
      if (gpgpu_flush_cache) {
         int all_threads_complete = 1 ; 
         for (i=0;i<gpu_n_shader;i++) {
            if (sc[i]->not_completed == 0) {
               shader_cache_flush(sc[i]);
            } else {
               all_threads_complete = 0 ; 
            }
         }
         if (all_threads_complete) {
            printf("Flushed L1 caches...\n");
            if (gpgpu_cache_dl2_opt) {
               int dlc = 0;
               for (i=0;i<gpu_n_mem;i++) {
                  dlc = L2c_cache_flush(dram[i]);
                  printf("Dirty lines flushed from L2 %d is %d  \n", i, dlc  );
               }
            }
         }
      }

      if (!(gpu_sim_cycle % gpu_stat_sample_freq)) {
         time_t days, hrs, minutes, sec;
         time_t curr_time;
         time(&curr_time);
         unsigned long long  elapsed_time = MAX(curr_time - simulation_starttime, 1);
         days    = elapsed_time/(3600*24);
         hrs     = elapsed_time/3600 - 24*days;
         minutes = elapsed_time/60 - 60*(hrs + 24*days);
         sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));
         printf("cycles: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s", 
                gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, 
                (double)gpu_sim_insn/(double)gpu_sim_cycle,
                (unsigned)((gpu_tot_sim_insn+gpu_sim_insn) / elapsed_time),
                (unsigned)days,(unsigned)hrs,(unsigned)minutes,(unsigned)sec,
                ctime(&curr_time));
         fflush(stdout);
         memlatstat_lat_pw();
         visualizer_printstat();
         if (gpgpu_runtime_stat && (gpu_runtime_stat_flag != 0) ) {
            if (gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
               for (i=0;i<gpu_n_mem;i++) {
                  dram_print_stat(dram[i],stdout);
               }
               printf("maxmrqlatency = %d \n", max_mrq_latency);
               printf("maxmflatency = %d \n", max_mf_latency);
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_DWF_MAP) {
               printf("DWF_MS: ");
               for (i=0;i<gpu_n_shader;i++) {
                  printf("%u ",acc_dyn_pcs[i]);
               }
               printf("\n");
               print_thread_pc( stdout );
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO) {
               shader_print_runtime_stat( stdout );
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_WARP_DIS) {
               print_shader_cycle_distro( stdout );
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_L1MISS) {
               shader_print_l1_miss_stat( stdout );
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_PDOM ) {
               if (pdom_sched_type) {
                  printf ("pdom_original_warps_count %d \n",n_pdom_sc_orig_stat );
                  printf ("pdom_single_warps_count %d \n",n_pdom_sc_single_stat );
               }
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_SCHED ) {
               printf("Average Num. Warps Issuable per Shader:\n");
               for (i=0;i<gpu_n_shader;i++) {
                  printf("%2.2f ", (float) num_warps_issuable_pershader[i]/ gpu_stat_sample_freq);
                  num_warps_issuable_pershader[i] = 0;
               }
               printf("\n");
            }
         }
      }

      for (i=0;i<gpu_n_mem;i++) {
         acc_mrq_length[i] += dram_que_length(dram[i]);
      }
      if (!(gpu_sim_cycle % 20000)) {
         // deadlock detection 
         if (gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
            gpu_deadlock = 1;
         } else {
            last_gpu_sim_insn = gpu_sim_insn;
         }
      }
      try_snap_shot(gpu_sim_cycle);
      spill_log_to_file (stdout, 0, gpu_sim_cycle);
   }
}

void dump_regs(unsigned sid, unsigned tid)
{
   if ( sid >= gpu_n_shader ) {
      printf("shader %u is out of range\n",sid);
      return;
   }
   if ( tid >= gpu_n_thread_per_shader ) {
      printf("thread %u is out of range\n",tid);
      return;
   }

   shader_core_ctx_t *s = sc[sid];

   ptx_dump_regs( s->thread[tid].ptx_thd_info );
}

extern int ptx_thread_done( void *thr );

void shader_dump_istream_state(shader_core_ctx_t *shader, FILE *fout )
{
   int i, t=0;
   fprintf( fout, "\n");
   for ( t=0; t < gpu_n_thread_per_shader/warp_size; t++ ) {
      int tid = t*warp_size;
      if ( shader->thread[ tid ].n_completed < warp_size ) {
         fprintf( fout, "  %u:%3u fetch state = c:%u a4f:%u bw:%u (completed: ", shader->sid, tid, 
                shader->thread[tid].n_completed,
                shader->thread[tid].n_avail4fetch,
                shader->thread[tid].n_waiting_at_barrier  );

         for ( i = tid; i < (t+1)*warp_size; i++ ) {
            if ( ptx_thread_done(shader->thread[i].ptx_thd_info) ) {
               fprintf(fout,"1");
            } else {
               fprintf(fout,"0");
            }
            if ( (((i+1)%4) == 0) && (i+1) < (t+1)*warp_size ) {
               fprintf(fout,",");
            }
         }
         fprintf(fout,")\n");
      }
   }
}

void dump_pipeline_impl( int mask, int s, int m ) 
{
/*
   You may want to use this function while running GPGPU-Sim in gdb.
   One way to do that is add the following to your .gdbinit file:
 
      define dp
         call dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
      end
 
   Then, typing "dp 3" will show the contents of the pipeline for shader core 3.
*/

   printf("Dumping pipeline state...\n");
   int i;
   if(!mask) mask = 0xFFFFFFFF;
   for (i=0;i<gpu_n_shader;i++) {
      if(s != -1) {
         i = s;
      }
      if(mask&1) shader_display_pipeline(sc[i], stdout, 1, mask & 0x2E );
      if(mask&0x40) shader_dump_istream_state(sc[i], stdout);
      if(mask&0x100) mshr_print(stdout, sc[i]);
      if(s != -1) {
         break;
      }
   }
   if(mask&0x10000) {
      for (i=0;i<gpu_n_mem;i++) {
         if(m != -1) {
            i=m;
         }
         printf("DRAM / memory controller %u:\n", i);
         if(mask&0x100000) dram_print_stat(dram[i],stdout);
         if(mask&0x1000000)   dram_visualize( dram[i] );
         if(m != -1) {
            break;
         }
      }
   }
   fflush(stdout);
}

void dump_pipeline()
{
   dump_pipeline_impl(0,-1,-1);
}
