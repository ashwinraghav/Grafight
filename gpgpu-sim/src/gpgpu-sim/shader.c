/* 
 * shader.c
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Ivan Sham, Henry Wong, Dan O'Connor, Henry Tran and the 
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

#include <float.h>
#include "shader.h"
#include "gpu-sim.h"
#include "addrdec.h"
#include "dram.h"
#include "dwf.h"
#include "warp_tracker.h"
#include "cflogger.h"
#include "gpu-misc.h"
#include "../cuda-sim/ptx_sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/dram_callback.h"
#include "mem_fetch.h"
#include <string.h>

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a,b) (((a)>(b))?(a):(b))

extern unsigned int mergemiss;
extern unsigned int L1_read_miss;
extern unsigned int L1_texture_miss;
extern unsigned int L1_const_miss;

extern unsigned int finished_trace;
extern int gpgpu_perfect_mem;
extern int gpgpu_no_dl1;
extern char *gpgpu_cache_texl1_opt;
extern char *gpgpu_cache_constl1_opt;
extern char *gpgpu_cache_dl1_opt;
extern unsigned int gpu_n_thread_per_shader;
extern unsigned int gpu_n_mshr_per_thread;
extern unsigned int gpu_n_shader;
extern unsigned int gpu_n_mem;
extern int gpgpu_reg_bankconflict;
extern int gpgpu_dram_sched_queue_size;
extern int gpgpu_memlatency_stat;
extern dram_t **dram;
extern int *num_warps_issuable;
extern int *num_warps_issuable_pershader;

extern unsigned long long  gpu_sim_insn;
extern unsigned long long  gpu_sim_insn_no_ld_const;
extern unsigned long long  gpu_sim_insn_last_update;
extern unsigned long long  gpu_completed_thread;
extern unsigned long long  gpu_sim_cycle;
extern shader_core_ctx_t **sc;
extern unsigned int gpgpu_pre_mem_stages;
extern int gpgpu_no_divg_load;
extern unsigned int gpgpu_thread_swizzling;
extern unsigned int gpgpu_strict_simd_wrbk;
extern unsigned int warp_conflict_at_writeback;
extern unsigned int gpgpu_commit_pc_beyond_two;
extern int gpgpu_spread_blocks_across_cores;
extern int gpgpu_cflog_interval;

extern unsigned int gpu_stall_by_MSHRwb;
extern unsigned int gpu_stall_shd_mem;
extern unsigned int gpu_stall_sh2icnt;
unsigned int warp_size = 4; 
int pipe_simd_width;
extern unsigned int **totalbankaccesses; //bankaccesses[shader id][dram chip id][bank id]
extern unsigned int *MCB_accesses; //upon cache miss, tracks which memory controllers accessed by a warp
extern unsigned int *num_MCBs_accessed; //tracks how many memory controllers are accessed whenever any thread in a warp misses in cache
extern unsigned int *max_return_queue_length;

unsigned int *shader_cycle_distro;

unsigned int g_waiting_at_barrier = 0;

unsigned int gpgpu_shmem_size = 16384;
unsigned int gpgpu_shader_registers = 8192;
unsigned int gpgpu_shader_cta = 8;

unsigned int gpgpu_n_load_insn = 0;
unsigned int gpgpu_n_store_insn = 0;
unsigned int gpgpu_n_shmem_insn = 0;
unsigned int gpgpu_n_tex_insn = 0;
unsigned int gpgpu_n_const_insn = 0;
unsigned int gpgpu_n_param_insn = 0;
unsigned int gpgpu_multi_unq_fetches = 0;

extern int gpgpu_cache_wt_through;

int gpgpu_shmem_bkconflict = 0;
unsigned int gpgpu_n_shmem_bkconflict = 0;
int gpgpu_n_shmem_bank = 16;

int gpgpu_cache_bkconflict = 0;
unsigned int gpgpu_n_cache_bkconflict = 0;
unsigned int gpgpu_n_cmem_portconflict = 0;
int gpgpu_n_cache_bank = 16;

extern int gpu_runtime_stat_flag;
int gpgpu_warpdistro_shader = -1;

int gpgpu_interwarp_mshr_merge = 0;
int gpgpu_n_intrawarp_mshr_merge = 0;

extern int gpgpu_partial_write_mask;
int gpgpu_n_partial_writes = 0;

extern int gpgpu_n_mem_write_local;
extern int gpgpu_n_mem_write_global;

#ifndef MhZ
   #define MhZ *1000000
#endif
extern double core_freq;
extern double icnt_freq;
extern double dram_freq;
extern double l2_freq;

int gpgpu_shmem_port_per_bank = 4;
int gpgpu_cache_port_per_bank = 4;
int gpgpu_const_port_per_bank = 4;
int gpgpu_shmem_pipe_speedup = 2;

unsigned int gpu_max_cta_per_shader = 8;
unsigned int gpu_padded_cta_size = 32;
int gpgpu_local_mem_map = 1;


extern int pdom_sched_type;
extern int n_pdom_sc_orig_stat;
extern int n_pdom_sc_single_stat;
extern int gpgpu_cuda_sim;
extern unsigned long long  gpu_tot_sim_cycle;

extern unsigned g_max_regs_per_thread;
extern void ptx_decode_inst( void *thd, unsigned *op, int *i1, int *i2, int *i3, int *i4, int *o1, int *o2, int *o3, int *o4, unsigned *vectorin, unsigned *vectorout, unsigned *isatom );
extern unsigned ptx_get_inst_op( void *thd);
extern void ptx_exec_inst( void *thd, address_type *addr, unsigned *space, unsigned *data_size, dram_callback_t* callback, unsigned warp_active_mask);
extern int  ptx_branch_taken( void *thd );
extern void ptx_sim_free_sm( void** thread_info );
extern unsigned ptx_sim_init_thread( void** thread_info, int sid, unsigned tid,unsigned threads_left,unsigned num_threads);
extern unsigned ptx_sim_cta_size();
extern const struct gpgpu_ptx_sim_kernel_info* ptx_sim_kernel_info();
extern void set_option_gpgpu_spread_blocks_across_cores(int option);
extern int ptx_thread_done( void *thr );
extern unsigned ptx_thread_donecycle( void *thr );
extern int ptx_thread_get_next_pc( void *thd );
extern void* ptx_thread_get_next_finfo( void *thd );
extern int ptx_thread_at_barrier( void *thd );
extern int ptx_thread_all_at_barrier( void *thd );
extern unsigned long long ptx_thread_get_cta_uid( void *thd );
extern void ptx_thread_reset_barrier( void *thd );
extern void ptx_thread_release_barrier( void *thd );
extern void ptx_print_insn( address_type pc, FILE *fp );
extern int ptx_set_tex_cache_linesize( unsigned linesize);
extern void time_vector_update(unsigned int uid,int slot ,long int cycle,int type);

static inst_t nop_inst = {.pc=0, .op=NO_OP, //generic bubbles
                          .out={0,0,0,0}, .in={0,0,0,0}, .is_vectorin=0, .is_vectorout=0,
                          .memreqaddr=0, .hw_thread_id=-1, .wlane=-1,
                          .uid = (unsigned)-1,
                          .priority = (unsigned)-1,
                          .ptx_thd_info = NULL,
                          .warp_active_mask = 0,
                          .ts_cycle = 0,
                          .id_cycle = 0,
                          .ex_cycle = 0,
                          .mm_cycle = 0
                          };

int log2i(int n) {
   int lg;
   lg = -1;
   while (n) {
      n>>=1;lg++;
   }
   return lg;
}

//SEAN:  pipetrace support
extern char* g_pipetrace_out;
extern int g_pipetrace;  //pipetrace on/off

//struct to contain timestamps for existence in different stages
typedef struct pipe_stat_t {
  unsigned uid; //unique identifier for instruction
  unsigned long long int memreqaddr;
  short hw_thread_id;
  unsigned char inst_type;

  unsigned issued;
  unsigned in_fetch;
  unsigned in_decode;
  unsigned in_pre_exec;
  unsigned in_exec;
  unsigned in_pre_mem;
  unsigned in_mem;
  unsigned in_writeback;

  struct pipe_stat_t *prev;
  struct pipe_stat_t *next;
} pipe_stat;

pipe_stat *pipe_stat_last=NULL;
pipe_stat *pipe_stat_first=NULL;

void pipe_stat_write_file() {
  FILE * pfile;
  pipe_stat *curr;

  pfile = fopen(g_pipetrace_out, "w");
  //print header
  fprintf(pfile, "Inst_type MemReqAddr Thread_ID InstructionID issued fetch decode pre-exec exec pre-mem mem writeback\n");

  /*TEST
  static int counting=0;
  counting++;
  printf("%i printing pipe stats\n", counting);
  //TEST*/

  curr = pipe_stat_first;

  /*TEST
  while(curr != NULL) {
    if(curr->uid == 60000) { printf("Printing uid %u's trace\n", curr->uid); }
    if(curr->next->uid < curr->uid) break;  //Don't know why, but this enters a (infinite?) loop (w/ atomics at least) otherwise
    curr = curr->next;
  }
  //TEST*/

  while(curr != NULL) {
    fprintf(pfile, "%5c ", curr->inst_type);
    fprintf(pfile, "%14llu  ", curr->memreqaddr);
    fprintf(pfile, "%5hu  ", curr->hw_thread_id);
    fprintf(pfile, "%10i", curr->uid);
    fprintf(pfile, "%10u", curr->issued);
    fprintf(pfile, "%6u", curr->in_fetch);
    fprintf(pfile, "%7u", curr->in_decode);
    fprintf(pfile, "%8u", curr->in_pre_exec);
    fprintf(pfile, "%7u", curr->in_exec);
    fprintf(pfile, "%6u", curr->in_pre_mem);
    fprintf(pfile, "%6u", curr->in_mem);
    fprintf(pfile, "%7u", curr->in_writeback);
    fprintf(pfile, "\n");

    if((curr->next != NULL) && (curr->next->uid < curr->uid)) break;  //Don't know why, but this enters a (infinite?) loop (w/ atomics at least) otherwise
    curr = curr->next;
  }

  fclose(pfile);
}
//SEAN*/

shader_core_ctx_t* shader_create( char *name, int sid,
                                  unsigned int n_threads,
                                  unsigned int n_mshr,
                                  fq_push_t fq_push,
                                  fq_has_buffer_t fq_has_buffer,
                                  unsigned int model )
{
   shader_core_ctx_t *sc;
   int i, j;
   unsigned int shd_n_set;
   unsigned int shd_linesize;
   unsigned int shd_n_assoc;
   unsigned char shd_policy;

   unsigned int l1tex_cache_n_set; //L1 texture cache parameters
   unsigned int l1tex_cache_linesize;
   unsigned int l1tex_cache_n_assoc;
   unsigned char l1tex_cache_policy;

   unsigned int l1const_cache_n_set; //L1 constant cache parameters
   unsigned int l1const_cache_linesize;
   unsigned int l1const_cache_n_assoc;
   unsigned char l1const_cache_policy;

   if ( gpgpu_cuda_sim ) {
      unsigned cta_size = ptx_sim_cta_size();
      if ( cta_size > n_threads ) {
         printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
         printf("                 This can cause problems with applications that use __syncthreads.\n");
         printf("                 CTA size (x*y*z) = %u, n_threads = %u\n", cta_size, n_threads );
         printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
         printf("                 modify the CUDA source to decrease the kernel block size.\n");
         abort();
      }
   }

   sc = (shader_core_ctx_t*) calloc(1, sizeof(shader_core_ctx_t));

   sc->name = name;
   sc->sid = sid;

   sc->RR_k = 0;

   sc->model = model;

   sc->pipeline_reg = (inst_t**) calloc(warp_size, sizeof(inst_t*));
   for (j = 0; j<warp_size; j++) {
      sc->pipeline_reg[j] = (inst_t*) calloc(N_PIPELINE_STAGES, sizeof(inst_t));
      for (i=0; i<N_PIPELINE_STAGES; i++) {
         sc->pipeline_reg[j][i] = nop_inst;
      }
   }

   if (gpgpu_pre_mem_stages) {
      sc->pre_mem_pipeline = (inst_t**) calloc(pipe_simd_width, sizeof(inst_t*));
      for (j = 0; j<pipe_simd_width; j++) {
         sc->pre_mem_pipeline[j] = (inst_t*) calloc(gpgpu_pre_mem_stages+1, sizeof(inst_t));
         for (i=0; i<=gpgpu_pre_mem_stages; i++) {
            sc->pre_mem_pipeline[j][i] = nop_inst;
         }
      }
   }
   sc->n_threads = n_threads;
   sc->thread = (thread_ctx_t*) calloc(sizeof(thread_ctx_t), n_threads);
   sc->not_completed = 0;
   sc->n_active_cta = 0;
   for (i = 0; i<MAX_CTA_PER_SHADER; i++  ) {
      sc->cta_status[i]=0;
   }
   //Warp variable initializations
   sc->next_warp = 0;
   sc->branch_priority = 0;
   sc->max_branch_priority = malloc(sizeof(int)*n_threads);


   for (i = 0; i<n_threads; i++) {
      sc->max_branch_priority[i] = INT_MAX;
      sc->thread[i].id = i;

      sc->thread[i].warp_priority = sc->max_branch_priority[i];
      sc->thread[i].avail4fetch = 0;
      sc->thread[i].n_avail4fetch = 0;
      sc->thread[i].n_waiting_at_barrier = 0;
      sc->thread[i].m_waiting_at_barrier = 0;
      if ( (i%warp_size) == 0 ) {
         assert( n_threads % warp_size == 0 );
         sc->thread[i].n_completed = warp_size;
      }

      sc->thread[i].ptx_thd_info = NULL;
      ptx_sim_init_thread(&sc->thread[i].ptx_thd_info,sid,i,n_threads-i,n_threads);
   }

   sscanf(gpgpu_cache_dl1_opt,"%d:%d:%d:%c", 
          &shd_n_set, &shd_linesize, &shd_n_assoc, &shd_policy);
   sscanf(gpgpu_cache_texl1_opt,"%d:%d:%d:%c", 
          &l1tex_cache_n_set, &l1tex_cache_linesize, &l1tex_cache_n_assoc, &l1tex_cache_policy);
   sscanf(gpgpu_cache_constl1_opt,"%d:%d:%d:%c", 
          &l1const_cache_n_set, &l1const_cache_linesize, &l1const_cache_n_assoc, &l1const_cache_policy);
#define STRSIZE 32
   char L1c_name[STRSIZE];
   char L1texc_name[STRSIZE];
   char L1constc_name[STRSIZE];
   snprintf(L1c_name, STRSIZE, "L1c_%03d", sc->sid);
   sc->L1cache = shd_cache_create(L1c_name,shd_n_set,shd_n_assoc,shd_linesize,shd_policy,1,0 );
   shd_cache_bind_logger(sc->L1cache, sc->sid, get_shader_normal_cache_id());
   snprintf(L1texc_name, STRSIZE, "L1texc_%03d", sc->sid);
   sc->L1texcache = shd_cache_create(L1texc_name,l1tex_cache_n_set,l1tex_cache_n_assoc,l1tex_cache_linesize,l1tex_cache_policy,1,0 );
   shd_cache_bind_logger(sc->L1texcache, sc->sid, get_shader_texture_cache_id());
   snprintf(L1constc_name, STRSIZE, "L1constc_%03d", sc->sid);
   sc->L1constcache = shd_cache_create(L1constc_name,l1const_cache_n_set,l1const_cache_n_assoc,l1const_cache_linesize,l1const_cache_policy,1,0 );
   shd_cache_bind_logger(sc->L1constcache, sc->sid, get_shader_constant_cache_id());
   //at this point, should set the parameters used by addressing schemes of all textures
   ptx_set_tex_cache_linesize(l1tex_cache_linesize);

   sc->mshr = calloc(n_threads,sizeof(delay_queue*));
   for (i=0; i<n_threads; i++) {
      sc->mshr[i] = dq_create("mshr",0,0,n_mshr);
   }
   sc->n_mshr_used=0;
   sc->max_n_mshr_used = 0;
   sc->return_queue = dq_create("return_queue", 0, 0, 0);

   sc->fq_push = fq_push;
   sc->fq_has_buffer = fq_has_buffer;

   sc->pdom_warp = (pdom_warp_ctx_t*)calloc(n_threads / warp_size, sizeof(pdom_warp_ctx_t));
   for (i = 0; i < n_threads / warp_size; ++i) {
      sc->pdom_warp[i].m_stack_top = 0;
      sc->pdom_warp[i].m_pc = (address_type*)calloc(warp_size * 2, sizeof(address_type));
      sc->pdom_warp[i].m_calldepth = (unsigned int*)calloc(warp_size * 2, sizeof(unsigned int));
      sc->pdom_warp[i].m_active_mask = (unsigned int*)calloc(warp_size * 2, sizeof(unsigned int));
      sc->pdom_warp[i].m_recvg_pc = (address_type*)calloc(warp_size * 2, sizeof(address_type));
      sc->pdom_warp[i].m_branch_div_cycle = (unsigned long long *)calloc(warp_size * 2, sizeof(unsigned long long ));

      memset(sc->pdom_warp[i].m_pc, -1, warp_size * 2 * sizeof(address_type));
      memset(sc->pdom_warp[i].m_calldepth, 0, warp_size * 2 * sizeof(unsigned int));
      memset(sc->pdom_warp[i].m_active_mask, 0, warp_size * 2 * sizeof(unsigned int));
      memset(sc->pdom_warp[i].m_recvg_pc, -1, warp_size * 2 * sizeof(address_type));
   }

   sc->waiting_at_barrier = 0;

   sc->last_issued_thread = sc->n_threads - 1;

   sc->using_dwf = (sc->model == DWF);

   sc->using_rrstage = (sc->model == DWF);

   sc->using_commit_queue = (sc->model == DWF
                             || sc->model == POST_DOMINATOR || sc->model == NO_RECONVERGE);

   if (sc->using_commit_queue) {
      sc->thd_commit_queue = dq_create("thd_commit_queue", 0, 0, 0);
   }

   sc->shmem_size = gpgpu_shmem_size;
   sc->n_registers = gpgpu_shader_registers;
   sc->n_cta = gpgpu_shader_cta;

   sc->mem_stage_done_uncoalesced_stall = 0;

   return sc;
}

unsigned shader_reinit(shader_core_ctx_t *sc, int start_thread, int end_thread) 
{
   int i;
   unsigned result=0;

   if ( gpgpu_cuda_sim ) {
      unsigned cta_size = ptx_sim_cta_size();
      if ( cta_size > sc->n_threads ) {
         printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
         printf("                 This can cause problems with applications that use __syncthreads.\n");
         printf("                 CTA size (x*y*z) = %u, n_threads = %u\n", cta_size, sc->n_threads );
         printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
         printf("                 modify the CUDA source to decrease the kernel block size.\n");
         abort();
      }
   }
   //resetting
   if ((end_thread - start_thread) == sc->n_threads) {
      sc->not_completed = 0;
   }


   sc->next_warp = 0;
   sc->branch_priority = 0;

   for (i = start_thread; i<end_thread; i++)
      ptx_sim_free_sm(&sc->thread[i].ptx_thd_info);

   for (i = start_thread; i<end_thread; i++) {
      sc->max_branch_priority[i] = INT_MAX;

      sc->thread[i].warp_priority = sc->max_branch_priority[i];
      sc->thread[i].avail4fetch = 0;
      sc->thread[i].n_waiting_at_barrier = 0;
      sc->thread[i].m_waiting_at_barrier = 0;

      if (gpgpu_cuda_sim) {
         if (!gpgpu_spread_blocks_across_cores) {
            result += ptx_sim_init_thread(&sc->thread[i].ptx_thd_info,sc->sid,i,sc->n_threads-i,sc->n_threads);
         }
      }
      if ( (i%warp_size) == 0 ) {
         sc->thread[i].n_completed = warp_size;
         sc->thread[i].n_avail4fetch = 0;
      } else {
         assert( sc->thread[i].n_completed == 0 );
         assert( sc->thread[i].n_avail4fetch == 0 );
      }
      sc->thread[i].n_insn = 0;
   }

   for (i = start_thread / warp_size; i < end_thread / warp_size; ++i) {
      sc->pdom_warp[i].m_stack_top = 0;
      memset(sc->pdom_warp[i].m_pc, -1, warp_size * 2 * sizeof(address_type));
      memset(sc->pdom_warp[i].m_calldepth, 0, warp_size * 2 * sizeof(unsigned int));
      memset(sc->pdom_warp[i].m_active_mask, 0, warp_size * 2 * sizeof(unsigned int));
      memset(sc->pdom_warp[i].m_recvg_pc, -1, warp_size * 2 * sizeof(address_type));
      memset(sc->pdom_warp[i].m_branch_div_cycle, 0, warp_size * 2 * sizeof(unsigned long long ));
   }

   sc->waiting_at_barrier = 0;

   sc->last_issued_thread = end_thread - 1; 

   if (sc->using_commit_queue) {
      if (!gpgpu_spread_blocks_across_cores) //assertion no longer holds with multiple blocks per core  
         assert(dq_empty(sc->thd_commit_queue));
   }
   sc->pending_shmem_bkacc = 0;
   sc->pending_cache_bkacc = 0;
   sc->pending_cmem_acc = 0;

   sc->mem_stage_done_uncoalesced_stall = 0;

   return result;
}

// initialize a CTA in the shader core, currently only useful for PDOM and DWF
void shader_init_CTA(shader_core_ctx_t *shader, int start_thread, int end_thread)
{
   int i;
   int n_thread = end_thread - start_thread;
   address_type start_pc = ptx_thread_get_next_pc(shader->thread[start_thread].ptx_thd_info);
   if (shader->model == POST_DOMINATOR) {
      int start_warp = start_thread / warp_size;
      int end_warp = end_thread / warp_size + ((end_thread % warp_size)? 1 : 0);
      for (i = start_warp; i < end_warp; ++i) {
         shader->pdom_warp[i].m_stack_top = 0;
         memset(shader->pdom_warp[i].m_pc, -1, warp_size * 2 * sizeof(address_type));
         memset(shader->pdom_warp[i].m_calldepth, 0, warp_size * 2 * sizeof(unsigned int));
         memset(shader->pdom_warp[i].m_active_mask, 0, warp_size * 2 * sizeof(unsigned int));
         memset(shader->pdom_warp[i].m_recvg_pc, -1, warp_size * 2 * sizeof(address_type));
         memset(shader->pdom_warp[i].m_branch_div_cycle, 0, warp_size * 2 * sizeof(unsigned long long ));
         shader->pdom_warp[i].m_pc[0] = start_pc;
         shader->pdom_warp[i].m_calldepth[0] = 1;
         int t = 0;
         for (t = 0; t < warp_size; t++) {
            if ( i * warp_size + t < end_thread ) {
               shader->pdom_warp[i].m_active_mask[0] |= (1 << t);
            }
         }
      }
   } else if (shader->model == DWF) {
      dwf_init_CTA(shader->sid, start_thread, n_thread, start_pc);
   }

   for (i = start_thread; i<end_thread; i++) {
      shader->thread[i].in_scheduler = 1;
   } 
}


/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/

static char* MSHR_Status_str[] = {
   "INITIALIZED",
   "IN_ICNT2MEM",
   "IN_ICNTOL2QUEUE",
   "IN_L2TODRAMQUEUE",
   "IN_DRAM_REQ_QUEUE",
   "IN_DRAMRETURN_Q",
   "IN_DRAMTOL2QUEUE",
   "IN_L2TOICNTQUEUE_HIT",
   "IN_L2TOICNTQUEUE_MISS",
   "IN_ICNT2SHADER",
   "FETCHED",
   "STALLED_IN_ATOM_Q",
};

mshr_entry *g_mshr_free_queue = NULL;
mshr_entry *g_mshr_free_queue_head;
void init_mshr_pool()
{
   int i;
   static int max_mshr_entries; // this includes merged mshrs 
                                //(e.g. if you have 1 mshr that can handle 20 merged requests this number should be 20 not 1)
   if (!g_mshr_free_queue) {

      assert ( gpu_n_mshr_per_thread );
      max_mshr_entries = gpu_n_mshr_per_thread * gpu_n_thread_per_shader * gpu_n_shader ;

      g_mshr_free_queue = malloc(max_mshr_entries * sizeof(mshr_entry));
   }
   for ( i=0; i < (max_mshr_entries-1); i++ ) {
      g_mshr_free_queue[i].merged_requests = &g_mshr_free_queue[i+1];
   }
   g_mshr_free_queue[max_mshr_entries-1].merged_requests = NULL;
   g_mshr_free_queue_head = g_mshr_free_queue;
}

unsigned g_next_mshr_request_uid = 1;

mshr_entry* alloc_mshr_entry()
{
   assert( g_mshr_free_queue_head != NULL ); // if this fails check init_mshr_pool 
   mshr_entry *result = g_mshr_free_queue_head;
   result->request_uid = g_next_mshr_request_uid++;
   g_mshr_free_queue_head = g_mshr_free_queue_head->merged_requests;
   result->hw_thread_id = -1;
   result->addr = 0xDEADDEAD;
   result->reg = -1;
   result->reg2 = -1;
   result->reg3 = -1;
   result->reg4 = -1;
   result->merged_requests = NULL;
   result->merged = 0;
   result->fetched = 0;
   result->iswrite = 0;
   result->istexture = 0;
   result->isconst = 0;
   result->islocal = 0;
   result->isvector = 0;
   result->status = INITIALIZED;
   return result;
}

void free_mshr_entry( mshr_entry *r )
{
   r->merged_requests = g_mshr_free_queue_head;
   //printf("mshr free %d\n",r->request_uid);
   r->request_uid = (unsigned)(-((int)r->request_uid)); // for debugging
   g_mshr_free_queue_head = r;
}

void mshr_print(FILE* fp, shader_core_ctx_t *shader) {
   int i = 0;
   for (i=0; i< shader->n_threads; i++) {
      delay_queue* queue = shader->mshr[i];
      delay_data* ptr = queue->head;
      while (ptr) {
         mshr_entry *mshr = ptr->data;
         fprintf(fp, "MSHR(%d|%d): UID=%u R%d %s Addr:0x%llx IS:%d Fetched:%d Merged:%d Prio:%d Status:%s\n", 
                 shader->sid, i, mshr->request_uid, mshr->reg, 
                 (mshr->iswrite)? "=>" : "<=",
                 mshr->addr, mshr->hw_thread_id, mshr->fetched, 
                 mshr->merged, mshr->priority, MSHR_Status_str[mshr->status]);
         ptr = ptr->next;
      }
   }
}

void mshr_update_status(mshr_entry *mshr, enum mshr_status new_status ) {
   mshr_entry *merged_req;
   mshr->status = new_status;
#if DEBUGL1MISS 
   printf("cycle %d Addr %x  %d \n",gpu_sim_cycle,CACHE_TAG_OF_64(mshr->addr),new_status);
#endif
   merged_req = mshr->merged_requests;
   while (merged_req) {
      merged_req->status = new_status;
      merged_req = merged_req->merged_requests;
   }
}

mshr_entry* getMSHR_returnhead(shader_core_ctx_t* sc) {
   mshr_entry* retVal = NULL;
   if (!dq_empty(sc->return_queue)) {
      retVal = (mshr_entry*) sc->return_queue->head->data;
   }

   return retVal;
}

mshr_entry* fetchMSHR(delay_queue** mshr, shader_core_ctx_t* sc) {
   mshr_entry* retVal = NULL;

   retVal = (mshr_entry*) dq_pop(sc->return_queue);
   if ( retVal && retVal->fetched )
      sc->mshr_fetch_counter++;

   return retVal;

}

mshr_entry* shader_check_mshr4tag(shader_core_ctx_t* sc, unsigned long long int addr , int mem_type)
{
   delay_data* mshr_p = NULL;
   int i;

   for (i=0;i<sc->n_threads;i++) {
      mshr_p = sc->mshr[i]->head;
      while (mshr_p) {
         if (mshr_p->data) {
            switch (mem_type) {
            case DCACHE :  if (SHD_CACHE_TAG(((mshr_entry*)mshr_p->data)->addr,sc) == SHD_CACHE_TAG(addr,sc))
                  return(mshr_entry*)mshr_p->data;
               break;
            case CONSTC: if (SHD_CONSTCACHE_TAG(((mshr_entry*)mshr_p->data)->addr,sc) == SHD_CONSTCACHE_TAG(addr,sc))
                  return(mshr_entry*)mshr_p->data;
               break;
            case TEXTC:     if (SHD_TEXCACHE_TAG(((mshr_entry*)mshr_p->data)->addr,sc) == SHD_TEXCACHE_TAG(addr,sc))
                  return(mshr_entry*)mshr_p->data;
               break;
            default:       assert( 0);
            }
         }
         mshr_p = mshr_p->next;
      }
   }

   return NULL;
}

void shader_update_mshr(shader_core_ctx_t* sc, unsigned long long int fetched_addr, unsigned int mshr_idx, int mem_type ) 
{
   assert( mshr_idx < sc->n_threads );
   int found_addr = 0;
   //Each shader has N_THREADS number of MSHR, one per thread
   delay_data* mshr_p = sc->mshr[mshr_idx]->head; //SEAN:  head of shader's thread's mshr (which is implemented as delay queue)
   mshr_entry* mshr_e = NULL;
   while (mshr_p) {
      if (mshr_p->data) {
         mshr_e = ((mshr_entry*)mshr_p->data);
         switch (mem_type) {
         case DCACHE: found_addr = (SHD_CACHE_TAG(mshr_e->addr,sc) == fetched_addr); break;
         case CONSTC: found_addr = (SHD_CONSTCACHE_TAG(mshr_e->addr,sc) == fetched_addr); break;
         case TEXTC: found_addr = (SHD_TEXCACHE_TAG(mshr_e->addr,sc) == fetched_addr); break;
         default: assert(0); break;
         }
         if (found_addr) {
            do {
               assert( mshr_e->fetched != 1 );  // check for duplicate requests
               mshr_e->fetched = 1;
               dq_push(sc->return_queue,mshr_e);
               if (sc->return_queue->length > max_return_queue_length[sc->sid]) {
                  max_return_queue_length[sc->sid] = sc->return_queue->length;
               }
               mshr_e = mshr_e->merged_requests;
            } while ( mshr_e != NULL );
            return;
         }
      }
      mshr_p = mshr_p->next;
   }
   int CantFindMSHR = 0;
   assert(CantFindMSHR);
   abort(); // should have found MSHR for this request
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// register id for unused register slot in instruction
#define DNA       (0)

unsigned g_next_shader_inst_uid=1;

// check to see if the fetch stage need to be stalled
int shader_fetch_stalled(shader_core_ctx_t *shader)
{
   int i;
   int n_warp_parts = warp_size/pipe_simd_width;

   if (shader->warp_part2issue < n_warp_parts) {
     /*TEST
     printf("%llu:  Stalling 1\n", gpu_sim_cycle);
     //TEST*/
      return 1;
   }

   for (i=0; i<warp_size; i++) {
      if (shader->pipeline_reg[i][TS_IF].hw_thread_id != -1 ) {
     /*TEST
     printf("%llu:  Stalling 2\n", gpu_sim_cycle);
     //TEST*/
         return 1;  // stalled 
      }
   }
   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[i][IF_ID].hw_thread_id != -1 ) {
     /*TEST
     printf("%llu:  Stalling 3\n", gpu_sim_cycle);
     //TEST*/
         return 1;  // stalled 
      }
   }

   shader->warp_part2issue = 0; // reset pointer to first warp part
   shader->new_warp_TS = 1;

   return 0; // not stalled
}

// initalize the pipeline stage register to nops
void shader_clear_stage_reg(shader_core_ctx_t *shader, int stage)
{
   int i;
   for (i=0; i<warp_size; i++) {
      shader->pipeline_reg[i][stage] = nop_inst;
   }
}

// return the next pc of a thread 
address_type shader_thread_nextpc(shader_core_ctx_t *shader, int tid)
{
   assert( gpgpu_cuda_sim );
   address_type pc = ptx_thread_get_next_pc( shader->thread[tid].ptx_thd_info );
   return pc;
}

// issue thread to the warp 
// tid - thread id, warp_id - used by PDOM, wlane - position in warp
void shader_issue_thread(shader_core_ctx_t *shader, int tid, int wlane, unsigned active_mask )
{
  /*TEST
  static int count = 0;
  //TEST*/
   if ( gpgpu_cuda_sim ) {
      shader->pipeline_reg[wlane][TS_IF].hw_thread_id = tid;
      shader->pipeline_reg[wlane][TS_IF].wlane = wlane;
      shader->pipeline_reg[wlane][TS_IF].pc = ptx_thread_get_next_pc( shader->thread[tid].ptx_thd_info );
      shader->pipeline_reg[wlane][TS_IF].ptx_thd_info = shader->thread[tid].ptx_thd_info;
      shader->pipeline_reg[wlane][TS_IF].memreqaddr = 0;
      shader->pipeline_reg[wlane][TS_IF].uid = g_next_shader_inst_uid++;
      shader->pipeline_reg[wlane][TS_IF].warp_active_mask = active_mask;
      shader->pipeline_reg[wlane][TS_IF].ts_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
      //SEAN
      if(g_pipetrace) {
	if(pipe_stat_last == NULL) {
	  /*TEST
	  printf("pipe_stats list is empty\n");
	  //TEST*/
	  pipe_stat_last = (pipe_stat *) malloc(sizeof(pipe_stat));
	  pipe_stat_first = pipe_stat_last;
	} else {
	  pipe_stat *curr = (pipe_stat *) malloc(sizeof(pipe_stat));
	  curr->prev = pipe_stat_last;
	  pipe_stat_last->next = curr;
	  pipe_stat_last = curr;
	}
	/*TEST
	count++;
	if(!(count % 1000)) printf("COUNT = %i\n",count);
	//TEST*/
	pipe_stat_last->issued=0;
	pipe_stat_last->in_fetch=0;
	pipe_stat_last->in_decode=0;
	pipe_stat_last->in_pre_exec=0;
	pipe_stat_last->in_exec=0;
	pipe_stat_last->in_pre_mem=0;
	pipe_stat_last->in_mem=0;
	pipe_stat_last->in_writeback=0;
	pipe_stat_last->uid = shader->pipeline_reg[wlane][TS_IF].uid;
	pipe_stat_last->hw_thread_id = shader->pipeline_reg[wlane][TS_IF].hw_thread_id;
	/*TEST
	if(pipe_stat_last->uid == 60000) printf("Creating pipe_stat for uid %u\n", pipe_stat_last->uid);
	//TEST*/
	pipe_stat_last->issued = gpu_sim_cycle;
      }
   }
   assert( shader->thread[tid].avail4fetch > 0 );
   shader->thread[tid].avail4fetch--;
   assert( shader->thread[tid - (tid % warp_size)].n_avail4fetch > 0 );
   shader->thread[tid - (tid % warp_size)].n_avail4fetch--;
   /*TEST
   printf("%llu SEAN:  n_avail4fetch decremented (now = %i)\n", gpu_sim_cycle, shader->thread[tid - (tid % warp_size)].n_avail4fetch);
   //TEST*/
} //shader_issue_thread

void update_max_branch_priority(shader_core_ctx_t *shader, unsigned warp_hw_tid, unsigned grid_num )
{
   int i;
   int temp_max = 0;
   // This means that a group of threads has completed,
   // hence need to update max_priority
   for (i = 0; i<warp_size; i++) {
      if ( !ptx_thread_done( shader->thread[warp_hw_tid+i].ptx_thd_info ) ) {
         if (shader->thread[warp_hw_tid+i].warp_priority>=temp_max) {
            temp_max = shader->thread[warp_hw_tid+i].warp_priority;
         }
      }
   }
   for (i = 0; i<warp_size; i++) {
      shader->max_branch_priority[warp_hw_tid+i] = temp_max;
   }
}

void shader_fetch_simd_no_reconverge(shader_core_ctx_t *shader, unsigned int shader_number, int grid_num )
{
   int i;
   int tid;
   int new_tid = 0;
   address_type pc = 0;
   int warp_ok = 0;
   int n_warp = shader->n_threads/warp_size;
   int complete = 0;

   assert(gpgpu_cuda_sim);

   // First, check to see if entire program is completed, 
   // if it is, then break out of loop
   for (i=0; i<shader->n_threads; i++) {
      if (!ptx_thread_done( shader->thread[i].ptx_thd_info )) {
         complete = 0;
         break;
      } else {
         complete = 1;
      }
   }
   if (complete) {
      // printf("Shader has completed program.\n");
      return;
   }

   if (shader_fetch_stalled(shader)) {
      return; 
   }
   shader_clear_stage_reg(shader, TS_IF);

   // Finds a warp where all threads in it are available for fetching 
   // simultaneously(all threads are not yet in pipeline, or, the ones 
   // that are not available, are completed already
   for (i=0; i<n_warp; i++) {
      int n_completed = shader->thread[warp_size*shader->next_warp].n_completed;
      int n_avail4fetch = shader->thread[warp_size*shader->next_warp].n_avail4fetch;
      if (((n_completed) == warp_size) ||
          ((n_completed + n_avail4fetch) < warp_size) ) {
         //All threads in this warp have completed, hence go to next warp
         //Or, some of the threads are still in pipeline
         warp_ok = 0; // hey look, it's a silent register update / store instruction! (this operation is redundant)
         shader->next_warp = (shader->next_warp+1)%n_warp;
      } else {
         int n_waiting_at_barrier = shader->thread[warp_size*shader->next_warp].n_waiting_at_barrier;
         if ( n_waiting_at_barrier >= warp_size ) {
            warp_ok = 0; // hey look, it's a silent register update / store instruction! (this operation is redundant)
            continue;
         }
         warp_ok = 1;
         break;
      }
   }
   // None of the instructions from inside the warp can be scheduled -> should   
   // probably just stall, ie nops into pipeline
   if (!warp_ok) {
      shader_clear_stage_reg(shader, TS_IF);  // NOTE: is this needed?
      shader->next_warp = (shader->next_warp+1)%n_warp;  // NOTE: this is not round-robin.
      /*TEST
      printf("%llu:  Stalled 4\n", gpu_sim_cycle);
      //TEST*/
      return;
   }

   tid = warp_size*shader->next_warp;

   for (i = 0; i<warp_size; i++) {
      if (shader->thread[tid+i].warp_priority == shader->max_branch_priority[tid+i]) {
         pc = shader_thread_nextpc(shader, tid+i);
         new_tid = tid+i;
         break;
      }
   }
   //Determine which instructions inside this 'warp' will be scheduled together at this run
   //If they are cannot be scheduled together then 'save' their branch priority
   for (i = 0; i<warp_size; i++) {
      if (!ptx_thread_done( shader->thread[tid+i].ptx_thd_info )) {
         address_type next_pc;
         next_pc = shader_thread_nextpc(shader, tid+i);
         if (next_pc != pc ||
             shader->thread[tid+i].warp_priority != shader->max_branch_priority[tid+i] ||
             shader->thread[tid+i].m_waiting_at_barrier) {
            if (!ptx_thread_done( shader->thread[tid+i].ptx_thd_info )) {
               if ( !shader->thread[tid + i].m_waiting_at_barrier ) {
                  shader->thread[tid + i].warp_priority = shader->branch_priority;
               }
            }
         } else {
            shader_issue_thread(shader, tid+i, i,(unsigned)-1); 
         }
      }
   }
   shader->branch_priority++;

   shader->next_warp = (shader->next_warp+1)%n_warp;
}

int pdom_sched_find_next_warp (shader_core_ctx_t *shader,int pdom_sched_policy, int* ready_warps
                               , int ready_warp_count, int* last_warp, int w_comp_c, int w_pipe_c, int w_barr_c)
{
   int n_warp = shader->n_threads/warp_size;
   int i=0;
   int selected_warp = ready_warps[0];
   int found =0; 

   switch (pdom_sched_policy) {
   case 0: 
      selected_warp = ready_warps[0]; //first ok warp found
      found=1;
      break;
   case 1  ://random
      selected_warp = ready_warps[rand()%ready_warp_count];
      found=1;  
      break;
   case 8  :// execute the first available warp which is after the warp execued last time
      found=0;
      selected_warp = (last_warp[shader->sid] + 1 ) % n_warp;
      while (!found) {
         for (i=0;i<ready_warp_count;i++) {
            if (selected_warp==ready_warps[i]) {
               found=1;
            }
         }
         if (!found)
            selected_warp = (selected_warp + 1 ) % n_warp;
      }
      break;         
   default:
      assert(0);
   }
   if (found) {
      if (ready_warp_count==1) {
         n_pdom_sc_single_stat++;
      } else {
         n_pdom_sc_orig_stat++;
      }
      return selected_warp;
   } else {
      return -1;
   }
}

void shader_fetch_simd_postdominator(shader_core_ctx_t *shader, unsigned int shader_number, int grid_num) {
   int i;
   int warp_ok = 0;
   int n_warp = shader->n_threads/warp_size;
   int complete = 0;
   int tmp_warp;
   int warp_id;

   address_type check_pc = -1;

   assert(gpgpu_cuda_sim);

   // First, check to see if entire program is completed, 
   // if it is, then break out of loop
   for (i=0; i<shader->n_threads; i++) {
      if (!ptx_thread_done( shader->thread[i].ptx_thd_info )) {
         complete = 0;
         break;
      } else {
         complete = 1;
      }
   }
   if (complete) {
      return;
   }

   if (shader_fetch_stalled(shader)) { //check if fetch stage is stalled
      return; 
   }
   shader_clear_stage_reg(shader, TS_IF);

   int ready_warp_count = 0;
   int w_comp_c = 0 ;
   int w_pipe_c = 0 ;
   int w_barr_c = 0 ;
   static int * ready_warps = NULL;
   static int * tmp_ready_warps = NULL;
   if (!ready_warps) {
      ready_warps = (int*)calloc(n_warp,sizeof(int));
   }
   if (!tmp_ready_warps) {
      tmp_ready_warps = (int*)calloc(n_warp,sizeof(int));
   }
   for (i=0; i<n_warp; i++) {
      ready_warps[i]=-1;
      tmp_ready_warps[i]=-1;
   }

   static int* last_warp; //keeps track of last warp issued per shader
   if (!last_warp) {
      last_warp = (int*)calloc(gpu_n_shader,sizeof(int));
   }


   // Finds a warp where all threads in it are available for fetching 
   // simultaneously(all threads are not yet in pipeline, or, the ones 
   // that are not available, are completed already
   for (i=0; i<n_warp; i++) {
      int n_completed = shader->thread[warp_size*shader->next_warp].n_completed;
      int n_avail4fetch = shader->thread[warp_size*shader->next_warp].n_avail4fetch;

      if ((n_completed) == warp_size) {
         //All threads in this warp have completed 
         w_comp_c++;
      } else if ((n_completed+n_avail4fetch) < warp_size) {
         //some of the threads are still in pipeline
         w_pipe_c++;
      } else if ( shader->thread[warp_size*shader->next_warp].n_waiting_at_barrier == 
                  shader->thread[warp_size*shader->next_warp].n_avail4fetch ) {
         w_barr_c++;
      } else {  //SEAN:  Unless this branch is taken, the pipe will stall (i.e. warp_ok will never be set)
         // A valid warp is found at this point
         tmp_ready_warps[ready_warp_count] = shader->next_warp;
         ready_warp_count++;
      }
      shader->next_warp = (shader->next_warp + 1) % n_warp;
   }
   for (i=0;i<ready_warp_count;i++) {
      ready_warps[i]=tmp_ready_warps[i];
   }

   num_warps_issuable[ready_warp_count]++;
   num_warps_issuable_pershader[shader->sid]+= ready_warp_count;

   if (ready_warp_count) {
      tmp_warp = pdom_sched_find_next_warp (shader, pdom_sched_type ,ready_warps
                                            , ready_warp_count, last_warp, w_comp_c, w_pipe_c ,w_barr_c);
      if (tmp_warp != -1) {
         shader->next_warp = tmp_warp;
         warp_ok=1;  
      }
   }

   static int no_warp_issued; 
   // None of the instructions from inside the warp can be scheduled -> should  
   // probably just stall, ie nops into pipeline
   if (!warp_ok) {
      shader_clear_stage_reg(shader, TS_IF);  
      /*TEST
      printf("%llu:  Stalled 5\n", gpu_sim_cycle);
      //TEST*/
      shader->next_warp = (shader->next_warp+1) % n_warp;  
      no_warp_issued = 1 ; 
      return;
   }

   /************************************************************/
   //at this point we have a warp to execute which is pointed to by
   //shader->next_warp

   warp_id = shader->next_warp;
   last_warp[shader->sid] = warp_id;
   int wtid = warp_size*shader->next_warp;

   pdom_warp_ctx_t *scheduled_warp = &(shader->pdom_warp[warp_id]);

   int stack_top = scheduled_warp->m_stack_top;

   address_type top_pc = scheduled_warp->m_pc[stack_top];
   unsigned int top_active_mask = scheduled_warp->m_active_mask[stack_top];
   address_type top_recvg_pc = scheduled_warp->m_recvg_pc[stack_top];

   assert(top_active_mask != 0);

   const address_type null_pc = 0;
   int warp_diverged = 0;
   address_type new_recvg_pc = null_pc;
   while (top_active_mask != 0) {

      // extract a group of threads with the same next PC among the active threads in the warp
      address_type tmp_next_pc = null_pc;
      unsigned int tmp_active_mask = 0;
      void *first_active_thread=NULL;
      for (i = warp_size - 1; i >= 0; i--) {
         unsigned int mask = (1 << i);
         if ((top_active_mask & mask) == mask) { // is this thread active?
            if (ptx_thread_done( shader->thread[wtid+i].ptx_thd_info )) {
               top_active_mask &= ~mask; // remove completed thread from active mask
            } else if (tmp_next_pc == null_pc) {
               first_active_thread = shader->thread[wtid+i].ptx_thd_info;
               tmp_next_pc = shader_thread_nextpc(shader, wtid+i);
               tmp_active_mask |= mask;
               top_active_mask &= ~mask;
            } else if (tmp_next_pc == shader_thread_nextpc(shader, wtid+i)) {
               tmp_active_mask |= mask;
               top_active_mask &= ~mask;
            }
         }
      }

      // discard the new entry if its PC matches with reconvergence PC
      // that automatically reconverges the entry 
      if (tmp_next_pc == top_recvg_pc) continue;

      // this new entry is not converging
      // if this entry does not include thread from the warp, divergence occurs
      if (top_active_mask != 0 && warp_diverged == 0) {
         warp_diverged = 1;
         // modify the existing top entry into a reconvergence entry in the pdom stack
         new_recvg_pc = get_converge_point(top_pc,first_active_thread);
         if (new_recvg_pc != top_recvg_pc) {
            scheduled_warp->m_pc[stack_top] = new_recvg_pc;
            scheduled_warp->m_branch_div_cycle[stack_top] = gpu_sim_cycle;
            stack_top += 1;
            scheduled_warp->m_branch_div_cycle[stack_top] = 0;
         }
      }

      // discard the new entry if its PC matches with reconvergence PC
      if (warp_diverged && tmp_next_pc == new_recvg_pc) continue;

      // update the current top of pdom stack
      scheduled_warp->m_pc[stack_top] = tmp_next_pc;
      scheduled_warp->m_active_mask[stack_top] = tmp_active_mask;
      if (warp_diverged) {
         scheduled_warp->m_calldepth[stack_top] = 0;
         scheduled_warp->m_recvg_pc[stack_top] = new_recvg_pc;
      } else {
         scheduled_warp->m_recvg_pc[stack_top] = top_recvg_pc;
      }
      stack_top += 1; // set top to next entry in the pdom stack
   }
   scheduled_warp->m_stack_top = stack_top - 1;

   assert(scheduled_warp->m_stack_top >= 0);
   assert(scheduled_warp->m_stack_top < warp_size * 2);

   // schedule threads according to active mask on the top of pdom stack
   for (i = 0; i < warp_size; i++) {
      unsigned int mask = (1 << i);
      if ((scheduled_warp->m_active_mask[scheduled_warp->m_stack_top] & mask) == mask) {
         assert (!ptx_thread_done( shader->thread[wtid+i].ptx_thd_info ));
         shader_issue_thread(shader, wtid+i, i, scheduled_warp->m_active_mask[scheduled_warp->m_stack_top]);
      }
   }
   shader->next_warp = (shader->next_warp+1)%n_warp;

   // check if all issued threads have the same pc
   for (i = 0; i < warp_size; i++) {
      if ( shader->pipeline_reg[i][TS_IF].hw_thread_id != -1 ) {
         if ( check_pc == -1 ) {
            check_pc = shader->pipeline_reg[i][TS_IF].pc;
         } else {
            assert( check_pc == shader->pipeline_reg[i][TS_IF].pc );
         }
      }
   }
} //shader_fetch_simd_postdominator

void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc )
{
   unsigned warp_id = tid/warp_size;
   pdom_warp_ctx_t *warp_info = &(sc[sid]->pdom_warp[warp_id]);
   unsigned idx = warp_info->m_stack_top;
   *pc = warp_info->m_pc[idx];
   *rpc = warp_info->m_recvg_pc[idx];
}

void shader_fetch_mimd( shader_core_ctx_t *shader, unsigned int shader_number ) 
{
   int i, j;
   unsigned int last_issued_thread = 0;

   if (shader_fetch_stalled(shader)) {
      return; 
   }
   shader_clear_stage_reg(shader, TS_IF);

   // some form of barrel processing: 
   // - checking availability from the thread after the last issued thread
   for (i=0,j=0;i<shader->n_threads && j<warp_size;i++) {
      int thd_id = (i + shader->last_issued_thread + 1) % shader->n_threads;
      if (shader->thread[thd_id].avail4fetch && !shader->thread[thd_id].m_waiting_at_barrier ) {
         shader_issue_thread(shader, thd_id, j,(unsigned)-1);
         last_issued_thread = thd_id;
         j++;
      }
   }
   shader->last_issued_thread = last_issued_thread;
}

// seperate the incoming warp into multiple warps with seperate pcs
int split_warp_by_pc(int *tid_in, shader_core_ctx_t *shader, int **tid_split, address_type *pc) {
   int i,j;
   int n_pc = 0;
   static int *pc_cnt = NULL; // count the number of threads with the same pc

   assert(tid_in);
   assert(tid_split);
   assert(pc);
   memset(pc,0,sizeof(address_type)*warp_size);

   if (!pc_cnt) pc_cnt = (int*) malloc(sizeof(int)*warp_size);
   memset(pc_cnt,0,sizeof(int)*warp_size);

   // go through each thread in the given warp
   for (i=0; i<warp_size; i++) {
      if (tid_in[i] < 0) continue;
      int matched = 0;
      address_type thd_pc;
      thd_pc = shader_thread_nextpc(shader, tid_in[i]);

      // check to see if the pc has occured before
      for (j=0; j<n_pc; j++) {
         if (thd_pc == pc[j]) {
            tid_split[j][pc_cnt[j]] = tid_in[i];
            pc_cnt[j]++;
            matched = 1;
            break;
         }
      }
      // if not, put the tid in a seperate warp
      if (!matched) {
         assert(n_pc < warp_size);
         tid_split[n_pc][0] = tid_in[i];
         pc[n_pc] = thd_pc;
         pc_cnt[n_pc] = 1;
         n_pc++;
      }
   }
   return n_pc;
}

// see if this warp just executed the barrier instruction 
int warp_reached_barrier(int *tid_in, shader_core_ctx_t *shader)
{
   int reached_barrier = 0;
   int i;
   for (i=0; i<warp_size; i++) {
      if (tid_in[i] < 0) continue;
      if (shader->thread[tid_in[i]].m_reached_barrier) {
         reached_barrier = 1;
         break;
      }
   }
   return reached_barrier;
}

// seperate the incoming warp into multiple warps with seperate pcs and cta
int split_warp_by_cta(int *tid_in, shader_core_ctx_t *shader, int **tid_split, address_type *pc, int *cta) {
   int i,j;
   int n_pc = 0;
   static int *pc_cnt = NULL; // count the number of threads with the same pc

   assert(tid_in);
   assert(tid_split);
   assert(pc);
   memset(pc,0,sizeof(address_type)*warp_size);

   if (!pc_cnt) pc_cnt = (int*) malloc(sizeof(int)*warp_size);
   memset(pc_cnt,0,sizeof(int)*warp_size);

   // go through each thread in the given warp
   for (i=0; i<warp_size; i++) {
      if (tid_in[i] < 0) continue;
      int matched = 0;
      address_type thd_pc;
      thd_pc = shader_thread_nextpc(shader, tid_in[i]);

      int thd_cta = ptx_thread_get_cta_uid( shader->thread[tid_in[i]].ptx_thd_info );

      // check to see if the pc has occured before
      for (j=0; j<n_pc; j++) {
         if (thd_pc == pc[j] && thd_cta == cta[j]) {
            tid_split[j][pc_cnt[j]] = tid_in[i];
            pc_cnt[j]++;
            matched = 1;
            break;
         }
      }
      // if not, put the tid in a seperate warp
      if (!matched) {
         assert(n_pc < warp_size);
         tid_split[n_pc][0] = tid_in[i];
         pc[n_pc] = thd_pc;
         cta[n_pc] = thd_cta;
         pc_cnt[n_pc] = 1;
         n_pc++;
      }
   }
   return n_pc;
}

void shader_fetch_simd_dwf( shader_core_ctx_t *shader, unsigned int shader_number ) {
   int i;

   static int *tid_in = NULL;
   static int *tid_out = NULL;

   if (!tid_in) {
      tid_in = malloc(sizeof(int)*warp_size);
      memset(tid_in, -1, sizeof(int)*warp_size);
   }
   if (!tid_out) {
      tid_out = malloc(sizeof(int)*warp_size);
      memset(tid_out, -1, sizeof(int)*warp_size);
   }


   static int **tid_split = NULL;
   if (!tid_split) {
      tid_split = (int**)malloc(sizeof(int*)*warp_size);
      tid_split[0] = (int*)malloc(sizeof(int)*warp_size*warp_size);
      for (i=1; i<warp_size; i++) {
         tid_split[i] = tid_split[0] + warp_size * i;
      }
   }

   static address_type *thd_pc = NULL;
   if (!thd_pc) thd_pc = (address_type*)malloc(sizeof(address_type)*warp_size);
   static int *thd_cta = NULL;
   if (!thd_cta) thd_cta = (int*)malloc(sizeof(int)*warp_size);

   int warpupdate_bw = 1;
   while (!dq_empty(shader->thd_commit_queue) && warpupdate_bw > 0) {
      // grab a committed warp, split it into multiple BRUs (tid_split) by PC
      int *tid_commit = (int*)dq_pop(shader->thd_commit_queue);
      memset(tid_split[0], -1, sizeof(int)*warp_size*warp_size);
      memset(thd_pc, 0, sizeof(address_type)*warp_size);
      memset(thd_cta, -1, sizeof(int)*warp_size);

      int reached_barrier = warp_reached_barrier(tid_commit, shader);

      int n_warp_update;
      if (reached_barrier) {
         n_warp_update = split_warp_by_cta(tid_commit, shader, tid_split, thd_pc, thd_cta);
      } else {
         n_warp_update = split_warp_by_pc(tid_commit, shader, tid_split, thd_pc);
      }

      if (n_warp_update > 2) gpgpu_commit_pc_beyond_two++;
      warpupdate_bw -= n_warp_update;
      // put the splitted warp updates into the DWF scheduler
      for (i=0;i<n_warp_update;i++) {
         int j;
         for (j=0;j<warp_size;j++) {
            if (tid_split[i][j] < 0) continue;
            assert(shader->thread[tid_split[i][j]].avail4fetch);
            assert(!shader->thread[tid_split[i][j]].in_scheduler);
            shader->thread[tid_split[i][j]].in_scheduler = 1;
         }
         dwf_clear_accessed(shader->sid);
         if (reached_barrier) {
            dwf_update_warp_at_barrier(shader->sid, tid_split[i], thd_pc[i], thd_cta[i]);
         } else {
            dwf_update_warp(shader->sid, tid_split[i], thd_pc[i]);
         }
      }

      free_commit_warp(tid_commit);
   }

   // Track the #PC right after the warps are input to the scheduler
   dwf_update_statistics(shader->sid);
   dwf_clear_policy_access(shader->sid);

   if (shader_fetch_stalled(shader)) {
      return; 
   }
   shader_clear_stage_reg(shader, TS_IF);

   address_type scheduled_pc;
   dwf_issue_warp(shader->sid, tid_out, &scheduled_pc);

   for (i=0; i<warp_size; i++) {
      int issue_tid = tid_out[i];
      if (issue_tid >= 0) {
         shader_issue_thread(shader, issue_tid, i, (unsigned)-1);
         shader->thread[issue_tid].in_scheduler = 0;
         shader->thread[issue_tid].m_reached_barrier = 0;
         shader->last_issued_thread = issue_tid;
         assert(shader->pipeline_reg[i][TS_IF].pc == scheduled_pc);
      }
   }   
}

void print_shader_cycle_distro( FILE *fout ) 
{
   unsigned i;
   fprintf(fout, "Warp Occupancy Distribution:\n");
   fprintf(fout, "Stall:%d\t", shader_cycle_distro[0]);
   fprintf(fout, "W0_Idle:%d\t", shader_cycle_distro[1]);
   fprintf(fout, "W0_Mem:%d", shader_cycle_distro[2]);
   for (i=3; i<warp_size+3; i++) {
      fprintf(fout, "\tW%d:%d", i-2, shader_cycle_distro[i]);
   }
   fprintf(fout, "\n");
}

void inflight_memory_insn_add( shader_core_ctx_t *shader, inst_t *mem_insn)
{
   if (enable_ptx_file_line_stats) {
      ptx_file_line_stats_add_inflight_memory_insn(shader->sid, mem_insn->pc);
   }
}

void inflight_memory_insn_sub( shader_core_ctx_t *shader, inst_t *mem_insn)
{
   if (enable_ptx_file_line_stats) {
      ptx_file_line_stats_sub_inflight_memory_insn(shader->sid, mem_insn->pc);
   }
}

void report_exposed_memory_latency( shader_core_ctx_t *shader )
{
   if (enable_ptx_file_line_stats) {
      ptx_file_line_stats_commit_exposed_latency(shader->sid, 1);
   }
}

static int gpgpu_warp_occ_detailed = 0;
static int **warp_occ_detailed = NULL;

extern void check_stage_pcs( shader_core_ctx_t *shader, unsigned stage );
extern void check_pm_stage_pcs( shader_core_ctx_t *shader, unsigned stage );

void shader_fetch( shader_core_ctx_t *shader, unsigned int shader_number, int grid_num ) 
{
   assert(shader->model < NUM_SIMD_MODEL);
   unsigned i;
   int n_warp_parts = warp_size/pipe_simd_width;

   // check if decode stage is stalled
   int decode_stalled = 0;
   for (i = 0; i < pipe_simd_width; i++) {
     if (shader->pipeline_reg[i][IF_ID].hw_thread_id != -1 ) {
       /*TEST
       printf("%llu:  Stalled 6\n", gpu_sim_cycle);
       //TEST*/
       decode_stalled = 1;
     }
   }

   if (shader->gpu_cycle % n_warp_parts == 0) { //n_warp_parts always '1'?
     //n_warp_parts = warp_size/pipe_simd_width

      if (shader->model == POST_DOMINATOR || shader->model == NO_RECONVERGE) {
         int warpupdate_bw = 1; // number of warps to be unlocked per scheduler cycle
         while (!dq_empty(shader->thd_commit_queue) && warpupdate_bw > 0) {
	   /*TEST
	   printf("SEAN:  thd_commit_queue is not empty (and warpupdate_bw = %i)\n", warpupdate_bw);
	   //TEST*/
            // grab a committed warp and unlock it here
            int *tid_commit = (int*)dq_pop(shader->thd_commit_queue);
	    /*TEST
	    dq_print(shader->thd_commit_queue);
	    //TEST*/
            for ( i=0; i<warp_size; i++) {
               if (tid_commit[i] >= 0) {
                  shader->thread[tid_commit[i]].avail4fetch++;
                  assert(shader->thread[tid_commit[i]].avail4fetch <= 1);
                  assert( shader->thread[tid_commit[i] - (tid_commit[i]%warp_size)].n_avail4fetch < warp_size );
                  shader->thread[tid_commit[i] - (tid_commit[i]%warp_size)].n_avail4fetch++;
		  /*TEST
		  printf("%llu SEAN:  n_avail4fetch incremented (now = %i)\n", gpu_sim_cycle, shader->thread[tid_commit[i] - (tid_commit[i]%warp_size)].n_avail4fetch);
		  //TEST*/
               }
            }
            warpupdate_bw -= 1;
            free_commit_warp(tid_commit);
         }
      }

      switch (shader->model) {
      case NO_RECONVERGE:
         shader_fetch_simd_no_reconverge(shader, shader_number, grid_num );
         break;
      case POST_DOMINATOR:
         shader_fetch_simd_postdominator(shader, shader_number, grid_num);
         break;
      case MIMD:
         shader_fetch_mimd(shader, shader_number);
         break;
      case DWF:
         shader_fetch_simd_dwf(shader, shader_number);
         break;
      default:
         fprintf(stderr, "Unknown scheduler: %d\n", shader->model);
         assert(0);
         break;
      }

      static int *tid_out = NULL;
      if (!tid_out) {
         tid_out = (int*) malloc(sizeof(int) * warp_size);
      }
      memset(tid_out, -1, sizeof(int)*warp_size);

      if (!shader_cycle_distro) {
         shader_cycle_distro = (unsigned int*) calloc(warp_size + 3, sizeof(unsigned int));
      }

      if (gpgpu_no_divg_load && shader->new_warp_TS && !decode_stalled) {
         int n_thd_in_warp = 0;
         address_type pc_out = 0xDEADBEEF;
         for (i=0; i<warp_size; i++) {
            tid_out[i] = shader->pipeline_reg[i][TS_IF].hw_thread_id;
            if (tid_out[i] >= 0) {
               n_thd_in_warp += 1;
               pc_out = shader->pipeline_reg[i][TS_IF].pc;
            }
         }
         wpt_register_warp(tid_out, shader);
         if (gpu_runtime_stat_flag & GPU_RSTAT_DWF_MAP) {
            track_thread_pc( shader->sid, tid_out, pc_out );
         }
         if (gpgpu_cflog_interval != 0) {
            insn_warp_occ_log( shader->sid, pc_out, n_thd_in_warp);
            shader_warp_occ_log( shader->sid, n_thd_in_warp);
         }
         if ( gpgpu_warpdistro_shader < 0 || shader->sid == gpgpu_warpdistro_shader ) {
            shader_cycle_distro[n_thd_in_warp + 2] += 1;
            if (n_thd_in_warp == 0) {
               if (shader->pending_mem_access == 0) shader_cycle_distro[1]++;
            }
         }
         shader->new_warp_TS = 0;

         if ( gpgpu_warp_occ_detailed && 
              n_thd_in_warp && (shader->model == POST_DOMINATOR) ) {
            int n_warp = gpu_n_thread_per_shader / warp_size;
            if (!warp_occ_detailed) {
               warp_occ_detailed = malloc(sizeof(int*) * gpu_n_shader * n_warp);
               warp_occ_detailed[0] = calloc(sizeof(int), gpu_n_shader * n_warp * warp_size);
               for (i = 0; i < gpu_n_shader * n_warp; i++) {
                  warp_occ_detailed[i] = warp_occ_detailed[0] + i * warp_size;
               }
            }

            int wid = -1;
            for (i=0; i<warp_size; i++) {
               if (tid_out[i] >= 0) wid = tid_out[i] / warp_size;
            }
            assert(wid != -1);
            warp_occ_detailed[shader->sid * n_warp + wid][n_thd_in_warp - 1] += 1;

            if (shader->sid == 0 && wid == 16 && 0) {
               printf("wtrace[%08x] ", pc_out);
               for (i=0; i<warp_size; i++) {
                  printf("%03d ", tid_out[i]);
               }
               printf("\n");
            }
         }
      } else {
         if ( gpgpu_warpdistro_shader < 0 || shader->sid == gpgpu_warpdistro_shader ) {
            shader_cycle_distro[0] += 1;
         }
      }

      if (!decode_stalled) {
         for (i = 0; i < warp_size; i++) {
            int tid_tsif = shader->pipeline_reg[i][TS_IF].hw_thread_id;
            address_type pc_out = shader->pipeline_reg[i][TS_IF].pc;
            cflog_update_thread_pc(shader->sid, tid_tsif, pc_out);
         }
      }

      if (enable_ptx_file_line_stats && !decode_stalled) {
         int TS_stage_empty = 1;
         for (i = 0; i < warp_size; i++) {
            if (shader->pipeline_reg[i][TS_IF].hw_thread_id >= 0) {
               TS_stage_empty = 0;
               break;
            }
         }
         if (TS_stage_empty) {
            report_exposed_memory_latency(shader);
         }
      }
   }

   // if not, send the warp part to decode stage
   if (!decode_stalled && shader->warp_part2issue < n_warp_parts) {
      check_stage_pcs(shader,TS_IF);
      for (i = 0; i < pipe_simd_width; i++) {
         int wlane_idx = shader->warp_part2issue * pipe_simd_width + i;
         shader->pipeline_reg[i][IF_ID] = shader->pipeline_reg[wlane_idx][TS_IF];
         shader->pipeline_reg[i][IF_ID].if_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
         shader->pipeline_reg[wlane_idx][TS_IF] = nop_inst;
	 //SEAN
	 if(g_pipetrace) {
	   if(shader->pipeline_reg[i][IF_ID].uid != nop_inst.uid) {
	     pipe_stat *curr=pipe_stat_last;
	     while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][IF_ID].uid)) curr = curr->prev;    
	     assert(curr->uid == shader->pipeline_reg[i][IF_ID].uid);
	     curr->in_fetch = gpu_sim_cycle;
	   }
	 }
      }
      shader->warp_part2issue += 1;
   }
} //shader_fetch

inline int is_load ( op_type op ) {
   return op == LOAD_OP;
}

inline int is_store ( op_type op ) {
   return op == STORE_OP;
}

inline int is_tex ( int space ) {
   return((space) == TEX_DIRECTIVE);
}

inline int is_const ( int space ) {
   return((space) == CONST_DIRECTIVE || (space) == PARAM_DIRECTIVE);
}

inline int is_local ( int space ) {
   return((space) == LOCAL_DIRECTIVE);
}

inline int is_param ( int space ) {
   return((space) == PARAM_DIRECTIVE);
}

inline int is_shared ( int space ) {
   return((space) == SHARED_DIRECTIVE);
}

inline int shmem_bank ( address_type addr ) {
   return((int)(addr/((address_type)WORD_SIZE)) % gpgpu_n_shmem_bank);
}

inline int cache_bank ( address_type addr, shader_core_ctx_t *shader ) {
   return(int)( addr >> (address_type)shader->L1cache->line_sz_log2 ) & ( gpgpu_n_cache_bank - 1 );
}

void shader_decode( shader_core_ctx_t *shader, 
                    unsigned int shader_number,
                    unsigned int grid_num ) {

   address_type addr;
   dram_callback_t callback;
   op_type op;
   register int is_write;
   int tid;
   int i1, i2, i3, i4, o1, o2, o3, o4; //4 outputs needed for texture fetches in cuda-sim
   int i;
   int touched_priority=0;
   int warp_tid=0;
   unsigned space, data_size, vectorin, vectorout, /*SEAN*/ isatom;
   address_type regs_regs_PC = 0xDEADBEEF;
   address_type warp_current_pc = 0x600DBEEF;
   address_type warp_next_pc = 0x600DBEEF;
   int       warp_diverging = 0;

   for (i=0; i<pipe_simd_width;i++) {
      int next_stage = (shader->using_rrstage)? ID_RR:ID_EX;
      if (shader->pipeline_reg[i][next_stage].hw_thread_id != -1 ) {
	/*TEST
	printf("%llu:  Stalled 7\n", gpu_sim_cycle);
	//TEST*/
	return;  /* stalled */
      }
   }

   check_stage_pcs(shader,IF_ID);

   // decode the instruction 
   for (i=0; i<pipe_simd_width;i++) {

      if (shader->pipeline_reg[i][IF_ID].hw_thread_id == -1 )
         continue; /* bubble */

      /* get the next instruction to execute from fetch stage */
      tid = shader->pipeline_reg[i][IF_ID].hw_thread_id;

      if ( gpgpu_cuda_sim ) {
	 ptx_decode_inst( shader->thread[tid].ptx_thd_info, (unsigned*)&op, &i1, &i2, &i3, &i4, &o1, &o2, &o3, &o4, &vectorin, &vectorout, &isatom);
         shader->pipeline_reg[i][IF_ID].op = op;
         shader->pipeline_reg[i][IF_ID].pc = ptx_thread_get_next_pc( shader->thread[tid].ptx_thd_info );
         shader->pipeline_reg[i][IF_ID].ptx_thd_info = shader->thread[tid].ptx_thd_info;
	 shader->pipeline_reg[i][IF_ID].isatom = isatom;

      } else {
         abort();
      }
      // put the info into the shader instruction structure 
      // - useful in tracking instruction dependency (not needed for now)
      shader->pipeline_reg[i][IF_ID].in[0] = i1;
      shader->pipeline_reg[i][IF_ID].in[1] = i2;
      shader->pipeline_reg[i][IF_ID].in[2] = i3;
      shader->pipeline_reg[i][IF_ID].in[3] = i4;
      shader->pipeline_reg[i][IF_ID].out[0] = o1;
      shader->pipeline_reg[i][IF_ID].out[1] = o2;
      shader->pipeline_reg[i][IF_ID].out[2] = o3;
      shader->pipeline_reg[i][IF_ID].out[3] = o4;

   }
   for (i=0; i<pipe_simd_width;i++) {
      if (shader->pipeline_reg[i][IF_ID].hw_thread_id == -1 )
         continue; /* bubble */
      /* get the next instruction to execute from fetch stage */
      tid = shader->pipeline_reg[i][IF_ID].hw_thread_id;
      if ( gpgpu_cuda_sim ) {
	 ptx_decode_inst( shader->thread[tid].ptx_thd_info, (unsigned*)&op, &i1, &i2, &i3, &i4, &o1, &o2, &o3, &o4, &vectorin, &vectorout, &isatom );
         ptx_exec_inst( shader->thread[tid].ptx_thd_info, &addr, &space, &data_size, &callback, shader->pipeline_reg[i][IF_ID].warp_active_mask );
         shader->pipeline_reg[i][IF_ID].callback = callback;
         shader->pipeline_reg[i][IF_ID].space = space;

         if (is_local(space) && (is_load(op) || is_store(op))) {
            // During functional execution, each thread sees its own memory space for local memory, but these
            // need to be mapped to a shared address space for timing simulation.  We do that mapping here.
            addr -= 0x100;
            addr /=4;
            if (gpgpu_local_mem_map) {
               // Dnew = D*nTpC*nCpS*nS + nTpC*C + T%nTpC
               // C = S + nS*(T/nTpC)
               // D = data index; T = thread; C = CTA; S = shader core; p = per
               // keep threads in a warp contiguous
               // then distribute across memory space by CTAs from successive shader cores first, 
               // then by successive CTA in same shader core
               addr *= gpu_padded_cta_size * gpu_max_cta_per_shader * gpu_n_shader;
               addr += gpu_padded_cta_size * (shader_number + gpu_n_shader * (tid / gpu_padded_cta_size));
               addr += tid % gpu_padded_cta_size; 
            } else {
               // legacy mapping that maps the same address in the local memory space of all threads 
               // to a single contiguous address region 
               addr *= gpu_n_shader * gpu_n_thread_per_shader;
               addr += (gpu_n_thread_per_shader*shader->sid) + tid;
            }
            addr *= 4;
            addr += 0x100;
         }
         shader->pipeline_reg[i][IF_ID].is_vectorin = vectorin;
         shader->pipeline_reg[i][IF_ID].is_vectorout = vectorout;
         shader->pipeline_reg[i][IF_ID].data_size = data_size;
         warp_current_pc = shader->pipeline_reg[i][IF_ID].pc;
         regs_regs_PC = ptx_thread_get_next_pc( shader->thread[tid].ptx_thd_info );
      }

      shader->pipeline_reg[i][IF_ID].memreqaddr = addr;
      if ( op == LOAD_OP ) {
         shader->pipeline_reg[i][IF_ID].inst_type = LOAD_OP;
      } else if ( op == STORE_OP ) {
         shader->pipeline_reg[i][IF_ID].inst_type = STORE_OP;
      }

      if ( gpgpu_cuda_sim && ptx_thread_at_barrier( shader->thread[tid].ptx_thd_info ) ) {
         //assert( shader->model == MIMD );
         shader->thread[tid].m_waiting_at_barrier=1;
         shader->thread[tid].m_reached_barrier=1; // not reset at barrier release, but at the issue after that
         shader->thread[tid-(tid%warp_size)].n_waiting_at_barrier++;
         shader->waiting_at_barrier++;
         if (shader->model == DWF) {
            int cta_uid = ptx_thread_get_cta_uid( shader->thread[tid].ptx_thd_info );
            dwf_hit_barrier( shader->sid, cta_uid );
         }
         int release = ptx_thread_all_at_barrier( shader->thread[tid].ptx_thd_info ); //test if all threads arrived at the barrier
         if ( release ) { //All threads arrived at barrier...releasing
            int t;
            int cta_uid = ptx_thread_get_cta_uid( shader->thread[tid].ptx_thd_info );
            for ( t=0; t < gpu_n_thread_per_shader; ++t ) {
               if ( !ptx_thread_at_barrier( shader->thread[t].ptx_thd_info ) )
                  continue;
               int other_cta_uid = ptx_thread_get_cta_uid( shader->thread[t].ptx_thd_info );
               if ( other_cta_uid == cta_uid ) { //reseting @barrier tracking info
                  shader->thread[t].n_waiting_at_barrier=0;
                  shader->thread[t].m_waiting_at_barrier=0;
                  ptx_thread_reset_barrier( shader->thread[t].ptx_thd_info );
                  shader->waiting_at_barrier--;
               }
            }
            if (shader->model == DWF) {
               dwf_release_barrier( shader->sid, cta_uid );
            }
            ptx_thread_release_barrier( shader->thread[tid].ptx_thd_info );
         }
      } else {
         assert( !shader->thread[tid].m_waiting_at_barrier );
      }

      // put the info into the shader instruction structure 
      // - useful in tracking instruction dependency (not needed for now)
      shader->pipeline_reg[i][IF_ID].in[0] = i1;
      shader->pipeline_reg[i][IF_ID].in[1] = i2;
      shader->pipeline_reg[i][IF_ID].in[2] = i3;
      shader->pipeline_reg[i][IF_ID].in[3] = i4;
      shader->pipeline_reg[i][IF_ID].out[0] = o1;
      shader->pipeline_reg[i][IF_ID].out[1] = o2;
      shader->pipeline_reg[i][IF_ID].out[2] = o3;
      shader->pipeline_reg[i][IF_ID].out[3] = o4;

      if ( op == STORE_OP ) {
         is_write = TRUE;
      }

      if ( op == BRANCH_OP ) {
         int taken=0;
         assert( gpgpu_cuda_sim );
         taken = ptx_branch_taken(shader->thread[tid].ptx_thd_info);
      }

      // go to the next instruction 
      // - done implicitly in ptx_exec_inst()
      
      // branch divergence detection
      if (warp_next_pc != regs_regs_PC) {
         if (warp_next_pc == 0x600DBEEF) {
            warp_next_pc = regs_regs_PC;
         } else {
            warp_diverging = 1;
         }
      }

      // direct the instruction to the appropriate next stage (config dependent)
      int next_stage = (shader->using_rrstage)? ID_RR:ID_EX;
      shader->pipeline_reg[i][next_stage] = shader->pipeline_reg[i][IF_ID];
      shader->pipeline_reg[i][next_stage].id_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
      shader->pipeline_reg[i][IF_ID] = nop_inst;
      //SEAN
      if(g_pipetrace) {
	if(shader->pipeline_reg[i][next_stage].uid != nop_inst.uid) {
	  pipe_stat *curr=pipe_stat_last;
	  while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][next_stage].uid)) curr = curr->prev;    
	  assert(curr->uid == shader->pipeline_reg[i][next_stage].uid);
	  curr->in_decode = gpu_sim_cycle;
	  curr->memreqaddr = shader->pipeline_reg[i][next_stage].memreqaddr;
	  switch(shader->pipeline_reg[i][next_stage].inst_type) {
	    //	  case NO_OP:  curr->inst_type = 'N'; break;
	  case ALU_OP:  curr->inst_type = 'A'; break;
	  case LOAD_OP:  curr->inst_type = 'L'; break;
	  case STORE_OP:  curr->inst_type = 'S'; break;
	  case BRANCH_OP:  curr->inst_type = 'B'; break;
	  default:  curr->inst_type = 'U';
	  }
	}
      }
   }
   if ( shader->model == NO_RECONVERGE && touched_priority ) {
      update_max_branch_priority(shader,warp_tid,grid_num);
   }
   shader->n_diverge += warp_diverging;
   if (warp_diverging == 1) {
       assert(warp_current_pc != 0x600DBEEF); // guard against empty warp causing warp divergence
       ptx_file_line_stats_add_warp_divergence(warp_current_pc, 1);
   }
}

unsigned int n_regconflict_stall = 0;

int regfile_hash(signed thread_number, unsigned simd_size, unsigned n_banks) {
   if (gpgpu_thread_swizzling) {
      signed warp_ID = thread_number / simd_size;
      return((thread_number + warp_ID) % n_banks);
   } else {
      return(thread_number % n_banks);
   }
}

void shader_preexecute( shader_core_ctx_t *shader, 
                        unsigned int shader_number ) {
   int i;
   static int *thread_warp = NULL;

   if (!thread_warp) {
      thread_warp = (int*) malloc(sizeof(int) * pipe_simd_width);
   }

   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[i][RR_EX].hw_thread_id != -1 ) {
         //stalled, but can still service a register read
         if (shader->RR_k) {
            shader->RR_k--;  
         }
	 /*TEST
	 printf("%llu:  Stalled 8\n", gpu_sim_cycle);
	 //TEST*/
         return;  // stalled 
      }
   }

   // if there is still register read to service, stall
   if (shader->RR_k > 1) {
      shader->RR_k--;
      return; 
   }

   // if RR_k == 1, it was stalled previously and the register read is now done
   if (!shader->RR_k && gpgpu_reg_bankconflict) {
      for (i=0; i<pipe_simd_width; i++) {
         thread_warp[i] = 0;
      }
      for (i=0; i<pipe_simd_width; i++) {
         if (shader->pipeline_reg[i][ID_RR].hw_thread_id != -1 )
            thread_warp[regfile_hash(shader->pipeline_reg[i][ID_RR].hw_thread_id, 
                                     warp_size, pipe_simd_width)]++;
      }
      for (i=0; i<pipe_simd_width; i++) {
         if (thread_warp[i] > shader->RR_k) {
            shader->RR_k = thread_warp[i];
         }
      }
   }

   // if there are more than one register read at a bank, stall
   if (shader->RR_k > 1) {
      n_regconflict_stall++;
      shader->RR_k--;
      /*TEST
      printf("%llu:  Stalled 9\n", gpu_sim_cycle);
      //TEST*/
      return; 
   }

   check_stage_pcs(shader,ID_RR);

   shader->RR_k = 0; //setting RR_k to 0 to indicate RF conflict check next cycle
   for (i=0; i<pipe_simd_width;i++) {
      //SEAN
      if(g_pipetrace) {
	if(shader->pipeline_reg[i][ID_RR].uid != nop_inst.uid) {
	  pipe_stat *curr=pipe_stat_last;
	  while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][ID_RR].uid)) curr = curr->prev;    
	  assert(curr->uid == shader->pipeline_reg[i][ID_RR].uid);
	  curr->in_pre_exec = gpu_sim_cycle;
	}
      }

      if (shader->pipeline_reg[i][ID_RR].hw_thread_id == -1 )
         continue; //bubble 
      shader->pipeline_reg[i][ID_EX] = shader->pipeline_reg[i][ID_RR];
      shader->pipeline_reg[i][ID_RR] = nop_inst;
   }

}


void shader_execute( shader_core_ctx_t *shader, 
                     unsigned int shader_number ) {

   int i;

   for (i=0; i<pipe_simd_width; i++) {
      if (gpgpu_pre_mem_stages) {
         if (shader->pre_mem_pipeline[i][0].hw_thread_id != -1 ) {
	   /*TEST - alread here, I just removed commenting
	   printf("stalled in shader_execute\n");
	   //TEST*/
	   return;  // stalled 
         }
      } else {
	if (shader->pipeline_reg[i][EX_MM].hw_thread_id != -1 ) {
	  /*TEST
	  printf("stalled in shader_execute\n");
	  //TEST*/
	  return;  // stalled 
	}
      }
   }

   check_stage_pcs(shader,ID_EX);

   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[i][ID_EX].hw_thread_id == -1 )
         continue; // bubble 
      if (gpgpu_pre_mem_stages) {
         shader->pre_mem_pipeline[i][0] = shader->pipeline_reg[i][ID_EX];
         shader->pre_mem_pipeline[i][0].ex_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
      }
      else {
         shader->pipeline_reg[i][EX_MM] = shader->pipeline_reg[i][ID_EX];
         shader->pipeline_reg[i][EX_MM].ex_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
      }
      shader->pipeline_reg[i][ID_EX] = nop_inst;
      //SEAN
      if(g_pipetrace) {
	if(shader->pipeline_reg[i][EX_MM].uid != nop_inst.uid) {
	  pipe_stat *curr=pipe_stat_last;
	  while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][EX_MM].uid)) curr = curr->prev;    
	  assert(curr->uid == shader->pipeline_reg[i][EX_MM].uid);
	  curr->in_exec = gpu_sim_cycle;
	}
      }
   }  
}

void shader_pre_memory( shader_core_ctx_t *shader, 
                        unsigned int shader_number ) {
   int i,j;


   for (j = gpgpu_pre_mem_stages; j > 0; j--) {
      for (i=0; i<pipe_simd_width; i++) {
         if (shader->pre_mem_pipeline[i][j].hw_thread_id != -1 ) {
            return; 
         }
      }
      check_pm_stage_pcs(shader,j-1);
      for (i=0; i<pipe_simd_width; i++) {
         shader->pre_mem_pipeline[i][j] = shader->pre_mem_pipeline[i][j - 1];
         shader->pre_mem_pipeline[i][j - 1] = nop_inst;
      }
   }
   check_pm_stage_pcs(shader,gpgpu_pre_mem_stages);
   for (i=0;i<pipe_simd_width ;i++ ) {
     shader->pipeline_reg[i][EX_MM] = shader->pre_mem_pipeline[i][gpgpu_pre_mem_stages];
     //SEAN 
     if(g_pipetrace) {
       if(shader->pipeline_reg[i][EX_MM].uid != nop_inst.uid) {
	 pipe_stat *curr=pipe_stat_last;
	 while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][EX_MM].uid)) curr = curr->prev; 
	 assert(curr->uid == shader->pipeline_reg[i][EX_MM].uid);
	 curr->in_pre_mem = gpu_sim_cycle;   
       }
     }
   }

   if (gpgpu_pre_mem_stages) {
      for (i=0; i<pipe_simd_width; i++)
         shader->pre_mem_pipeline[i][0] = nop_inst;
   }
}

void shader_const_memory( shader_core_ctx_t *shader, unsigned int shader_number ) 
{                                                           
   int i;
   int rc_fail = 0; // resource allocation
   int bk_conflict = 0;
   int const_mem_access = 0; //1 if const cache/memory accessed
   int wb_stalled = 0; // check if next stage is stalled
   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[i][MM_WB].hw_thread_id != -1 ) {
         wb_stalled = 1;
	 /*TEST
	 printf("%llu:  Stalled 10\n", gpu_sim_cycle);
	 //TEST*/
         break;
      }
   }

   if (shader->pending_cmem_acc >= gpgpu_const_port_per_bank) {
      shader->pending_cmem_acc-=gpgpu_const_port_per_bank;
      if (shader->pending_cmem_acc > gpgpu_const_port_per_bank) {
         if (!wb_stalled) gpu_stall_shd_mem++; // correct the performance counter.
	 /*TEST
	 printf("%llu:  Stalled 11\n", gpu_sim_cycle);
	 //TEST*/
         return;  // stalled 
      }
   }

   if (wb_stalled) {
     /*TEST
     printf("%llu:  Stalled 12\n", gpu_sim_cycle);
     //TEST*/
     return; // don't proceed if next stage is stalled.
   }

   // select from the pipeline register of choice.
   static inst_t *const_insn = NULL;
   if (!const_insn)  const_insn = (inst_t*)malloc(pipe_simd_width * sizeof(inst_t));
   for (i=0; i<pipe_simd_width; i++) {
      if (is_const(shader->pipeline_reg[i][EX_MM].space) && is_load(shader->pipeline_reg[i][EX_MM].op)) {
         const_insn[i] = shader->pipeline_reg[i][EX_MM]; //only process it if it is a constant memory instruction
         if (!is_load(const_insn[i].op))
            assert(0); //constant memory is read only!
      } else
         const_insn[i] = nop_inst;
   }

   // allocate flag arrays for cache miss check
   static unsigned char *constcachehit = NULL;
   if (!constcachehit) constcachehit = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(constcachehit, 0, pipe_simd_width * sizeof(unsigned char));

   // allocate flag arrays for memory access type
   static unsigned char *isconstcache = NULL;
   if (!isconstcache) isconstcache = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(isconstcache, 0, pipe_simd_width * sizeof(unsigned char));

   // check for cache misses
   for (i=0; i<pipe_simd_width; i++) {

      if (const_insn[i].hw_thread_id == -1 )
         continue; // bubble 

      // Identify constant instruction from parameter load
      if ( (const_insn[i].hw_thread_id) != -1) {
         if (is_param(const_insn[i].space))
            gpgpu_n_param_insn++;
         else {
            gpgpu_n_const_insn++; 
            const_mem_access = 1;
         }
         isconstcache[i] = 1;

      }
   }


   static unsigned char *cmem_bkacc = NULL;
   if (!cmem_bkacc) cmem_bkacc = (unsigned char*)malloc(sizeof(unsigned char));
   memset(cmem_bkacc, 0, sizeof(unsigned char));
   static address_type *cmem_addr = NULL;
   if (!cmem_addr) cmem_addr = (address_type*)malloc(pipe_simd_width * sizeof(address_type));
   memset(cmem_addr, 0, pipe_simd_width * sizeof(address_type));
   if (shader->pending_cmem_acc == 0) { // 0 = check conflict for new insn
      for (i=0; i<pipe_simd_width; i++) {
         if (isconstcache[i]) {
            int b=0;
            int no_conflict = 0;
            for (b=0; b < shader->pending_cmem_acc; b++) { // search for coexist access to same address
               if (cmem_addr[b] == const_insn[i].memreqaddr)
                  no_conflict=1;
            }
            if (!no_conflict) { // new address
               cmem_addr[shader->pending_cmem_acc] = const_insn[i].memreqaddr;
               shader->pending_cmem_acc++;
            }
         }
      }
   }
   if (shader->pending_cmem_acc <= gpgpu_const_port_per_bank) { // last acc to shmem done, pass insn to next stage
      shader->pending_cmem_acc = 0; // do check for conflict next cycle
   }
   if (shader->pending_cmem_acc > gpgpu_const_port_per_bank) {
      gpgpu_n_cmem_portconflict++;
      rc_fail = 1;
      bk_conflict = 1;
   }


   int cache_hits_waiting = 0;
   // access cache tag if no bank conflict
   if (!bk_conflict) {
      for (i=0; i<pipe_simd_width; i++) {

         if (isconstcache[i]) {
            if (is_param(const_insn[i].space)) {
               constcachehit[i] = 1;
            } else {

               shd_cache_line_t *hit_cacheline = shd_cache_access(shader->L1constcache,
                                                                  const_insn[i].memreqaddr,
                                                                  WORD_SIZE, 0, //should always be a read
                                                                  shader->gpu_cycle);
               constcachehit[i] = (hit_cacheline != 0);
               if (gpgpu_perfect_mem) constcachehit[i] = 1;
            } 
         }
      }
   }

   // try to allocate resource for interconnect and mshr
   int require_MSHR;
   // record each unique fetch
   int num_unq_fetch;
   static unsigned long long int *unq_memaddr = NULL;
   static int *unq_bsize = NULL;
   static int *unq_type = NULL;  // READ or WRITE
   static int *unq_merge_count = NULL; //number of mem_insn with the same unq addr
   if (!unq_memaddr) unq_memaddr = (unsigned long long int *)malloc(pipe_simd_width * sizeof(unsigned long long int));
   if (!unq_bsize) unq_bsize = (int *)malloc(pipe_simd_width * sizeof(int));
   if (!unq_type) unq_type = (int *)malloc(pipe_simd_width * sizeof(int));
   if (!unq_merge_count) unq_merge_count = (int *)malloc(pipe_simd_width * sizeof(int));
   memset(unq_type, 0, pipe_simd_width * sizeof(int));
   memset(unq_merge_count, 0, pipe_simd_width * sizeof(int));
   memset(unq_memaddr, 0, pipe_simd_width * sizeof(unsigned long long int));
   memset(unq_bsize, 0, pipe_simd_width * sizeof(int));

   static mshr_entry **fetching = NULL;
   if (!fetching) fetching = (mshr_entry **)malloc(pipe_simd_width * sizeof(mshr_entry *));
   memset(fetching, 0, pipe_simd_width * sizeof(mshr_entry *));

   num_unq_fetch = 0;
   require_MSHR = 0;
   if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC ) {
      memset(MCB_accesses, 0, gpu_n_mem*4*sizeof(int));
   }
   for (i=0; i<pipe_simd_width; i++) {

      if (const_insn[i].hw_thread_id == -1 )
         continue; // bubble 

      if (shader->model == MIMD) require_MSHR = 0; // each thread would only use its private MSHR in MIMD

      // Specify what resource is required 
      // Normal LD: Fetch Entry(if not yet issued) + MSHR
      if ( !constcachehit[i]) {//if cache miss and need to bring data into cache

         if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
            addrdec_t tlx;
            addrdec_tlx(SHD_CONSTCACHE_TAG(const_insn[i].memreqaddr, shader), &tlx);
            totalbankaccesses[tlx.chip][tlx.bk]++;
            MCB_accesses[tlx.chip*4 + tlx.bk]=1;
         }

         // Checks if there is an identical fetch request 
         // from a previous missed constant load 
         if (gpgpu_interwarp_mshr_merge & CONST_MSHR_MERGE) {
            fetching[i] = shader_check_mshr4tag(shader, const_insn[i].memreqaddr,CONSTC);
         } else {
            fetching[i] = NULL;
         }
         if (fetching[i] == NULL) {
            // check for redundent fetch in the same warp
            int w, fetching_in_same_warp;
            fetching_in_same_warp = 0;
            for (w=0;w<i;w++) {
               if (SHD_CONSTCACHE_TAG(const_insn[i].memreqaddr,shader) == SHD_CONSTCACHE_TAG(const_insn[w].memreqaddr,shader)) {
                  fetching_in_same_warp = 1;
               }
            }
            if (!fetching_in_same_warp) {
               unq_memaddr[num_unq_fetch] = SHD_CONSTCACHE_TAG(const_insn[i].memreqaddr,shader);
               unq_bsize[num_unq_fetch] = READ_PACKET_SIZE;
               unq_type[num_unq_fetch] = 1; //read
               num_unq_fetch += 1;
            }
         }
         require_MSHR++;
      }

      if ( require_MSHR && dq_full(shader->mshr[const_insn[i].hw_thread_id]) ) {
         // can't allocate all the resources - stall and retry next cycle
         rc_fail = 1;
	 /*TEST
	 printf("%llu:  Stalled 13\n", gpu_sim_cycle);
	 //TEST*/
      }
   } //end for 
   for ( i=0;i<num_unq_fetch;i++ ) {
      int w;
      for (w=0;w<pipe_simd_width;w++) {
         if (SHD_CONSTCACHE_TAG(unq_memaddr[i],shader) == SHD_CONSTCACHE_TAG(const_insn[w].memreqaddr,shader)) {
            unq_merge_count[i]++;
            gpgpu_n_intrawarp_mshr_merge++;
         }
      }
   }
   if (num_unq_fetch > 1) gpgpu_multi_unq_fetches++;

   if (const_mem_access && (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC )) {
      int MCBs_accessed = 0; //memory controller banks
      for (i=0;i < gpu_n_mem*4 ; i++) {
         MCBs_accessed += MCB_accesses[i];
      }
      num_MCBs_accessed[MCBs_accessed]++;

   }
   // Resource needs to be available for all the fetch requests (more than one) 
   if (num_unq_fetch
       && !(shader->fq_has_buffer(unq_memaddr, unq_bsize, num_unq_fetch, shader->sid)) ) {
      rc_fail = 1;
      gpu_stall_sh2icnt++;
   }

   // tracking down the mshrs allocated by this warp (for coalescing)
   static mshr_entry **same_warp_mshr = NULL;
   if (!same_warp_mshr) same_warp_mshr = (mshr_entry **)malloc(pipe_simd_width * sizeof(mshr_entry*));
   memset(same_warp_mshr, 0, pipe_simd_width * sizeof(mshr_entry*));
   if (rc_fail ) {
      // can't allocate all the resources - stall and retry next cycle
      gpu_stall_shd_mem++;
      /*TEST
      printf("%llu Stalling shd mem 14\n", gpu_sim_cycle);
      //TEST*/
      for (i=0; i<pipe_simd_width; i++) {
         if ( (const_insn[i].hw_thread_id != -1 ) ) {
            // correcting statistic for resource stall
            if (!bk_conflict) {
               shd_cache_undo_stats( shader->L1constcache, !constcachehit[i] );
            }
            gpgpu_n_const_insn--;
         }

      }
   } else {
      for (i=0; i<pipe_simd_width; i++) {
         if (const_insn[i].hw_thread_id == -1 )
            continue; // bubble 

         if (!constcachehit[i]) { //cache miss
            L1_const_miss++;
            shader->pending_mem_access++;
            // Push request to MSHR and Fetching queue:
            // The request is always pushed to MSHR as a load always need to be 
            // reactivated when its data arrives. 

            mshr_entry *mshr_e = alloc_mshr_entry();
            mshr_e->inst_uid = const_insn[i].uid;
            mshr_e->addr = const_insn[i].memreqaddr;
	    //SEAN
	    mshr_e->is_atom = const_insn[i].isatom;
            same_warp_mshr[i] = mshr_e; // attaching the mshr entry to this warp
            mshr_e->isvector = 0;
            mshr_e->reg = const_insn[i].out[0];
            mshr_e->hw_thread_id = const_insn[i].hw_thread_id;
            mshr_e->priority = shader->mshr_up_counter;
            mshr_e->inst = const_insn[i];
            mshr_e->iswrite = 0;
            mshr_e->istexture = 0;
            mshr_e->isconst = 1;
            mshr_e->islocal = 0;
            inflight_memory_insn_add(shader, &const_insn[i]);

            // need to check again for inter-warp fetch merge
            if (gpgpu_interwarp_mshr_merge & CONST_MSHR_MERGE) {
               fetching[i] = shader_check_mshr4tag(shader, const_insn[i].memreqaddr,CONSTC);
            } else {
               fetching[i] = NULL;
               if (fetching[i] == NULL) {
                  // check for redundant fetch in the same warp
                  int w, fetching_in_same_warp;
                  fetching_in_same_warp = 0;
                  for (w=0;w<i;w++) {
                     if (SHD_CACHE_TAG(const_insn[i].memreqaddr,shader) == SHD_CACHE_TAG(const_insn[w].memreqaddr,shader)) {
                        fetching_in_same_warp = 1;
                        fetching[i] = same_warp_mshr[w];
                        break;
                     }
                  }
               }
            }
            if ( fetching[i] != NULL ) {
               // merge with pending request(s)
               mergemiss++;
               shader->thread[const_insn[i].hw_thread_id].n_l1_mrghit_ac++;
               shd_cache_mergehit(shader->L1constcache, const_insn[i].memreqaddr);
               mshr_entry *others = fetching[i]->merged_requests;
               mshr_e->merged_requests = others;
               fetching[i]->merged_requests = mshr_e;
               mshr_e->status = fetching[i]->status;
               mshr_e->fetched = fetching[i]->fetched;
               mshr_e->merged = 1;
               if (mshr_e->fetched ) {
                  dq_push(shader->return_queue,mshr_e);
                  if (shader->return_queue->length > max_return_queue_length[shader->sid]) {
                     max_return_queue_length[shader->sid] = shader->return_queue->length;
                  }
               }
            }

            // Try pushing the load into the MSHR. 
            // It should always success, as resource check did not fail
            if (!dq_push(shader->mshr[const_insn[i].hw_thread_id], mshr_e)) assert(0);

            // Pushing to MSHR is successful. Issue the memory fetching 
            // for this cache line only if it is not already done 
            // by a previous load instruction. 
            if ( fetching[i] == NULL ) {
               shader->fq_push( SHD_CONSTCACHE_TAG(mshr_e->addr,shader),
                                shader->L1constcache->line_sz,
                                0, NO_PARTIAL_WRITE, shader->sid, const_insn[i].hw_thread_id, mshr_e, 
                                cache_hits_waiting, CONST_ACC_R, const_insn[i].pc);
               shader->n_mshr_used++;
               shader->mshr_up_counter++;
               if (shader->n_mshr_used > shader->max_n_mshr_used) shader->max_n_mshr_used = shader->n_mshr_used;
            }

            // turn into an nop (taking the LD/ST insn out of the pipeline)
            shader->pipeline_reg[i][EX_MM] = nop_inst;
         }
      }
      check_stage_pcs(shader,EX_MM);
      for (i=0; i<pipe_simd_width; i++) {
         if (const_insn[i].hw_thread_id == -1 )
            continue; // bubble 

         shader->pipeline_reg[i][MM_WB] = shader->pipeline_reg[i][EX_MM];
         shader->pipeline_reg[i][MM_WB].mm_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
         shader->pipeline_reg[i][EX_MM] = nop_inst;      
	 //SEAN
	 if(g_pipetrace) {
	   if(shader->pipeline_reg[i][MM_WB].uid != nop_inst.uid) {
	     pipe_stat *curr=pipe_stat_last;
	     while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][MM_WB].uid)) curr = curr->prev;    
	     assert(curr->uid == shader->pipeline_reg[i][MM_WB].uid);
	     curr->in_mem = gpu_sim_cycle;
	   }
	 }
      }  
      // reflect the change to EX|MM pipeline register to the pre_mem stage
      if (gpgpu_pre_mem_stages) {
         for (i=0;i<pipe_simd_width ;i++ )
            shader->pre_mem_pipeline[i][gpgpu_pre_mem_stages] = shader->pipeline_reg[i][EX_MM];
      }
   }
}


void shader_texture_memory( shader_core_ctx_t *shader, unsigned int shader_number ) 
{                                                           
   int i;
   int rc_fail = 0; // resource allocation
   int bk_conflict = 0;
   int tex_mem_access = 0; //1 if texture cache/memory accessed
   int wb_stalled = 0; // check if next stage is stalled
   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[i][MM_WB].hw_thread_id != -1 ) {
	 /*TEST
	 printf("%llu:  Stalled 15\n", gpu_sim_cycle);
	 //TEST*/
         wb_stalled = 1;
         break;
      }
   }
   if (wb_stalled) {
     /*TEST
     printf("%llu:  Stalled 16\n", gpu_sim_cycle);
     //TEST*/
     return; // don't proceed if next stage is stalled.
   }

   // select from the pipeline register of choice.
   static inst_t *tex_insn = NULL;
   if (!tex_insn)  tex_insn = (inst_t*)malloc(pipe_simd_width * sizeof(inst_t));
   for (i=0; i<pipe_simd_width; i++) {
      if (is_tex(shader->pipeline_reg[i][EX_MM].space) && is_load(shader->pipeline_reg[i][EX_MM].op)) {
         tex_insn[i] = shader->pipeline_reg[i][EX_MM]; //only process it if it is a texture instruction
         if (!is_load(tex_insn[i].op))
            assert(0);
         tex_mem_access = 1;
      } else
         tex_insn[i] = nop_inst;
   }

   // allocate flag arrays for cache miss check
   static unsigned char *texcachehit = NULL;
   if (!texcachehit) texcachehit = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(texcachehit, 0, pipe_simd_width * sizeof(unsigned char));

   // allocate flag arrays for memory access type
   static unsigned char *istexcache = NULL;
   if (!istexcache) istexcache = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(istexcache, 0, pipe_simd_width * sizeof(unsigned char));

   // check for cache misses
   for (i=0; i<pipe_simd_width; i++) {

      if (tex_insn[i].hw_thread_id == -1 )
         continue; // bubble 

      // Identify texture instruction 
      if ( (tex_insn[i].hw_thread_id) != -1) {
         // check L1 texture cache for data 
         gpgpu_n_tex_insn ++; 
         istexcache[i] = 1;
      }
   }

   int cache_hits_waiting = 0;
   // access cache tag if no bank conflict
   if (!bk_conflict) {
      for (i=0; i<pipe_simd_width; i++) {

         if (istexcache[i]) {
            shd_cache_line_t *hit_cacheline = shd_cache_access(shader->L1texcache,
                                                               tex_insn[i].memreqaddr,
                                                               WORD_SIZE, 0, //should always be a read
                                                               shader->gpu_cycle);
            texcachehit[i] = (hit_cacheline != 0);
            if (gpgpu_perfect_mem) texcachehit[i] = 1;
         }
      }
   }

   // try to allocate resource for interconnect and mshr
   int require_MSHR;
   // record each unique fetch
   int num_unq_fetch;
   static unsigned long long int *unq_memaddr = NULL;
   static int *unq_bsize = NULL;
   static int *unq_type = NULL;  // READ or WRITE
   static int *unq_merge_count = NULL; //number of mem_insn with the same unq addr
   if (!unq_memaddr) unq_memaddr = (unsigned long long int *)malloc(pipe_simd_width * sizeof(unsigned long long int));
   if (!unq_bsize) unq_bsize = (int *)malloc(pipe_simd_width * sizeof(int));
   if (!unq_type) unq_type = (int *)malloc(pipe_simd_width * sizeof(int));
   if (!unq_merge_count) unq_merge_count = (int *)malloc(pipe_simd_width * sizeof(int));
   memset(unq_memaddr, 0, pipe_simd_width * sizeof(unsigned long long int));
   memset(unq_bsize, 0, pipe_simd_width * sizeof(int));
   memset(unq_type, 0, pipe_simd_width * sizeof(int));
   memset(unq_merge_count, 0, pipe_simd_width * sizeof(int));
   static mshr_entry **fetching = NULL;
   if (!fetching) fetching = (mshr_entry **)malloc(pipe_simd_width * sizeof(mshr_entry *));
   memset(fetching, 0, pipe_simd_width * sizeof(mshr_entry *));

   num_unq_fetch = 0;
   require_MSHR = 0;
   if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC ) {
      memset(MCB_accesses, 0, gpu_n_mem*4*sizeof(int));
   }
   for (i=0; i<pipe_simd_width; i++) {

      if (tex_insn[i].hw_thread_id == -1 )
         continue; // bubble 

      if (shader->model == MIMD) require_MSHR = 0; // each thread would only use its private MSHR in MIMD

      // Specify what resource is required 
      // Normal LD: Fetch Entry(if not yet issued) + MSHR
      if ( !texcachehit[i]) {//if cache miss and need to bring data into cache

         if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
            addrdec_t tlx;
            addrdec_tlx(SHD_TEXCACHE_TAG(tex_insn[i].memreqaddr, shader), &tlx);
            totalbankaccesses[tlx.chip][tlx.bk]++; //bankaccesses[shader id][dram chip id][bank id]
            MCB_accesses[tlx.chip*4 + tlx.bk]=1;
         }

         // Checks if there is an identical fetch request 
         // from a previous missed texture load 
         if (gpgpu_interwarp_mshr_merge & TEX_MSHR_MERGE) {
            fetching[i] = shader_check_mshr4tag(shader, tex_insn[i].memreqaddr,TEXTC);
         } else {
            fetching[i] = NULL;
         }
         if (fetching[i] == NULL) {
            // check for redundant fetch in the same warp
            int w, fetching_in_same_warp;
            fetching_in_same_warp = 0;
            for (w=0;w<i;w++) {
               if (SHD_TEXCACHE_TAG(tex_insn[i].memreqaddr,shader) == SHD_TEXCACHE_TAG(tex_insn[w].memreqaddr,shader)) {
                  fetching_in_same_warp = 1;
               }
            }
            if (!fetching_in_same_warp) {
               unq_memaddr[num_unq_fetch] = SHD_TEXCACHE_TAG(tex_insn[i].memreqaddr,shader);
               unq_bsize[num_unq_fetch] = READ_PACKET_SIZE; 
               unq_type[num_unq_fetch] = 1; //read
               num_unq_fetch += 1;
            }
         }
         require_MSHR++;
      }

      if ( require_MSHR && dq_full(shader->mshr[tex_insn[i].hw_thread_id]) ) {
	// can't allocate all the resources - stall and retry next cycle
	/*TEST
	printf("%llu:  Stalled 17\n", gpu_sim_cycle);
	//TEST*/
	rc_fail = 1;
      }
   } //end for
   for ( i=0;i<num_unq_fetch;i++ ) {
      int w;
      for (w=0;w<pipe_simd_width;w++) {
         if (SHD_TEXCACHE_TAG(unq_memaddr[i],shader) == SHD_TEXCACHE_TAG(tex_insn[w].memreqaddr,shader)) {
            unq_merge_count[i]++;
            gpgpu_n_intrawarp_mshr_merge++;
         }
      }
   }

   if (num_unq_fetch > 1) gpgpu_multi_unq_fetches++;

   if (tex_mem_access &&(gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC )) {
      int MCBs_accessed = 0;
      for (i=0;i < gpu_n_mem*4 ; i++) {
         MCBs_accessed += MCB_accesses[i];
      }
      num_MCBs_accessed[MCBs_accessed]++;
   }
   // Resource needs to be available for all the fetch requests (more than one) 
   if (num_unq_fetch
       && !(shader->fq_has_buffer(unq_memaddr, unq_bsize, num_unq_fetch, shader->sid)) ) {
      rc_fail = 1;
      gpu_stall_sh2icnt++;
   }

   // tracking down the mshrs allocated by this warp (for coalescing)
   static mshr_entry **same_warp_mshr = NULL;
   if (!same_warp_mshr) same_warp_mshr = (mshr_entry **)malloc(pipe_simd_width * sizeof(mshr_entry*));
   memset(same_warp_mshr, 0, pipe_simd_width * sizeof(mshr_entry*));
   if (rc_fail) {
      // can't allocate all the resources - stall and retry next cycle
      gpu_stall_shd_mem++;
      /*TEST
      printf("%llu Stalling shd mem 18\n", gpu_sim_cycle);
      //TEST*/
      for (i=0; i<pipe_simd_width; i++) {
         if ( (tex_insn[i].hw_thread_id != -1 ) ) {
            // correcting statistic for resource stall
            if (!bk_conflict) {
               shd_cache_undo_stats( shader->L1texcache, !texcachehit[i] );
            }
            gpgpu_n_tex_insn--;
         }

      }
   } else {
      for (i=0; i<pipe_simd_width; i++) {
         if (tex_insn[i].hw_thread_id == -1 )
            continue; // bubble 

         if (!texcachehit[i]) { //cache miss
            L1_texture_miss++;
            shader->pending_mem_access++;
            // Push request to MSHR and Fetching queue:
            // The request is always pushed to MSHR as a load always need to be 
            // reactivated when its data arrives.

            mshr_entry *mshr_e = alloc_mshr_entry();
            mshr_e->inst_uid = tex_insn[i].uid;
            same_warp_mshr[i] = mshr_e; // attaching the mshr entry to this warp
            mshr_e->addr = tex_insn[i].memreqaddr;
	    //SEAN
	    mshr_e->is_atom = tex_insn[i].isatom;
            mshr_e->isvector = 1;
            mshr_e->reg = tex_insn[i].out[0];
            mshr_e->reg2 = tex_insn[i].out[1];
            mshr_e->reg3 = tex_insn[i].out[2];
            mshr_e->reg4 = tex_insn[i].out[3];
            mshr_e->hw_thread_id = tex_insn[i].hw_thread_id;
            mshr_e->priority = shader->mshr_up_counter;
            mshr_e->inst = tex_insn[i];
            mshr_e->iswrite = 0;
            mshr_e->istexture = 1;
            mshr_e->isconst = 0;
            inflight_memory_insn_add(shader, &tex_insn[i]);

            // need to check again for inter-warp fetch merge
            if (gpgpu_interwarp_mshr_merge & TEX_MSHR_MERGE) {
               fetching[i] = shader_check_mshr4tag(shader, tex_insn[i].memreqaddr,TEXTC);
            } else {
               fetching[i] = NULL;
               if (fetching[i] == NULL) {
                  // check for redundant fetch in the same warp
                  int w, fetching_in_same_warp;
                  fetching_in_same_warp = 0;
                  for (w=0;w<i;w++) {
                     if (SHD_CACHE_TAG(tex_insn[i].memreqaddr,shader) == SHD_CACHE_TAG(tex_insn[w].memreqaddr,shader)) {
                        fetching_in_same_warp = 1;
                        fetching[i] = same_warp_mshr[w];
                        break;
                     }
                  }
               }
            }
            if ( fetching[i] != NULL ) {
               // merge with pending request(s)
               mergemiss++;
               shader->thread[tex_insn[i].hw_thread_id].n_l1_mrghit_ac++;
               shd_cache_mergehit(shader->L1texcache, tex_insn[i].memreqaddr);
               mshr_entry *others = fetching[i]->merged_requests;
               mshr_e->merged_requests = others;
               fetching[i]->merged_requests = mshr_e;
               mshr_e->status = fetching[i]->status;
               mshr_e->fetched = fetching[i]->fetched;
               mshr_e->merged = 1;
               if (mshr_e->fetched) {
                  dq_push(shader->return_queue,mshr_e);
                  if (shader->return_queue->length > max_return_queue_length[shader->sid]) {
                     max_return_queue_length[shader->sid] = shader->return_queue->length;
                  }
               }
            }

            // Try pushing the load into the MSHR. 
            // It should always success, as resource check did not fail
            if (!dq_push(shader->mshr[tex_insn[i].hw_thread_id], mshr_e)) assert(0);

            // Pushing to MSHR is successful. Issue the memory fetching 
            // for this cache line only if it is not already done 
            // by a previous load instruction. 
            if ( fetching[i] == NULL ) {
               shader->fq_push( SHD_TEXCACHE_TAG(mshr_e->addr,shader),
                                shader->L1texcache->line_sz,
                                0, NO_PARTIAL_WRITE, shader->sid, tex_insn[i].hw_thread_id, mshr_e, 
                                cache_hits_waiting, TEXTURE_ACC_R, tex_insn[i].pc);
               shader->n_mshr_used++;                 

               shader->mshr_up_counter++;
               if (shader->n_mshr_used > shader->max_n_mshr_used) shader->max_n_mshr_used = shader->n_mshr_used;
            }

            shader->pipeline_reg[i][EX_MM] = nop_inst;
         }

      }
      check_stage_pcs(shader,EX_MM);
      for (i=0; i<pipe_simd_width; i++) {
         if (tex_insn[i].hw_thread_id == -1 )
            continue; // bubble

         shader->pipeline_reg[i][MM_WB] = shader->pipeline_reg[i][EX_MM];
         shader->pipeline_reg[i][MM_WB].mm_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
         shader->pipeline_reg[i][EX_MM] = nop_inst;  
	 //SEAN
	 if(g_pipetrace) {
	   if(shader->pipeline_reg[i][MM_WB].uid != nop_inst.uid) {
	     pipe_stat *curr=pipe_stat_last;
	     while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][MM_WB].uid)) curr = curr->prev;    
	     assert(curr->uid == shader->pipeline_reg[i][MM_WB].uid);
	     curr->in_mem = gpu_sim_cycle;
	   }
	 }
      }
      // reflect the change to EX|MM pipeline register to the pre_mem stage
      if (gpgpu_pre_mem_stages) {
         for (i=0;i<pipe_simd_width ;i++ )
            shader->pre_mem_pipeline[i][gpgpu_pre_mem_stages] = shader->pipeline_reg[i][EX_MM];
      }
   }
}

void shader_memory( shader_core_ctx_t *shader, unsigned int shader_number ) 
{
   int i;
   int rc_fail = 0; // resource allocation
   int bk_conflict = 0; // bank conflict flag
   int is_mem_access = 0; //if any threads passing through are memory accesses, this gets set to 1
   int wb_stalled = 0; // check if next stage is stalled
   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[i][MM_WB].hw_thread_id != -1 ) {
	 /*TEST
	 printf("%llu:  Stalled 19\n", gpu_sim_cycle);
	 //TEST*/
         wb_stalled = 1;
         break;
      }
   }

   if (shader->pending_shmem_bkacc > gpgpu_shmem_port_per_bank) {
      shader->pending_shmem_bkacc -= gpgpu_shmem_port_per_bank;
      if (shader->pending_shmem_bkacc > gpgpu_shmem_port_per_bank) {
         if (!wb_stalled) gpu_stall_shd_mem++; // correct the performance counter. 
	 /*TEST
	 printf("%llu:  Stalled 20\n", gpu_sim_cycle);
	 //TEST*/
         return;  // stalled 
      }
   }
   if (shader->pending_cache_bkacc > gpgpu_cache_port_per_bank) {
      shader->pending_cache_bkacc -= gpgpu_cache_port_per_bank;
      if (shader->pending_cache_bkacc > gpgpu_cache_port_per_bank) {
         if (!wb_stalled) gpu_stall_shd_mem++; // correct the performance counter. 
	 /*TEST
	 printf("%llu:  Stalled 21\n", gpu_sim_cycle);
	 //TEST*/
         return;  // stalled 
      }
   }

   if (wb_stalled) {
     /*TEST
     printf("%llu:  Stalled 22\n", gpu_sim_cycle);
     //TEST*/
     return; // don't preceed if next stage is stalled.
   }

   // select from the pipeline register of choice.
   static inst_t *mem_insn = NULL;
   if (!mem_insn)  mem_insn = (inst_t*)malloc(pipe_simd_width * sizeof(inst_t));
   for (i=0; i<pipe_simd_width; i++) {
      if (!((is_tex(shader->pipeline_reg[i][EX_MM].space) 
             || is_const(shader->pipeline_reg[i][EX_MM].space)) && is_load(shader->pipeline_reg[i][EX_MM].op))) {
         mem_insn[i] = shader->pipeline_reg[i][EX_MM]; //only process it if it is not a texture instruction
      } else
         mem_insn[i] = nop_inst;
   }

   // allocate flag arrays for cache miss check
   static unsigned char *cachehit = NULL;
   if (!cachehit) cachehit = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(cachehit, 0, pipe_simd_width * sizeof(unsigned char));

   // allocate flag arrays for memory access type
   static unsigned char *iscache = NULL;
   if (!iscache) iscache = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(iscache, 0, pipe_simd_width * sizeof(unsigned char));
   static unsigned char *iswrite = NULL;
   if (!iswrite) iswrite = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(iswrite, 0, pipe_simd_width * sizeof(unsigned char));

   // allocate flag arrays for shared memory check
   static unsigned char *isshmem = NULL;
   if (!isshmem) isshmem = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(isshmem, 0, pipe_simd_width * sizeof(unsigned char));

   // allocate flag arrays for atomic operation check
   static unsigned char *isatommem = NULL;
   if (!isatommem) isatommem = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
   memset(isatommem, 0, pipe_simd_width * sizeof(unsigned char));

   // check for cache misses
   for (i=0; i<pipe_simd_width; i++) {

      if (mem_insn[i].hw_thread_id == -1 )
         continue; // bubble 

      //SEAN
      if(mem_insn[i].isatom) {
	isatommem[i] = 1;
      }

      //SEAN - this should probably be more explicit
      // Instead of relying on the fact that only atomics have callbacks, this should be handled by a check for an instruction being an atomic
      // The question is:  does this 'if' enforce the 'continue' only if it's an atomic or any time there's an instruction with a callback function (and the isatommem is just appended because only atomics have callbacks.  In other words:  can I just add 'continue' to 'if' statement above?
      if ( mem_insn[i].callback.function != NULL ) {
	//         isatommem[i] = 1;
         continue;
      }

      // Identify load/store instruction 
      if (( is_load(mem_insn[i].op) || is_store(mem_insn[i].op) )) {
         // check L1 cache for data 
         iswrite[i] = is_store(mem_insn[i].op);
         gpgpu_n_load_insn += (iswrite[i])? 0:1;
         gpgpu_n_store_insn += (iswrite[i])? 1:0;


         if (gpgpu_cuda_sim && is_shared(mem_insn[i].space)) {
            cachehit[i] = 1;
            isshmem[i] = 1;
            gpgpu_n_shmem_insn++;
         } else {
            if ( is_load(mem_insn[i].op)) is_mem_access = 1;
            iscache[i] = 1;
         }
      }
   }

   // check for cache bank conflict (even if there is no L1D $, this models stalls for coalescing)
   if (gpgpu_cache_bkconflict) {
      static unsigned char *cache_bkacc = NULL;
      if (!cache_bkacc) cache_bkacc = (unsigned char*)malloc(gpgpu_n_cache_bank * sizeof(unsigned char));
      memset(cache_bkacc, 0, gpgpu_n_cache_bank * sizeof(unsigned char));
      static address_type *cache_addr = NULL;
      if (!cache_addr) cache_addr = (address_type*)malloc(gpgpu_n_cache_bank * pipe_simd_width * sizeof(address_type));
      memset(cache_addr, 0, gpgpu_n_cache_bank * pipe_simd_width * sizeof(address_type));

      if (shader->pending_cache_bkacc == 0) { // 0 = check conflict for new insn
         if (shader->mem_stage_done_uncoalesced_stall == 0) {
            address_type gmem_insn_pc = -1;
            int gmem_laneid = -1;

            for (i=0; i<pipe_simd_width; i++) {
               if (iscache[i]) {
                  int cc_bank = cache_bank( mem_insn[i].memreqaddr, shader );
                  int b;
                  for (b=0; b < cache_bkacc[cc_bank]; b++) { // search for coexist access to same address
                     if ( SHD_CACHE_TAG(cache_addr[b],shader) 
                          == SHD_CACHE_TAG(mem_insn[i].memreqaddr,shader) )
                        break;
                  }
                  if ( b >= cache_bkacc[cc_bank] ) { // new address to the same bank
                     cache_addr[cache_bkacc[cc_bank]] = SHD_CACHE_TAG(mem_insn[i].memreqaddr,shader);
                     cache_bkacc[cc_bank]++;
                     if ( cache_bkacc[cc_bank] > shader->pending_cache_bkacc )
                        shader->pending_cache_bkacc = cache_bkacc[cc_bank];
                  }
                  gmem_insn_pc = mem_insn[i].pc;
                  gmem_laneid = i;
               }
            }

            if (gmem_insn_pc != -1 && (shader->pending_cache_bkacc > gpgpu_cache_port_per_bank)) {
               assert(gmem_laneid != -1);
               ptx_file_line_stats_add_uncoalesced_gmem(mem_insn[gmem_laneid].ptx_thd_info, gmem_insn_pc, shader->pending_cache_bkacc);
            }
            shader->mem_stage_done_uncoalesced_stall = 1; // indicate that this warp instruction should not stall for coalescing again
         }
      } else if (shader->pending_cache_bkacc <= gpgpu_cache_port_per_bank) { // last acc to shmem done, pass insn to next stage
         shader->pending_cache_bkacc = 0; // do check for conflict next cycle
      }
      if (shader->pending_cache_bkacc > gpgpu_cache_port_per_bank) {
         gpgpu_n_cache_bkconflict++;
         rc_fail = 1;
         bk_conflict = 1;
      }
   }

   // check for shared memory bank conflict
   if (gpgpu_shmem_bkconflict) {
      int mem_pipe_simd_width = pipe_simd_width / gpgpu_shmem_pipe_speedup;

      static unsigned char *shmem_bkacc = NULL;
      if (!shmem_bkacc) shmem_bkacc = (unsigned char*)malloc(gpgpu_n_shmem_bank * sizeof(unsigned char));
      static address_type *shmem_addr = NULL;
      if (!shmem_addr) shmem_addr = (address_type*)malloc(gpgpu_n_shmem_bank * pipe_simd_width * sizeof(address_type));

      if (shader->pending_shmem_bkacc == 0) { // 0 = check conflict for new insn
         address_type smem_insn_pc = -1;
         int smem_laneid = -1;
         int m;
         
         for (m = 0; m < gpgpu_shmem_pipe_speedup; m++) {
            unsigned char n_pending_shmem_bkacc = 0;
            memset(shmem_bkacc, 0, gpgpu_n_shmem_bank * sizeof(unsigned char));
            memset(shmem_addr, 0, gpgpu_n_shmem_bank * pipe_simd_width * sizeof(address_type));

            for (i = mem_pipe_simd_width * m; i < (mem_pipe_simd_width * (m + 1)); i++) {
               if (isshmem[i]) {
                  int sm_bank = shmem_bank(mem_insn[i].memreqaddr);
                  int b;
                  for (b=0; b < shmem_bkacc[sm_bank]; b++) { // search for coexist access to same address in the same bank
                     if (shmem_addr[sm_bank * gpgpu_n_shmem_bank + b] == mem_insn[i].memreqaddr)
                        break;
                  }
                  if ( b >= shmem_bkacc[sm_bank] ) { // new address to the same bank
                     shmem_addr[sm_bank * gpgpu_n_shmem_bank + shmem_bkacc[sm_bank]] = mem_insn[i].memreqaddr;
                     shmem_bkacc[sm_bank]++;
                     if ( shmem_bkacc[sm_bank] > n_pending_shmem_bkacc ) //find max pending access to any bank 
                        n_pending_shmem_bkacc = shmem_bkacc[sm_bank];
                  }
                  smem_insn_pc = mem_insn[i].pc;
                  smem_laneid = i;
               }
            }
            shader->pending_shmem_bkacc += n_pending_shmem_bkacc;
         }

         if (smem_insn_pc != -1) {
            assert(smem_laneid != -1);
            ptx_file_line_stats_add_smem_bank_conflict(mem_insn[smem_laneid].ptx_thd_info, smem_insn_pc, shader->pending_shmem_bkacc);
         }
      }
      if (shader->pending_shmem_bkacc <= gpgpu_shmem_port_per_bank) { // last acc to shmem done, pass insn to next stage
         shader->pending_shmem_bkacc = 0; // do check for conflict next cycle
      }
      if (shader->pending_shmem_bkacc > gpgpu_shmem_port_per_bank) {
         gpgpu_n_shmem_bkconflict++;
         rc_fail = 1;
         bk_conflict = 1;
      }
   }

   int cache_hits_waiting = 0;
   // access cache tag if no bank conflict AND if there is a dl1 cache
   if (!bk_conflict && !gpgpu_no_dl1) {
      for (i=0; i<pipe_simd_width; i++) {

         if (iscache[i]) {
            shd_cache_line_t *hit_cacheline = shd_cache_access(shader->L1cache,
                                                               mem_insn[i].memreqaddr,
                                                               WORD_SIZE, iswrite[i],
                                                               shader->gpu_cycle);
            cachehit[i] = (hit_cacheline != 0);
            if (cachehit[i]) {
               cache_hits_waiting++;
            }
            if (gpgpu_perfect_mem) cachehit[i] = 1;

         }
      }
   }

   // try to allocate resource for interconnect and mshr
   int require_MSHR;
   // record each unique fetch
   int num_unq_fetch;
   static unsigned long long int *unq_memaddr = NULL;
   static int *unq_bsize = NULL;
   static int *unq_type = NULL;  // READ or WRITE
   static int *unq_merge_count = NULL; //number of mem_insn with the same unq addr
   if (!unq_memaddr) unq_memaddr = (unsigned long long int *)malloc(pipe_simd_width * sizeof(unsigned long long int));
   if (!unq_bsize) unq_bsize = (int *)malloc(pipe_simd_width * sizeof(int));
   if (!unq_type) unq_type = (int *)malloc(pipe_simd_width * sizeof(int));
   if (!unq_merge_count) unq_merge_count = (int *)malloc(pipe_simd_width * sizeof(int));
   memset(unq_memaddr, 0, pipe_simd_width * sizeof(unsigned long long int));
   memset(unq_bsize, 0, pipe_simd_width * sizeof(int));
   memset(unq_type, 0, pipe_simd_width * sizeof(int));
   memset(unq_merge_count, 0, pipe_simd_width * sizeof(int));
   static mshr_entry **fetching = NULL;
   if (!fetching) fetching = (mshr_entry **)malloc(pipe_simd_width * sizeof(mshr_entry *));
   memset(fetching, 0, pipe_simd_width * sizeof(mshr_entry *));

   num_unq_fetch = 0;
   require_MSHR = 0;
   if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC ) {
      memset(MCB_accesses, 0, gpu_n_mem*4*sizeof(int));
   }
   for (i=0; i<pipe_simd_width; i++) {

      if (mem_insn[i].hw_thread_id == -1 )
         continue; // bubble 

      if (isshmem[i])
         continue; // no resource check for shared memory access

      if (shader->model == MIMD) require_MSHR = 0; // each thread would only use its private MSHR in MIMD

      // Specify what resource is required 
      // Normal LD/ST: Fetch Entry(if not yet issued) + MSHR
      if ( (is_load(mem_insn[i].op) || (!(gpgpu_cache_wt_through || gpgpu_no_dl1) && is_store(mem_insn[i].op)) ) //if cache miss and need to bring data into cache
           && (!cachehit[i] || gpgpu_no_dl1)) {
         if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
            addrdec_t tlx;
            addrdec_tlx(SHD_CACHE_TAG(mem_insn[i].memreqaddr, shader), &tlx);
            totalbankaccesses[tlx.chip][tlx.bk]++; //bankaccesses[shader id][dram chip id][bank id]
            MCB_accesses[tlx.chip*4 + tlx.bk]=1;
         }

         // Checks if there is an identical fetch request 
         // from a previous missed load 
         if (gpgpu_interwarp_mshr_merge & GLOBAL_MSHR_MERGE) {
            fetching[i] = shader_check_mshr4tag(shader, mem_insn[i].memreqaddr,DCACHE);
         } else {
            fetching[i] = NULL;
         }
         if (fetching[i] == NULL) {
            // check for redundent fetch in the same warp
            int w, fetching_in_same_warp;
            fetching_in_same_warp = 0;
            for (w=0;w<i;w++) {
               if (SHD_CACHE_TAG(mem_insn[i].memreqaddr,shader) == SHD_CACHE_TAG(mem_insn[w].memreqaddr,shader)) {
                  fetching_in_same_warp = 1;
               }
            }
            if (!fetching_in_same_warp) {
               unq_memaddr[num_unq_fetch] = SHD_CACHE_TAG(mem_insn[i].memreqaddr,shader);
               unq_bsize[num_unq_fetch] = READ_PACKET_SIZE; //write back cache never directly write to memory
               unq_type[num_unq_fetch] = 1; //read
               num_unq_fetch += 1;
            }
         }
         require_MSHR++;

      } else if ( gpgpu_no_dl1 && !isshmem[i] && is_store(mem_insn[i].op) ) {
         if (gpgpu_memlatency_stat) {
            addrdec_t tlx;
            addrdec_tlx(SHD_CACHE_TAG(mem_insn[i].memreqaddr, shader), &tlx);
            totalbankaccesses[tlx.chip][tlx.bk]++; //bankaccesses[shader id][dram chip id][bank id]  WF: should this really be done here?
         }

         // coalescing memory writes
         int w, fetching_in_same_warp;
         fetching_in_same_warp = 0;
         for (w=0;w<num_unq_fetch;w++) {
            if (SHD_CACHE_TAG(mem_insn[i].memreqaddr,shader) == SHD_CACHE_TAG(unq_memaddr[w],shader)) {
               fetching_in_same_warp = 1;
            }
         }
         if (!fetching_in_same_warp) {
            unq_memaddr[num_unq_fetch] = SHD_CACHE_TAG(mem_insn[i].memreqaddr,shader);
            unq_bsize[num_unq_fetch] = shader->L1cache->line_sz; //write the whole line...
            if (gpgpu_partial_write_mask) {
               unq_bsize[num_unq_fetch] += WRITE_PACKET_SIZE + WRITE_MASK_SIZE;
            }
            unq_type[num_unq_fetch] = 2; //write
            num_unq_fetch += 1;
         }

      } else if ( gpgpu_cache_wt_through && !isshmem[i] && is_store(mem_insn[i].op) ) {
         if (gpgpu_memlatency_stat) {
            addrdec_t tlx;
            addrdec_tlx(SHD_CACHE_TAG(mem_insn[i].memreqaddr, shader), &tlx);
            totalbankaccesses[tlx.chip][tlx.bk]++; //bankaccesses[shader id][dram chip id][bank id]
         }

         unq_memaddr[num_unq_fetch] = mem_insn[i].memreqaddr;
         unq_bsize[num_unq_fetch] = READ_PACKET_SIZE + WORD_SIZE; //write through cache packet
         unq_type[num_unq_fetch] = 2; //write
         num_unq_fetch += 1;

      }

      if ( require_MSHR && dq_full(shader->mshr[mem_insn[i].hw_thread_id]) ) {
         // can't allocate all the resources - stall and retry next cycle
	 /*TEST
	 printf("%llu:  Stalled 23\n", gpu_sim_cycle);
	 //TEST*/
         rc_fail = 1;
      }

   }   //end for
   if (num_unq_fetch > 1) gpgpu_multi_unq_fetches++;
   for ( i=0;i<num_unq_fetch;i++ ) {
      int w;
      for (w=0;w<pipe_simd_width;w++) {
         if (SHD_CACHE_TAG(unq_memaddr[i],shader) == SHD_CACHE_TAG(mem_insn[w].memreqaddr,shader)) {
            unq_merge_count[i]++;
            gpgpu_n_intrawarp_mshr_merge++;
         }
      }
   }

   if (is_mem_access && (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC )) {
      int MCBs_accessed = 0;
      for (i=0;i < gpu_n_mem*4 ; i++) {
         MCBs_accessed += MCB_accesses[i];
      }
      num_MCBs_accessed[MCBs_accessed]++;
   }
   // Resource needs to be available for all the fetch requests (more than one) 
   if (num_unq_fetch
       && !(shader->fq_has_buffer(unq_memaddr, unq_bsize, num_unq_fetch, shader->sid)) ) {
      rc_fail = 1;
      gpu_stall_sh2icnt++;
   }

   // tracking down the mshrs allocated by this warp (for coalescing)
   static mshr_entry **same_warp_mshr = NULL;
   if (!same_warp_mshr) same_warp_mshr = (mshr_entry **)malloc(pipe_simd_width * sizeof(mshr_entry*));
   memset(same_warp_mshr, 0, pipe_simd_width * sizeof(mshr_entry*));
   if (rc_fail) {
      // can't allocate all the resources - stall and retry next cycle
      gpu_stall_shd_mem++;
      /*TEST
      printf("%llu Stalling shd mem 24\n", gpu_sim_cycle);
      //TEST*/
      for (i=0; i<pipe_simd_width; i++) {
         if ( is_load(mem_insn[i].op) || is_store(mem_insn[i].op) ) {
            // correcting statistic for resource stall
            if ( isshmem[i] ) {
               gpgpu_n_shmem_insn--;
            } else {
               if (!bk_conflict) {
                  shd_cache_undo_stats( shader->L1cache, !cachehit[i] );
               }
            }
            gpgpu_n_load_insn -= (iswrite[i])? 0:1;
            gpgpu_n_store_insn -= (iswrite[i])? 1:0;
         }
      }
   } else {
      // create the memory fetch request and mshr entry for cache miss
      // and pass instruction from EX_MM to MM_WB for cache hit
      for (i=0; i<pipe_simd_width; i++) {

         if (mem_insn[i].hw_thread_id == -1 )
            continue; // bubble 

         // Identify load/store instruction 
         if (isshmem[i]) {
            // do nothing!
         } else if ( is_load(mem_insn[i].op) 
                     || (!(gpgpu_cache_wt_through || gpgpu_no_dl1) && is_store(mem_insn[i].op)) ) {

            // note that an atomic memory operation is always treated as a miss
	   if (!gpgpu_perfect_mem && (!cachehit[i] || gpgpu_no_dl1 || isatommem[i])) { //cache miss
               L1_read_miss++;
               shader->thread[mem_insn[i].hw_thread_id].n_l1_mis_ac++;
               shader->pending_mem_access++;

               // Push request to MSHR and Fetching queue:
               // The request is always pushed to MSHR as a load always need to be 
               // reactivated when its data arrives. 
               mshr_entry *mshr_e = alloc_mshr_entry();
               mshr_e->inst_uid = mem_insn[i].uid;
               same_warp_mshr[i] = mshr_e; // attaching the mshr entry to this warp
               mshr_e->addr = mem_insn[i].memreqaddr;
	       //SEAN
	       mshr_e->is_atom = mem_insn[i].isatom;
               mshr_e->isvector = mem_insn[i].is_vectorout; //cuda supports vector loads
               mshr_e->reg = mem_insn[i].out[0];
               if (mem_insn[i].is_vectorout) {
                  mshr_e->reg2 = mem_insn[i].out[1];
                  mshr_e->reg3 = mem_insn[i].out[2];
                  mshr_e->reg4 = mem_insn[i].out[3];
               }
               mshr_e->hw_thread_id = mem_insn[i].hw_thread_id;
               mshr_e->priority = shader->mshr_up_counter;
               mshr_e->inst = mem_insn[i];
               mshr_e->iswrite = iswrite[i];
               mshr_e->istexture = 0;
               mshr_e->isconst = 0;
               mshr_e->islocal = is_local(mem_insn[i].space);
               inflight_memory_insn_add(shader, &mem_insn[i]);

               // need to check again for inter-warp fetch merge
               if ((gpgpu_interwarp_mshr_merge & GLOBAL_MSHR_MERGE) && !isatommem[i]) {
                  fetching[i] = shader_check_mshr4tag(shader, mem_insn[i].memreqaddr,DCACHE);
               } else {
                  fetching[i] = NULL;
                  if (fetching[i] == NULL && !isatommem[i]) {
                     // check for redundent fetch in the same warp
                     int w, fetching_in_same_warp;
                     fetching_in_same_warp = 0;
                     for (w=0;w<i;w++) {
                        if (SHD_CACHE_TAG(mem_insn[i].memreqaddr,shader) == SHD_CACHE_TAG(mem_insn[w].memreqaddr,shader)) {
                           fetching_in_same_warp = 1;
                           fetching[i] = same_warp_mshr[w];
                           break;
                        }
                     }
                  }
               }
               if ( fetching[i] != NULL ) {
                  // merge with pending request(s) if merge limit not reached
                  mergemiss++;
                  shader->thread[mem_insn[i].hw_thread_id].n_l1_mrghit_ac++;
                  shd_cache_mergehit(shader->L1cache, mem_insn[i].memreqaddr);
                  mshr_entry *others = fetching[i]->merged_requests;
                  mshr_e->merged_requests = others;
                  fetching[i]->merged_requests = mshr_e;
                  mshr_e->status = fetching[i]->status;
                  mshr_e->fetched = fetching[i]->fetched;
                  mshr_e->merged = 1;
                  if (mshr_e->fetched) {
                     dq_push(shader->return_queue,mshr_e);
                     if (shader->return_queue->length > max_return_queue_length[shader->sid]) {
                        max_return_queue_length[shader->sid] = shader->return_queue->length;
                     }
                  }
               }

               // Try pushing the load into the MSHR. 
               // It should always success, as resource check did not fail
               if (!dq_push(shader->mshr[mem_insn[i].hw_thread_id], mshr_e)) assert(0);

               // Pushing to MSHR is successful. Issue the memory fetching 
               // for this cache line only if it is not already done 
               // by a previous load instruction. 
               if ( fetching[i] == NULL ) {
                  enum mem_access_type mem_acc = is_local(mem_insn[i].space)? LOCAL_ACC_R : GLOBAL_ACC_R;
                  shader->fq_push( SHD_CACHE_TAG(mshr_e->addr,shader),
                                   shader->L1cache->line_sz,
                                   (iswrite[i] ? 1:0), NO_PARTIAL_WRITE, shader->sid, mem_insn[i].hw_thread_id, mshr_e, 
                                   cache_hits_waiting, mem_acc, mem_insn[i].pc);

		  //TEST
		  if(iswrite[i]) {
		    printf("SEAN:  Pushed write response up\n");
		  }
		  //TEST*/
		  /*TEST
		  if(!mshr_e->iswrite) {
		    printf("SEAN: %llu data read request pushed from core (1).  Time %llu (%u)\n", mshr_e->addr, gpu_sim_cycle, shader->gpu_cycle);
		  }
		  else {
		    printf("SEAN: %llu data write request pushed from core (1).  Time %llu (%u)\n", mshr_e->addr, gpu_sim_cycle, shader->gpu_cycle);
		  }
		  //TEST*/
                  shader->n_mshr_used++;

                  shader->mshr_up_counter++;
                  if (shader->n_mshr_used > shader->max_n_mshr_used) shader->max_n_mshr_used = shader->n_mshr_used;
               }

               shader->pipeline_reg[i][EX_MM] = nop_inst;               

            }
         } else if ( gpgpu_no_dl1 && !gpgpu_perfect_mem &&
                     !isshmem[i] && is_store(mem_insn[i].op) ) {
            if (num_unq_fetch > 0) {
               // coalesced memory write command
               num_unq_fetch -= 1;

               // generate partial write mask and record one of the threads that is writing
               int w;
               int writer_tid = -1;
               unsigned char write_is_local = 0;
               unsigned long long int partial_write_mask = 0;
               for (w=0;w<warp_size;w++) {
                  if (SHD_CACHE_TAG(mem_insn[w].memreqaddr,shader) == unq_memaddr[num_unq_fetch]) {
                     int data_offset = mem_insn[w].memreqaddr & ((unsigned long long int)shader->L1cache->line_sz - 1);
                     unsigned long long int mask = (1 << mem_insn[w].data_size) - 1;
                     partial_write_mask |= (mask << data_offset);

                     write_is_local = is_local(mem_insn[w].space);

                     writer_tid = w;
                  }
               }
               assert(writer_tid != -1);

               if (partial_write_mask != 0xFFFFFFFF && partial_write_mask != 0xFFFFFFFFFFFFFFFFULL) {
                  gpgpu_n_partial_writes++;
               }

               enum mem_access_type mem_acc = (write_is_local)? LOCAL_ACC_W : GLOBAL_ACC_W;

               if (write_is_local) {
                  gpgpu_n_mem_write_local++;
               } else {
                  gpgpu_n_mem_write_global++;
               }

               // send out the coalesced memory write command with the write mask
               shader->fq_push( unq_memaddr[num_unq_fetch],
                                unq_bsize[num_unq_fetch], 1, partial_write_mask, 
                                shader->sid, mem_insn[writer_tid].hw_thread_id, NULL, 0,
                                mem_acc, mem_insn[writer_tid].pc);
	       /*TEST
	       //if(mf->type == RD_REQ) {
		 printf("SEAN: %llu data request pushed from core (2).  Time %llu (%u)\n", unq_memaddr[num_unq_fetch], gpu_sim_cycle, shader->gpu_cycle);
		 }
	       else {
		 printf("SEAN: %llu data write request pushed from core (2).  Time %llu (%u)\n", unq_memaddr[num_unq_fetch], gpu_sim_cycle, shader->gpu_cycle);
	       }
	       //TEST*/
            }
         } else if ( (gpgpu_cache_wt_through||gpgpu_no_dl1) && !gpgpu_perfect_mem &&
                     !isshmem[i] && is_store(mem_insn[i].op) ) {
            // write through memory command
            enum mem_access_type mem_acc = (mem_insn[i].space)? LOCAL_ACC_W : GLOBAL_ACC_W;
            shader->fq_push( mem_insn[i].memreqaddr,
                             READ_PACKET_SIZE + WORD_SIZE, 1, NO_PARTIAL_WRITE, 
                             shader->sid, mem_insn[i].hw_thread_id, NULL, 0,
                             mem_acc, mem_insn[i].pc);
	    /*TEST
	    //	    if(mf->type == RD_REQ) {
	      printf("SEAN: %llu data request pushed from core (3).  Time %llu (%u)\n", mem_insn[i].memreqaddr, gpu_sim_cycle, shader->gpu_cycle);
	    }
	    else {
	      printf("SEAN: %llu data write request pushed from core (3).  Time %llu (%u)\n", mem_insn[i].memreqaddr, gpu_sim_cycle, shader->gpu_cycle);
	    }
	    //TEST*/
         }
      }
   }

   if (!rc_fail) {
      check_stage_pcs(shader,EX_MM);
      // and pass instruction from EX_MM to MM_WB for cache hit
      for (i=0; i<pipe_simd_width; i++) {
         if (mem_insn[i].hw_thread_id == -1 )
            continue; // bubble 
         shader->pipeline_reg[i][MM_WB] = shader->pipeline_reg[i][EX_MM];
         shader->pipeline_reg[i][MM_WB].mm_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
         shader->pipeline_reg[i][EX_MM] = nop_inst; 
	 //SEAN
	 if(g_pipetrace) {
	   if(shader->pipeline_reg[i][MM_WB].uid != nop_inst.uid) {
	     pipe_stat *curr=pipe_stat_last;
	     while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][MM_WB].uid)) curr = curr->prev;    
	     assert(curr->uid == shader->pipeline_reg[i][MM_WB].uid);
	     curr->in_mem = gpu_sim_cycle;
	   }
	 }
      }    

      // turn on coalescing checker and its stalling mechanism again
      shader->mem_stage_done_uncoalesced_stall = 0;
   }

   // reflect the change to EX|MM pipeline register to the pre_mem stage
   if (gpgpu_pre_mem_stages) {
      check_stage_pcs(shader,EX_MM);
      for (i=0;i<pipe_simd_width ;i++ )
         shader->pre_mem_pipeline[i][gpgpu_pre_mem_stages] = shader->pipeline_reg[i][EX_MM];
   }
} //shader_memory



int writeback_l1_miss =0 ;

void register_cta_thread_exit(shader_core_ctx_t *shader, int tid )
{
   if (gpgpu_cuda_sim && gpgpu_spread_blocks_across_cores) {
      unsigned padded_cta_size = ptx_sim_cta_size();
      if (padded_cta_size%warp_size) {
         padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);
      }
      int cta_num = tid/padded_cta_size;
      assert( shader->cta_status[cta_num] > 0 );
      shader->cta_status[cta_num]--;
      if (!shader->cta_status[cta_num]) {
         shader->n_active_cta--;
         shader_CTA_count_unlog(shader->sid, 1);
         printf("Shader %d finished CTA #%d (%lld,%lld)\n", shader->sid, cta_num, gpu_sim_cycle, gpu_tot_sim_cycle );
      }
   }
}

typedef struct {
   unsigned pc;
   unsigned long latency;
   void *ptx_thd_info;
} insn_latency_info;

void obtain_insn_latency_info(insn_latency_info *latinfo, inst_t *insn)
{
   latinfo->pc = insn->pc;
   latinfo->latency = gpu_tot_sim_cycle + gpu_sim_cycle - insn->ts_cycle;
   latinfo->ptx_thd_info = insn->ptx_thd_info;
}

void shader_writeback( shader_core_ctx_t *shader, unsigned int shader_number, int grid_num ) 
{
   static int *unlock_tid = NULL;
   static int *freed_warp = NULL;
   static int *mshr_tid = NULL;
   static int *pl_tid = NULL;
   static mshr_entry **mshr_head = NULL;
   static insn_latency_info *unlock_lat_info = NULL;
   static insn_latency_info *mshr_lat_info = NULL;
   static insn_latency_info *pl_lat_info = NULL;

   int tid;
   op_type op;
   int o1, o2, o3, o4;
   unsigned char stalled_by_MSHR = 0;
   unsigned char writeback_by_MSHR = 0;
   unsigned char mshr_fetched = 0;
   unsigned char w2rf = 0;
   int i;
   mshr_entry *mshr_returnhead = NULL;

   if ( unlock_tid == NULL ) {
      unlock_tid = (int*) malloc(sizeof(int)*pipe_simd_width);
      mshr_tid = (int*) malloc(sizeof(int)*pipe_simd_width);
      pl_tid = (int*) malloc(sizeof(int)*pipe_simd_width);
      mshr_head = (mshr_entry**) calloc(sizeof(mshr_entry*),pipe_simd_width);
      freed_warp = (int *) malloc(sizeof(int)*pipe_simd_width);
      mshr_lat_info = (insn_latency_info*) malloc(sizeof(insn_latency_info) * pipe_simd_width);
      pl_lat_info = (insn_latency_info*) malloc(sizeof(insn_latency_info) * pipe_simd_width);
   }
   for (i=0; i<pipe_simd_width; i++) {
      unlock_tid[i] = -1;
      mshr_tid[i] = -1;
      pl_tid[i] = -1;
      mshr_head[i] = NULL;
      freed_warp[i] = 0;
   }
   unlock_lat_info = NULL;

   check_stage_pcs(shader,MM_WB); //SEAN:  checks warp consistency (defaulted to turned off, however => nothing actually happens with this call).

   /* Generate Condition for instruction writeback to register file.
      A load miss *instruction* does not reach writeback until the data is fetched */
   for (i=0; i<pipe_simd_width; i++) {
      tid = shader->pipeline_reg[i][MM_WB].hw_thread_id;
      w2rf |= (tid >= 0); //SEAN:  will be '1' as long as tid != -1
      pl_tid[i] = tid;
   }
   int mshr_warpid = -1;
   for (i=0; i<pipe_simd_width; i++) {
      mshr_returnhead = getMSHR_returnhead(shader);
      if ((shader->model == POST_DOMINATOR || shader->model == NO_RECONVERGE) && gpgpu_strict_simd_wrbk) { //SEAN:  gpgpu_strict_simd_wrbk defaults to '0'
         if (mshr_returnhead) {
            if (mshr_warpid == -1) {
               mshr_warpid = mshr_returnhead->hw_thread_id / warp_size;
            } else {
               // restricting the threads from mshr to be in the same warp
               assert(mshr_returnhead->fetched);
               if (mshr_warpid != (mshr_returnhead->hw_thread_id / warp_size)) {
                  warp_conflict_at_writeback++;
                  break;
               }
            }
            mshr_returnhead = NULL;
         }
      }

      // Generate Condition for MSHR has fetched data 
      mshr_head[i] = fetchMSHR(shader->mshr, shader);
      mshr_fetched = 0;
      if (mshr_head[i]) {
         mshr_fetched = mshr_head[i]->fetched;
         mshr_tid[i] = mshr_head[i]->hw_thread_id;
         obtain_insn_latency_info(&mshr_lat_info[i], &(mshr_head[i]->inst));
         inflight_memory_insn_sub(shader, &mshr_head[i]->inst);
      }

      /* arbitrate between two source of commit: instr or mshr */
      /* Note: now the priority is given to the instr, but
        if this is not the case, we should stall at WB */
      if ( w2rf && !mshr_fetched ) { //SEAN:  inflight insn wants to write to RF & no mshr return also wants to write
         // instruction needs to be written to destination register 
         // and there is nothing from the MSHR, proceed with the writeback
      } else if ( !w2rf && mshr_fetched ) { //SEAN:  no inflight insn trying to write to RF, but there is return data from the MSHR
         // all nop in this stage => no need to unlock and writeback

         removeEntry(mshr_head[i], shader->mshr, shader->n_threads);
         writeback_by_MSHR = 1;
	 /*TEST
	 printf("SEAN (%llu):  load data returned for tid %i and written to RF\n", gpu_sim_cycle, mshr_tid[i]);
	 //TEST*/
	 //SEAN
	 if(g_pipetrace) {
	   for (i=0; i<pipe_simd_width; i++) {
	     if((mshr_head[i]) && (mshr_head[i]->inst.uid != nop_inst.uid)) {
	       pipe_stat *curr=pipe_stat_last;
	       while((curr != NULL) && (curr->uid != mshr_head[i]->inst.uid)) curr = curr->prev;    
	       assert(curr->uid == mshr_head[i]->inst.uid);
	       curr->in_writeback = gpu_sim_cycle;
	     }
	   }
	 }

      } else if ( w2rf && mshr_fetched ) { //SEAN:  inflight insn *&* MSHR return both want to write to RF => stall inflight insn?
         // stall the pipeline if a load from MSHR is ready to commit
         assert (mshr_head[i]->hw_thread_id >= 0);

         removeEntry(mshr_head[i], shader->mshr, shader->n_threads);
         writeback_by_MSHR = 1;
         stalled_by_MSHR = 1;
	 /*TEST
	 printf("%llu:  Stalled by MSHR 24\n", gpu_sim_cycle);
	 //TEST*/
	 //SEAN
	 if(g_pipetrace) {
	   for (i=0; i<pipe_simd_width; i++) {
	     if((mshr_head[i]) && (mshr_head[i]->inst.uid != nop_inst.uid)) {
	       pipe_stat *curr=pipe_stat_last;
	       while((curr != NULL) && (curr->uid != mshr_head[i]->inst.uid)) curr = curr->prev;    
	       assert(curr->uid == mshr_head[i]->inst.uid);
	       curr->in_writeback = gpu_sim_cycle;
	     }
	   }
	 }

      }
   }
   if (stalled_by_MSHR) {
      gpu_stall_by_MSHRwb++;
   }
   shd_cache_line_t *hit_cacheline;
   if (writeback_by_MSHR) {
      for (i=0; i<pipe_simd_width; i++) {
         if (mshr_head[i]) {
            time_vector_update(mshr_head[i]->inst_uid,MR_WRITEBACK,gpu_sim_cycle+gpu_tot_sim_cycle,RD_REQ);
            assert (mshr_head[i]->hw_thread_id >= 0);
            // this is just to make the line dirty, shouldn't be counted towards the statistics
            if (!gpgpu_no_dl1) {
               if (mshr_head[i]->istexture) {
                  hit_cacheline = shd_cache_access(shader->L1texcache, 
                                                   mshr_head[i]->addr, WORD_SIZE, 
                                                   mshr_head[i]->iswrite, //should always be 0
                                                   shader->gpu_cycle); 
                  shd_cache_undo_stats( shader->L1texcache, !hit_cacheline );
                  shader->pending_mem_access--;
               } else if (mshr_head[i]->isconst) {
                  hit_cacheline = shd_cache_access(shader->L1constcache, 
                                                   mshr_head[i]->addr, WORD_SIZE, 
                                                   mshr_head[i]->iswrite, //should always be 0
                                                   shader->gpu_cycle); 
                  shd_cache_undo_stats( shader->L1constcache, !hit_cacheline );
                  shader->pending_mem_access--;
               } else {
                  hit_cacheline = shd_cache_access(shader->L1cache, 
                                                   mshr_head[i]->addr, WORD_SIZE, 
                                                   mshr_head[i]->iswrite, 
                                                   shader->gpu_cycle); 
                  shd_cache_undo_stats( shader->L1constcache, !hit_cacheline );
                  if (!hit_cacheline) {
                     writeback_l1_miss++;
                  }
                  shader->pending_mem_access--;
               }
            }
            shader->n_mshr_used--;
            free_mshr_entry(mshr_head[i]);
            mshr_head[i] = NULL;
            unlock_tid[i] = mshr_tid[i];
         }
      }
      unlock_lat_info = mshr_lat_info;
   } else { //!writeback_by_MSHR
      for (i=0; i<pipe_simd_width; i++) {
         op = shader->pipeline_reg[i][MM_WB].op;
         tid = shader->pipeline_reg[i][MM_WB].hw_thread_id;
         o1 = shader->pipeline_reg[i][MM_WB].out[0];
         o2 = shader->pipeline_reg[i][MM_WB].out[1];
         o3 = shader->pipeline_reg[i][MM_WB].out[2];
         o4 = shader->pipeline_reg[i][MM_WB].out[3];

         unlock_tid[i] = pl_tid[i];
         obtain_insn_latency_info(&pl_lat_info[i], &shader->pipeline_reg[i][MM_WB]);
      }
      unlock_lat_info = pl_lat_info;
   }
   int thd_unlocked = 0;
   for (i=0; i<pipe_simd_width; i++) {
      // NOTE: no need to check for next-stage stall at the last stage
      if (unlock_tid[i] >= 0 ) { // not unlocking an invalid thread (ie. due to a bubble)
         // thread completed if it is going to fetching beyond code boundry 
         if ( gpgpu_cuda_sim && ptx_thread_done(shader->thread[unlock_tid[i]].ptx_thd_info) ) {

            finished_trace += 1;
            shader->not_completed -= 1;
            gpu_completed_thread += 1;

            int warp_tid = unlock_tid[i]-(unlock_tid[i]%warp_size);
            if (!(shader->thread[warp_tid].n_completed < warp_size)) {
               printf("shader[%d]->thread[%d].n_completed = %d; warp_size = %d\n", 
                      shader->sid,warp_tid, shader->thread[warp_tid].n_completed, warp_size);
            }
            assert( shader->thread[warp_tid].n_completed < warp_size );
            shader->thread[warp_tid].n_completed++;
            if ( shader->model == NO_RECONVERGE ) {
               update_max_branch_priority(shader,warp_tid,grid_num);
            }

            if (gpgpu_no_divg_load) {
               int amask = wpt_signal_complete(unlock_tid[i], shader);
               freed_warp[i] = (amask != 0)? 1 : 0;
            } else {
               register_cta_thread_exit(shader, unlock_tid[i] );
            }
         } else { //thread is not finished yet
            // program is not finished yet, allow more fetch 
            if (gpgpu_no_divg_load) {
	      freed_warp[i] = wpt_signal_avail(unlock_tid[i], shader);
            } else {
               shader->thread[unlock_tid[i]].avail4fetch++;
               assert(shader->thread[unlock_tid[i]].avail4fetch <= 1);
               assert( shader->thread[unlock_tid[i] - (unlock_tid[i]%warp_size)].n_avail4fetch < warp_size );
               shader->thread[unlock_tid[i] - (unlock_tid[i]%warp_size)].n_avail4fetch++;
               thd_unlocked = 1;
	       /*TEST
	       printf("%llu SEAN:  n_avail4fetch incremented (and thd unlocked in shader_writeback) (now = %i)\n", gpu_sim_cycle, shader->thread[unlock_tid[i] - (unlock_tid[i]%warp_size)].n_avail4fetch);
	       //TEST*/
            }
         }

         // At any rate, a real instruction is committed
         // - don't count cache miss
         if ( shader->pipeline_reg[i][MM_WB].inst_type != NO_OP_FLAG ) {
            gpu_sim_insn++;
            if ( !is_const(shader->pipeline_reg[i][MM_WB].space) ) 
               gpu_sim_insn_no_ld_const++;
            gpu_sim_insn_last_update = gpu_sim_cycle;
            shader->num_sim_insn++;
            shader->thread[unlock_tid[i]].n_insn++;
            shader->thread[unlock_tid[i]].n_insn_ac++;
         }

         if (enable_ptx_file_line_stats) {
            unsigned pc = unlock_lat_info[i].pc;
            unsigned long latency = unlock_lat_info[i].latency;
            ptx_file_line_stats_add_latency(unlock_lat_info[i].ptx_thd_info, pc, latency);
         }
      }
   }
   if (shader->using_commit_queue && thd_unlocked) {
      int *tid_unlocked = alloc_commit_warp();
      memcpy(tid_unlocked, unlock_tid, sizeof(int)*pipe_simd_width); //NOTE: this may be warp_size
      dq_push(shader->thd_commit_queue,(void*)tid_unlocked);
      /*TEST
      printf("SEAN:  Pushed tid %i to thread commit queue\n", *tid_unlocked);
      //TEST*/
   }
   /*TEST
   else if (!thd_unlocked) {
     printf("SEAN:  thread not pushed to thd_commit_queue because thd_unlocked = %i\n", thd_unlocked);
   } else {
     printf("SEAN:  thread not pushed to thd_commit_queue because using_commit_queue = %i\n", shader->using_commit_queue);
   }
   //TEST*/
 
   //SEAN
   if(g_pipetrace) {
     for (i=0; i<pipe_simd_width; i++) {
       if(shader->pipeline_reg[i][MM_WB].uid != nop_inst.uid) {
	 pipe_stat *curr=pipe_stat_last;
	 while((curr != NULL) && (curr->uid != shader->pipeline_reg[i][MM_WB].uid)) curr = curr->prev;    
	 assert(curr->uid == shader->pipeline_reg[i][MM_WB].uid);
	 curr->in_writeback = gpu_sim_cycle;
       }
     }
   }

   /* The pipeline can be stalled by MSHR */
   if (!stalled_by_MSHR) {
      for (i=0; i<pipe_simd_width; i++) {
         shader->pipeline_reg[i][WB_RT] = shader->pipeline_reg[i][MM_WB];
         shader->pipeline_reg[i][MM_WB] = nop_inst;
      }
   }
} //shader_writeback

void shader_print_runtime_stat( FILE *fout ) {
   int i;

   fprintf(fout, "SHD_INSN: ");
   for (i=0;i<gpu_n_shader;i++) {
      fprintf(fout, "%u ",sc[i]->num_sim_insn);
   }
   fprintf(fout, "\n");
   fprintf(fout, "SHD_THDS: ");
   for (i=0;i<gpu_n_shader;i++) {
      fprintf(fout, "%u ",sc[i]->not_completed);
   }
   fprintf(fout, "\n");
   fprintf(fout, "SHD_DIVG: ");
   for (i=0;i<gpu_n_shader;i++) {
      fprintf(fout, "%u ",sc[i]->n_diverge);
   }
   fprintf(fout, "\n");

   fprintf(fout, "THD_INSN: ");
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_insn);
   }
   fprintf(fout, "\n");
}

void shader_print_l1_miss_stat( FILE *fout ) {
   int i;

   fprintf(fout, "THD_INSN_AC: ");
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_insn_ac);
   }
   fprintf(fout, "\n");

   fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_l1_mis_ac);
   }
   fprintf(fout, "\n");

   fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_l1_mis_ac - sc[0]->thread[i].n_l1_mrghit_ac);
   }
   fprintf(fout, "\n");

   fprintf(fout, "T_L1_Acc: "); //l1 access per thread
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_l1_access_ac);
   }
   fprintf(fout, "\n");

   //per warp
   int temp =0; 
   fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      temp += sc[0]->thread[i].n_l1_mis_ac;
      if (i%warp_size == (warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp=0;
   fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      temp += (sc[0]->thread[i].n_l1_mis_ac - sc[0]->thread[i].n_l1_mrghit_ac);
      if (i%warp_size == (warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp =0;
   fprintf(fout, "W_L1_Acc: "); //l1 access per warp
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      temp += sc[0]->thread[i].n_l1_access_ac;
      if (i%warp_size == (warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");

}

void shader_print_stage(shader_core_ctx_t *shader, unsigned int stage, 
                        FILE *fout, int stage_width, int print_mem, int mask ) 
{
   int i, j, warp_id = -1;

   for (i=0; i<stage_width; i++) {
      if (shader->pipeline_reg[i][stage].hw_thread_id > -1) {
         warp_id = shader->pipeline_reg[i][stage].hw_thread_id / warp_size;
         break;
      }
   }
   i = (i>=stage_width)? 0 : i;

   fprintf(fout,"0x%04x ", shader->pipeline_reg[i][stage].pc );

   if( mask & 2 ) {
      fprintf(fout, "(" );
      for (j=0; j<stage_width; j++)
         fprintf(fout, "%03d ", shader->pipeline_reg[j][stage].hw_thread_id);
      fprintf(fout, "): ");
   } else {
      fprintf(fout, "w%02d[", warp_id);
      for (j=0; j<stage_width; j++) 
         fprintf(fout, "%c", ((shader->pipeline_reg[j][stage].hw_thread_id != -1)?'1':'0') );
      fprintf(fout, "]: ");
   }

   if( warp_id != -1 && shader->model == POST_DOMINATOR ) {
      pdom_warp_ctx_t *warp=&(shader->pdom_warp[warp_id]);
      if( warp->m_recvg_pc[warp->m_stack_top] == -1 ) {
         fprintf(fout," rp:--- ");
      } else {
         fprintf(fout," rp:0x%03x ", warp->m_recvg_pc[warp->m_stack_top] );
      }
   }

   ptx_print_insn( shader->pipeline_reg[i][stage].pc, fout );

   if( mask & 0x10 ) {
      if ( (shader->pipeline_reg[i][stage].op == STORE_OP ||
            shader->pipeline_reg[i][stage].op == LOAD_OP) && print_mem )
         fprintf(fout, "  mem: 0x%016llx", shader->pipeline_reg[i][stage].memreqaddr);
   }
   fprintf(fout, "\n");
}

void shader_print_pre_mem_stages(shader_core_ctx_t *shader, FILE *fout, int print_mem, int mask ) 
{
   int i, j, pms;
   int warp_id;

   if (!gpgpu_pre_mem_stages) return;

   for (pms = 0; pms <= gpgpu_pre_mem_stages - 1; pms++) {
      fprintf(fout, "PM[%01d] = ", pms);

      warp_id = -1;

      for (i=0; i<pipe_simd_width; i++) {
         if (shader->pre_mem_pipeline[i][pms].hw_thread_id > -1) {
            warp_id = shader->pre_mem_pipeline[i][pms].hw_thread_id / warp_size;
            break;
         }
      }
      i = (i>=pipe_simd_width)? 0 : i;

      fprintf(fout,"0x%04x ", shader->pre_mem_pipeline[i][pms].pc );

      if( mask & 2 ) {
         fprintf(fout, "(" );
         for (j=0; j<pipe_simd_width; j++)
            fprintf(fout, "%03d ", shader->pre_mem_pipeline[j][pms].hw_thread_id);
         fprintf(fout, "): ");
      } else {
         fprintf(fout, "w%02d[", warp_id);
         for (j=0; j<pipe_simd_width; j++)
            fprintf(fout, "%c", ((shader->pre_mem_pipeline[j][pms].hw_thread_id != -1)?'1':'0') );
         fprintf(fout, "]: ");
      }

      if( warp_id != -1 && shader->model == POST_DOMINATOR ) {
         pdom_warp_ctx_t *warp=&(shader->pdom_warp[warp_id]);
         if( warp->m_recvg_pc[warp->m_stack_top] == -1 ) {
            printf(" rp:--- ");
         } else {
            printf(" rp:0x%03x ", warp->m_recvg_pc[warp->m_stack_top] );
         }
      }

      ptx_print_insn( shader->pre_mem_pipeline[i][pms].pc, fout );

      if( mask & 0x10 ) {
         if ( ( shader->pre_mem_pipeline[i][pms].op == LOAD_OP ||
                shader->pre_mem_pipeline[i][pms].op == STORE_OP ) && print_mem )
            fprintf(fout, "  mem: 0x%016llx", shader->pre_mem_pipeline[i][pms].memreqaddr);
      }
      fprintf(fout, "\n");
   }
}

extern const char * ptx_get_fname( unsigned PC );

void shader_display_pipeline(shader_core_ctx_t *shader, FILE *fout, int print_mem, int mask ) 
{
   // call this function from within gdb to print out status of pipeline
   // if you encounter a bug, or to visualize pipeline operation
   // (this is a good way to "verify" your pipeline model makes sense!)

   fprintf(fout, "=================================================\n");
   fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", shader->sid, 
           gpu_tot_sim_cycle, gpu_sim_cycle, shader->not_completed);
   fprintf(fout, "=================================================\n");

   if ( (mask & 4) && shader->model == POST_DOMINATOR ) {
      int i, n, m, j;
      fprintf(fout,"warp status:\n");
      n = shader->n_threads / warp_size;
      for ( i=0; i < n; i++) {
         unsigned nactive = 0;
         for (j=0; j<warp_size; j++ ) {
            unsigned tid = i*warp_size + j;
            int done = ptx_thread_done( shader->thread[tid].ptx_thd_info );
            nactive += (ptx_thread_done( shader->thread[tid].ptx_thd_info )?0:1);
            if ( done && (mask & 8) ) {
               unsigned done_cycle = ptx_thread_donecycle( shader->thread[tid].ptx_thd_info );
               if ( done_cycle ) {
                  printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle );
               }
            }
         }
         if ( nactive == 0 ) {
            continue;
         }
         pdom_warp_ctx_t *warp=&(shader->pdom_warp[i]);
         unsigned k;
         for ( k=0; k <= warp->m_stack_top; k++ ) {
            if ( k==0 ) {
               fprintf(fout, "w%02d (%2u thds active): %2u ", i, nactive, k );
            } else {
               fprintf(fout, "                      %2u ", k );
            }
            for (m=1,j=0; j<warp_size; j++, m<<=1)
               fprintf(fout, "%c", ((warp->m_active_mask[k] & m)?'1':'0') );
            fprintf(fout, " pc: %4u", warp->m_pc[k] );
            if ( warp->m_recvg_pc[k] == -1 ) {
               fprintf(fout," rp: ---- cd: %2u ", warp->m_calldepth[k] );
            } else {
               fprintf(fout," rp: %4u cd: %2u ", warp->m_recvg_pc[k], warp->m_calldepth[k] );
            }
            if ( warp->m_branch_div_cycle[k] != 0 ) {
               fprintf(fout," bd@%6u ", (unsigned) warp->m_branch_div_cycle[k] );
            } else {
               fprintf(fout,"           " );
            }
            //fprintf(fout," func=\'%s\' ", ptx_get_fname( warp->m_pc[k] ) );
            ptx_print_insn( warp->m_pc[k], fout );
            fprintf(fout,"\n");
         }
      }
      fprintf(fout,"\n");
   }

   if ( mask & 0x20 ) {
      fprintf(fout, "TS/IF = ");
      shader_print_stage(shader, TS_IF, fout, warp_size, print_mem, mask);
   }

   fprintf(fout, "IF/ID = ");
   shader_print_stage(shader, IF_ID, fout, pipe_simd_width, print_mem, mask );

   if (shader->using_rrstage) {
      fprintf(fout, "ID/RR = ");
      shader_print_stage(shader, ID_RR, fout, pipe_simd_width, print_mem, mask);
   }

   fprintf(fout, "ID/EX = ");
   shader_print_stage(shader, ID_EX, fout, pipe_simd_width, print_mem, mask);

   shader_print_pre_mem_stages(shader, fout, print_mem, mask);

   if (!gpgpu_pre_mem_stages)
      fprintf(fout, "EX/MEM= ");
   else
      fprintf(fout, "PM/MEM= ");
   shader_print_stage(shader, EX_MM, fout, pipe_simd_width, print_mem, mask);

   fprintf(fout, "MEM/WB= ");
   shader_print_stage(shader, MM_WB, fout, pipe_simd_width, print_mem, mask);

   fprintf(fout, "\n");
}

void shader_dump_thread_state(shader_core_ctx_t *shader, FILE *fout )
{
   int i, w=0;
   fprintf( fout, "\n");
   for ( w=0; w < gpu_n_thread_per_shader/warp_size; w++ ) {
      int tid = w*warp_size;
      if ( shader->thread[ tid ].n_completed < warp_size ) {
         fprintf( fout, "  %u:%3u fetch state = c:%u a4f:%u bw:%u (completed: ", shader->sid, tid, 
                  shader->thread[tid].n_completed,
                  shader->thread[tid].n_avail4fetch,
                  shader->thread[tid].n_waiting_at_barrier );

         for ( i = tid; i < (w+1)*warp_size; i++ ) {
            if ( gpgpu_cuda_sim && ptx_thread_done(shader->thread[i].ptx_thd_info) ) {
               fprintf(fout,"1");
            } else {
               fprintf(fout,"0");
            }
            if ( (((i+1)%4) == 0) && (i+1) < (w+1)*warp_size ) {
               fprintf(fout,",");
            }
         }
         fprintf(fout,")\n");
      }
   }
}

void shader_dp(shader_core_ctx_t *shader, int print_mem) {
   shader_display_pipeline(shader, stdout, print_mem, 7 );
}

unsigned int max_cta_per_shader( shader_core_ctx_t *shader)
{
   unsigned int result;
   unsigned int padded_cta_size;

   padded_cta_size = ptx_sim_cta_size();
   if (padded_cta_size%warp_size) {
      padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);
      //printf("padded_cta_size=%u\n", padded_cta_size);
   }

   //Limit by n_threads/shader
   unsigned int result_thread = shader->n_threads / padded_cta_size;

   const struct gpgpu_ptx_sim_kernel_info *kernel_info = ptx_sim_kernel_info();

   //Limit by shmem/shader
   unsigned int result_shmem = (unsigned)-1;
   if (kernel_info->smem > 0)
      result_shmem = shader->shmem_size / kernel_info->smem;

   //Limit by register count, rounded up to multiple of 4.
   unsigned int result_regs = (unsigned)-1;
   if (kernel_info->regs > 0)
      result_regs = shader->n_registers / (padded_cta_size * ((kernel_info->regs+3)&~3));

   //Limit by CTA
   unsigned int result_cta = shader->n_cta;

   result = result_thread;
   result = min(result, result_shmem);
   result = min(result, result_regs);
   result = min(result, result_cta);

   static const struct gpgpu_ptx_sim_kernel_info* last_kinfo = NULL;
   if (last_kinfo != kernel_info) {   //Only print out stats if kernel_info struct changes
      last_kinfo = kernel_info;
      printf ("CTA/core = %u, limited by:", result);
      if (result == result_thread) printf (" threads");
      if (result == result_shmem) printf (" shmem");
      if (result == result_regs) printf (" regs");
      if (result == result_cta) printf (" cta_limit");
      printf ("\n");
   }

   if (result < 1) {
      printf ("Error: max_cta_per_shader(\"%s\") returning %d. Kernel requires more resources than shader has?\n", shader->name, result);
      abort();
   }
   return result;
}

void shader_cycle( shader_core_ctx_t *shader, 
                   unsigned int shader_number,
                   int grid_num ) 
{
   // last pipeline stage
   shader_writeback(shader, shader_number, grid_num);

   // three parallel stages (only one does something on a given cycle)
   shader_const_memory   (shader, shader_number);
   shader_memory   (shader, shader_number);
   shader_texture_memory   (shader, shader_number);

   // empty stage
   if (gpgpu_pre_mem_stages)
      shader_pre_memory(shader, shader_number);

   shader_execute  (shader, shader_number);
   if (shader->using_rrstage) {
      // model register bank conflicts 
      // (see Fung et al. MICRO'07 paper or ACM TACO paper)
      shader_preexecute (shader, shader_number);
   }

   shader_decode   (shader, shader_number, grid_num);

   shader_fetch    (shader, shader_number, grid_num);
}

// performance counter that are not local to one shader
void shader_print_accstats( FILE* fout ) 
{
   fprintf(fout, "gpgpu_n_load_insn  = %d\n", gpgpu_n_load_insn);
   fprintf(fout, "gpgpu_n_store_insn = %d\n", gpgpu_n_store_insn);
   fprintf(fout, "gpgpu_n_shmem_insn = %d\n", gpgpu_n_shmem_insn);
   fprintf(fout, "gpgpu_n_tex_insn = %d\n", gpgpu_n_tex_insn);
   fprintf(fout, "gpgpu_n_const_mem_insn = %d\n", gpgpu_n_const_insn);
   fprintf(fout, "gpgpu_n_param_mem_insn = %d\n", gpgpu_n_param_insn);

   fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", gpgpu_n_shmem_bkconflict);
   fprintf(fout, "gpgpu_n_cache_bkconflict = %d\n", gpgpu_n_cache_bkconflict);   

   fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n", gpgpu_n_intrawarp_mshr_merge);
   fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", gpgpu_n_cmem_portconflict);

   fprintf(fout, "gpgpu_n_writeback_l1_miss = %d\n", writeback_l1_miss);

   fprintf(fout, "gpgpu_n_partial_writes = %d\n", gpgpu_n_partial_writes);

   if (warp_occ_detailed) {
      int s, w, t;
      int n_warp = gpu_n_thread_per_shader / warp_size;

      for (s = 0; s<gpu_n_shader; s++)
         for (w = 0; w<n_warp; w++) {
            fprintf(fout, "wod[%d][%d]=", s, w);
            for (t = 0; t<warp_size; t++) {
               fprintf(fout, "%d ", warp_occ_detailed[s * n_warp + w][t]);
            }
            fprintf(fout, "\n");
         }
   }
}

// Flushes all content of the cache to memory
void shader_cache_flush(shader_core_ctx_t* sc) 
{
   unsigned int i;
   unsigned int set;
   unsigned long long int flush_addr;

   shd_cache_t *cp = sc->L1cache;
   shd_cache_line_t *pline;

   for (i=0; i<cp->nset*cp->assoc; i++) {
      pline = &(cp->lines[i]);
      set = i / cp->assoc;
      if ((pline->status & (DIRTY|VALID)) == (DIRTY|VALID)) {
         flush_addr = pline->addr;

         sc->fq_push(flush_addr, sc->L1cache->line_sz, 1, NO_PARTIAL_WRITE, sc->sid, 0, NULL, 0, GLOBAL_ACC_W, -1);

         pline->status &= ~VALID;
         pline->status &= ~DIRTY;
      } else if (pline->status & VALID) {
         pline->status &= ~VALID;
      }
   }
}
