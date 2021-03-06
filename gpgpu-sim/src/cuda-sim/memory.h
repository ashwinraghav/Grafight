/* 
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung and the University of 
 * British Columbia, Vancouver, BC V6T 1Z4, All Rights Reserved.
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

#ifndef memory_h_INCLUDED
#define memory_h_INCLUDED

#ifdef __GNUC__
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
   #include <unordered_map>
   #define mem_map std::unordered_map
   #define MEM_MAP_RESIZE(hash_size) m_data.rehash(hash_size)
#else
   #include <ext/hash_map>
   namespace std {
      using namespace __gnu_cxx;
   }
   #define mem_map std::hash_map
   #define MEM_MAP_RESIZE(hash_size) m_data.resize(hash_size)
#endif
#else
   #include <map>
   #define mem_map std::map
   #define MEM_MAP_RESIZE(hash_size) 
#endif

#include <assert.h>
#include <string.h>
#include <string>
#include "../util.h"

typedef address_type mem_addr_t;

#define MEM_BLOCK_SIZE (4*1024)

template<unsigned BSIZE> class mem_storage {
public:
   mem_storage( const mem_storage &another )
   {
      m_data = new unsigned char[ BSIZE ];
      memcpy(m_data,another.m_data,BSIZE);
   }
   mem_storage()
   {
      m_data = new unsigned char[ BSIZE ];
   }
   ~mem_storage()
   {
      delete[] m_data;
   }

   void write( unsigned offset, size_t length, const unsigned char *data )
   {
      assert( offset + length <= BSIZE );
      memcpy(m_data+offset,data,length);
   }

   void read( unsigned offset, size_t length, unsigned char *data ) const
   {
      assert( offset + length <= BSIZE );
      memcpy(data,m_data+offset,length);
   }

private:
   unsigned m_nbytes;
   unsigned char *m_data;
};

class memory_space
{
public:
   virtual ~memory_space() {}
   virtual void write( mem_addr_t addr, size_t length, const void *data ) = 0;
   virtual void read( mem_addr_t addr, size_t length, void *data ) const = 0;;
};

template<unsigned BSIZE> class memory_space_impl : public memory_space {
public:
   memory_space_impl( std::string name, unsigned hash_size );

   virtual void write( mem_addr_t addr, size_t length, const void *data );
   virtual void read( mem_addr_t addr, size_t length, void *data ) const;

private:
   std::string m_name;
   unsigned m_log2_block_size;
   typedef mem_map<mem_addr_t,mem_storage<BSIZE> > map_t;
   map_t m_data;
};

#endif
