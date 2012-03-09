/*
   Written By Ashwin Raghav
   Twitter @ashwinraghav
   blog.ashwinraghav.com
   github.com/ashwinraghav
   If you want to copy the code, by all means DO	
 */

#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void Kernel( Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_graph_visited, int* g_cost, bool *g_over, int no_of_nodes) 
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		g_graph_visited[tid]=true;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
		{
			int id = g_graph_edges[i];
			if(!g_graph_visited[id])
			{
				g_cost[id]=g_cost[tid]+1;
				g_graph_mask[id]=true;
				//Change the loop stop value such that loop continues
				*g_over=true;
			}
		}
	}
}

#endif 
