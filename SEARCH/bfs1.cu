/*
   Written By Ashwin Raghav
   Twitter @ashwinraghav
   blog.ashwinraghav.com
   github.com/ashwinraghav
   If you want to copy the code, by all means DO	
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#define MAX_THREADS_PER_BLOCK 256

int no_of_nodes;
int edge_list_size;
FILE *fp;
long long start_timer();
long long stop_timer(long long start_time, char *name);
void store_results(char* file_name);
void store_destinations(char file_name[], bool *node_dest);
//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
        char type[100];
};

#include <kernel.cu>

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
	//CUT_EXIT(argc, argv);
	return 0;
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{

	//    CUT_DEVICE_INIT();

	printf("Reading File\n");
	static char *input_file_name;
	//printf("argc=%d\n", argc);
	if (argc == 2 ) {
		input_file_name = argv[1];
		printf("Input file: %s\n", input_file_name);
	}
	else 
	{
		input_file_name = "SampleGraph.txt";
		printf("No input file specified, defaulting to SampleGraph.txt\n");
	}
	//Read in Graph from a file
	fp = fopen(input_file_name,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno; 
        char node_type[100];
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d %s", &start, &edgeno, node_type);
                strcpy(h_graph_nodes[i].type, node_type);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);

	//set the source node as true in the mask
	h_graph_mask[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");

        getchar();
	//Copy the Node list to device memory
	Node* d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes); 
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice);

	//Copy the Edge List to device Memory
	int* d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size);
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);

	//Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes);
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);

	//Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes);
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);
	
        //Copy the Mask to device memory
	bool* d_graph_dest;
	bool* h_graph_dest = (bool*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_graph_dest[i]=false;
	cudaMalloc( (void**) &d_graph_dest, sizeof(bool)*no_of_nodes);
	cudaMemcpy( d_graph_dest, h_graph_dest, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);

	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;

	// allocate device memory for result
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

	//make a bool to check if the execution is over
	bool *d_over;
	cudaMalloc( (void**) &d_over, sizeof(bool));

	printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	//start the timer
	long long timer;
	timer = start_timer();
	int k=0;
	bool stop;

        int type_size = 3;
        char *types[type_size];
        types[0] = "person";
        types[1] = "person";
        types[2] = "restaurant";
        
	//Call the Kernel untill all the elements of Frontier are not false
	while(k < type_size)
	{
		//if no thread changes this value then the loop stops
		stop=false;
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);
                bool final_step = false;

                if(type_size == k)
                        final_step = true;
                            
                Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_graph_visited, d_graph_dest, d_cost, d_over, no_of_nodes, types[0], final_step);

		cudaThreadSynchronize();
		cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
		k++;
	}
	//while(stop);


	printf("Kernel Executed %d times\n",k);

	// copy result from device to host
	cudaMemcpy(h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_graph_dest, d_graph_dest, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);

        store_destinations("result.txt", h_graph_dest);

	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
	stop_timer(timer, "Total Processing time");
}

void store_destinations(char file_name[], bool *node_dest)
{
        FILE *fpo = fopen(file_name, "w");
        for(int i=0; i < no_of_nodes; i++ )
        {
                if(node_dest[i])
                {
                        fprintf(fpo, "node %d\n", i);
                }
        }
        fclose(fpo);

}

void store_results(char file_name[])
{
	//Store the result into a file
	FILE *fpo = fopen(file_name,"w");
	//for(int i=0;i<no_of_nodes;i++)
	//	fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	//printf("Result stored in result.txt\n");
}
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

long long stop_timer(long long start_time, char *label) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
	printf("%s: %.5f sec\n", label, ((float) (end_time - start_time)) / (1000 * 1000));
	return end_time - start_time;
}

