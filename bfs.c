/*
	Written By Ashwin Raghav
	Twitter @ashwinraghav
	blog.ashwinraghav.com
	github.com/ashwinraghav

*/
#include <stdio.h>

#define N 10

void bfs(int adj[][N],int visited[],int start)
{
	int q[N],rear=-1,front=-1,i;
	q[++rear]=start;
	visited[start]=1;
	while(rear != front)
	{
		start = q[++front];
		if(start==9)
			printf("10\t");
		else
			printf("%c \t",start+49); //change to 65 in case of alphabets

		for(i=0;i<N;i++)
		{
			if(adj[start][i] && !visited[i])
			{
				q[++rear]=i;
				visited[i]=1;
			}
		}
	}
}

int main()
{
	int visited[N]={0};
	int adj[N][N]={
		{0,1,1,0,0,0,0,0,0,1},
		{0,0,0,0,1,0,0,0,0,1},
		{0,0,0,0,1,0,1,0,0,0},
		{1,0,1,0,0,1,1,0,0,1},
		{0,0,0,0,0,0,1,1,0,0},
		{0,0,0,1,0,0,0,1,0,0},
		{0,0,0,0,0,0,0,1,1,1},
		{0,0,1,0,0,0,0,0,0,0},
		{0,0,0,1,0,0,0,0,0,0},
		{0,0,1,0,0,0,0,1,1,0}};

	bfs(adj,visited,0);
	return 0;


}

