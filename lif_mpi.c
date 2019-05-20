#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
//#include <omp.h>

#define PI 3.14159265358979323846



void take_screenshot(FILE *fp, double *u, unsigned int *discharge_counter, double t, int N)
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			fprintf(fp, "%d\t%d\t%f\t%u\t%f\n", i, j, u[i*N+j], discharge_counter[i*N+j], t);
		}
	}
}



void calculate_next_u(double *u, double *u_new, unsigned int *discharge_counter, unsigned int *refractory_counter, int N_r, int chunk, 
int start, double a, double b, double dt, double u_th, double t, unsigned int *neighbors_list, int refractory_period)
{
	int i,offset,neighbor;
	double sum; 

	for (i=0; i<chunk; i++)
	{
		sum = 0;
		offset = i*N_r;

		if (!refractory_counter[i])
		{
			for (neighbor=0; neighbor<N_r; neighbor++)
			{
				sum += u[neighbors_list[offset + neighbor]];
			}

			u_new[i] = dt + a*u[start+i] + b*sum;

			if (u_new[i] >= u_th)
			{
				u_new[i] = 0;
				refractory_counter[i] = (unsigned int) refractory_period;
				discharge_counter[i]++;
			}
		}
		else
		{
			refractory_counter[i]--;
		}   
	}
}


void find_neighbors(unsigned int *neighbors_list, char** Base_Matrix, int start, int chunk, int N, int N_f, int N_r)
{
	int i,k,l,count,x,y,x_offset,y_offset,x_index,y_index,central_neuron_index;

	central_neuron_index = N_f / 2;

	for (i=0; i<chunk; i++)
	{
		count = 0;
		x = (start + i)/N;
		y = (start + i)%N;
		x_offset = central_neuron_index - x + N;
		y_offset = central_neuron_index - y + N;

		for (k = 0; k < N; k++)
		{
			x_index = (k + x_offset) % N;
			for (l = 0; l < N; l++)
			{
				y_index = (l + y_offset) % N;
				if (Base_Matrix[x_index][y_index] == ' ')
				{
					neighbors_list[i*N_r + count] = k*N + l;
					count++;                    
				}
			}
		}
	}
}

void initialize_u(double* u, int N, double u_th, int seed, unsigned int* refractory_counter, int previous_exists, FILE *fp2)
{
	int i;

	srand(seed);

	if (previous_exists)
	{
		fread(u,sizeof *u,N*N,fp2);
		fread(refractory_counter,sizeof *refractory_counter,N*N,fp2);
	}
	else
	{
		for (i = 0; i < N*N; i++)
		{
			u[i] = u_th* (double)rand() / (double)RAND_MAX;	
		}
	}
}


char ** create_base_matrix(int N, int N_f)
{
	int i, j, dim, d;
	int depth = (int)(log(N_f) / log(3));
	char **Base_Matrix, *ptr;

	Base_Matrix = (char**)malloc(sizeof(char*)*N + sizeof(char)*N*N);
	ptr = (char*) (Base_Matrix + N); 

	for(i = 0; i < N; i++)
	{
		Base_Matrix[i] = (ptr + N * i);
	} 


	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			Base_Matrix[i][j] = '0';
		}
	}

	for (i = 0, dim = 1; i < depth; i++, dim *= 3)
		;

	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			for (d = dim / 3; d; d /= 3)
				if ((i % (d * 3)) / d == 1 && (j % (d * 3)) / d == 1)
					break;
			Base_Matrix[i][j] = d ? '0' : ' ';
		}
	}

	return Base_Matrix;
}


int main(int argc, char *argv[])
{
	int i,comm_size, my_rank, start, chunk, *recvcounts, *displs, previous_exists = 0;
	int N, N_f, N_r, refractory_period, seed, screenshot_timer, loop_counter, loop_end, screenshot = 30000;
	unsigned int *discharge_counter, *refractory_counter, *neighbors_list;
	double *u, *u_new, a, b, t = 0;
	const double t_end = 4020, dt = 0.001, u_th = 0.98;
	double sigma;
	char **Base_Matrix;
	char filename[50], init[50];
	pid_t pid;
	FILE *fp, *fp2;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if (my_rank == 0)
	{
		if (argc != 6)
		{
			printf("Not enough input arguments.\n");
			exit(1);
		}
		N = atoi(argv[1]);
		N_f = atoi(argv[2]);
		sigma = atof(argv[3]);
		refractory_period = atoi(argv[4]);
		seed = atoi(argv[5]);
		if (sigma<0)
		{
			snprintf(filename, sizeof(filename), "lif_%dt_%df_-%d%d%dsg_%dpr_%dsd.dat",N,N_f,(int)(-1*sigma),((int)((-1*sigma) * 10))%10,((int)((-1*sigma)*100))%10,refractory_period,seed);
			snprintf(init, sizeof(init), "lif_%dt_%df_-%d%d%dsg_%dpr_%dsd_init",N,N_f,(int)(-1*sigma),((int)((-1*sigma) * 10))%10,((int)((-1*sigma)*100))%10,refractory_period,seed);
		}
		else
		{
			snprintf(filename, sizeof(filename), "lif_%dt_%df_%d%d%dsg_%dpr_%dsd.dat",N,N_f,(int)sigma,((int)(sigma * 10))%10,((int)(sigma*100))%10,refractory_period,seed);
			snprintf(init, sizeof(init), "lif_%dt_%df_%d%d%dsg_%dpr_%dsd_init",N,N_f,(int)sigma,((int)(sigma * 10))%10,((int)(sigma*100))%10,refractory_period,seed);
		}
		
		if( access( init, F_OK ) != -1 )
		{
			previous_exists = 1;
			fp = fopen(filename, "a");
			fp2 = fopen(init, "r+b");
			fread(&t,sizeof t,1,fp2);
		}
		else
		{
			fp = fopen(filename, "w");
			fp2 = fopen(init, "wb");
		}    
	}

	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N_f, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&refractory_period, 1, MPI_INT, 0, MPI_COMM_WORLD);

	N_r = (int)pow(8, log(N_f) / log(3));

	//Split Workload
	chunk = N*N/comm_size;
	if (my_rank < (N*N)%comm_size)
	{
		chunk++;
	}

	u = (double*) malloc(N*N*sizeof *u);
	if (my_rank == 0)
	{
		discharge_counter = (unsigned int*) calloc(N*N,sizeof *discharge_counter);
		refractory_counter = (unsigned int*) calloc(N*N,sizeof *refractory_counter);
	}
	else
	{
		discharge_counter = (unsigned int*) calloc(chunk,sizeof *discharge_counter);
		refractory_counter = (unsigned int*) calloc(chunk,sizeof *refractory_counter);
	}	
	neighbors_list = (unsigned int*) malloc(chunk*N_r*sizeof *neighbors_list);
	u_new = (double*) malloc(chunk*sizeof *u_new);
	recvcounts = (int*) malloc(comm_size*sizeof *recvcounts);
	displs = (int*) malloc(comm_size*sizeof *displs);

	if(u == NULL || discharge_counter == NULL || refractory_counter == NULL 
		|| neighbors_list == NULL || u_new == NULL || recvcounts == NULL || displs == NULL)                     
	{
		printf("Error! memory not allocated.\n");
		free(u);
		free(discharge_counter);
		free(refractory_counter);
		free(neighbors_list);
		free(u_new);
		free(recvcounts);
		free(displs);
		MPI_Finalize();
		exit(0);
	}

	chunk = N*N/comm_size;
	start = 0;
	for(i=0;i<comm_size;i++)
	{
		if(i<(N*N)%comm_size)
		{
			recvcounts[i] = chunk+1;
			displs[i] = start;
			start += chunk+1;
		}
		else
		{
			recvcounts[i] = chunk;
			displs[i] = start;
			start += chunk;
		}
	}

	chunk = recvcounts[my_rank];
	start = displs[my_rank];

	Base_Matrix = create_base_matrix(N,N_f);
	find_neighbors(neighbors_list, Base_Matrix, start, chunk, N, N_f, N_r);
	free(Base_Matrix);

	if (my_rank == 0)
	{
		initialize_u(u,N,u_th,seed,refractory_counter,previous_exists,fp2);
	}

	MPI_Bcast(u, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	screenshot_timer = 1;

	if (previous_exists)
	{
		if (my_rank == 0)
			MPI_Scatterv(refractory_counter, recvcounts, displs, MPI_UNSIGNED, MPI_IN_PLACE, chunk, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		else
			MPI_Scatterv(refractory_counter, recvcounts, displs, MPI_UNSIGNED, refractory_counter, chunk, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

		screenshot_timer = screenshot+1;
	}

	

	a = 1 + (sigma-1)*dt;
	b = -sigma*dt/N_r;

	loop_end = (int)(t_end / dt);

	for (loop_counter = 0; loop_counter < loop_end; loop_counter++, t += dt)
	{

		if (!(--screenshot_timer))
		{
			if (my_rank == 0)
			{
				MPI_Gatherv(MPI_IN_PLACE, chunk, MPI_UNSIGNED, discharge_counter, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
				MPI_Gatherv(MPI_IN_PLACE, chunk, MPI_UNSIGNED, refractory_counter, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			}
			else
			{
				MPI_Gatherv(discharge_counter, chunk, MPI_UNSIGNED, discharge_counter, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
				MPI_Gatherv(refractory_counter, chunk, MPI_UNSIGNED, refractory_counter, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
			}

			if (my_rank == 0)
			{
				wait(NULL);
				pid = fork();
				if (pid < 0)
				{
					perror("error: fork");
					exit(1);
				}
				if(pid == 0)
				{
					take_screenshot(fp,u,discharge_counter,t,N);
					printf("\rProgress: %.3f%%",(1.0*loop_counter)/loop_end*100);
					fflush(stdout);
					exit(0);
				}
				
				rewind(fp2);
				fwrite(&t,sizeof t,1,fp2);
				fwrite(u,sizeof *u,N*N,fp2);
				fwrite(refractory_counter,sizeof *refractory_counter,N*N,fp2);
				fflush(fp2);
			}

			memset(discharge_counter, 0, chunk * sizeof *discharge_counter);
			screenshot_timer = screenshot;
		}

		calculate_next_u(u,u_new,discharge_counter,refractory_counter,N_r,chunk,
						start,a,b,dt,u_th,t,neighbors_list, refractory_period);

		MPI_Allgatherv(u_new, chunk, MPI_DOUBLE, u, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);	
	}

	if (my_rank == 0)
	{
		MPI_Gatherv(MPI_IN_PLACE, chunk, MPI_UNSIGNED, discharge_counter, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		MPI_Gatherv(MPI_IN_PLACE, chunk, MPI_UNSIGNED, refractory_counter, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Gatherv(discharge_counter, chunk, MPI_UNSIGNED, discharge_counter, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		MPI_Gatherv(refractory_counter, chunk, MPI_UNSIGNED, refractory_counter, recvcounts, displs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	}

	if (my_rank==0)
	{
		wait(NULL);
		take_screenshot(fp,u,discharge_counter,t,N);
		rewind(fp2);
		fwrite(&t,sizeof t,1,fp2);
		fwrite(u,sizeof *u,N*N,fp2);
		fwrite(refractory_counter,sizeof *refractory_counter,N*N,fp2);
		fflush(fp2);
		fclose(fp);
		fclose(fp2);
		printf("\nCompletion %s\n",filename);
	}


	MPI_Finalize();
	free(u);
	free(discharge_counter);
	free(refractory_counter);
	free(neighbors_list);
	free(u_new);
	free(recvcounts);
	free(displs);
	return 0;
}
