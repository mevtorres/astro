//#include <omp.h>
#include <time.h>
#include <stdio.h>
#include "headers/params.h"
#include "device_corner_turn_kernel.cu"
#include "timer.h"

//{{{ Corner-turn 

void corner_turn(unsigned short *d_input, float *d_output, int nchans, int nsamp)
{

	//{{{ Simple corner turn on the GPU 
	GpuTimer timer;

	size_t divisions_in_t = CT;
	size_t divisions_in_f = CF;
	size_t num_blocks_t = nsamp / divisions_in_t;
	size_t num_blocks_f = nchans / divisions_in_f;
	float ftemp;

	printf("\nCORNER TURN!");
	printf("\n%d %d", nsamp, nchans);
	printf("\n%zu %zu", divisions_in_t, divisions_in_f);
	printf("\n%zu %zu", num_blocks_t, num_blocks_f);

	dim3 threads_per_block(divisions_in_t, divisions_in_f);
	dim3 num_blocks(num_blocks_t, num_blocks_f);

	timer.Start();
	
	simple_corner_turn_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, nchans, nsamp);
	cudaDeviceSynchronize();
	swap<<<num_blocks, threads_per_block>>>(d_input, d_output, nchans, nsamp);
	cudaDeviceSynchronize();
	
	timer.Stop();
	double time = (timer.Elapsed())/1000.0;
	printf("\nPerformed CT: %lf (GPU estimate)", time);
	ftemp = (float) (((float) divisions_in_t)*((float) divisions_in_f)*((float) num_blocks_t)*((float) num_blocks_f));
	printf("\nCT Gops based on %.2f ops per channel per tsamp: %f", 10.0, ( ( 10.0*(ftemp) ) / ( time ) ) / 1000000000.0);
	printf("\nCT Device memory bandwidth in GB/s: %lf", ( ( sizeof(float) + sizeof(unsigned short) ) * ( ftemp ) ) / ( time ) / 1000000000.0);

	//cudaMemcpy(d_input, d_output, inputsize, cudaMemcpyDeviceToDevice);

	//}}}

}

//}}}

