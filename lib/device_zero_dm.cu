
//#include <omp.h>
#include <time.h>
#include <stdio.h>
#include "headers/params.h"
#include "device_zero_dm_kernel.cu"

//{{{ zero_dm

void zero_dm(unsigned short *d_input, int nchans, int nsamp, int nbits) {

	int sum_threads     = 256;
	int num_blocks_t    = nsamp;

	printf("\nCORNER TURN!");
	printf("\n%d %d %d", nsamp, nchans, sum_threads);
	printf("\n%d %d", CT, 1);
	printf("\n%d %d", num_blocks_t, 1);

	dim3 threads_per_block(sum_threads, 1);
	dim3 num_blocks(num_blocks_t,1);

	clock_t start_t, end_t;
	start_t = clock();

	float normalization_factor = ((pow(2,nbits)-1)/2);

	zero_dm_kernel<<< num_blocks, threads_per_block, sum_threads*4 >>>(d_input, nchans, nsamp, normalization_factor);
	cudaDeviceSynchronize();

	end_t = clock();
	double time = (double)(end_t-start_t) / CLOCKS_PER_SEC;
	printf("\nPerformed ZDM: %lf (GPU estimate)", time);

	//}}}

}

//}}}

