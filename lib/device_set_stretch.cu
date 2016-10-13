#include <omp.h>
#include <stdio.h>
#include "AstroAccelerate/params.h"
#include "device_set_stretch_kernel.cu"
#include "helper_cuda.h"

<<<<<<< HEAD
//{{{ Dopler Stretch 

=======
>>>>>>> fe80b9c735d1c898047cbb64bcf8da05cd6a21da
void set_stretch_gpu(cudaEvent_t event, cudaStream_t stream, int samps, float mean, float *d_input) {

	int divisions_in_t  = 32;
	int num_blocks_t    = samps/divisions_in_t;

	dim3 threads_per_block(divisions_in_t);
	dim3 num_blocks(num_blocks_t);

	cudaStreamWaitEvent(stream, event, 0);
	set_stretch_kernel<<< num_blocks, threads_per_block, 0, stream >>>(samps, mean, d_input);
	getLastCudaError("stretch_kernel failed");
	cudaEventRecord(event, stream);
<<<<<<< HEAD
}

//}}}

=======
}
>>>>>>> fe80b9c735d1c898047cbb64bcf8da05cd6a21da
