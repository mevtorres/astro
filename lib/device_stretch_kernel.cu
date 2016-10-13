#ifndef STRETCH_KERNEL_H_
#define STRETCH_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "AstroAccelerate/params.h"
#include <cufft.h>

<<<<<<< HEAD
//{{{ stretch
=======
>>>>>>> fe80b9c735d1c898047cbb64bcf8da05cd6a21da
__global__ void stretch_kernel(int acc, int samps, float tsamp, float *d_input, float *d_output, float t_zero, float multiplier, float tsamp_inverse)
{

	int t  = blockIdx.x * blockDim.x + threadIdx.x;
	
	float p_time = t*(t_zero  + (multiplier * (t-1.0f)));

	int stretch_index = __float2int_rz(p_time*tsamp_inverse);
					
	if(stretch_index >= 0 && stretch_index <samps) d_output[stretch_index] = d_input[t];
}

<<<<<<< HEAD
//}}}

#endif

=======
#endif
>>>>>>> fe80b9c735d1c898047cbb64bcf8da05cd6a21da
