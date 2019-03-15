#include "aa_device_configure_add.hpp"


#include <stdio.h>

__global__ void add(){
	printf("ID : %d\n", threadIdx.x);
}


namespace astroaccelerate{

void call_kernel_add(){
	printf("Calling function add...");
		add<<<1,1>>>();
	printf(" done.");
}
}
