#ifndef ASTROACCELERATE_BIN_H_
#define ASTROACCELERATE_BIN_H_

extern void bin_gpu(unsigned short *d_input, float *d_output, int nchans, int nsamp);
extern void bin_gpu_stream(unsigned short *d_input, float *d_output, int nchans, int nsamp, cudaStream_t stream);

#endif

