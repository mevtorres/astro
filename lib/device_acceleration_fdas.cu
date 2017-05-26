//
#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
//#include <omp.h>
//
#include <errno.h>
#include <string.h>
#include <sys/time.h>
//#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
//
#include "headers/params.h"
#include "helper_cuda.h"
#include "headers/fdas.h"
#include "headers/fdas_host.h"

void acceleration_fdas(int range,
					   int nsamp,
					   int max_ndms,
					   int processed,
					   int nboots,
					   int num_trial_bins,
					   int navdms,
					   float narrow,
					   float wide,
					   int nsearch,
					   float aggression,
					   float cutoff,
					   float ***output_buffer,
					   int *ndms,
					   int *inBin,
					   float *dm_low,
					   float *dm_high,
					   float *dm_step,
					   float tsamp,
					   int enable_custom_fft,
					   int enable_inbin,
					   int enable_norm,
					   float sigma_constant,
					   int enable_output_ffdot_plan,
					   int enable_output_fdas_list)
{

	fdas_params params;
	// fdas_new_acc_sig acc_sig;
	cmd_args cmdargs;
	fdas_gpuarrays gpuarrays1, gpuarrays2;
	fdas_cufftplan fftplans1, fftplans2;
	cudaStream_t stream1, stream2;
	cudaError_t res1, res2;
	//float *acc_signal = NULL;
	struct timeval t_start, t_end;
	double t_gpu = 0.0, t_gpu_i = 0.0;

	//set default arguments
	cmdargs.nharms = 1; //
	cmdargs.nsig = 0; //
	cmdargs.duty = 0.10; //
	cmdargs.iter = 1; //
	cmdargs.writef = 1; //
	cmdargs.zval = 4; //
	cmdargs.mul = 1024; //
	cmdargs.wsig = 0; //
	cmdargs.search = 1; //
	cmdargs.thresh = 10.0; //
	cmdargs.freq0 = 100.5; //
	cmdargs.sigamp = 0.1; //
	cmdargs.basic = 0; //
	cmdargs.kfft = 1; //
	if (enable_custom_fft == 1){
		cmdargs.basic = 0; //
		cmdargs.kfft = 1; //
	}
	else{
		cmdargs.basic = 1; //
		cmdargs.kfft  = 0; //
	}
	//
	if (enable_inbin == 1)
		cmdargs.inbin = 1; //
	else
		cmdargs.inbin = 0; //
	//
	if (enable_norm == 1)
		cmdargs.norm = 1; //
	else
		cmdargs.norm = 0; //

	//get signal parameters
	/*acc_sig.nsamps = cmdargs.mul * 8192;  //
	acc_sig.freq0 = cmdargs.freq0; //
	acc_sig.mul = cmdargs.mul; 	//
	acc_sig.zval = cmdargs.zval; //
	acc_sig.nsig = cmdargs.nsig; //
	acc_sig.nharms = cmdargs.nharms; //
	acc_sig.duty = cmdargs.duty; //
	acc_sig.sigamp = cmdargs.sigamp; //
*/
	int nearest = (int) floorf(log2f((float) processed));
	printf("\nnearest:\t%d", nearest);
	int samps = (int) powf(2.0, nearest);
	processed=samps;
	printf("\nsamps:\t%d", samps);

	params.nsamps = samps;

	/// Print params.h
	fdas_print_params_h();

	// prepare signal
	params.offset = presto_z_resp_halfwidth((double) ZMAX, 0); //array offset when we pick signal points for the overlp-save method
	printf(" Calculated overlap-save offsets: %d\n", params.offset);

	//
	params.sigblock = KERNLEN - 2 * params.offset + 1;
	params.scale = 1.0f / (float) (KERNLEN);
	params.rfftlen = params.nsamps / 2 + 1;
	params.nblocks = params.rfftlen / params.sigblock;
	params.siglen = params.nblocks * params.sigblock;
	params.extlen = params.nblocks * KERNLEN; //signal array extended to array of separate N=KERNLEN segments
	params.ffdotlen = params.siglen * NKERN; // total size of ffdot complex plane in fourier bins
	params.ffdotlen_cpx = params.extlen * NKERN; // total size of ffdot powers plane in fourier bins
	params.max_list_length = params.ffdotlen/4;
	
	if (cmdargs.inbin)
		params.ffdotlen = params.ffdotlen * 2;

	if (cmdargs.search)
	{
		printf("\nnumber of convolution templates NKERN = %d, template length = %d, acceleration step in bins = %f, zmax = %d, template size power of 2 = %d, scale=%f \n",
				NKERN, KERNLEN, ACCEL_STEP, ZMAX, NEXP, params.scale);

		if (cmdargs.basic)
			printf("\nBasic algorithm:\n---------\n");

		else if (cmdargs.kfft)
			printf("\nCustom algorithm:\n-------\n");

		printf("\nnsamps = %d\ncpx signal length = %d\ntotal length: initial = %d, extended = %d\nconvolution signal segment length = %d\ntemplate length = %d\n# convolution blocks = %d\nffdot length = %u\n",
				params.nsamps, params.rfftlen, params.siglen, params.extlen,
				params.sigblock, KERNLEN, params.nblocks, params.ffdotlen);
		//
		if (cmdargs.basic)
			printf("ffdot length cpx pts = %u\n", params.ffdotlen_cpx);

		//memory required
		size_t mem_ffdot = params.ffdotlen * sizeof(float); // total size of ffdot plane powers in bytes
		size_t mem_ffdot_cpx = params.ffdotlen_cpx * sizeof(float2); // total size of ffdot plane powers in bytes
		size_t mem_kern_array = KERNLEN * NKERN * sizeof(float2);
		size_t mem_max_list_size = params.max_list_length*4*sizeof(float);
		size_t mem_signals = (params.nsamps * sizeof(float)) + (params.siglen * sizeof(float2)) + (params.extlen * sizeof(float2));

		//Determining memory usage
		size_t mem_tot_needed = 0;
		double gbyte = 1024.0 * 1024.0 * 1024.0;
		float mbyte = gbyte / 1024.0;
		size_t mfree, mtotal;

		if (cmdargs.basic)
			mem_tot_needed = mem_ffdot + mem_ffdot_cpx + mem_kern_array + mem_signals + mem_max_list_size; // KA added + mem_max_list_size
		if (cmdargs.kfft)
			mem_tot_needed = mem_ffdot + mem_kern_array + mem_signals + mem_max_list_size; // KA added + mem_max_list_size
		checkCudaErrors(cudaMemGetInfo(&mfree, &mtotal));

		// get available memory info
		printf( "Total memory for this device: %.2f GB\nAvailable memory on this device for data upload: %.2f GB \n", mtotal / gbyte, mfree / gbyte);

		//Allocating gpu arrays
		gpuarrays1.mem_insig = params.nsamps * sizeof(float);
		gpuarrays1.mem_rfft = params.rfftlen * sizeof(float2);
		gpuarrays1.mem_extsig = params.extlen * sizeof(float2);
		gpuarrays1.mem_ffdot = mem_ffdot;
		gpuarrays1.mem_ffdot_cpx = mem_ffdot_cpx;
		gpuarrays1.mem_ipedge = params.nblocks * 2;
		gpuarrays1.mem_max_list_size = mem_max_list_size;
		gpuarrays2.mem_insig = params.nsamps * sizeof(float);
		gpuarrays2.mem_rfft = params.rfftlen * sizeof(float2);
		gpuarrays2.mem_extsig = params.extlen * sizeof(float2);
		gpuarrays2.mem_ffdot = mem_ffdot;
		gpuarrays2.mem_ffdot_cpx = mem_ffdot_cpx;
		gpuarrays2.mem_ipedge = params.nblocks * 2;
		gpuarrays2.mem_max_list_size = mem_max_list_size;

		printf("Total memory needed on GPU for arrays to process 1 DM: %.4f GB\nfloat ffdot plane (for power spectrum) = %.4f GB.\nTemplate array %.4f GB\nOne dimensional signals %.4f\n1 GB = %f",
				(double) mem_tot_needed / gbyte, (double) (mem_ffdot) / gbyte, (double) mem_kern_array / gbyte, (double) mem_signals / gbyte, gbyte);
		//
		if (mem_tot_needed >= (mfree - gbyte / 10)) {
			printf("\nNot enough memory available on the device to process this array\nPlease use a shorter signal\nExiting program...\n\n");
			exit(1);
		}
		getLastCudaError("\nCuda Error\n");

		fdas_alloc_gpu_arrays_stream(&gpuarrays1, &cmdargs);
		getLastCudaError("\nCuda Error\n");
		fdas_alloc_gpu_arrays_stream(&gpuarrays2, &cmdargs);
		getLastCudaError("\nCuda Error\n");

		res1 = cudaStreamCreate(&stream1);
		res2 = cudaStreamCreate(&stream2);

		// Calculate kernel templates on CPU and upload-fft on GPU
		printf("\nCreating acceleration templates with KERNLEN=%d, NKERN = %d zmax=%d... ",	KERNLEN, NKERN, ZMAX);
		fdas_create_acc_kernels(gpuarrays1.d_kernel, &cmdargs);
		getLastCudaError("\nCuda Error\n");
		fdas_create_acc_kernels(gpuarrays2.d_kernel, &cmdargs);
		getLastCudaError("\nCuda Error\n");
		printf(" done.\n");

		//Create cufft plans
		fdas_cuda_create_fftplans(&fftplans1, &params);
		getLastCudaError("\nCuda Error\n");
		fdas_cuda_create_fftplans(&fftplans2, &params);
		getLastCudaError("\nCuda Error\n");

		// peak finding
		int ibin=1;
		if (cmdargs.inbin) ibin=2;
		unsigned int list_size;
		float *d_MSD1, *d_MSD2;
		float h_MSD1[3], h_MSD2[3];
		if ( cudaSuccess != cudaMalloc((void**) &d_MSD1, sizeof(float)*3)) printf("Allocation error!\n");
		unsigned int *gmem_fdas_peak_pos1;
		if ( cudaSuccess != cudaMalloc((void**) &gmem_fdas_peak_pos1, 1*sizeof(int))) printf("Allocation error!\n");
		cudaMemset((void*) gmem_fdas_peak_pos1, 0, sizeof(int));
		if ( cudaSuccess != cudaMalloc((void**) &d_MSD2, sizeof(float)*3)) printf("Allocation error!\n");
		unsigned int *gmem_fdas_peak_pos2;
		if ( cudaSuccess != cudaMalloc((void**) &gmem_fdas_peak_pos2, 1*sizeof(int))) printf("Allocation error!\n");
		cudaMemset((void*) gmem_fdas_peak_pos2, 0, sizeof(int));

		// Starting main acceleration search
		//cudaGetLastError(); //reset errors
		printf("\n\nStarting main acceleration search\n\n");

		int iter=cmdargs.iter;
		int titer=1;

		// FFT
		for (int i = 0; i < range; i++) {
			processed=samps/inBin[i];
			for (int dm_count = 0; dm_count < 10; ++dm_count) {

				/*********************************** stream1 *********************************************************/
				//first time PCIe transfer and print timing
				gettimeofday(&t_start, NULL); //don't time transfer
				checkCudaErrors( cudaMemcpyAsync(gpuarrays1.d_in_signal, output_buffer[i][dm_count], processed*sizeof(float), cudaMemcpyHostToDevice, stream1));

				gettimeofday(&t_end, NULL);
				t_gpu = (double) (t_end.tv_sec + (t_end.tv_usec / 1000000.0)  - t_start.tv_sec - (t_start.tv_usec/ 1000000.0)) * 1000.0;
				t_gpu_i = (t_gpu /(double)titer);
				printf("\n\nAverage vector transfer time of %d float samples (%.2f Mb) from 1000 iterations: %f ms\n\n", params.nsamps, (float)(gpuarrays1.mem_insig)/mbyte, t_gpu_i);

				cudaProfilerStart(); //exclude cuda initialization ops
				if(cmdargs.basic) {
					gettimeofday(&t_start, NULL); //don't time transfer
				    fdas_cuda_basic_stream(&fftplans1, &gpuarrays1, &cmdargs, &params, stream1);

				    gettimeofday(&t_end, NULL);
				    t_gpu = (double) (t_end.tv_sec + (t_end.tv_usec / 1000000.0)  - t_start.tv_sec - (t_start.tv_usec/ 1000000.0)) * 1000.0;
				    t_gpu_i = (t_gpu / (double)iter);
				    printf("\n\nConvolution using basic algorithm with cuFFT\nTotal process took: %f ms per iteration \nTotal time %d iterations: %f ms\n", t_gpu_i, iter, t_gpu);
				}

				#ifndef NOCUST
				if (cmdargs.kfft) {
					printf("\nMain: running FDAS with custom fft\n");
					gettimeofday(&t_start, NULL); //don't time transfer
					fdas_cuda_customfft_stream(&fftplans1, &gpuarrays1, &cmdargs, &params, stream1);


					gettimeofday(&t_end, NULL);
					t_gpu = (double) (t_end.tv_sec + (t_end.tv_usec / 1000000.0)  - t_start.tv_sec - (t_start.tv_usec/ 1000000.0)) * 1000.0;
					t_gpu_i = (t_gpu / (double)iter);
					printf("\n\nConvolution using custom FFT:\nTotal process took: %f ms\n per iteration \nTotal time %d iterations: %f ms\n", t_gpu_i, iter, t_gpu);
				}
				#endif
				// Calculating base level noise and peak find
				if(cmdargs.basic || cmdargs.kfft){
					//------------- Testing BLN
					//float signal_mean, signal_sd;
					//------------- Testing BLN
					
					cudaMemsetAsync((void*) gmem_fdas_peak_pos1, 0, sizeof(int), stream1);
					
					// This might be bit iffy since when interbining is done values are correlated
					printf("Dimensions for BLN: ibin:%d; siglen:%d;\n", ibin, params.siglen);
					BLN_stream(gpuarrays1.d_ffdot_pwr, d_MSD1, 32, 32, NKERN, ibin*params.siglen, 0, sigma_constant, stream1);
					////------------- Testing BLN
					//checkCudaErrors(cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost));
					//signal_mean=h_MSD[0]; signal_sd=h_MSD[1];
					//MSD_limited(gpuarrays.d_ffdot_pwr, d_MSD, NKERN, ibin*params.siglen, 0);
					//checkCudaErrors(cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost));
					//printf("BLN: mean:%f; sd:%f || MSD: mean:%f; sd:%f\n", signal_mean, signal_sd, h_MSD[0], h_MSD[1]);
					////------------- Testing BLN
					
					PEAK_FIND_FOR_FDAS_stream(gpuarrays1.d_ffdot_pwr, gpuarrays1.d_fdas_peak_list, d_MSD1, NKERN, ibin*params.siglen, cmdargs.thresh, params.max_list_length, gmem_fdas_peak_pos1, dm_count*dm_step[i] + dm_low[i], stream1);
					//PEAK_FIND_FOR_FDAS(gpuarrays1.d_ffdot_pwr, gpuarrays1.d_fdas_peak_list, d_MSD1, NKERN, ibin*params.siglen, cmdargs.thresh, params.max_list_length, gmem_fdas_peak_pos1, dm_count*dm_step[i] + dm_low[i]);

					cudaStreamSynchronize(stream1);

					//
					checkCudaErrors( cudaMemcpyAsync(gpuarrays2.d_in_signal, output_buffer[i][dm_count+1], processed*sizeof(float), cudaMemcpyHostToDevice, stream2));

					checkCudaErrors(cudaMemcpyAsync(h_MSD1, d_MSD1, 3*sizeof(float), cudaMemcpyDeviceToHost, stream1));
					checkCudaErrors(cudaMemcpyAsync(&list_size, gmem_fdas_peak_pos1, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));
					if (enable_output_fdas_list == 1)
					{
						if(list_size>0)
							fdas_write_list(&gpuarrays1, &cmdargs, &params, h_MSD1, dm_low[i], dm_count, dm_step[i], list_size);
					}
				}
				if (enable_output_ffdot_plan == 1)
				{
					fdas_write_ffdot_stream(&gpuarrays1, &cmdargs, &params, dm_low[i], dm_count, dm_step[i], stream1);
				}

				// increment loop index
				dm_count++;

				/*********************************** stream2 *********************************************************/

				cudaProfilerStart(); //exclude cuda initialization ops
				if(cmdargs.basic) {
					gettimeofday(&t_start, NULL); //don't time transfer
				    fdas_cuda_basic_stream(&fftplans2, &gpuarrays2, &cmdargs, &params, stream2);

				    gettimeofday(&t_end, NULL);
				    t_gpu = (double) (t_end.tv_sec + (t_end.tv_usec / 1000000.0)  - t_start.tv_sec - (t_start.tv_usec/ 1000000.0)) * 1000.0;
				    t_gpu_i = (t_gpu / (double)iter);
				    printf("\n\nConvolution using basic algorithm with cuFFT\nTotal process took: %f ms per iteration \nTotal time %d iterations: %f ms\n", t_gpu_i, iter, t_gpu);
				}

				#ifndef NOCUST
				if (cmdargs.kfft) {
					printf("\nMain: running FDAS with custom fft\n");
					gettimeofday(&t_start, NULL); //don't time transfer
					fdas_cuda_customfft_stream(&fftplans2, &gpuarrays2, &cmdargs, &params, stream2);

					gettimeofday(&t_end, NULL);
					t_gpu = (double) (t_end.tv_sec + (t_end.tv_usec / 1000000.0)  - t_start.tv_sec - (t_start.tv_usec/ 1000000.0)) * 1000.0;
					t_gpu_i = (t_gpu / (double)iter);
					printf("\n\nConvolution using custom FFT:\nTotal process took: %f ms\n per iteration \nTotal time %d iterations: %f ms\n", t_gpu_i, iter, t_gpu);
				}
				#endif
				// Calculating base level noise and peak find
				if(cmdargs.basic || cmdargs.kfft){
					//------------- Testing BLN
					//float signal_mean, signal_sd;
					//------------- Testing BLN
					cudaMemsetAsync((void*) gmem_fdas_peak_pos2, 0, sizeof(int), stream2);

					// This might be bit iffy since when interbining is done values are correlated
					printf("Dimensions for BLN: ibin:%d; siglen:%d;\n", ibin, params.siglen);
					BLN_stream(gpuarrays2.d_ffdot_pwr, d_MSD1, 32, 32, NKERN, ibin*params.siglen, 0, sigma_constant, stream2);
					////------------- Testing BLN
					//checkCudaErrors(cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost));
					//signal_mean=h_MSD[0]; signal_sd=h_MSD[1];
					//MSD_limited(gpuarrays.d_ffdot_pwr, d_MSD, NKERN, ibin*params.siglen, 0);
					//checkCudaErrors(cudaMemcpy(h_MSD, d_MSD, 3*sizeof(float), cudaMemcpyDeviceToHost));
					//printf("BLN: mean:%f; sd:%f || MSD: mean:%f; sd:%f\n", signal_mean, signal_sd, h_MSD[0], h_MSD[1]);
					////------------- Testing BLN

					PEAK_FIND_FOR_FDAS_stream(gpuarrays2.d_ffdot_pwr, gpuarrays2.d_fdas_peak_list, d_MSD2, NKERN, ibin*params.siglen, cmdargs.thresh, params.max_list_length, gmem_fdas_peak_pos2, dm_count*dm_step[i] + dm_low[i], stream2);

					cudaStreamSynchronize(stream2);

					checkCudaErrors(cudaMemcpyAsync(h_MSD2, d_MSD2, 3*sizeof(float), cudaMemcpyDeviceToHost, stream2));
					checkCudaErrors(cudaMemcpyAsync(&list_size, gmem_fdas_peak_pos2, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream2));
					if (enable_output_fdas_list == 1)
					{
						if(list_size>0)
							fdas_write_list(&gpuarrays2, &cmdargs, &params, h_MSD2, dm_low[i], dm_count, dm_step[i], list_size);
					}
				}
				if (enable_output_ffdot_plan == 1)
				{
					fdas_write_ffdot_stream(&gpuarrays2, &cmdargs, &params, dm_low[i], dm_count, dm_step[i], stream2);
				}
			}
		}

		cufftDestroy(fftplans1.realplan);
	    cufftDestroy(fftplans1.forwardplan);
		cufftDestroy(fftplans2.realplan);
	    cufftDestroy(fftplans2.forwardplan);
	    // releasing GPU arrays
	    fdas_free_gpu_arrays_stream(&gpuarrays1, &cmdargs);
	    fdas_free_gpu_arrays_stream(&gpuarrays2, &cmdargs);
		res1 = cudaStreamDestroy(stream1);
		res2 = cudaStreamDestroy(stream2);
		//
		cudaFree(d_MSD1);
		cudaFree(gmem_fdas_peak_pos1);
		cudaFree(d_MSD2);
		cudaFree(gmem_fdas_peak_pos2);
	}
}
