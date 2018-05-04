//#define GPU_ANALYSIS_DEBUG
//#define MSD_BOXCAR_TEST
//#define GPU_PARTIAL_TIMER
#define GPU_TIMER

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "headers/params.h"

#include "headers/device_BC_plan.h"
#include "headers/device_peak_find.h"
#include "headers/device_MSD_plane_profile.h"
#include "headers/device_SPS_long.h"
#include "headers/device_threshold.h"

#include "timer.h"

typedef float* FP;
//TODO:
// Make BC_plan for arbitrary long pulses, by reusing last element in the plane

/*
void Create_list_of_boxcar_widths(std::vector<int> *boxcar_widths, std::vector<int> *BC_widths, int max_boxcar_width){
	int DIT_value, DIT_factor, width;
	DIT_value = 1;
	DIT_factor = 2;
	width = 0;
	for(int f=0; f<(int) BC_widths->size(); f++){
		for(int b=0; b<BC_widths->operator[](f); b++){
			width = width + DIT_value;
			if(width<=max_boxcar_width){
				boxcar_widths->push_back(width);
			}
		}
		DIT_value = DIT_value*DIT_factor;
	}
}
*/

void Create_DM_list(std::vector<size_t> *DM_list, size_t nTimesamples, size_t nDMs){
	size_t max_timesamples, DMs_per_cycle, nRepeats, nRest, itemp;
	size_t free_mem, total_mem;
	size_t nElements = nTimesamples*nDMs;
	cudaMemGetInfo(&free_mem,&total_mem);
	printf("  Memory required by boxcar filters:%0.3f MB\n",(4.5*nElements*sizeof(float) + 2*nElements*sizeof(ushort))/(1024.0*1024) );
	printf("  Memory available:%0.3f MB \n", ((float) free_mem)/(1024.0*1024.0) );
	
	max_timesamples=(free_mem*0.95)/(5.5*sizeof(float) + 2*sizeof(ushort));
	DMs_per_cycle = max_timesamples/nTimesamples;
	itemp = (int) (DMs_per_cycle/THR_WARPS_PER_BLOCK);
	DMs_per_cycle = itemp*THR_WARPS_PER_BLOCK;
	nRepeats = nDMs/DMs_per_cycle;
	nRest = nDMs - nRepeats*DMs_per_cycle;
	
	for(int f=0; f<nRepeats; f++) DM_list->push_back(DMs_per_cycle);
	if(nRest>0) DM_list->push_back(nRest);
	
	if( DM_list->size() > 1 ) 
		printf("  SPS will run %zu batches each containing %d DM trials. Remainder %d DM trials\n", DM_list->size(), DMs_per_cycle, nRest);
	else 
		printf("  SPS will run %zu batch containing %d DM trials.\n", DM_list->size(), nRest);
}

/*
// Extend this to arbitrary size plans
void Create_PD_plan(std::vector<PulseDetection_plan> *PD_plan, std::vector<int> *BC_widths, int nDMs, int nTimesamples){
	int Elements_per_block, itemp, nRest;
	PulseDetection_plan PDmp;
	
	if(BC_widths->size()>0){
		PDmp.shift        = 0;
		PDmp.output_shift = 0;
		PDmp.startTaps    = 0;
		PDmp.iteration    = 0;
		
		PDmp.decimated_timesamples = nTimesamples;
		PDmp.dtm = (nTimesamples>>(PDmp.iteration+1));
		PDmp.dtm = PDmp.dtm - (PDmp.dtm&1);
		
		PDmp.nBoxcars = BC_widths->operator[](0);
		Elements_per_block = PD_NTHREADS*2 - PDmp.nBoxcars;
		itemp = PDmp.decimated_timesamples;
		PDmp.nBlocks = itemp/Elements_per_block;
		nRest = itemp - PDmp.nBlocks*Elements_per_block;
		if(nRest>0) PDmp.nBlocks++;
		PDmp.unprocessed_samples = PDmp.nBoxcars + 6;
		if(PDmp.decimated_timesamples<PDmp.unprocessed_samples) PDmp.nBlocks=0;
		PDmp.total_ut = PDmp.unprocessed_samples;
		
		
		PD_plan->push_back(PDmp);
		
		for(int f=1; f< (int) BC_widths->size(); f++){
			// These are based on previous values of PDmp
			PDmp.shift        = PDmp.nBoxcars/2;
			PDmp.output_shift = PDmp.output_shift + PDmp.decimated_timesamples;
			PDmp.startTaps    = PDmp.startTaps + PDmp.nBoxcars*(1<<PDmp.iteration);
			PDmp.iteration    = PDmp.iteration + 1;
			
			// Definition of new PDmp values
			PDmp.decimated_timesamples = PDmp.dtm;
			PDmp.dtm = (nTimesamples>>(PDmp.iteration+1));
			PDmp.dtm = PDmp.dtm - (PDmp.dtm&1);
			
			PDmp.nBoxcars = BC_widths->operator[](f);
			Elements_per_block=PD_NTHREADS*2 - PDmp.nBoxcars;
			itemp = PDmp.decimated_timesamples;
			PDmp.nBlocks = itemp/Elements_per_block;
			nRest = itemp - PDmp.nBlocks*Elements_per_block;
			if(nRest>0) PDmp.nBlocks++;
			PDmp.unprocessed_samples = PDmp.unprocessed_samples/2 + PDmp.nBoxcars + 6; //
			if(PDmp.decimated_timesamples<PDmp.unprocessed_samples) PDmp.nBlocks=0;
			PDmp.total_ut = PDmp.unprocessed_samples*(1<<PDmp.iteration);
			
			PD_plan->push_back(PDmp);
		}
	}
}
*/

/*
int Get_max_iteration(int max_boxcar_width, std::vector<int> *BC_widths, int *max_width_performed){
	int startTaps, iteration;
	
	startTaps = 0;
	iteration = 0;
	for(int f=0; f<(int) BC_widths->size(); f++){
		startTaps = startTaps + BC_widths->operator[](f)*(1<<f);
		if(startTaps>=max_boxcar_width) {
			iteration = f+1;
			break;
		}
	}
	
	if(max_boxcar_width>startTaps) {
		iteration=(int) BC_widths->size();
	}
	
	*max_width_performed=startTaps;
	return(iteration);
}
*/


void Copy_data_for_global_analysis(float *d_input, float **dedispersed_data, size_t nTimesamples, size_t nTS_per_chunk, size_t TS_shift, size_t nDMs){
	for(size_t f=0; f<nDMs; f++){
		checkCudaErrors( cudaMemcpy( &d_input[f*nTS_per_chunk], &((dedispersed_data[f])[TS_shift]), nTS_per_chunk*sizeof(float), cudaMemcpyHostToDevice));
	}
	printf("\n");
}


//-----------------------------------+-------------------------------------
//-----------------------------------+-------------------------------------

void analysis_global_GPU(float ***output_buffer, int nRanges, size_t nProcessedTimesamples, float *dm_low, float *dm_high, float *dm_step, int *inBin, int *ndms, float tsamp, float max_boxcar_width_in_sec, float sigma_cutoff, int candidate_algorithm, int enable_sps_baselinenoise, float OR_sigma_multiplier){
	//--------> Definition of SPDT boxcar plan
	int max_boxcar_width = (int) (max_boxcar_width_in_sec/tsamp);
	int t_BC_widths[10]={PD_MAXTAPS,16,16,16,8,8,8,8,8,8};
	std::vector<int> BC_widths(t_BC_widths,t_BC_widths+sizeof(t_BC_widths)/sizeof(int));
	
	//--------> Timers
	GpuTimer DDTR_timer, range_timer, chunk_timer, MSD_timer, timer;
	double DDTR_time=0, range_time=0, chunk_time=0, MSD_time=0, SPDT_time=0, PF_time=0;
	DDTR_timer.Start();
	
	//--------> Loop through dedispersion ranges
	for(int range=0; range<nRanges; range++){
		range_timer.Start();
		
		//--------> Number of time samples and DM trials for this range
		size_t nTimesamples = nProcessedTimesamples/inBin[range];
		size_t nDMs = ndms[range];
		printf("\n-------------------| Range %d |----------------------\n", range);
		printf("  nTimesamples: %zu; nDMs: %zu; inBin: %d\n", nTimesamples, nDMs, inBin[range]);
		
		//--------> Set up boxcars
		std::vector<int> h_boxcar_widths;
		int max_width_performed = 0;
		int max_iteration = Get_max_iteration(max_boxcar_width/inBin[range], &BC_widths, &max_width_performed);
		printf("  Selected iteration:%d; maximum boxcar width requested:%d; maximum boxcar width performed:%d;\n", max_iteration, max_boxcar_width/inBin[range], max_width_performed);
		Create_list_of_boxcar_widths(&h_boxcar_widths, &BC_widths, max_width_performed);
		
		//--------------------------------------------------------------------------
		//---------- Determine what can be processed in available memory
		size_t free_mem, total_mem;
		cudaMemGetInfo(&free_mem, &total_mem);
		size_t mem_per_timesample = (nDMs + ((size_t) ceil((float) nDMs/4.0))*3)*sizeof(float);
		size_t max_nTS_in_memory = free_mem/mem_per_timesample;
		printf("  Maximum number of time-samples which fit into memory: %zu\n", max_nTS_in_memory);
		//--------------------------------------------------------------------------<
		
		//--------------------------------------------------------------------------
		//---------- Separating data into chunks if necessary
		int nRepeats = ceil( (float) ((float) nTimesamples)/((float) max_nTS_in_memory));
		size_t chunk_size = (size_t) (nTimesamples/nRepeats);
		size_t tail_size = nTimesamples - chunk_size*(nRepeats-1);
	
		std::vector<size_t> nTS_per_chunk;
		for(int f=0; f<(nRepeats-1); f++){
			nTS_per_chunk.push_back(chunk_size);
		}
		nTS_per_chunk.push_back(tail_size);
		printf("  nRepeats: %d; chunk_size: %zu; tail_size: %zu\n", nRepeats, chunk_size, tail_size);
		printf("  Number of time-samples per iteration: "); for(int f=0; f<(int) nTS_per_chunk.size(); f++) printf("%zu ", nTS_per_chunk[f]); printf("\n");
		//--------------------------------------------------------------------------<
		
		
		//--------------------------------------------------------------------------
		//---------- MSD persistent variables
		size_t MSD_profile_size_in_bytes, MSD_DIT_profile_size_in_bytes, workarea_size_in_bytes;
		Get_MSD_plane_profile_memory_requirements(&MSD_profile_size_in_bytes, &MSD_DIT_profile_size_in_bytes, &workarea_size_in_bytes, nTS_per_chunk[0], nDMs, &h_boxcar_widths);
		float *d_MSD_interpolated;		cudaMalloc((void **) &d_MSD_interpolated, MSD_profile_size_in_bytes);
		float *d_MSD_DIT_continuous;	cudaMalloc((void **) &d_MSD_DIT_continuous, MSD_DIT_profile_size_in_bytes);
		float *d_input;					cudaMalloc((void **) &d_input, nTS_per_chunk[0]*nDMs*sizeof(float));
		
		float *h_peak_list;
		h_peak_list = (float *) malloc(nTS_per_chunk[0]*nDMs*sizeof(float));
		std::vector<FP> peak_pointers;
		std::vector<int> peak_sizes;
		//--------------------------------------------------------------------------<
		
		//--------------------------------------------------------------------------
		//---------- Loop through chunks		
		size_t TS_shift = 0;
		for(int chunk=0; chunk<(int) nTS_per_chunk.size(); chunk++){
			chunk_timer.Start();
			cudaMemset((void*) d_MSD_interpolated, 0, MSD_profile_size_in_bytes);
			Copy_data_for_global_analysis(d_input, output_buffer[range], nTimesamples, nTS_per_chunk[chunk], TS_shift, nDMs);
			checkCudaErrors(cudaGetLastError());
			
			//--------------------------------------------------------------------------
			//---------- Preparing plan for single pulse detection
			std::vector<size_t> DM_list;
			Create_DM_list(&DM_list, nTS_per_chunk[chunk], nDMs);
			size_t local_max_list_size = (DM_list[0]*nTimesamples)/4;
			//--------------------------------------------------------------------------<
			
			timer.Start();
			//--------------------------------------------------------------------------
			//---------- Calculation of MSD
			double dit_time, MSD_only_time;
			cudaMemset((void*) d_MSD_interpolated, 0, MSD_profile_size_in_bytes);
			
			cudaMemGetInfo(&free_mem, &total_mem);
			float *temporary_workarea;
			if(workarea_size_in_bytes>free_mem) {printf("Not enough memory!"); exit(2);}
			if ( cudaSuccess != cudaMalloc((void **) &temporary_workarea, workarea_size_in_bytes)) printf("Allocation error! MSD\n");
			
			MSD_plane_profile(d_MSD_interpolated, d_input, d_MSD_DIT_continuous, temporary_workarea, false, nTS_per_chunk[chunk], nDMs, &h_boxcar_widths, TS_shift*tsamp*inBin[range], dm_low[range], dm_high[range], OR_sigma_multiplier, enable_sps_baselinenoise, true, &MSD_time, &dit_time, &MSD_only_time);
			printf("    MSD time: Total: %f ms; DIT: %f ms; MSD: %f ms;\n", MSD_time, dit_time, MSD_only_time);
			
			cudaFree(temporary_workarea);
			//--------------------------------------------------------------------------<
			timer.Stop();
			MSD_time += timer.Elapsed();
			printf("MSD time: %f; Total: %f;\n", timer.Elapsed(), MSD_time);
			
			checkCudaErrors(cudaGetLastError());
			
			if(DM_list.size()>0){
				int peak_pos = 0;
				size_t DMs_per_cycle = DM_list[0];
				
				float *d_peak_list;
				if ( cudaSuccess != cudaMalloc((void**) &d_peak_list, sizeof(float)*DMs_per_cycle*nTimesamples)) printf("Allocation error! peaks\n");
				
				float *d_decimated;
				if ( cudaSuccess != cudaMalloc((void **) &d_decimated,  sizeof(float)*(((DMs_per_cycle*nTimesamples)/2)+PD_MAXTAPS) )) printf("Allocation error! dedispered\n");
				
				float *d_boxcar_values;
				if ( cudaSuccess != cudaMalloc((void **) &d_boxcar_values,  sizeof(float)*DMs_per_cycle*nTimesamples)) printf("Allocation error! boxcars\n");
				
				float *d_output_SNR;
				if ( cudaSuccess != cudaMalloc((void **) &d_output_SNR, sizeof(float)*2*DMs_per_cycle*nTimesamples)) printf("Allocation error! SNR\n");
				
				ushort *d_output_taps;
				if ( cudaSuccess != cudaMalloc((void **) &d_output_taps, sizeof(ushort)*2*DMs_per_cycle*nTimesamples)) printf("Allocation error! taps\n");
				
				int *gmem_peak_pos;
				cudaMalloc((void**) &gmem_peak_pos, 1*sizeof(int));
				cudaMemset((void*) gmem_peak_pos, 0, sizeof(int));

				size_t DM_shift = 0;
				for(int f=0; f<DM_list.size(); f++) {
					std::vector<PulseDetection_plan> PD_plan;
					Create_PD_plan(&PD_plan, &BC_widths, 1, nTS_per_chunk[chunk]);
					
					//-------------- SPDT
					timer.Start();
					SPDT_search_long_MSD_plane(&d_input[DM_shift*nTimesamples], d_boxcar_values, d_decimated, d_output_SNR, d_output_taps, d_MSD_interpolated, &PD_plan, max_iteration, nTimesamples, DM_list[f]);
					timer.Stop();
					SPDT_time += timer.Elapsed();
					printf("    SPDT took:%f ms\n", timer.Elapsed());
					//-------------- SPDT
					
					checkCudaErrors(cudaGetLastError());
					
					#ifdef GPU_ANALYSIS_DEBUG
					printf("    BC_shift:%d; DMs_per_cycle:%d; f*DMs_per_cycle:%d; max_iteration:%d;\n", DM_shift*nTimesamples, DM_list[f], DM_shift, max_iteration);
					#endif
					
					if(candidate_algorithm==1){
						//-------------- Thresholding
						timer.Start();
						THRESHOLD(d_output_SNR, d_output_taps, d_peak_list, gmem_peak_pos, sigma_cutoff, DM_list[f], nTimesamples, DM_shift, &PD_plan, max_iteration, local_max_list_size);
						timer.Stop();
						PF_time += timer.Elapsed();
						printf("    Thresholding took:%f ms\n", timer.Elapsed());
						//-------------- Thresholding
					}
					else {
						//-------------- Peak finding
						timer.Start();
						PEAK_FIND(d_output_SNR, d_output_taps, d_peak_list, DM_list[f], nTimesamples, sigma_cutoff, local_max_list_size, gmem_peak_pos, DM_shift, &PD_plan, max_iteration);
						timer.Stop();
						PF_time = timer.Elapsed();
						printf("    Peak finding took:%f ms\n", timer.Elapsed());
						//-------------- Peak finding
					}
					
					checkCudaErrors(cudaGetLastError());
					
					checkCudaErrors(cudaMemcpy(&peak_pos, gmem_peak_pos, sizeof(int), cudaMemcpyDeviceToHost));
					printf("    peak_pos:%d; local_max:%d;\n", peak_pos, local_max_list_size);
					if( peak_pos>=local_max_list_size ) {
						printf("    Maximum list size reached! Increase list size or increase sigma cutoff.\n");
						peak_pos=local_max_list_size;
					}
					checkCudaErrors(cudaMemcpy(h_peak_list, d_peak_list, peak_pos*4*sizeof(float), cudaMemcpyDeviceToHost));
					checkCudaErrors(cudaGetLastError());
					
					peak_pointers.push_back(NULL);
					int index = peak_pointers.size()-1;
					peak_pointers[index] = new float[peak_pos*4];
					//------------------------> Output
					peak_sizes.push_back(peak_pos);
					for (int count = 0; count < peak_pos; count++){
						peak_pointers[index][4*count]     = h_peak_list[4*count]*dm_step[range] + dm_low[range];
						peak_pointers[index][4*count + 1] = h_peak_list[4*count + 1]*tsamp + TS_shift*tsamp*inBin[range];
						peak_pointers[index][4*count + 2] = h_peak_list[4*count + 2];
						peak_pointers[index][4*count + 3] = h_peak_list[4*count + 3]*inBin[range];
					}
					
					// copy to vector of pointers with scaling

					DM_shift = DM_shift + DM_list[f];
					checkCudaErrors(cudaMemset((void*) gmem_peak_pos, 0, sizeof(int)));
					peak_pos = 0;
					checkCudaErrors(cudaGetLastError());
				}
			
				checkCudaErrors(cudaFree(d_peak_list));
				checkCudaErrors(cudaFree(d_decimated));
				checkCudaErrors(cudaFree(d_boxcar_values));
				checkCudaErrors(cudaFree(d_output_SNR));
				checkCudaErrors(cudaFree(d_output_taps));
				checkCudaErrors(cudaFree(gmem_peak_pos));
				checkCudaErrors(cudaGetLastError());
			}
			else printf("Error not enough memory to search for pulses\n");
			
			TS_shift = TS_shift + nTS_per_chunk[chunk];
			checkCudaErrors(cudaGetLastError());
			chunk_timer.Stop();
			chunk_time += chunk_timer.Elapsed();
			printf("Chunk time: %f; Total: %f\n", chunk_timer.Elapsed(), chunk_time);
			checkCudaErrors(cudaGetLastError());
		} // Chunks from ranges
		free(h_peak_list);
		checkCudaErrors(cudaGetLastError());
		//--------------------------------------------------------------------------
		//---------- Exporting peaks
		FILE *fp_out;
		char filename[200];
		
		if(candidate_algorithm==1) 
			sprintf(filename, "SPS_T-t_%.2f-dm_%.2f-%.2f.dat", 0, dm_low[range], dm_high[range]);
		else 
			sprintf(filename, "SPS_P-t_%.2f-dm_%.2f-%.2f.dat", 0, dm_low[range], dm_high[range]);
		
		if (( fp_out = fopen(filename, "wb") ) == NULL)	{
			fprintf(stderr, "Error opening output file!\n");
			exit(0);
		}
		
		for(size_t f=0; f<peak_pointers.size(); f++){
			fwrite(peak_pointers[f], peak_sizes[f]*sizeof(float), 4, fp_out);
		}
		
		fclose(fp_out);
		//--------------------------------------------------------------------------<
		
		//--------------------------------------------------------------------------
		//---------- Deallocations
		cudaFree(d_MSD_interpolated);
		cudaFree(d_MSD_DIT_continuous);
		cudaFree(d_input);
		checkCudaErrors(cudaGetLastError());
		
		for(size_t f=0; f<peak_pointers.size(); f++){
			delete[] peak_pointers[f];
		}
		peak_pointers.clear();
		peak_sizes.clear();
		//--------------------------------------------------------------------------<

		range_timer.Stop();
		range_time += range_timer.Elapsed();
		printf("Range time: %f; Total: %f\n", range_timer.Elapsed(), range_time);
		checkCudaErrors(cudaGetLastError());
	} //Ranges
	
	DDTR_timer.Stop();
	DDTR_time = DDTR_timer.Elapsed();
	printf("Total DDTR plan time: %f\n", DDTR_time);	
	checkCudaErrors(cudaGetLastError());
}
