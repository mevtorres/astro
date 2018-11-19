//
//  aa_permitted_pipelines_2.hpp
//  aapipeline
//
//  Created by Cees Carels on Monday 19/11/2018.
//  Copyright © 2018 Astro-Accelerate. All rights reserved.
//

#ifndef ASTRO_ACCELERATE_AA_PERMITTED_PIPELINES_2_HPP
#define ASTRO_ACCELERATE_AA_PERMITTED_PIPELINES_2_HPP


#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <stdio.h>
#include "aa_compute.hpp"
#include "aa_ddtr_strategy.hpp"
#include "aa_ddtr_plan.hpp"
#include "aa_filterbank_metadata.hpp"
#include "aa_device_load_data.hpp"
#include "aa_bin_gpu.hpp"
#include "aa_zero_dm.hpp"
#include "aa_zero_dm_outliers.hpp"
#include "aa_corner_turn.hpp"
#include "device_rfi.hpp"
#include "aa_dedisperse.hpp"

namespace astroaccelerate {
  template<aa_compute::modules zero_dm_type, bool enable_old_rfi>
  class aa_permitted_pipelines_2 {
  public:
    aa_permitted_pipelines_2(const aa_ddtr_strategy &ddtr_strategy,
			     unsigned short *const input_buffer);
    
    ~aa_permitted_pipelines_2() {
      if(!memory_cleanup) {
	cleanup();
      }
    }

    aa_permitted_pipelines_2(const aa_permitted_pipelines_1 &) = delete;

    bool setup() {
      return set_data();
    }

    bool next(std::vector<float> &output_buffer) {
      return run_pipeline(output_buffer, true);
    }
    
    bool cleanup() {
      if(!memory_cleanup) {
	cudaFree(d_input);
	cudaFree(d_output);

	size_t t_processed_size = m_ddtr_strategy.t_processed().size();
	for(size_t i = 0; i < t_processed_size; i++) {
	  free(t_processed[i]);
	}
	free(t_processed);
      }
      return true;
    }
  private:
    int                **t_processed;
    aa_ddtr_strategy   m_ddtr_strategy;
    unsigned short     *m_input_buffer;
    int                num_tchunks;
    std::vector<float> dm_shifts;
    float              *dmshifts;
    int                maxshift;
    int                max_ndms;
    int                nchans;
    int                nbits;
    int                enable_zero_dm;
    int                enable_zero_dm_with_outliers;
    int                failsafe;
    long int           inc;
    float              tsamp;
    float              tsamp_original;
    int                maxshift_original;
    size_t             range;
    float              tstart_local;

    unsigned short     *d_input;
    float              *d_output;

    std::vector<float> dm_low;
    std::vector<float> dm_high;
    std::vector<float> dm_step;
    std::vector<int>   inBin;

    bool memory_cleanup;
    
    //Loop counter variables
    int t;

    void allocate_memory_gpu(const int &maxshift, const int &max_ndms, const int &nchans, int **const t_processed, unsigned short **const d_input, float **const d_output) {

      int time_samps = t_processed[0][0] + maxshift;
      printf("\n\n\n%d\n\n\n", time_samps);
      size_t gpu_inputsize = (size_t) time_samps * (size_t) nchans * sizeof(unsigned short);

      checkCudaErrors( cudaMalloc((void **) d_input, gpu_inputsize) );

      size_t gpu_outputsize = 0;
      if (nchans < max_ndms) {
	gpu_outputsize = (size_t)time_samps * (size_t)max_ndms * sizeof(float);
      }
      else {
	gpu_outputsize = (size_t)time_samps * (size_t)nchans * sizeof(float);
      }

      checkCudaErrors( cudaMalloc((void **) d_output, gpu_outputsize) );
      cudaMemset(*d_output, 0, gpu_outputsize);

    }

    bool set_data() {
      num_tchunks = m_ddtr_strategy.num_tchunks();
      size_t t_processed_size = m_ddtr_strategy.t_processed().size();

      t_processed = new int*[t_processed_size];
      for(size_t i = 0; i < t_processed_size; i++) {
	t_processed[i] = new int[m_ddtr_strategy.t_processed().at(i).size()];
      }

      for(size_t i = 0; i < t_processed_size; i++) {
	for(size_t j = 0; j < m_ddtr_strategy.t_processed().at(i).size(); j++) {
	  t_processed[i][j] = m_ddtr_strategy.t_processed().at(i).at(j);
	}
      }

      dm_shifts                       = m_ddtr_strategy.dmshifts();
      dmshifts                        = dm_shifts.data();
      maxshift                        = m_ddtr_strategy.maxshift();
      max_ndms                        = m_ddtr_strategy.max_ndms();
      nchans                          = m_ddtr_strategy.metadata().nchans();
      nbits                           = m_ddtr_strategy.metadata().nbits();
      enable_zero_dm                  = 0;
      enable_zero_dm_with_outliers    = 0;
      failsafe                        = 0;
      inc                             = 0;
      tsamp                           = m_ddtr_strategy.metadata().tsamp();
      tsamp_original                  = tsamp;
      maxshift_original               = maxshift;
      range                           = m_ddtr_strategy.range();
      tstart_local                    = 0.0;

      //Allocate GPU memory
      d_input                         = NULL;
      d_output                        = NULL;

      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
      //Put the dm low, high, step struct contents into separate arrays again.
      //This is needed so that the kernel wrapper functions don't need to be modified.
      dm_low.resize(m_ddtr_strategy.range());
      dm_high.resize(m_ddtr_strategy.range());
      dm_step.resize(m_ddtr_strategy.range());
      inBin.resize(m_ddtr_strategy.range());
      for(size_t i = 0; i < m_ddtr_strategy.range(); i++) {
	dm_low[i]   = m_ddtr_strategy.dm(i).low;
	dm_high[i]  = m_ddtr_strategy.dm(i).high;
	dm_step[i]  = m_ddtr_strategy.dm(i).step;
	inBin[i]    = m_ddtr_strategy.dm(i).inBin;
      }
      return true;
    }

    inline void save_data_offset(float *device_pointer, int device_offset, float *host_pointer, int host_offset, size_t size) {
      cudaMemcpy(host_pointer + host_offset, device_pointer + device_offset, size, cudaMemcpyDeviceToHost);
    }

    bool run_pipeline(std::vector<float> &output_buffer, const bool dump_ddtr_output) {
      printf("NOTICE: Pipeline start/resume run_pipeline_1.\n");
      if(t >= num_tchunks) return false;//In this case, there are no more chunks to process.
      printf("\nNOTICE: t_processed:\t%d, %d", t_processed[0][t], t);

      const int *ndms = m_ddtr_strategy.ndms_data();

      checkCudaErrors(cudaGetLastError());
      load_data(-1, inBin.data(), d_input, &m_input_buffer[(long int) ( inc * nchans )], t_processed[0][t], maxshift, nchans, dmshifts);
      checkCudaErrors(cudaGetLastError());

      if(zero_dm_type == aa_compute::modules::zero_dm) {
	zero_dm(d_input, nchans, t_processed[0][t]+maxshift, nbits);
      }

      checkCudaErrors(cudaGetLastError());


      if(zero_dm_type == aa_compute::modules::zero_dm_with_outliers) {
	zero_dm_outliers(d_input, nchans, t_processed[0][t]+maxshift);
      }

      checkCudaErrors(cudaGetLastError());

      corner_turn(d_input, d_output, nchans, t_processed[0][t] + maxshift);

      checkCudaErrors(cudaGetLastError());

      if(enable_old_rfi) {
	printf("\nPerforming old GPU rfi...");
	rfi_gpu(d_input, nchans, t_processed[0][t]+maxshift);
      }

      checkCudaErrors(cudaGetLastError());

      int oldBin = 1;
      for(size_t dm_range = 0; dm_range < range; dm_range++) {
	printf("\n\nNOTICE: %f\t%f\t%f\t%d\n", m_ddtr_strategy.dm(dm_range).low, m_ddtr_strategy.dm(dm_range).high, m_ddtr_strategy.dm(dm_range).step, m_ddtr_strategy.ndms(dm_range));
	printf("\nAmount of telescope time processed: %f\n", tstart_local);

	maxshift = maxshift_original / inBin[dm_range];

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	load_data(dm_range, inBin.data(), d_input, &m_input_buffer[(long int) ( inc * nchans )], t_processed[dm_range][t], maxshift, nchans, dmshifts);

	checkCudaErrors(cudaGetLastError());


	if (inBin[dm_range] > oldBin) {
	  bin_gpu(d_input, d_output, nchans, t_processed[dm_range - 1][t] + maxshift * inBin[dm_range]);
	  ( tsamp ) = ( tsamp ) * 2.0f;
	}

	checkCudaErrors(cudaGetLastError());

	dedisperse(dm_range, t_processed[dm_range][t], inBin.data(), dmshifts, d_input, d_output, nchans, &tsamp, dm_low.data(), dm_step.data(), ndms, nbits, failsafe);

	if(dump_ddtr_output) {
	  //Resize vector to contain the output array
	  size_t total_samps = 0;
	  for (int k = 0; k < num_tchunks; k++) {
	    total_samps += t_processed[dm_range][k];
	  }
	  output_buffer.resize(total_samps);
	  for (int k = 0; k < ndms[dm_range]; k++) {
	    save_data_offset(d_output, k * t_processed[dm_range][t], output_buffer.data(), inc / inBin[dm_range], sizeof(float) * t_processed[dm_range][t]);
	  }
	}
	checkCudaErrors(cudaGetLastError());

	oldBin = inBin[dm_range];
      }

      inc = inc + t_processed[0][t];
      printf("\nNOTICE: INC:\t%ld\n", inc);
      tstart_local = ( tsamp_original * inc );
      tsamp = tsamp_original;
      maxshift = maxshift_original;

      ++t;
      printf("NOTICE: Pipeline ended run_pipeline_1 over chunk %d / %d.\n", t, num_tchunks);
      return true;
    }    
  };

  template<> inline aa_permitted_pipelines_2<aa_compute::modules::zero_dm, false>::aa_permitted_pipelines_2(const aa_ddtr_strategy &ddtr_strategy,
													    unsigned short *const input_buffer) :    m_ddtr_strategy(ddtr_strategy),
																		     m_input_buffer(input_buffer),
																		     memory_cleanup(false),
																		     t(0) {
    
  }

  template<> inline aa_permitted_pipelines_2<aa_compute::modules::zero_dm, true>::aa_permitted_pipelines_2(const aa_ddtr_strategy &ddtr_strategy,
													   unsigned short *const input_buffer) :    m_ddtr_strategy(ddtr_strategy),
																		    m_input_buffer(input_buffer),
																		    memory_cleanup(false),
																		    t(0) {
    
  }
  
  template<> inline aa_permitted_pipelines_2<aa_compute::modules::zero_dm_with_outliers, false>::aa_permitted_pipelines_2(const aa_ddtr_strategy &ddtr_strategy,
															  unsigned short *const input_buffer) :    m_ddtr_strategy(ddtr_strategy),
																				   m_input_buffer(input_buffer),
																				   memory_cleanup(false),
																				   t(0) {
    
  }
  
  template<> inline aa_permitted_pipelines_2<aa_compute::modules::zero_dm_with_outliers, true>::aa_permitted_pipelines_2(const aa_ddtr_strategy &ddtr_strategy,
															 unsigned short *const input_buffer) :    m_ddtr_strategy(ddtr_strategy),
																				  m_input_buffer(input_buffer),
																				  memory_cleanup(false),
																				  t(0) {
    
  }
  
}//namespace astroaccelerate
  
#endif /* ASTRO_ACCELERATE_AA_PERMITTED_PIPELINES_2_HPP */