#ifndef ASTRO_ACCELERATE_AA_PERMITTED_PIPELINES_3_HPP
#define ASTRO_ACCELERATE_AA_PERMITTED_PIPELINES_3_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <stdio.h>
#include "aa_compute.hpp"
#include "aa_ddtr_strategy.hpp"
#include "aa_ddtr_plan.hpp"
#include "aa_analysis_plan.hpp"
#include "aa_analysis_strategy.hpp"
#include "aa_periodicity_strategy.hpp"

#include "aa_filterbank_metadata.hpp"
#include "aa_device_load_data.hpp"
#include "aa_bin_gpu.hpp"
#include "aa_zero_dm.hpp"
#include "aa_zero_dm_outliers.hpp"
#include "aa_corner_turn.hpp"
#include "aa_device_rfi.hpp"
#include "aa_dedisperse.hpp"

#include "aa_gpu_timer.hpp"

#include "aa_device_analysis.hpp"
#include "aa_device_periods.hpp"
#include "aa_pipeline_runner.hpp"

#include "aa_gpu_timer.hpp"

namespace astroaccelerate {

  /**
   * \class aa_permitted_pipelines_3 aa_permitted_pipelines_3.hpp "include/aa_permitted_pipelines_3.hpp"
   * \brief Templated class to run dedispersion and analysis and periodicity.
   * \details The class is templated over the zero_dm_type (aa_compute::module_option::zero_dm or aa_compute::module_option::zero_dm_with_outliers).
   * \author Cees Carels.
   * \date 28 November 2018.
   */

  template<aa_compute::module_option zero_dm_type, bool enable_old_rfi>
  class aa_permitted_pipelines_3 : public aa_pipeline_runner {
  public:
    aa_permitted_pipelines_3(const aa_ddtr_strategy &ddtr_strategy,
			     const aa_analysis_strategy &analysis_strategy,
			     const aa_periodicity_strategy &periodicity_strategy,
			     unsigned short const*const input_buffer) {
      
    }
    
    ~aa_permitted_pipelines_3() {
      //Only call cleanup if memory had been allocated during setup,
      //and if the memory was not already cleaned up usingthe cleanup method.
      if(memory_allocated && !memory_cleanup) {
	cleanup();
      }      
    }

    aa_permitted_pipelines_3(const aa_permitted_pipelines_3 &) = delete;

    /** \brief Method to setup and allocate memory for the pipeline containers. */
    bool setup() override {
      if(!memory_allocated) {
	return set_data();
      }

      if(memory_allocated) {
	return true;
      }
      
      return false;
    }

    /** \brief Override base class next() method to process next time chunk. */
    bool next() override {
      if(memory_allocated) {
	return run_pipeline();
      }

      return false;
    }
    
    /** \brief De-allocate memory for this pipeline instance. */
    bool cleanup() {
      if(memory_allocated && !memory_cleanup) {
	std::cout << "Doing cleanup" << std::endl;
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(m_d_MSD_workarea);
        cudaFree(m_d_MSD_output_taps);
	cudaFree(m_d_MSD_interpolated);

	size_t t_processed_size = m_ddtr_strategy.t_processed().size();
	for(size_t i = 0; i < t_processed_size; i++) {
	  free(t_processed[i]);
	}
	free(t_processed);

	memory_cleanup = true;
      }
      return true;
    }
  private:
    float              ***m_output_buffer;
    int                **t_processed;
    aa_ddtr_strategy        m_ddtr_strategy;
    aa_analysis_strategy    m_analysis_strategy;
    aa_periodicity_strategy m_periodicity_strategy;
    unsigned short     const*const m_input_buffer;
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

    bool memory_allocated;
    bool memory_cleanup;
    bool periodicity_did_run;
    
    //Loop counter variables
    int t;
    aa_gpu_timer       m_timer;
    
    float  *m_d_MSD_workarea         = NULL;
    float  *m_d_MSD_interpolated     = NULL;
    ushort *m_d_MSD_output_taps      = NULL;
    
    /** \brief Allocate the GPU memory needed for dedispersion. */
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
    
    /**
     * \brief Allocate memory for MSD.
     */
    void allocate_memory_MSD(float **const d_MSD_workarea, unsigned short **const d_MSD_output_taps, float **const d_MSD_interpolated,
			     const unsigned long int &MSD_maxtimesamples, const size_t &MSD_profile_size) {
      checkCudaErrors(cudaMalloc((void **) d_MSD_workarea,        MSD_maxtimesamples*5.5*sizeof(float)));
      checkCudaErrors(cudaMalloc((void **) &(*d_MSD_output_taps), sizeof(ushort)*2*MSD_maxtimesamples));
      checkCudaErrors(cudaMalloc((void **) d_MSD_interpolated,    sizeof(float)*MSD_profile_size));
    }

    /**
     * \brief Allocate a 3D array that is an output buffer that stores dedispersed array data.
     * \details This array is used by periodicity.
     */
    void allocate_memory_cpu_output() {
      size_t estimate_outputbuffer_size = 0;
      size_t outputsize = 0;
      const size_t range = m_ddtr_strategy.range();
      const int *ndms = m_ddtr_strategy.ndms_data();
      
      for(size_t i = 0; i < range; i++) {
        for(int j = 0; j < m_ddtr_strategy.num_tchunks(); j++) {
	  estimate_outputbuffer_size += (size_t)(t_processed[i][j]*sizeof(float)*ndms[i]);
        }
      }
      
      outputsize = 0;
      m_output_buffer = (float ***) malloc(range * sizeof(float **));
      for(size_t i = 0; i < range; i++) {
        int total_samps = 0;
        for(int k = 0; k < num_tchunks; k++) {
	  total_samps += t_processed[i][k];
        }
        m_output_buffer[i] = (float **) malloc(ndms[i] * sizeof(float *));
        for (int j = 0; j < ndms[i]; j++) {
	  m_output_buffer[i][j] = (float *) malloc(( total_samps ) * sizeof(float));
        }
        outputsize += ( total_samps ) * ndms[i] * sizeof(float);
      }
    }
    
    /** \brief Method that allocates all memory for this pipeline. */
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

      //Allocate GPU memory for dedispersion
      allocate_memory_gpu(maxshift, max_ndms, nchans, t_processed, &d_input, &d_output);
      //Allocate GPU memory for SPS (i.e. analysis)
      allocate_memory_MSD(&m_d_MSD_workarea, &m_d_MSD_output_taps, &m_d_MSD_interpolated, m_analysis_strategy.MSD_data_info(), m_analysis_strategy.MSD_profile_size_in_bytes());
      //Allocate memory for CPU output for periodicity
      allocate_memory_cpu_output();
      
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
      memory_allocated = true;
      return true;
    }

    /** \brief Transfer data from the device to the host. */
    inline void save_data_offset(float *device_pointer, int device_offset, float *host_pointer, int host_offset, size_t size) {
      cudaMemcpy(host_pointer + host_offset, device_pointer + device_offset, size, cudaMemcpyDeviceToHost);
    }

    /** \brief Transfer data from the device to the host. */
    inline void save_data(float *const device_pointer, float *const host_pointer, const size_t &size) {
      cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost);
    }

    /**
     * \brief Run the pipeline by processing the next time chunk of data.
     * \details Process any flags for dumping output or providing it back to the user.
     * \returns A boolean to indicate whether further time chunks are available to process (true) or not (false).
     */
    bool run_pipeline() {
      printf("NOTICE: Pipeline start/resume run_pipeline_3.\n");
      if(t >= num_tchunks) {
	return periodicity();
	m_timer.Stop();
        float time = m_timer.Elapsed() / 1000;
        printf("\n\n === OVERALL DEDISPERSION THROUGHPUT INCLUDING SYNCS AND DATA TRANSFERS ===\n");
        printf("\n(Performed Brute-Force Dedispersion: %g (GPU estimate)", time);
        printf("\nAmount of telescope time processed: %f", tstart_local);
        printf("\nNumber of samples processed: %ld", inc);
        printf("\nReal-time speedup factor: %lf\n", ( tstart_local ) / time);
	return false; // In this case, there are no more chunks to process.
      }
      else if(t == 0) {
	m_timer.Start();
      }
      printf("\nNOTICE: t_processed:\t%d, %d", t_processed[0][t], t);
      
      const int *ndms = m_ddtr_strategy.ndms_data();
      
      checkCudaErrors(cudaGetLastError());
      load_data(-1, inBin.data(), d_input, &m_input_buffer[(long int) ( inc * nchans )], t_processed[0][t], maxshift, nchans, dmshifts);
      checkCudaErrors(cudaGetLastError());

      if(zero_dm_type == aa_compute::module_option::zero_dm) {
	zero_dm(d_input, nchans, t_processed[0][t]+maxshift, nbits);
      }

      checkCudaErrors(cudaGetLastError());

      if(zero_dm_type == aa_compute::module_option::zero_dm_with_outliers) {
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

	checkCudaErrors(cudaGetLastError());

	//Add analysis
	unsigned int *h_peak_list_DM;
	unsigned int *h_peak_list_TS;
	float        *h_peak_list_SNR;
	unsigned int *h_peak_list_BW;
	size_t        max_peak_size;
	size_t        peak_pos;
	max_peak_size   = (size_t) ( ndms[dm_range]*t_processed[dm_range][t]/2 );
	h_peak_list_DM  = (unsigned int*) malloc(max_peak_size*sizeof(unsigned int));
	h_peak_list_TS  = (unsigned int*) malloc(max_peak_size*sizeof(unsigned int));
	h_peak_list_SNR = (float*) malloc(max_peak_size*sizeof(float));
	h_peak_list_BW  = (unsigned int*) malloc(max_peak_size*sizeof(unsigned int));
	peak_pos=0;
	const bool dump_to_disk	= false;
        const bool dump_to_user	= true;
	analysis_output output;
	analysis_GPU(h_peak_list_DM,
		     h_peak_list_TS,
		     h_peak_list_SNR,
		     h_peak_list_BW,
		     &peak_pos,
		     max_peak_size,
		     dm_range,
		     tstart_local,
		     t_processed[dm_range][t],
		     inBin[dm_range],
		     &maxshift,
		     max_ndms,
		     ndms,
		     m_analysis_strategy.sigma_cutoff(),
		     m_analysis_strategy.sigma_constant(),
		     m_analysis_strategy.max_boxcar_width_in_sec(),
		     d_output,
		     dm_low.data(),
		     dm_high.data(),
		     dm_step.data(),
		     tsamp,
		     m_analysis_strategy.candidate_algorithm(),
		     m_d_MSD_workarea,
		     m_d_MSD_output_taps,
		     m_d_MSD_interpolated,
		     m_analysis_strategy.MSD_data_info(),
		     m_analysis_strategy.enable_sps_baseline_noise(),
		     dump_to_disk,
		     dump_to_user,
		     output);
	
	free(h_peak_list_DM);
	free(h_peak_list_TS);
	free(h_peak_list_SNR);
	free(h_peak_list_BW);
	
	oldBin = inBin[dm_range];
      }
      
      inc = inc + t_processed[0][t];
      printf("\nNOTICE: INC:\t%ld\n", inc);
      tstart_local = ( tsamp_original * inc );
      tsamp = tsamp_original;
      maxshift = maxshift_original;

      ++t;
      printf("NOTICE: Pipeline ended run_pipeline_3 over chunk %d / %d.\n", t, num_tchunks);
      return true;
    }

    bool periodicity() {
      if(periodicity_did_run) return false;
      cleanup();
      aa_gpu_timer timer;
      timer.Start();
      const int *ndms =	m_ddtr_strategy.ndms_data();
      GPU_periodicity(m_ddtr_strategy.range(),
		      m_ddtr_strategy.metadata().nsamples(),
		      max_ndms,
		      inc,
		      m_periodicity_strategy.sigma_cutoff(),
		      m_output_buffer,
		      ndms,
		      inBin.data(),
		      dm_low.data(),
		      dm_high.data(),
		      dm_step.data(),
		      tsamp_original,
		      m_periodicity_strategy.nHarmonics(),
		      m_analysis_strategy.candidate_algorithm(),
		      m_analysis_strategy.enable_sps_baseline_noise(),
		      m_analysis_strategy.sigma_constant());
      
      timer.Stop();
      float time = timer.Elapsed()/1000;
      printf("\n\n === OVERALL PERIODICITY THROUGHPUT INCLUDING SYNCS AND DATA TRANSFERS ===\n");
      
      printf("\nPerformed Periodicity Location: %f (GPU estimate)", time);
      printf("\nAmount of telescope time processed: %f", tstart_local);
      printf("\nNumber of samples processed: %ld", inc);
      printf("\nReal-time speedup factor: %f\n", ( tstart_local ) / ( time ));
      periodicity_did_run = true;

      for(size_t i = 0; i < range; i++) {
	for(int j = 0; j < ndms[i]; j++) {
	  free(m_output_buffer[i][j]);
	}
	free(m_output_buffer[i]);
      }
      free(m_output_buffer);
      
      return true;
    }
  };

  template<> inline aa_permitted_pipelines_3<aa_compute::module_option::zero_dm, false>::aa_permitted_pipelines_3(const aa_ddtr_strategy &ddtr_strategy,
														  const aa_analysis_strategy &analysis_strategy,
														  const aa_periodicity_strategy &periodicity_strategy,
														  unsigned short const*const input_buffer) : m_ddtr_strategy(ddtr_strategy),
																			     m_analysis_strategy(analysis_strategy),
																			     m_periodicity_strategy(periodicity_strategy),
																			     m_input_buffer(input_buffer),
																			     memory_allocated(false),
																			     memory_cleanup(false),
																			     periodicity_did_run(false),
																			     t(0),
																			     m_d_MSD_workarea(NULL),
																			     m_d_MSD_interpolated(NULL),
																			     m_d_MSD_output_taps(NULL) {
    
    
  }
  
  template<> inline aa_permitted_pipelines_3<aa_compute::module_option::zero_dm, true>::aa_permitted_pipelines_3(const aa_ddtr_strategy &ddtr_strategy,
														 const aa_analysis_strategy &analysis_strategy,
														 const aa_periodicity_strategy &periodicity_strategy,
														 unsigned short const*const input_buffer) : m_ddtr_strategy(ddtr_strategy),
																			    m_analysis_strategy(analysis_strategy),
																			    m_periodicity_strategy(periodicity_strategy),
																			    m_input_buffer(input_buffer),
																			    memory_allocated(false),
																			    memory_cleanup(false),
																			    periodicity_did_run(false),
																			    t(0),
																			    m_d_MSD_workarea(NULL),
																			    m_d_MSD_interpolated(NULL),
																			    m_d_MSD_output_taps(NULL) {
    
  }
  
  template<> inline aa_permitted_pipelines_3<aa_compute::module_option::zero_dm_with_outliers, false>::aa_permitted_pipelines_3(const aa_ddtr_strategy &ddtr_strategy,
																const aa_analysis_strategy &analysis_strategy,
																const aa_periodicity_strategy &periodicity_strategy,
																unsigned short const*const input_buffer) : m_ddtr_strategy(ddtr_strategy),
																					   m_analysis_strategy(analysis_strategy),
																					   m_periodicity_strategy(periodicity_strategy),
																					   m_input_buffer(input_buffer),
																					   memory_allocated(false),
																					   memory_cleanup(false),
																					   periodicity_did_run(false),
																					   t(0),
																					   m_d_MSD_workarea(NULL),
																					   m_d_MSD_interpolated(NULL),
																					   m_d_MSD_output_taps(NULL) {
    
    
  }
  
  template<> inline aa_permitted_pipelines_3<aa_compute::module_option::zero_dm_with_outliers, true>::aa_permitted_pipelines_3(const aa_ddtr_strategy &ddtr_strategy,
															       const aa_analysis_strategy &analysis_strategy,
															       const aa_periodicity_strategy &periodicity_strategy,
															       unsigned short const*const input_buffer) : m_ddtr_strategy(ddtr_strategy),
																					  m_analysis_strategy(analysis_strategy),
																					  m_periodicity_strategy(periodicity_strategy),
																					  m_input_buffer(input_buffer),
																					  memory_allocated(false),
																					  memory_cleanup(false),
																					  periodicity_did_run(false),
																					  t(0),
																					  m_d_MSD_workarea(NULL),
																					  m_d_MSD_interpolated(NULL),
																					  m_d_MSD_output_taps(NULL) {
    
  }

  template<> inline aa_permitted_pipelines_3<aa_compute::module_option::empty, true>::aa_permitted_pipelines_3(const aa_ddtr_strategy &ddtr_strategy,
													       const aa_analysis_strategy &analysis_strategy,
													       const aa_periodicity_strategy &periodicity_strategy,
													       unsigned short const*const input_buffer) : m_ddtr_strategy(ddtr_strategy),
																			  m_analysis_strategy(analysis_strategy),
																			  m_periodicity_strategy(periodicity_strategy),
																			  m_input_buffer(input_buffer),
																			  memory_allocated(false),
																			  memory_cleanup(false),
																			  periodicity_did_run(false),
																			  t(0),
																			  m_d_MSD_workarea(NULL),
																			  m_d_MSD_interpolated(NULL),
																			  m_d_MSD_output_taps(NULL) {
    
  }

  template<> inline aa_permitted_pipelines_3<aa_compute::module_option::empty, false>::aa_permitted_pipelines_3(const aa_ddtr_strategy &ddtr_strategy,
														const aa_analysis_strategy &analysis_strategy,
														const aa_periodicity_strategy &periodicity_strategy,
														unsigned short const*const input_buffer) : m_ddtr_strategy(ddtr_strategy),
																			   m_analysis_strategy(analysis_strategy),
																			   m_periodicity_strategy(periodicity_strategy),
																			   m_input_buffer(input_buffer),
																			   memory_allocated(false),
																			   memory_cleanup(false),
																			   periodicity_did_run(false),
																			   t(0),
																			   m_d_MSD_workarea(NULL),
																			   m_d_MSD_interpolated(NULL),
																			   m_d_MSD_output_taps(NULL) {
    
  }
  
} // namespace astroaccelerate
  
#endif // ASTRO_ACCELERATE_AA_PERMITTED_PIPELINES_3_HPP
