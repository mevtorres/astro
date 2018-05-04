#ifndef ASTROACCELERATE_GPUANALYSISGLOBAL_H_
#define ASTROACCELERATE_GPUANALYSISGLOBAL_H_

void analysis_global_GPU(float ***output_buffer, int nRanges, size_t nProcessedTimesamples, float *dm_low, float *dm_high, float *dm_step, int *inBin, int *ndms, float tsamp, float max_boxcar_width_in_sec, float sigma_cutoff, int candidate_algorithm, int enable_sps_baselinenoise, float OR_sigma_multiplier);

#endif
