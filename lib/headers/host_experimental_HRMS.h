#ifndef ASTROACCELERATE_EX_HRMS_H_
#define ASTROACCELERATE_EX_HRMS_H_

void Do_PHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff);

void Do_SHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff);

void Do_EHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff);

void Do_LEHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff);	

#endif
