#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>

class General_HRMS {
public:
	int nHarmonics;
	size_t nSamples;
	size_t nDMs;
	float mean;
	float sd;
	double *mean_array;
	double *stdev_array;
	
	//---------------------------------------------------------------------------------
	//-------> Kahan MSD
	void d_kahan_summation(float *signal, size_t nDMs, size_t nTimesamples, size_t offset, float *result, float *error){
		double sum;
		double sum_error;
		double a,b;
		
		sum=0;
		sum_error=0;
		for(size_t d=0;d<nDMs; d++){
			for(size_t s=0; s<(nTimesamples-offset); s++){
				a=signal[(size_t) (d*nTimesamples + s)]-sum_error;
				b=sum+a;
				sum_error=(b-sum);
				sum_error=sum_error-a;
				sum=b;
			}
		}
		*result=sum;
		*error=sum_error;
	}

	void d_kahan_sd(float *signal, size_t nDMs, size_t nTimesamples, size_t offset, double mean, float *result, float *error){
		double sum;
		double sum_error;
		double a,b,dtemp;
		
		sum=0;
		sum_error=0;
		for(size_t d=0;d<nDMs; d++){
			for(size_t s=0; s<(nTimesamples-offset); s++){
				dtemp=(signal[(size_t) (d*nTimesamples + s)]-sum_error - mean);
				a=dtemp*dtemp;
				b=sum+a;
				sum_error=(b-sum);
				sum_error=sum_error-a;
				sum=b;
			}
		}
		*result=sum;
		*error=sum_error;
	}

	void MSD_Kahan(float *h_input, size_t nDMs, size_t nTimesamples, size_t offset, double *mean, double *sd){
		float error, signal_mean, signal_sd;
		size_t nElements=nDMs*(nTimesamples-offset);
		
		d_kahan_summation(h_input, nDMs, nTimesamples, offset, &signal_mean, &error);
		signal_mean=signal_mean/nElements;
		
		d_kahan_sd(h_input, nDMs, nTimesamples, offset, signal_mean, &signal_sd, &error);
		signal_sd=sqrt(signal_sd/nElements);

		*mean=signal_mean;
		*sd=signal_sd;
	}
	//-------> Kahan MSD
	//---------------------------------------------------------------------------------
	
	void Find_MSD(float *input){
		double ts_mean, ts_stdev;
		MSD_Kahan(input, nDMs, nSamples, 0, &ts_mean, &ts_stdev);
		printf("MSD: mean: %f; stdev: %f;\n", ts_mean, ts_stdev);
		mean = ts_mean;
		sd   = ts_stdev;
	}
	
	int Calculate_Boxcar(float *output, float *input, size_t width){
		size_t d,s,t;
		float ftemp;
		
		for(d=0; d<nDMs; d++){
			for(s=0; s<nSamples-width + 1; s++){
				ftemp=0;
				for(t=0; t<width; t++){
					ftemp+=input[d*nSamples + s + t];
				}
				output[d*nSamples + s]=ftemp;
			}
		}
		return(width-1);
	}
	
	void Find_MSD_array(float *input){
		printf("Starting MSD calculation\n");
		if(mean_array!=NULL && stdev_array!=NULL && nHarmonics > 1) {
			float *tempdata;
			tempdata = new float[nSamples*nDMs];
			mean_array[0]=0; stdev_array[0]=0;
			
			// First harmonic
			printf("  h: 1; ");
			MSD_Kahan(input, nDMs, nSamples, 0, &mean_array[1], &stdev_array[1]);
			
			// Higher harmonics
			for(int h=2; h<=nHarmonics; h++){
				printf("%d; ", h);
				int remainder;
				remainder = Calculate_Boxcar(tempdata, input, h);
				MSD_Kahan(tempdata, nDMs, nSamples, remainder, &mean_array[h], &stdev_array[h]);
			}
			printf("\n");
			delete[] tempdata;
		}
		else {
			printf("\nERROR!!!! in Find_MSD_array\n");
		}
		printf("MSD calculation finished!\n");
	}
	
	void Allocate_MSD_arrays(){
		mean_array = new double[nHarmonics+1];
		stdev_array = new double[nHarmonics+1];
	}
	
	//---------------------------------------------------------------------------------
	//-------> Thresholding?
	
	General_HRMS(){
		nHarmonics = 0;
		nSamples = 0;
		nDMs = 0;
		mean = 0;
		sd = 0;
		mean_array = NULL;
		stdev_array = NULL;
	}
	
	~General_HRMS(){
		if(mean_array!=NULL) 
			delete[] mean_array;
		if(stdev_array!=NULL)
			delete[] stdev_array;
	}
};

//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------

class AlternativeSuper_HRMS:public General_HRMS {
	// This calculates HRMS by taking fundamental frequency index 'n' and taking H_i = H_{i-1} + max(f[i*n], f[i*n+1],...,f[i*n+i]);
public:
	float *result;
	int *result_harmonics;
	
	AlternativeSuper_HRMS(int t_nHarmonics, size_t t_nDMs, size_t t_nSamples){
		result = new float[t_nDMs*t_nSamples];
		result_harmonics = new int[t_nDMs*t_nSamples];
		nHarmonics = t_nHarmonics;
		nDMs = t_nDMs;
		nSamples = t_nSamples;
	}
	
	float find_max(float *input, size_t size){
		float max=0;
		for(size_t f=0; f<size; f++){
			max=std::max(max,input[f]);
		}
		return(max);
	}
	
	size_t maxDecimate(float *output, float *input, int DIT_factor, size_t size){
		float max;
		size_t mod_size = size/DIT_factor;
		for(size_t f=0; f<mod_size; f++){
			max = find_max(&input[f*DIT_factor],DIT_factor);
			output[f] = max;
		}
		return(mod_size);
	}
	
	void Do_HRMS(float *data){
		printf("\n-----------------------------------------\n");
		printf("---> MaxDIT HRMS...\n");
		size_t DITpos;
		float temp_SNR;
		float *DIT_data;
		double *local_mean, *local_StDev;
		local_mean = new double[nHarmonics+1];
		local_StDev = new double[nHarmonics+1];
		
		printf("Calculating MSD...\n");
		Find_MSD(data);
		
		// Determine max size of the temporary data;
		size_t data_size = 0;
		for(int h=2; h<=nHarmonics; h++){
			data_size = data_size + (size_t) (nSamples/h);
		}
		printf("Data required for maxDITs is: %zu = %f x input size for single DM trial\n", data_size, ((float) data_size)/((float) nSamples));
		DIT_data = new float[data_size*nDMs];
		
		printf("\nCalculating maxDIT...\n");
		DITpos = 0;
		for(int h=2; h<=nHarmonics; h++){
			size_t DIT_Samples = nSamples/h;
			for(size_t d=0; d<nDMs; d++){
				maxDecimate(&DIT_data[DITpos + d*DIT_Samples], &data[d*nSamples], h, nSamples);
			}
			DITpos = DITpos + DIT_Samples*nDMs;
		}
		
		float *harmonics;
		harmonics = new float[nSamples*nDMs];
		
		// First harmonic
		for(size_t d=0; d<nDMs; d++){
			for(size_t t=0; t<nSamples; t++) {
				harmonics[d*nSamples + t] = data[d*nSamples + t];
			}
		}
		MSD_Kahan(harmonics, nDMs, nSamples, 0, &local_mean[1], &local_StDev[1]);
		printf("Mean: %f; sd: %f;\n", local_mean[1], local_StDev[1]);
		for(size_t d=0; d<nDMs; d++){
			for(size_t t=0; t<nSamples; t++) {
				result[t] = (harmonics[t] - local_mean[1])/local_StDev[1];
				result_harmonics[t] = 1;
			}
		}
		
		// Higher harmonics
		DITpos = 0;
		for(int h=2; h<=nHarmonics; h++){
			size_t DIT_Samples = nSamples/h;
			for(size_t d=0; d<nDMs; d++){
				for(size_t t=0; t<DIT_Samples; t++){
					harmonics[d*nSamples + t] = harmonics[d*nSamples + t] + DIT_data[DITpos + d*DIT_Samples + t];
				}
			}
			MSD_Kahan(harmonics, nDMs, DIT_Samples, 0, &local_mean[h], &local_StDev[h]);
			printf("Mean: %f; sd: %f;\n", local_mean[h], local_StDev[h]);
			for(size_t d=0; d<nDMs; d++){
				for(size_t t=0; t<DIT_Samples; t++){
					temp_SNR = (harmonics[d*nSamples + t] - local_mean[h])/(local_StDev[h]);
					if(temp_SNR>result[d*nSamples + t]) {
						result[d*nSamples + t] = temp_SNR;
						result_harmonics[d*nSamples + t] = h;
					}
				}
			}
			DITpos = DITpos + DIT_Samples*nDMs;
		}
		delete[] harmonics;		
		delete[] DIT_data;
		delete[] local_mean;
		delete[] local_StDev;
		printf("------------------<\n");
	}
	
	void Export(size_t fft_nSamples, double sampling_time, const char *filename){
		double sampling_frequency = 1.0/sampling_time;
		std::ofstream FILEOUT;
		FILEOUT.open(filename);
		for(size_t f=0; f<nSamples; f++){
			FILEOUT << (((double) f)*sampling_frequency)/((double) fft_nSamples) << " " << result[f] << " " << result_harmonics[f] << std::endl;
		}
		FILEOUT.close();
	}
	
	~AlternativeSuper_HRMS(){
		delete[] result;
		delete[] result_harmonics;
	}
};

//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------

void Exp_HRMS_Export(std::vector<float> *list, char *filename){
	size_t el = 4;
	size_t nCandidates = list->size()/el;
	
	FILE *fp_out;
	if((fp_out = fopen(filename, "wb")) == NULL) {
		fprintf(stderr, "Error opening output file!\n");
		exit(0);
	}

	fwrite(&list->operator[](0), nCandidates*sizeof(float), el, fp_out);
	fclose(fp_out);
}

void Exp_HRMS_Process(std::vector<float> *list, size_t nTimesamples, float dm_step, float dm_low, float sampling_time, float mod){
	size_t el = 4;
	size_t nCandidates = list->size()/el;

	for(size_t c=0; c<nCandidates; c++) {
		list->operator[](c*el+0) = list->operator[](c*el+0)*dm_step + dm_low;
		list->operator[](c*el+1) = list->operator[](c*el+1)*(1.0/(sampling_time*((float) nTimesamples)*mod));
		list->operator[](c*el+2) = list->operator[](c*el+2);
		list->operator[](c*el+3) = list->operator[](c*el+3);
	}
}

void Exp_HRMS_Threshold(std::vector<float> *candidate_list, float *hrms_result, int *hrms_result_harmonic, float sigma_cutoff, size_t nTimesamples, size_t nDMs, size_t dm_shift){
	printf("nDMs: %zu; nTimesamples:%zu;\n",nDMs, nTimesamples);
	for(size_t d=0; d<nDMs; d++){
		for(size_t t=0; t<nTimesamples; t++){
			size_t pos = d*nTimesamples + t;
			if(hrms_result[pos]>sigma_cutoff) {
				//printf("Candidate: [%f; %f; %f; %f]\n", (float) (d*1.0f + dm_shift*1.0), (float) t, hrms_result[pos], (int) hrms_result_harmonic[pos]);
				candidate_list->push_back((float) (d + dm_shift));
				candidate_list->push_back((float) t);
				candidate_list->push_back(hrms_result[pos]);
				candidate_list->push_back(hrms_result_harmonic[pos]);
			}
		}
	}
}

void Exp_HRMS(){
}

void Do_SHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff){
	printf("SHRMS brekeke!!!\n");
	
	//---------------------------------> Harmonic summing
	size_t nSamples = (nTimesamples>>1);
	AlternativeSuper_HRMS ASHRMS(nHarmonics, nDMs, nSamples);
	ASHRMS.Do_HRMS(input);
	
	//---------> Thresholding, processing and export
	std::vector<float> candidate_list;
	printf("nCandidates before: %zu;\n", candidate_list.size());
	Exp_HRMS_Threshold(&candidate_list, ASHRMS.result, ASHRMS.result_harmonics, sigma_cutoff, nSamples, nDMs, dm_shift);
	printf("nCandidates after threshold: %zu;\n", candidate_list.size());
	Exp_HRMS_Process(&candidate_list, nTimesamples, dm_step, dm_low, sampling_time, mod);
	printf("nCandidates after process: %zu;\n", candidate_list.size());
	char filename[100];
	sprintf(filename, "SHRMS-dm_%.2f-%.2f.dat", dm_low + dm_shift*dm_step, dm_low + (dm_shift + nDMs)*dm_step);
	printf("Filename: %s;\n", filename);
	Exp_HRMS_Export(&candidate_list, filename);
}

void Do_EHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff){
	printf("EHRMS brekeke!!!\n");
}

void Do_LEHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff){
	printf("LEHRMS brekeke!!!\n");
}
