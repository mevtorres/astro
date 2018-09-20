#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <omp.h>

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
			printf("Harmonic: %d; Mean: %f; StDev: %f;\n", 1, mean_array[1], stdev_array[1]);
			// Higher harmonics
			for(int h=2; h<=nHarmonics; h++){
				printf("%d; ", h);
				int remainder;
				remainder = Calculate_Boxcar(tempdata, input, h);
				MSD_Kahan(tempdata, nDMs, nSamples, remainder, &mean_array[h], &stdev_array[h]);
				printf("Harmonic: %d; Mean: %f; StDev: %f;\n", h, mean_array[h], stdev_array[h]);
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

class Point_HRMS:public General_HRMS {
	// This selects a fundamental frequency index 'n' and add to it its multiples k*n so H_0=f[n]; H_1=f[n]+f[2*n]; H_2=f[n]+f[2*n]+f[3*n] => H_i=H_{i-1} + f[i*n];
public:
	float *result;
	int *result_harmonics;
	
	Point_HRMS(int t_nHarmonics, size_t t_nDMs, size_t t_nSamples){
		result = new float[t_nDMs*t_nSamples];
		result_harmonics = new int[t_nDMs*t_nSamples];
		nHarmonics = t_nHarmonics;
		nDMs = t_nDMs;
		nSamples = t_nSamples;
	}
	
	void Do_HRMS(float *data){
		printf("\n-----------------------------------------\n");
		printf("---> Point HRMS...\n");
		printf("Calculating MSD...\n");
		Allocate_MSD_arrays();
		Find_MSD_array(data);
		float SNR, temp_SNR, HS;
		int max_harmonics;
		
		printf("Harmonic summing...\n");
		for(size_t d=0; d<nDMs; d++){
			for(size_t t=0; t<nSamples; t++){
				HS = data[d*nSamples + t];
				SNR = (HS - mean_array[1])/stdev_array[1];
				max_harmonics = 0;
			
				for(int h=2; h<=nHarmonics; h++){
					if( (h*t)<nSamples){
						HS = HS + data[d*nSamples + (h*t)];
						temp_SNR = (HS - mean_array[h])/(stdev_array[h]);
			
						if(temp_SNR>SNR) {
							SNR = temp_SNR;
							max_harmonics = h;
						}
					}
				}
					
				result[d*nSamples + t] = SNR;
				result_harmonics[d*nSamples + t] = max_harmonics;
			}
		}
		
		printf("-----------------<\n");
	}
	
	void Export(size_t fft_nSamples, double sampling_time, const char *filename){
		double sampling_frequency = 1.0/sampling_time;
		std::ofstream FILEOUT;
		FILEOUT.open(filename);
		for(size_t d=0; d<nDMs; d++){
			for(size_t t=0; t<nSamples; t++){
				FILEOUT << (((double) t)*sampling_frequency)/((double) fft_nSamples) << " " << result[d*nSamples + t] << " " << result_harmonics[d*nSamples + t] << " " << d << std::endl;
			}
		}
		FILEOUT.close();
	}
	
	~Point_HRMS(){
		delete[] result;
		delete[] result_harmonics;
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


class LimitedExhaustive_HRMS:public General_HRMS {
	// This calculates Exhaustive HRMS
public:
	float *result;
	int *result_harmonics;
	
	LimitedExhaustive_HRMS(int t_nHarmonics, size_t t_nDMs, size_t t_nSamples){
		result = new float[t_nDMs*t_nSamples];
		result_harmonics = new int[t_nDMs*t_nSamples];
		nHarmonics = t_nHarmonics;
		nDMs = t_nDMs;
		nSamples = t_nSamples;
	}
	
	float find_max(float *input, size_t size, size_t *max_position){
		float max=0;
		if(size>0){
			max = input[0]; *max_position = 0;
			for(size_t f=0; f<size; f++){
				if( input[f]>max ) { max = input[f]; *max_position=f;}
			}
		}
		return(max);
	}
	
	
	void Perform_limited_search(float *data, size_t t, size_t d, int nHarmonics, float *SNR, size_t *SNR_harmonic){
		float *mHS; // maximum SNR value per harmonic
		float *partial_sums;
		int partial_sums_size = (nHarmonics*(1+nHarmonics))/2;
		int data_shift = 0, old_data_shift = 0;
		float t_SNR = 0;
		size_t t_SNR_harmonic = 0;
		
		partial_sums = new float[partial_sums_size];
		mHS = new float[nHarmonics];
		
		// insert initial value 
		partial_sums[0] = data[d*nSamples + t];
		old_data_shift = data_shift;
		data_shift = data_shift + 1;
		
		
		// creating following iterations
		for(int f=1; f<nHarmonics; f++){
			// zeroth value
			if( (f+1)*t < nSamples ) {
				partial_sums[data_shift + 0] = partial_sums[old_data_shift + 0] + data[d*nSamples + (f+1)*t];
			}
			else {
				partial_sums[data_shift + 0] = 0;
			}
						
			// 1+ values
			for(int i = 1; i<f; i++){
				if( (f+1)*t + i < nSamples ){
					//printf("To produce [%d;%d] data [%d;%d] and [%d;%d] are accessed.\n", f, i, f-1, i-1, f-1, i);
					partial_sums[data_shift + i] = std::max(partial_sums[old_data_shift + i - 1], partial_sums[old_data_shift + i]) + data[d*nSamples + (f+1)*t + i];
				}
				else {
					partial_sums[data_shift + i] = 0;
				}
			}
			
			// last values
			if( (f+1)*t + f < nSamples ){
				partial_sums[data_shift + f] = partial_sums[old_data_shift + f - 1] + data[d*nSamples + (f+1)*t + f];
				//printf("To produce [%d;%d] data [%d;%d] are accessed.\n", f, f, f-1, f-1);
			}
			else {
				partial_sums[data_shift + f] = 0;
			}
			
			old_data_shift = data_shift;
			data_shift = data_shift + f + 1;
		}
		
		// find max for each harmonics
		mHS[0] = partial_sums[0];
		data_shift = 1;
		for(int f=1; f<nHarmonics; f++){
			size_t tpos;
			mHS[f] = find_max(&partial_sums[data_shift], f+1, &tpos);
			data_shift = data_shift + f + 1;
		}
		
		// find highest SNR
		for(int f=0; f<nHarmonics; f++) mHS[f] = (mHS[f]-mean_array[f+1])/(stdev_array[f+1]);
		t_SNR = find_max(mHS, nHarmonics, &t_SNR_harmonic);
		
		delete[] mHS;
		delete[] partial_sums;
		
		*SNR_harmonic = t_SNR_harmonic;
		*SNR = t_SNR;
	}
	
	void Do_HRMS(float *data){
		printf("\n-----------------------------------------\n");
		printf("---> Limited exhaustive HRMS...\n");
		
		printf("Calculating MSD...\n");
		Allocate_MSD_arrays();
		Find_MSD_array(data);
		
		printf("Harmonic summing...\n");
		#pragma omp parallel for
		for(size_t d=0; d<nDMs; d++){
			for(size_t t=1; t<nSamples; t++){
				float SNR = 0;
				size_t SNR_width = 0;
			
				Perform_limited_search(data, t, d, nHarmonics, &SNR, &SNR_width);
			
				result[d*nSamples + t] = SNR;
				result_harmonics[d*nSamples + t] = SNR_width;
			}
		}
		printf("\n------------------------------<\n");
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
	
	~LimitedExhaustive_HRMS(){
		delete[] result;
		delete[] result_harmonics;
	}
};


//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------

class Exhaustive_HRMS:public General_HRMS {
	// This calculates Exhaustive HRMS
public:
	float *result;
	int *result_harmonics;
	
	Exhaustive_HRMS(int t_nHarmonics, size_t t_nDMs, size_t t_nSamples){
		result = new float[t_nDMs*t_nSamples];
		result_harmonics = new int[t_nDMs*t_nSamples];
		nHarmonics = t_nHarmonics;
		nDMs = t_nDMs;
		nSamples = t_nSamples;
	}
	
	float find_max(float *input, size_t size, size_t *max_position){
		float max=0;
		if(size>0){
			max = input[0]; *max_position = 0;
			for(size_t f=0; f<size; f++){
				if( input[f]>max ) { max = input[f]; *max_position=f;}
			}
		}
		return(max);
	}
	
	int RfSNR(float *mHS, float *data, float pHS, size_t t, size_t d, int depth, int shift){
		if(depth==nHarmonics) return(0);
		int new_depth, new_shift;
		
		size_t pos = (depth+1)*t + shift;
		if(pos<nSamples){
			float HS = pHS + data[d*nSamples + pos];
			if(HS>mHS[depth]) mHS[depth]=HS;
			// left
			new_depth = depth + 1;
			new_shift = shift;
			RfSNR(mHS, data, HS, t, d, new_depth, new_shift);
			
			// right
			new_depth = depth + 1;
			new_shift = shift + 1;		
			RfSNR(mHS, data, HS, t, d, new_depth, new_shift);
		}
		
		return(0);
	}
	
	void Find_highest_SNR(float *data, size_t t, size_t d, float *SNR, int *SNR_harmonic){
		float *mHS; // maximum SNR value per harmonic
		mHS = new float[nHarmonics];
		for(int f=0; f<nHarmonics; f++) mHS[f]=0;
		
		RfSNR(mHS, data, 0, t, d, 0, 0);
		
		size_t t_SNR_harmonic = 0;
		float  t_SNR = 0;
		for(int f=0; f<nHarmonics; f++) mHS[f] = (mHS[f]-mean_array[f+1])/(stdev_array[f+1]);
		t_SNR = find_max(mHS, nHarmonics, &t_SNR_harmonic);
		delete[] mHS;
		
		*SNR_harmonic = t_SNR_harmonic;
		*SNR = t_SNR;
	}
	
	void Do_HRMS(float *data){
		printf("\n-----------------------------------------\n");
		printf("---> Exhaustive HRMS...\n");
		
		printf("Calculating MSD...\n");
		Allocate_MSD_arrays();
		Find_MSD_array(data);
		
		printf("Harmonic summing...\n");
		for(int d=0; d<nDMs; d++){
			printf("|");
			fflush(stdout);
			//#pragma omp parallel for
			for(int t=1; t<nSamples; t++){
				float SNR = 0;
				int SNR_width = 0;
				Find_highest_SNR(data, t, d, &SNR, &SNR_width);
				result[d*nSamples + t] = SNR;
				result_harmonics[d*nSamples + t] = SNR_width;
			}
		}
		printf("\n");
		printf("----------------------<\n");
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
	
	~Exhaustive_HRMS(){
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

void Do_PHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff){	
	//---------------------------------> Harmonic summing
	size_t nSamples = (nTimesamples>>1);
	Point_HRMS PHRMS(nHarmonics, nDMs, nSamples);
	PHRMS.Do_HRMS(input);
	
	//---------> Thresholding, processing and export
	std::vector<float> candidate_list;
	printf("nCandidates before: %zu;\n", candidate_list.size());
	Exp_HRMS_Threshold(&candidate_list, PHRMS.result, PHRMS.result_harmonics, sigma_cutoff, nSamples, nDMs, dm_shift);
	printf("nCandidates after threshold: %zu;\n", candidate_list.size());
	Exp_HRMS_Process(&candidate_list, nTimesamples, dm_step, dm_low, sampling_time, mod);
	printf("nCandidates after process: %zu;\n", candidate_list.size());
	char filename[100];
	sprintf(filename, "PHRMS-dm_%.2f-%.2f.dat", dm_low + dm_shift*dm_step, dm_low + (dm_shift + nDMs)*dm_step);
	printf("Filename: %s;\n", filename);
	Exp_HRMS_Export(&candidate_list, filename);
}

void Do_SHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff){
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
	//---------------------------------> Harmonic summing
	size_t nSamples = (nTimesamples>>1);
	Exhaustive_HRMS EHRMS(nHarmonics, nDMs, nSamples);
	EHRMS.Do_HRMS(input);
	
	//---------> Thresholding, processing and export
	std::vector<float> candidate_list;
	printf("nCandidates before: %zu;\n", candidate_list.size());
	Exp_HRMS_Threshold(&candidate_list, EHRMS.result, EHRMS.result_harmonics, sigma_cutoff, nSamples, nDMs, dm_shift);
	printf("nCandidates after threshold: %zu;\n", candidate_list.size());
	Exp_HRMS_Process(&candidate_list, nTimesamples, dm_step, dm_low, sampling_time, mod);
	printf("nCandidates after process: %zu;\n", candidate_list.size());
	char filename[100];
	sprintf(filename, "EHRMS-dm_%.2f-%.2f.dat", dm_low + dm_shift*dm_step, dm_low + (dm_shift + nDMs)*dm_step);
	printf("Filename: %s;\n", filename);
	Exp_HRMS_Export(&candidate_list, filename);
}

void Do_LEHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod, float sigma_cutoff){
	//---------------------------------> Harmonic summing
	size_t nSamples = (nTimesamples>>1);
	LimitedExhaustive_HRMS LEHRMS(nHarmonics, nDMs, nSamples);
	LEHRMS.Do_HRMS(input);
	
	//---------> Thresholding, processing and export
	std::vector<float> candidate_list;
	printf("nCandidates before: %zu;\n", candidate_list.size());
	Exp_HRMS_Threshold(&candidate_list, LEHRMS.result, LEHRMS.result_harmonics, sigma_cutoff, nSamples, nDMs, dm_shift);
	printf("nCandidates after threshold: %zu;\n", candidate_list.size());
	Exp_HRMS_Process(&candidate_list, nTimesamples, dm_step, dm_low, sampling_time, mod);
	printf("nCandidates after process: %zu;\n", candidate_list.size());
	char filename[100];
	sprintf(filename, "LEHRMS-dm_%.2f-%.2f.dat", dm_low + dm_shift*dm_step, dm_low + (dm_shift + nDMs)*dm_step);
	printf("Filename: %s;\n", filename);
	Exp_HRMS_Export(&candidate_list, filename);
}
