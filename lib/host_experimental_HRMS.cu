#include <stdio.h>

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
		if(mean_array!=NULL && stdev_array!=NULL && nHarmonics < 1) {
			float *tempdata;
			tempdata = new float[nSamples];
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





void Exp_HRMS_Process(){
}

void Exp_HRMS_Threshold(){
}

void Exp_HRMS_??(){
}

void Do_SHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod){
	printf("SHRMS brekeke!!!\n");
}

void Do_EHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod){
	printf("EHRMS brekeke!!!\n");
}

void Do_LEHRMS(float *input, size_t nTimesamples, size_t nDMs, size_t dm_shift, int nHarmonics, float dm_step, float dm_low, float dm_high, float sampling_time, float mod){
	printf("LEHRMS brekeke!!!\n");
}