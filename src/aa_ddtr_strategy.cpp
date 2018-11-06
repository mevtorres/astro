//
//  aa_ddtr_strategy.cpp
//  aapipeline
//
//  Created by Cees Carels on Tuesday 23/10/2018.
//  Copyright © 2018 Astro-Accelerate. All rights reserved.
//

#include "aa_ddtr_strategy.hpp"

aa_ddtr_strategy::aa_ddtr_strategy() : m_ready(false), is_setup(false), m_maxshift(0), m_num_tchunks(0), m_total_ndms(0), m_max_dm(0.0), m_maxshift_high(0), m_max_ndms(0) {
    
}

aa_ddtr_strategy::aa_ddtr_strategy(const aa_ddtr_plan &plan, const aa_filterbank_metadata &metadata, const size_t &free_memory, const bool &enable_analysis) : m_ready(false), is_setup(false), m_maxshift(0), m_num_tchunks(0), m_total_ndms(0), m_max_dm(0.0), m_maxshift_high(0), m_max_ndms(0) {
    strategy(plan, metadata, free_memory, enable_analysis);
}

bool aa_ddtr_strategy::strategy(const aa_ddtr_plan &plan, const aa_filterbank_metadata &metadata, const size_t &free_memory, const bool &enable_analysis) {
    /**
     * This method relies on defining points when nsamps is a multiple of
     * nchans - bin on the diagonal or a fraction of it.
     */
    std::cout << "Calculating strategy" << std::endl;
    const float power         = 2.0;  //This variable is set to 2.0 in host_main_function, and used only here (unless it is modified in get_user_input when the input_file.txt is read).
    
    //Part of the filterbank metadata
    const int nchans  = metadata.nchans();
    const int nsamp   = metadata.nsamples();
    const float fch1  = metadata.fch1();
    const float foff  = metadata.foff();
    const float tsamp = metadata.tsamp();
    
    if(!plan.range()) {
        //No user requested dm settings have been added, this is an invalid aa_ddtr_plan.
        return false;
    }
    
    //Plan requested DM settings
    const size_t range = plan.range();
    m_ndms.resize(range);
    
    m_dmshifts.resize(nchans);
    
    //Strategy set DM settings
    str_dm.resize(range);
    
    const size_t gpu_memory = free_memory;
    
    const double SPDT_fraction = 3.0/4.0; // 1.0 for MSD plane profile validation
    
    //Calculate maxshift, the number of dms for this bin and the highest value of dm to be calculated in this bin
    
    if (power != 2.0) {
        // Calculate time independent dm shifts
        for (int c = 0; c < nchans; c++) {
            m_dmshifts[c] = 4148.741601f * ( ( 1.0 / pow(( fch1 + ( foff * c ) ), power) ) - ( 1.0 / pow(fch1, power) ) );
        }
    }
    else {
        // Calculate time independent dm shifts
        for (int c = 0; c < nchans; c++) {
            m_dmshifts[c] = (float) ( 4148.741601f * ( ( 1.0 / pow((double) ( fch1 + ( foff * c ) ), power) ) - ( 1.0 / pow((double) fch1, power) ) ) );
        }
    }
    
    for(int i = 0; i < (int)range; i++)    {
        float n;
        modff(      ( ( (int) ( (  plan.user_dm(i).high - plan.user_dm(i).low ) / plan.user_dm(i).step ) + SDIVINDM ) / SDIVINDM )     , &n); // This calculates number of SDIVINDM blocks per DM range
        m_ndms[i] = (int) ( (int) n * SDIVINDM ); // This is number of DM trial per DM range
        if (m_max_ndms < m_ndms[i])
            m_max_ndms = m_ndms[i]; // looking for maximum number of DM trials for memory allocation
        m_total_ndms = m_total_ndms + m_ndms[i];
    }
    printf("\nMaximum number of dm trials in any of the range steps:\t%d", m_max_ndms);
    
    str_dm[0].low = plan.user_dm(0).low;                        //
    str_dm[0].high = str_dm[0].low + ( m_ndms[0] * ( plan.user_dm(0).step ) );   // Redefines DM plan to suit GPU
    str_dm[0].step = plan.user_dm(0).step;                      //
    for (size_t i = 1; i < range; i++)    {
        str_dm[i].low = str_dm[i-1].high;
        str_dm[i].high = str_dm[i].low + m_ndms[i] * plan.user_dm(i).step;
        str_dm[i].step = plan.user_dm(i).step;
        
        if (plan.user_dm(i-1).inBin > 1) {
            m_maxshift = (int) ceil(( ( str_dm[i-1].low + str_dm[i-1].step * m_ndms[i - 1] ) * m_dmshifts[nchans - 1] ) / ( tsamp ));
            m_maxshift = (int) ceil((float) ( m_maxshift + ( (float) ( SDIVINT*2*SNUMREG ) ) ) / (float) plan.user_dm(i-1).inBin) / (float) ( SDIVINT*2*SNUMREG );
            m_maxshift = ( m_maxshift ) * ( SDIVINT*2*SNUMREG ) * plan.user_dm(i-1).inBin;
            if (( m_maxshift ) > m_maxshift_high)
                m_maxshift_high = ( m_maxshift );
        }
    }
    
    if (plan.user_dm(range-1).inBin > 1) {
        m_maxshift = (int) ceil(( ( str_dm[range-1].low + str_dm[range-1].step * m_ndms[range - 1] ) * m_dmshifts[nchans - 1] ) / ( tsamp ));
        m_maxshift = (int) ceil((float) ( m_maxshift + ( (float) ( SDIVINT*2*SNUMREG ) ) ) / (float) plan.user_dm(range-1).inBin) / (float) ( SDIVINT*2*SNUMREG );
        m_maxshift = m_maxshift * ( SDIVINT*2*SNUMREG ) * plan.user_dm(range-1).inBin;
        if (( m_maxshift ) > m_maxshift_high)
            m_maxshift_high = ( m_maxshift );
    }
    
    if (m_maxshift_high == 0)    {
        m_maxshift_high = (int) ceil(( ( str_dm[range-1].low + str_dm[range-1].step * ( m_ndms[range - 1] ) ) * m_dmshifts[nchans - 1] ) / tsamp);
    }
    m_max_dm = ceil(str_dm[range-1].high);
    
    m_maxshift = ( m_maxshift_high + ( SNUMREG * 2 * SDIVINT ) );
    printf("\nRange:\t%lu, MAXSHIFT:\t%d, Scrunch value:\t%d", range - 1, m_maxshift, plan.user_dm(range-1).inBin);
    printf("\nMaximum dispersive delay:\t%.2f (s)", m_maxshift * tsamp);
    
    if (m_maxshift >= nsamp)    {
        printf("\n\nERROR!! Your maximum DM trial exceeds the number of samples you have.\nReduce your maximum DM trial\n\n");
        return false;
    }
    
    printf("\nDiagonal DM:\t%f", ( tsamp * nchans * 0.0001205 * powf(( fch1 + ( foff * ( nchans / 2 ) ) ), 3.0) ) / ( -foff * nchans ));
    if (m_maxshift >= nsamp)    {
        printf("ERROR!! Your maximum DM trial exceeds the number of samples you have.\nReduce your maximum DM trial");
        return false;
    }
    
    /* Four cases:
     * 1) nchans < m_max_ndms & nsamp fits in GPU RAM
     * 2) nchans > m_max_ndms & nsamp fits in GPU RAM
     * 3) nchans < m_max_ndms & nsamp does not fit in GPU RAM
     * 4) nchans > m_max_ndms & nsamp does not fit in GPU RAM
     */
    
    unsigned int max_tsamps;
    // Allocate memory to store the t_processed ranges:
    m_t_processed = (int **) malloc(range * sizeof(int *));
    
    if (nchans < ( m_max_ndms )) {
        // This means that we can cornerturn into the allocated output buffer
        // without increasing the memory needed
        
        // Maximum number of samples we can fit in our GPU RAM is then given by:
        //max_tsamps = (unsigned int) ( (*gpu_memory) / ( sizeof(unsigned short) * ( (*m_max_ndms) + nchans ) ) ); // maximum number of timesamples we can fit into GPU memory
        size_t SPDT_memory_requirements = (enable_analysis ? (sizeof(float)*(m_max_ndms)*SPDT_fraction) : 0 );
        max_tsamps = (unsigned int) ( (gpu_memory) / ( sizeof(unsigned short)*nchans + sizeof(float)*(m_max_ndms) + SPDT_memory_requirements )); // maximum number of timesamples we can fit into GPU memory
        
        // Check that we dont have an out of range maxshift:
        if ((unsigned int)( m_maxshift ) > max_tsamps)    {
            printf("\nERROR: The selected GPU does not have enough memory for this number of dispersion trials.");
            printf("\nReduce maximum dm or increase the size of dm step");
            return false;
        }
        
        // Next check to see if nsamp fits in GPU RAM:
        if ((unsigned int)nsamp < max_tsamps)    {
            // We have case 1)
            // Allocate memory to hold the values of nsamps to be processed
            unsigned long int local_t_processed = (unsigned long int) floor(( (double) ( nsamp - (m_maxshift) ) / (double) plan.user_dm(range-1).inBin ) / (double) ( SDIVINT*2*SNUMREG )); //number of timesamples per block
            local_t_processed = local_t_processed * ( SDIVINT*2*SNUMREG ) * plan.user_dm(range-1).inBin;
            for (size_t i = 0; i < range; i++)    {
                m_t_processed[i] = (int *) malloc(sizeof(int)); // TODO: change to size_t
                m_t_processed[i][0] = (int) floor(( (float) ( local_t_processed ) / (float) plan.user_dm(i).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
                m_t_processed[i][0] = m_t_processed[i][0] * ( SDIVINT*2*SNUMREG );
            }
            ( m_num_tchunks ) = 1;
            printf("\nIn 1\n");
        }
        else {
            // We have case 3)
            // Work out how many time samples we can fit into ram
            int samp_block_size = max_tsamps - ( m_maxshift );
            
            // Work out how many blocks of time samples we need to complete the processing
            // upto nsamp-maxshift
            //int num_blocks = (int) floor(( (float) nsamp - ( *maxshift ) )) / ( (float) ( samp_block_size ) ) + 1;
            
            // Find the common integer amount of samples between all bins
            int local_t_processed = (int) floor(( (float) ( samp_block_size ) / (float) plan.user_dm(range-1).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
            local_t_processed = local_t_processed * ( SDIVINT*2*SNUMREG ) * plan.user_dm(range-1).inBin;
            
            int num_blocks = (int) floor(( (float) nsamp - (float)( m_maxshift ) )) / ( (float) ( local_t_processed ) );
            
            // Work out the remaining fraction to be processed
            int remainder =  nsamp -  (num_blocks*local_t_processed ) - (m_maxshift) ;
            remainder = (int) floor((float) remainder / (float) plan.user_dm(range-1).inBin) / (float) ( SDIVINT*2*SNUMREG );
            remainder = remainder * ( SDIVINT*2*SNUMREG ) * plan.user_dm(range-1).inBin;
            
            for (size_t i = 0; i < range; i++)    {
                // Allocate memory to hold the values of nsamps to be processed
                m_t_processed[i] = (int *) malloc((num_blocks + 1) * sizeof(int));
                // Remember the last block holds less!
                for (int j = 0; j < num_blocks; j++) {
                    m_t_processed[i][j] = (int) floor(( (float) ( local_t_processed ) / (float) plan.user_dm(i).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
                    m_t_processed[i][j] = m_t_processed[i][j] * ( SDIVINT*2*SNUMREG );
                }
                // fractional bit
                m_t_processed[i][num_blocks] = (int) floor(( (float) ( remainder ) / (float) plan.user_dm(i).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
                m_t_processed[i][num_blocks] = m_t_processed[i][num_blocks] * ( SDIVINT*2*SNUMREG );
            }
            ( m_num_tchunks ) = num_blocks + 1;
            printf("\nIn 3\n");
            printf("\nnum_blocks:\t%d", num_blocks);
        }
    }
    else {
        // This means that we cannot cornerturn into the allocated output buffer
        // without increasing the memory needed. Set the output buffer to be as large as the input buffer:
        
        // Maximum number of samples we can fit in our GPU RAM is then given by:
        //max_tsamps = (unsigned int) ( ( *gpu_memory ) / ( nchans * ( sizeof(float) + 2 * sizeof(unsigned short) ) ) );
        size_t SPDT_memory_requirements = (enable_analysis ? (sizeof(float)*(m_max_ndms)*SPDT_fraction) : 0 );
        max_tsamps = (unsigned int) ( ( gpu_memory ) / ( nchans * ( sizeof(float) + sizeof(unsigned short) )+ SPDT_memory_requirements ));
        
        // Check that we dont have an out of range maxshift:
        if (( m_maxshift ) > (int)max_tsamps) {
            printf("\nERROR: The selected GPU does not have enough memory for this number of dispersion trials.");
            printf("\nReduce maximum dm or increase the size of dm step");
            return false;
        }
        
        // Next check to see if nsamp fits in GPU RAM:
        if ((unsigned int)nsamp < max_tsamps) {
            // We have case 2)
            // Allocate memory to hold the values of nsamps to be processed
            int local_t_processed = (int) floor(( (float) ( nsamp - ( m_maxshift ) ) / (float) plan.user_dm(range-1).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
            local_t_processed = local_t_processed * ( SDIVINT*2*SNUMREG ) * plan.user_dm(range-1).inBin;
            for (size_t i = 0; i < range; i++) {
                m_t_processed[i] = (int *) malloc(sizeof(int));
                m_t_processed[i][0] = (int) floor(( (float) ( local_t_processed ) / (float) plan.user_dm(i).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
                m_t_processed[i][0] = m_t_processed[i][0] * ( SDIVINT*2*SNUMREG );
            }
            ( m_num_tchunks ) = 1;
            printf("\nIn 2\n");
        }
        else {
            // We have case 4)
            // Work out how many time samples we can fit into ram
            int samp_block_size = max_tsamps - ( m_maxshift );
            
            // Work out how many blocks of time samples we need to complete the processing
            // upto nsamp-maxshift
            //int num_blocks = (int) floor(( (float) nsamp - (float) ( *maxshift ) ) / ( (float) samp_block_size ));
            
            // Find the common integer amount of samples between all bins
            int local_t_processed = (int) floor(( (float) ( samp_block_size ) / (float) plan.user_dm(range-1).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
            local_t_processed = local_t_processed * ( SDIVINT*2*SNUMREG ) * plan.user_dm(range-1).inBin;
            
            // samp_block_size was not used to calculate remainder instead there is local_t_processed which might be different
            int num_blocks = (int) floor(( (float) nsamp - (float) ( m_maxshift ) ) / ( (float) local_t_processed ));
            
            // Work out the remaining fraction to be processed
            int remainder = nsamp - ( num_blocks * local_t_processed ) - ( m_maxshift );
            remainder = (int) floor((float) remainder / (float) plan.user_dm(range-1).inBin) / (float) ( SDIVINT*2*SNUMREG );
            remainder = remainder * ( SDIVINT*2*SNUMREG ) * plan.user_dm(range-1).inBin;
            
            for (size_t i = 0; i < range; i++)    {
                // Allocate memory to hold the values of nsamps to be processed
                m_t_processed[i] = (int *) malloc(( num_blocks + 1 ) * sizeof(int));
                // Remember the last block holds less!
                for (int j = 0; j < num_blocks; j++) {
                    m_t_processed[i][j] = (int) floor(( (float) ( local_t_processed ) / (float) plan.user_dm(i).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
                    m_t_processed[i][j] = m_t_processed[i][j] * ( SDIVINT*2*SNUMREG );
                }
                // fractional bit
                m_t_processed[i][num_blocks] = (int) floor(( (float) ( remainder ) / (float) plan.user_dm(i).inBin ) / (float) ( SDIVINT*2*SNUMREG ));
                m_t_processed[i][num_blocks] = m_t_processed[i][num_blocks] * ( SDIVINT*2*SNUMREG );
            }
            ( m_num_tchunks ) = num_blocks + 1;
            printf("\nIn 4\n");
        }
    }
    printf("\nMaxshift memory needed:\t%lu MB", nchans * ( m_maxshift ) * sizeof(unsigned short) / 1024 / 1024);
    if (nchans < ( m_max_ndms ))    {
        printf("\nOutput memory needed:\t%lu MB", ( m_max_ndms ) * ( m_maxshift ) * sizeof(float) / 1024 / 1024);
    }
    else {
        printf("\nOutput memory needed:\t%lu MB\n", nchans * ( m_maxshift ) * sizeof(float) / 1024 / 1024);
    }
    
    //Strategy does not change inBin, outBin.
    //Re-assign original inBin, outBin to the strategy.
    for(size_t i = 0; i < str_dm.size(); i++) {
        str_dm.at(i).inBin = plan.user_dm(i).inBin;
        str_dm.at(i).outBin = plan.user_dm(i).outBin;
    }
    
    m_ready = true;
    return true;
}

void aa_ddtr_strategy::allocate_memory_cpu_output() {
    size_t estimate_outputbuffer_size = 0;
    size_t outputsize = 0;
    const size_t range = str_dm.size();
    for(size_t i = 0; i < range; i++) {
        for(int j = 0; j < m_num_tchunks; j++) {
            estimate_outputbuffer_size += (size_t)(m_t_processed[i][j]*sizeof(float)*m_ndms[i]);
        }
    }
    
    outputsize = 0;
    output_buffer = (float ***) malloc(range * sizeof(float **));
    for(size_t i = 0; i < range; i++) {
        int total_samps = 0;
        for(int k = 0; k < m_num_tchunks; k++) {
            total_samps += m_t_processed[i][k];
        }
        output_buffer[i] = (float **) malloc(m_ndms[i] * sizeof(float *));
        for (int j = 0; j < m_ndms[i]; j++) {
            output_buffer[i][j] = (float *) malloc(( total_samps ) * sizeof(float));
        }
        outputsize += ( total_samps ) * m_ndms[i] * sizeof(float);
    }
}

bool aa_ddtr_strategy::setup() {
    /**
     * Performs memory allocations and other setup before dedispersion can be run.
     */
    
    //output_buffer is used by the pipeline, but is it also used by the user?
    allocate_memory_cpu_output();
    
    is_setup = true;
    return true;
}