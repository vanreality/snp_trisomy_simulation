#!/usr/bin/bash

# Define root directory
root_dir="/lustre1/cqyi/syfan/snp_nipt/results/snp_trisomy_simulation"

# Define array of simulation data directories
snp_sets=(
    "957"
    "25092" 
    "3k"
    "all"
)

# Define array of labels
labels=("disomy" "trisomy")

# Get mode: "cfDNA", "cfDNA+WBC", or "cfDNA+model"
mode=$1

simulated_ff_min=0.01
simulated_ff_max=0.10
simulated_ff_step=0.01
num_repeats=30

# Loop through each simulation directory
for snp_set in "${snp_sets[@]}"; do
    input_dir="${root_dir}/simulated_sequencing_data_${snp_set}"
    output_dir="${root_dir}/lr_result_${mode}_${snp_set}"
    
    # Loop through each label
    for label in "${labels[@]}"; do
        for depth in $(seq 80 20 200); do
            # Submit SLURM job
            sbatch run_lr_calculator.slurm \
            "${input_dir}" \
            "${label}" \
            "${depth}" \
            "${output_dir}" \
            "${mode}" \
            "${simulated_ff_min}" \
            "${simulated_ff_max}" \
            "${simulated_ff_step}" \
            "${num_repeats}"
        done
    done
done
