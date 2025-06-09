#!/usr/bin/bash

# Define root directory
root_dir="/lustre1/cqyi/syfan/snp_nipt/results/snp_trisomy_simulation"

# Define array of simulation data directories
sim_dirs=(
    "simulated_sequencing_data_957"
    "simulated_sequencing_data_25092" 
    "simulated_sequencing_data_2k"
    "simulated_sequencing_data_3k"
    "simulated_sequencing_data_all"
)

# Define array of labels
labels=("disomy" "trisomy")

# Loop through each simulation directory
for sim_dir in "${sim_dirs[@]}"; do
    input_dir="${root_dir}/${sim_dir}"
    
    # Loop through each label
    for label in "${labels[@]}"; do
        # Submit SLURM job
        sbatch run_lr_calculator.slurm "${input_dir}" "${label}"
    done
done