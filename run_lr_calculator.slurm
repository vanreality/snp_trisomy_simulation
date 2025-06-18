#!/usr/bin/bash
#SBATCH -J lr_calculator
#SBATCH --partition=cn-long
#SBATCH -o lr_calculator.log
#SBATCH -e lr_calculator.log
#SBATCH --cpus-per-task=32

input_dir=$1
label=$2
depth=$3
output_dir=$4
mode=$5
simulated_ff_min=$6
simulated_ff_max=$7
simulated_ff_step=$8
num_repeats=$9

for ff in $(seq ${simulated_ff_min} ${simulated_ff_step} ${simulated_ff_max}); do
    if [ "${mode}" = "cfDNA_model" ]; then
        ff_min=$(echo "define max(a, b) { if (a > b) return a else return b }; max(${ff}*3-0.05, 0.001)" | bc -l)
        ff_max=$(echo "${ff}*3+0.05" | bc -l)
    else
        ff_min="0.001"
        ff_max="0.11"
    fi
    for idx in $(seq 0 $((${num_repeats}-1))); do
        singularity exec -B /lustre1,/lustre2,/appsnew /lustre1/cqyi/syfan/images/common_tools.sif \
        python3 lr_calculator.py \
        -i ${input_dir}/${label}_${depth}_${ff}_${idx}.tsv.gz \
        -o ${output_dir} \
        --ncpus 32 \
        --ff-min ${ff_min} \
        --ff-max ${ff_max} \
        --ff-step 0.001 \
        --mode ${mode}
    done
done
