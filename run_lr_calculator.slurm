#!/usr/bin/bash
#SBATCH -J lr_calculator
#SBATCH --partition=cn-long
#SBATCH -o lr_calculator.log
#SBATCH -e lr_calculator.log
#SBATCH --cpus-per-task=32

input_dir=$1
label=$2
depth=$3

for ff in $(seq 0.01 0.01 0.10); do
    for idx in {0..29}; do
        singularity exec -B /lustre1,/lustre2,/appsnew /lustre1/cqyi/syfan/images/common_tools.sif \
        python3 lr_calculator.py \
        -i ${input_dir}/${label}_${depth}_${ff}_${idx}.tsv.gz \
        -o ${input_dir} \
        --ncpus 32 \
        --ff-min 0.01 \
        --ff-max 0.12 \
        --ff-step 0.01
    done
done
