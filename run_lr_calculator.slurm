#!/usr/bin/bash
#SBATCH -J lr_calculator
#SBATCH --partition=cn-long
#SBATCH -o lr_calculator.log
#SBATCH -e lr_calculator.log
#SBATCH --cpus-per-task=32

input_dir=$1
label=$2

for depth in $(seq 50 10 200); do
    for repeat_idx in {0..2}; do
        for ff in $(seq 0.005 0.005 0.2); do
            singularity exec -B /lustre1,/lustre2,/appsnew /lustre1/cqyi/syfan/images/common_tools.sif \
            python3 lr_calculator.py \
            -i ${input_dir}/${label}_${depth}_${ff}_${repeat_idx}.tsv.gz \
            -o ${input_dir}
        done
    done
done
