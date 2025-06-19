#!/usr/bin/bash
#SBATCH -J simulate_data
#SBATCH --partition=cn-long
#SBATCH -o simulate_data.log
#SBATCH -e simulate_data.log
#SBATCH --cpus-per-task=32

snp_path=$1
output_dir=$2

singularity exec -B /lustre1,/lustre2,/appsnew /lustre1/cqyi/syfan/images/common_tools.sif \
python3 simulate_sequencing_data.py \
  --n_repeats 30 \
  --model_accuracy 0.81 \
  --trisomy_chr chr16 \
  --min_depth 80 \
  --max_depth 200 \
  --num_depth 7 \
  --min_ff 0.01 \
  --max_ff 0.1 \
  --num_ff 10 \
  --potential_snp_path $snp_path \
  --output_dir $output_dir
