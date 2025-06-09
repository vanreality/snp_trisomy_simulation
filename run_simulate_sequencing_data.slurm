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
  --n_repeats 20 \
  --model_accuracy 0.81 \
  --trisomy_chr chr16 \
  --min_depth 50 \
  --max_depth 200 \
  --num_depth 16 \
  --min_ff 0.005 \
  --max_ff 0.2 \
  --num_ff 40 \
  --potential_snp_path $snp_path \
  --output_dir $output_dir