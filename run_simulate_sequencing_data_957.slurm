#!/usr/bin/bash
#SBATCH -J simulate_data_957
#SBATCH --partition=cn-long
#SBATCH -o simulate_data_957.log
#SBATCH -e simulate_data_957.log
#SBATCH --cpus-per-task=32

singularity exec -B /lustre1,/lustre2,/appsnew /lustre1/cqyi/syfan/images/common_tools.sif \
python3 simulate_sequencing_data.py \
  --n_repeats 10 \
  --model_accuracy 0.81 \
  --trisomy_chr chr16 \
  --min_depth 50 \
  --max_depth 150 \
  --num_depth 11 \
  --min_ff 0.005 \
  --max_ff 0.2 \
  --num_ff 40 \
  --potential_snp_path /lustre1/cqyi/syfan/snp_nipt/results/potential_panel_region/filtered_senddmr_igtc_ChinaMAP.tsv \
  --output_dir /lustre1/cqyi/syfan/snp_nipt/results/snp_trisomy_simulation/simulated_sequencing_data_957