root_dir="/lustre1/cqyi/syfan/snp_nipt/results/snp_trisomy_simulation"

sbatch run_simulate_sequencing_data.slurm ${root_dir}/potential_panel_region/add_chr16_4654.tsv ${root_dir}/simulated_sequencing_data_all
sbatch run_simulate_sequencing_data.slurm ${root_dir}/potential_panel_region/add_chr16_3000.tsv ${root_dir}/simulated_sequencing_data_3k
sbatch run_simulate_sequencing_data.slurm ${root_dir}/potential_panel_region/merged_probes_ChinaMAP_filtered.tsv ${root_dir}/simulated_sequencing_data_25092
sbatch run_simulate_sequencing_data.slurm ${root_dir}/potential_panel_region/filtered_senddmr_igtc_ChinaMAP.tsv ${root_dir}/simulated_sequencing_data_957



