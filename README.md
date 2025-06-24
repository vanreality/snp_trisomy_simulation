# SNP-NIPT-nf

This project implements a Nextflow pipeline for SNP-based Non-Invasive Prenatal Testing (NIPT). The pipeline processes sequencing data to estimate fetal fraction and calculate log-likelihood ratios for detecting fetal trisomy.

## Workflow

The main workflow is defined in the `snp_nipt.nf` file. It consists of three main processes:

1.  **Input Processing**: The pipeline can start from a single large parquet file or a samplesheet of per-sample parquet files. If a single file is provided, it's split into per-sample files.
2.  **SNP Pileup Generation**: For each sample, it extracts SNP information from the parquet file and generates a pileup file in `tsv.gz` format. This step requires a reference genome and a list of known SNPs.
3.  **Likelihood-Ratio Calculation**: For each sample, it uses the pileup data to estimate the fetal fraction and calculate the log-likelihood ratio for trisomy on each chromosome.

## Requirements

*   [Nextflow](https://www.nextflow.io/docs/latest/getstarted.html) (`>=21.10.3`)
*   [Singularity](https://sylabs.io/guides/3.5/user-guide/quick_start.html)

## Usage

To run the pipeline, you need to provide input data and specify some parameters.

### Parameters

#### Input Data

You must provide input data using one of the following parameters:

*   `--input_parquet`: Path to a single parquet file containing data for multiple samples.
*   `--input_parquet_samplesheet`: Path to a CSV samplesheet specifying sample IDs and paths to their parquet files.
    Format:
    ```csv
    sample,parquet
    sample1,path/to/sample1.parquet
    sample2,path/to/sample2.parquet
    ```
*   `--input_pileup_samplesheet`: Path to a CSV samplesheet specifying sample IDs and paths to their pileup files. This skips the SNP extraction step.
    Format:
    ```csv
    sample,pileup
    sample1,path/to/sample1.pileup.tsv.gz
    sample2,path/to/sample2.pileup.tsv.gz
    ```

#### Reference and SNP data

*   `--fasta`: Path to the reference genome FASTA file.
*   `--fasta_index`: Path to the reference genome FASTA index file (`.fai`).
*   `--snp_list`: Path to a file containing a list of SNPs to be analyzed.

#### Other Parameters

*   `--outdir`: The directory where results will be saved.
*   `--filter_mode`: The mode for filtering SNPs (`hard_filter` or `prob_weighted`). Default: `hard_filter`.
*   `--threshold`: A threshold value for filtering. Default: `0.5`.
*   `--lr_mode`: The mode for LR calculation (`cfDNA`, `cfDNA+WBC`, `cfDNA+model`). Default: `cfDNA+model`.

### Command-line Example

```bash
nextflow run snp_nipt.nf -profile singularity \
    --input_parquet <path_to_parquet> \
    --fasta <path_to_fasta> \
    --fasta_index <path_to_fasta_index> \
    --snp_list <path_to_snp_list> \
    --outdir results
```

## Output

The pipeline generates the following output files for each sample in the specified `--outdir`:

The output directory will have the following structure, with subdirectories for each main step of the workflow:

```
<outdir>/
├── calculate/
│   └── <sample_id>_pileup_lr.tsv
├── extract/
│   ├── <sample_id>_pileup.tsv.gz
│   └── <sample_id>_raw_snp_calls.tsv.gz
└── split/
    └── <sample_id>.parquet
```

### Output File Descriptions

*   **`split/`**: Contains per-sample parquet files, created if the pipeline is started with a single `--input_parquet` file.
*   **`extract/`**: Contains results from the SNP extraction process.
    *   `<sample_id>_pileup.tsv.gz`: Pileup data for each SNP.
    *   `<sample_id>_raw_snp_calls.tsv.gz`: Raw SNP calls from the initial extraction.
*   **`calculate/`**: Contains the final analysis results.
    *   `<sample_id>_pileup_lr.tsv`: A tab-separated file with the estimated fetal fraction and the log-likelihood ratio (LR) for each chromosome.

        **Format of `<sample_id>_pileup_lr.tsv`:**

        | Chrom |   LR   | Fetal Fraction |
        |:------|:------:|:--------------:|
        | chr1  |  0.52  |      0.12      |
        | chr2  |  1.23  |      0.12      |
        | ...   |  ...   |      ...       |

## Data Simulation Utility

The `bin` directory contains a Python script `simulate_sequencing_data.py` for generating simulated SNP sequencing data. This can be useful for testing the pipeline or for research purposes.

### Usage

```bash
python3 bin/simulate_sequencing_data.py [OPTIONS]
```

### Main Options

*   `--n_repeats`: Number of repetitions per parameter combination.
*   `--model_accuracy`: Model accuracy for simulations.
*   `--trisomy_chr`: Chromosome to simulate trisomy for (e.g., `chr21`).
*   `--min_depth`, `--max_depth`, `--num_depth`: Parameters for sequencing depth range.
*   `--min_ff`, `--max_ff`, `--num_ff`: Parameters for fetal fraction range.
*   `--potential_snp_path`: Path to a TSV file with variant data.
*   `--output_dir`: Directory to save the simulated data.
*   `--n_processes`: Number of processes for parallel execution.

The simulated data is saved in parquet format in the specified output directory, organized by simulation parameters.
