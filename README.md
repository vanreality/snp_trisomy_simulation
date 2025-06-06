# SNP Sequencing Data Simulator for cfDNA Analysis

A robust Python tool for simulating SNP sequencing data from cell-free DNA (cfDNA) samples of pregnant women carrying either diploid or triploid fetuses. This simulator models the complex mixture of maternal and fetal DNA in blood samples, accounting for various biological and technical factors.

## Features

### 🧬 Biological Modeling
- **Diploid inheritance**: Normal Mendelian inheritance patterns
- **Trisomy inheritance**: Simulation of chromosomal trisomy (e.g., chr16, chr18, chr21)
- **Hardy-Weinberg equilibrium**: Realistic population-level allele frequencies
- **Parental genotype generation**: Based on population allele frequencies

### 🔬 Technical Simulation
- **Sequencing depth variation**: Poisson-distributed read depths
- **Fetal fraction modeling**: Configurable proportion of fetal cfDNA
- **Machine learning classifier effects**: Simulation of read classification accuracy
- **Multiple parameter combinations**: Systematic exploration of parameter space

### 🚀 Performance & Robustness
- **Parallel processing**: Multi-core execution for efficiency
- **Comprehensive error handling**: Robust validation and error management
- **Rich CLI interface**: Beautiful command-line interface with progress tracking
- **Detailed logging**: Configurable logging for debugging and monitoring

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages (install via pip)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd snp_simulate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python simulate_sequencing_data.py --help
```

## Usage

### Basic Usage
```bash
python simulate_sequencing_data.py \
    --potential_snp_path data/variants.tsv \
    --output_dir results \
    --n_repeats 10
```

### Advanced Configuration
```bash
python simulate_sequencing_data.py \
    --n_repeats 50 \
    --model_accuracy 0.85 \
    --trisomy_chr chr21 \
    --min_depth 30 \
    --max_depth 200 \
    --num_depth 15 \
    --min_ff 0.01 \
    --max_ff 0.08 \
    --num_ff 12 \
    --potential_snp_path data/filtered_variants.tsv \
    --output_dir results/trisomy21_analysis \
    --n_processes 8 \
    --verbose
```

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--n_repeats` | int | 1 | Number of simulation replicates per parameter combination |
| `--model_accuracy` | float | 0.81 | Classifier accuracy (0.0-1.0) |
| `--trisomy_chr` | str | chr16 | Chromosome to simulate trisomy for |
| `--min_depth` | int | 50 | Minimum sequencing depth |
| `--max_depth` | int | 150 | Maximum sequencing depth |
| `--num_depth` | int | 11 | Number of depth points to simulate |
| `--min_ff` | float | 0.005 | Minimum fetal fraction |
| `--max_ff` | float | 0.05 | Maximum fetal fraction |
| `--num_ff` | int | 10 | Number of fetal fraction points |
| `--potential_snp_path` | path | filtered_senddmr_igtc_ChinaMAP.tsv | Input SNP data file |
| `--output_dir` | path | results | Output directory |
| `--n_processes` | int | CPU-1 | Number of parallel processes |
| `--verbose` | flag | False | Enable verbose logging |

## Input Data Format

The input SNP data file should be a tab-separated file with the following columns:
1. `chr` - Chromosome (e.g., chr1, chr2, ...)
2. `pos` - Position (integer)
3. `id` - Variant ID
4. `ref` - Reference allele
5. `alt` - Alternative allele
6. `qual` - Quality score
7. `filter` - Filter status
8. `info` - INFO field containing AF=<frequency>

Example:
```
chr1    1000000    rs123456    A    G    99    PASS    AF=0.25;...
chr1    1000100    rs123457    C    T    95    PASS    AF=0.15;...
```

## Output Files

The simulator generates TSV files for each parameter combination:

### Disomy Files
`disomy_{depth}_{fetal_fraction}_{repeat}.tsv`

### Trisomy Files
`trisomy_{depth}_{fetal_fraction}_{repeat}.tsv`

### Output Columns
- `chr`, `pos`, `ref`, `alt` - Variant information
- `af` - Allele frequency
- `maternal_gt`, `paternal_gt`, `fetal_gt` - Genotypes
- `current_depth` - Simulated sequencing depth
- `maternal_ref_reads`, `maternal_alt_reads` - Maternal read counts
- `fetal_ref_reads`, `fetal_alt_reads` - Fetal read counts
- `cfDNA_ref_reads`, `cfDNA_alt_reads` - Combined cfDNA read counts
- `filtered_fetal_ref`, `filtered_fetal_alt` - Correctly classified fetal reads
- `misclassified_maternal_ref`, `misclassified_maternal_alt` - Misclassified maternal reads
- `is_trisomy_chr` - Whether variant is on trisomy chromosome (trisomy files only)

## Simulation Logic

### 1. Genotype Generation
- Parental genotypes generated using Hardy-Weinberg equilibrium
- Fetal genotypes follow Mendelian inheritance (diploid) or include extra chromosome (trisomy)

### 2. Sequencing Simulation
- Total depth drawn from Poisson distribution
- Fetal vs maternal reads allocated based on fetal fraction
- Individual allele counts simulated using binomial distribution

### 3. Classifier Simulation
- Models imperfect separation of maternal and fetal reads
- True positive rate = model accuracy
- False positive rate = 1 - model accuracy

### 4. Trisomy Modeling
- Extra chromosome randomly inherited from mother or father
- Fetal fraction increased by 50% for trisomy chromosomes
- Genotype encoding: 0-3 alternate alleles instead of 0-2

## Error Handling

The simulator includes comprehensive error handling for:
- Invalid input parameters
- Missing or corrupted data files
- Numerical edge cases (zero depth, invalid allele frequencies)
- File I/O errors
- Memory and processing errors

## Performance Optimization

- **Vectorized operations**: NumPy arrays for efficient computation
- **Parallel processing**: Multiprocessing for parameter combinations
- **Memory management**: Efficient data structures and garbage collection
- **Thread-safe RNG**: Separate random number generators for parallel execution

## Logging and Monitoring

- **Rich progress bars**: Visual progress tracking
- **Detailed logging**: Configurable log levels and output
- **Error reporting**: Clear error messages and stack traces
- **Statistics display**: Summary tables for input data and parameters

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this simulator in your research, please cite:

```
SNP Sequencing Data Simulator for cfDNA Analysis
[Add publication details when available]
```

## Contact

For questions, issues, or contributions, please contact [contact information]. 