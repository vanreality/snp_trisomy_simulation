import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from numpy.random import default_rng
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from functools import partial
from itertools import chain
import click


# Initialize a single RNG
rng = default_rng()


def extract_AF(x):
    """Parse the AF (allele frequency) field out of the INFO column."""
    AF = x['info'].split('AF=')[1].split(';')[0]
    return float(AF)


def generate_genotype(af):
    """Generates a genotype based on allele frequency (Hardy-Weinberg).

    Args:
        af (float): Allele frequency of the alternative allele.

    Returns:
        int: Genotype code (0 = hom-ref, 1 = het, 2 = hom-alt).
    """
    p_hom_ref = (1 - af) ** 2
    p_het = 2 * af * (1 - af)
    p_hom_alt = af ** 2

    return rng.choice([0, 1, 2], p=[p_hom_ref, p_het, p_hom_alt])


def get_disomy_fetal_genotype(maternal_gt, paternal_gt):
    """Determines the fetal genotype given parental genotypes, using RNG.

    Args:
        maternal_gt (int): Maternal genotype (0, 1, or 2).
        paternal_gt (int): Paternal genotype (0, 1, or 2).

    Returns:
        int: Fetal genotype (0, 1, or 2).
    """
    if maternal_gt == 0:
        if paternal_gt == 0:
            return 0
        elif paternal_gt == 1:
            return rng.choice([0, 1])      # R,R or R,A
        else:  # paternal_gt == 2
            return 1                       # R,A
    elif maternal_gt == 1:
        if paternal_gt == 0:
            return rng.choice([0, 1])     # R,R or R,A
        elif paternal_gt == 1:
            return rng.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
        else:  # paternal_gt == 2
            return rng.choice([1, 2])     # R,A or A,A
    else:  # maternal_gt == 2
        if paternal_gt == 0:
            return 1                       # R,A
        elif paternal_gt == 1:
            return rng.choice([1, 2])     # R,A or A,A
        else:
            return 2                       # A,A


def get_trisomy_fetal_genotype(maternal_gt, paternal_gt):
    """Determines the trisomy fetal genotype given parental genotypes.

    The extra (third) chromosome is randomly inherited from the mother or father.
    The genotype is encoded as the count of alternate alleles (0-3).

    Args:
        maternal_gt (int): Maternal genotype (0, 1, or 2).
        paternal_gt (int): Paternal genotype (0, 1, or 2).

    Returns:
        int: Trisomy fetal genotype (0 = 0 alternate alleles, 1, 2, or 3).
    """
    # First get a disomy genotype (0, 1, or 2 alt alleles)
    disomy_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt)
    # Convert disomy genotype code to count of alternate alleles
    # (0→0, 1→1, 2→2)
    disomy_alt_count = disomy_gt

    # Randomly choose whether the extra chromosome comes from mother or father
    origin = rng.choice(['maternal', 'paternal'])
    if origin == 'maternal':
        # Sample one allele from the mother's two alleles
        if maternal_gt == 0:
            extra_alt = 0
        elif maternal_gt == 1:
            # One ref, one alt → 50:50 chance
            extra_alt = rng.choice([0, 1])
        else:  # maternal_gt == 2
            extra_alt = 1
    else:
        # Sample one allele from the father's two alleles
        if paternal_gt == 0:
            extra_alt = 0
        elif paternal_gt == 1:
            extra_alt = rng.choice([0, 1])
        else:  # paternal_gt == 2
            extra_alt = 1

    # Trisomy genotype is total count of alternate alleles (0-3)
    trisomy_alt_count = disomy_alt_count + extra_alt
    return trisomy_alt_count


def simulate_disomy_sequencing(depth, fetal_fraction, maternal_gt, fetal_gt):
    """Simulates disomy sequencing reads (binomial).

    Args:
        depth (int): Total sequencing depth for this SNP.
        fetal_fraction (float): Proportion of fetal cfDNA.
        maternal_gt (int): Maternal genotype (0,1,2).
        fetal_gt (int): Fetal genotype (0,1,2).

    Returns:
        tuple[int,int,int,int,int,int]: (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads).
    """
    # Simulate maternal and fetal reads counts
    fetal_reads = rng.binomial(depth, fetal_fraction)
    maternal_reads = depth - fetal_reads

    # Simulate maternal allele counts
    maternal_alt_prob = maternal_gt / 2
    maternal_alt_reads = rng.binomial(maternal_reads, maternal_alt_prob)
    maternal_ref_reads = maternal_reads - maternal_alt_reads

    # Simulate fetal allele counts
    fetal_alt_prob = fetal_gt / 2
    fetal_alt_reads = rng.binomial(fetal_reads, fetal_alt_prob)
    fetal_ref_reads = fetal_reads - fetal_alt_reads

    # Calculate cfDNA allele frequency
    cfDNA_alt_reads = maternal_alt_reads + fetal_alt_reads
    cfDNA_ref_reads = maternal_ref_reads + fetal_ref_reads

    return maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads


def simulate_trisomy_sequencing(depth, fetal_fraction, maternal_gt, fetal_gt):
    """Simulates trisomy sequencing reads (binomial).

    Args:
        depth (int): Total sequencing depth for this SNP.
        fetal_fraction (float): Proportion of fetal cfDNA.
        maternal_gt (int): Maternal genotype (0,1,2).
        fetal_gt (int): Fetal genotype (0,1,2).

    Returns:
        tuple[int,int,int,int,int,int]: (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads).
    """
    # Simulate maternal and fetal reads counts
    fetal_reads = rng.binomial(depth, fetal_fraction * 1.5)
    maternal_reads = depth - fetal_reads

    # Simulate maternal allele counts
    maternal_alt_prob = maternal_gt / 2
    maternal_alt_reads = rng.binomial(maternal_reads, maternal_alt_prob)
    maternal_ref_reads = maternal_reads - maternal_alt_reads

    # Simulate fetal allele counts
    fetal_alt_prob = fetal_gt / 3
    fetal_alt_reads = rng.binomial(fetal_reads, fetal_alt_prob)
    fetal_ref_reads = fetal_reads - fetal_alt_reads

    # Calculate cfDNA allele frequency
    cfDNA_alt_reads = maternal_alt_reads + fetal_alt_reads
    cfDNA_ref_reads = maternal_ref_reads + fetal_ref_reads

    return maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads


def classifier_filtering(model_accuracy, maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads):
    """Use a ML or LLM model to filter out maternal reads
    Args:
        model_accuracy (float): Accuracy of the model.
        maternal_ref_reads (int): Number of maternal reference reads.
        maternal_alt_reads (int): Number of maternal alternate reads.
        fetal_ref_reads (int): Number of fetal reference reads.
        fetal_alt_reads (int): Number of fetal alternate reads.

    Returns:
        tuple[int,int,int,int]: (correct_fetal_ref, correct_fetal_alt, misclassified_maternal_ref, misclassified_maternal_alt).
    """
    # True positive
    correct_fetal_ref = rng.binomial(fetal_ref_reads, model_accuracy)
    correct_fetal_alt = rng.binomial(fetal_alt_reads, model_accuracy)

    # False positive
    misclassified_maternal_ref = rng.binomial(maternal_ref_reads, 1 - model_accuracy)
    misclassified_maternal_alt = rng.binomial(maternal_alt_reads, 1 - model_accuracy)

    return correct_fetal_ref, correct_fetal_alt, misclassified_maternal_ref, misclassified_maternal_alt


def run_single_disomy_simulation_set(variants_df, depth_lambda, fetal_fraction_value, model_accuracy):
    """Runs multiple disomy simulation replicates for a given (depth, fetal_fraction).

    Args:
        variants_df (pd.DataFrame): Must have column 'AF'.
        depth_lambda (int): Poisson λ for per-SNP depth.
        fetal_fraction_value (float): Fraction of fetal cfDNA.
        model_accuracy (float): Accuracy of the model.

    Returns:
        pd.DataFrame: DataFrame containing simulation results for each variant.
    """
    records = []
    for _, row in variants_df.iterrows():
        af = row['af']

        maternal_gt = generate_genotype(af)
        paternal_gt = generate_genotype(af)
        fetal_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt)

        # Draw a Poisson depth for this SNP
        current_depth = rng.poisson(depth_lambda)
        if current_depth == 0:
            continue

        (maternal_ref_reads, 
            maternal_alt_reads, 
            fetal_ref_reads, 
            fetal_alt_reads, 
            cfDNA_ref_reads, 
            cfDNA_alt_reads) = simulate_disomy_sequencing(current_depth, fetal_fraction_value, maternal_gt, fetal_gt)
        (filtered_fetal_ref, 
            filtered_fetal_alt, 
            misclassified_maternal_ref, 
            misclassified_maternal_alt, 
            ) = classifier_filtering(model_accuracy, maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads)
        
        records.append({
            'chr': row['chr'],
            'pos': row['pos'],
            'ref': row['ref'],
            'alt': row['alt'],
            'af': af,
            'maternal_gt': maternal_gt,
            'paternal_gt': paternal_gt,
            'fetal_gt': fetal_gt,
            'current_depth': current_depth,
            'maternal_ref_reads': maternal_ref_reads,
            'maternal_alt_reads': maternal_alt_reads,
            'fetal_ref_reads': fetal_ref_reads,
            'fetal_alt_reads': fetal_alt_reads,
            'cfDNA_ref_reads': cfDNA_ref_reads,
            'cfDNA_alt_reads': cfDNA_alt_reads,
            'filtered_fetal_ref': filtered_fetal_ref,
            'filtered_fetal_alt': filtered_fetal_alt,
            'misclassified_maternal_ref': misclassified_maternal_ref,
            'misclassified_maternal_alt': misclassified_maternal_alt,
        })

    return pd.DataFrame(records)


def run_single_trisomy_simulation_set(variants_df, depth_lambda, fetal_fraction_value, model_accuracy, trisomy_chr):
    """Runs multiple disomy simulation replicates for a given (depth, fetal_fraction).

    Args:
        variants_df (pd.DataFrame): Must have column 'AF'.
        depth_lambda (int): Poisson λ for per-SNP depth.
        fetal_fraction_value (float): Fraction of fetal cfDNA.
        model_accuracy (float): Accuracy of the model.

    Returns:
        pd.DataFrame: DataFrame containing simulation results for each variant.
    """
    records = []
    for _, row in variants_df.iterrows():
        af = row['af']

        maternal_gt = generate_genotype(af)
        paternal_gt = generate_genotype(af)

        if row['chr'] == trisomy_chr:
            fetal_gt = get_trisomy_fetal_genotype(maternal_gt, paternal_gt)
        else:
            fetal_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt)

        # Draw a Poisson depth for this SNP
        current_depth = rng.poisson(depth_lambda)
        if current_depth == 0:
            continue

        # Simulate sequencing reads
        if row['chr'] == trisomy_chr:
            (maternal_ref_reads, 
                maternal_alt_reads, 
                fetal_ref_reads, 
                fetal_alt_reads, 
                cfDNA_ref_reads, 
                cfDNA_alt_reads) = simulate_trisomy_sequencing(current_depth, fetal_fraction_value, maternal_gt, fetal_gt)
        else:
            (maternal_ref_reads, 
                maternal_alt_reads, 
                fetal_ref_reads, 
                fetal_alt_reads, 
                cfDNA_ref_reads, 
                cfDNA_alt_reads) = simulate_disomy_sequencing(current_depth, fetal_fraction_value, maternal_gt, fetal_gt)
            
        # Filtering using the model
        (filtered_fetal_ref, 
            filtered_fetal_alt, 
            misclassified_maternal_ref, 
            misclassified_maternal_alt, 
            ) = classifier_filtering(model_accuracy, maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads)
        
        records.append({
            'chr': row['chr'],
            'pos': row['pos'],
            'ref': row['ref'],
            'alt': row['alt'],
            'af': af,
            'maternal_gt': maternal_gt,
            'paternal_gt': paternal_gt,
            'fetal_gt': fetal_gt,
            'current_depth': current_depth,
            'maternal_ref_reads': maternal_ref_reads,
            'maternal_alt_reads': maternal_alt_reads,
            'fetal_ref_reads': fetal_ref_reads,
            'fetal_alt_reads': fetal_alt_reads,
            'cfDNA_ref_reads': cfDNA_ref_reads,
            'cfDNA_alt_reads': cfDNA_alt_reads,
            'filtered_fetal_ref': filtered_fetal_ref,
            'filtered_fetal_alt': filtered_fetal_alt,
            'misclassified_maternal_ref': misclassified_maternal_ref,
            'misclassified_maternal_alt': misclassified_maternal_alt,
        })

    return pd.DataFrame(records)


def run_simulation(args):
    """
    Wrapper function for parallel execution
    Args:
        args: Tuple of (depth, ff, repeat_idx, trisomy_chr, potential_df, model_accuracy, output_dir)
    """
    depth, ff, repeat_idx, trisomy_chr, potential_df, model_accuracy, output_dir = args

    # Run disomy simulation
    disomy_simulated_df = run_single_disomy_simulation_set(
        potential_df, depth, ff, model_accuracy
    )
    disomy_simulated_df.to_csv(f"{output_dir}/disomy_{depth}_{ff}_{repeat_idx}.tsv", sep='\t', index=False, header=True)

    # Run trisomy simulation
    trisomy_simulated_df = run_single_trisomy_simulation_set(
        potential_df, depth, ff, model_accuracy, trisomy_chr
    )
    trisomy_simulated_df.to_csv(f"{output_dir}/trisomy_{depth}_{ff}_{repeat_idx}.tsv", sep='\t', index=False, header=True)


@click.command()
@click.option('--n_repeats', default=1, type=int, help='Number of repetitions per parameter combination')
@click.option('--model_accuracy', default=0.81, type=float, help='Model accuracy for simulations')
@click.option('--trisomy_chr', default='chr16', type=str, help='Chromosome to simulate trisomy for')
@click.option('--min_depth', default=50, type=int, help='Minimum sequencing depth')
@click.option('--max_depth', default=150, type=int, help='Maximum sequencing depth')
@click.option('--num_depth', default=11, type=int, help='Number of depth points')
@click.option('--min_ff', default=0.005, type=float, help='Minimum fetal fraction')
@click.option('--max_ff', default=0.05, type=float, help='Maximum fetal fraction')
@click.option('--num_ff', default=10, type=int, help='Number of fetal fraction points')
@click.option('--potential_snp_path', default='filtered_senddmr_igtc_ChinaMAP.tsv', 
              type=click.Path(exists=True), help='Path to variant data TSV file')
@click.option('--output_dir', default='results', 
              type=str, help='Output path for tsv file')
def main(n_repeats, model_accuracy, trisomy_chr, min_depth, max_depth, num_depth,
         min_ff, max_ff, num_ff, potential_snp_path, output_dir):
    # Load variant data
    potential_df = pd.read_csv(
        potential_snp_path,
        sep='\t',
        names=['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info']
    )
    potential_df['af'] = potential_df.apply(extract_AF, axis=1)
    
    # Generate parameter ranges
    depth_range = np.linspace(min_depth, max_depth, num=num_depth, dtype=int)
    ff_range = np.linspace(min_ff, max_ff, num=num_ff)
    
    # Generate all parameter combinations
    param_combinations = [
        (depth, ff, repeat_idx, trisomy_chr, potential_df, model_accuracy, output_dir)
        for depth in depth_range
        for ff in ff_range
        for repeat_idx in range(n_repeats)
    ]
    
    # Use multiprocessing
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        pool.map(run_simulation, param_combinations)


if __name__ == '__main__':
    main()