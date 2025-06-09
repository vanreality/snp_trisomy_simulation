#!/usr/bin/env python3
"""
Fetal Fraction Estimation and Likelihood Ratio Calculator for Aneuploidy Detection

This script analyzes cell-free DNA (cfDNA) sequencing data to:
1. Estimate fetal fraction from maternal plasma samples
2. Calculate likelihood ratios for trisomy vs. disomy detection

The analysis uses SNP read counts and population allele frequencies to perform
statistical inference on fetal chromosomal abnormalities.
"""

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
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint


# Initialize rich console for beautiful output
console = Console()


# Global helper functions for multiprocessing (need to be at module level)
def _log_binomial_coeff_global(n: int, k: int) -> float:
    """Compute log(C(n,k)) using log-gamma function for numerical stability."""
    if k < 0 or k > n or n < 0:
        return float('-inf')
    if k == 0 or k == n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _log_binomial_pmf_global(k: int, n: int, p: float) -> float:
    """Compute log-probability mass function of Binomial(n, p) at k."""
    if n < 0 or k < 0 or k > n:
        return float('-inf')
    if p <= 0.0:
        return 0.0 if k == 0 else float('-inf')
    if p >= 1.0:
        return 0.0 if k == n else float('-inf')
    
    log_coeff = _log_binomial_coeff_global(n, k)
    return log_coeff + k * math.log(p) + (n - k) * math.log(1 - p)


def _log_sum_exp_global(log_values: np.ndarray) -> float:
    """Compute log(sum(exp(x_i))) for numerical stability."""
    if len(log_values) == 0:
        return float('-inf')
    max_val = np.max(log_values)
    if max_val == float('-inf'):
        return float('-inf')
    return max_val + math.log(np.sum(np.exp(log_values - max_val)))


def _compute_ff_likelihood_chunk(args: Tuple) -> Dict[float, float]:
    """
    Worker function to compute log-likelihood for a chunk of fetal fraction values.
    
    This function processes a subset of FF values in parallel to speed up computation.
    
    Args:
        args: Tuple containing (f_values_chunk, shared_data)
            - f_values_chunk: Array of FF values to process
            - shared_data: Tuple of (p_arr, n_arr, alt_arr, log_prior_G, maternal_alt_frac, fetal_alt_frac)
    
    Returns:
        Dictionary mapping FF values to their computed log-likelihoods
    """
    f_values_chunk, shared_data = args
    p_arr, n_arr, alt_arr, log_prior_G, maternal_alt_frac, fetal_alt_frac = shared_data
    
    chunk_results = {}
    num_snps = len(p_arr)
    
    for FF in f_values_chunk:
        total_log_lik = 0.0
        
        # Process each SNP for this FF value
        for i in range(num_snps):
            n_i = n_arr[i]
            k_i = alt_arr[i]
            
            # Compute mixture probabilities for all genotypes
            mu_values = (1 - FF) * maternal_alt_frac + FF * fetal_alt_frac[i, :]
            
            # Compute log probabilities for each genotype
            log_probs = np.zeros(3)
            for g in range(3):
                mu_g = mu_values[g]
                log_pmf = _log_binomial_pmf_global(k_i, n_i, mu_g)
                log_probs[g] = log_prior_G[i, g] + log_pmf
            
            # Use log-sum-exp to compute log-likelihood for this SNP
            log_L_i = _log_sum_exp_global(log_probs)
            total_log_lik += log_L_i
        
        chunk_results[FF] = total_log_lik
    
    return chunk_results


def estimate_fetal_fraction(
    background_chr_df: pd.DataFrame, 
    f_min: float = 0.001, 
    f_max: float = 0.5, 
    f_step: float = 0.001,
    ncpus: int = cpu_count()
) -> Tuple[float, Dict[float, float]]:
    """
    Estimate fetal fraction (FF) from maternal cfDNA SNP read counts using maximum likelihood estimation.
    
    This function implements a mixture model approach where:
    - Maternal genotypes follow Hardy-Weinberg equilibrium
    - Fetal genotypes depend on maternal and paternal contributions
    - Observed alt-allele fractions are modeled as a mixture of maternal and fetal contributions
    
    The likelihood is computed by marginalizing over all possible maternal genotype combinations
    and selecting the fetal fraction that maximizes the total log-likelihood across all SNPs.
    
    Args:
        background_chr_df (pd.DataFrame): DataFrame containing SNP information with columns:
            - 'chr': chromosome identifier (string or int)
            - 'pos': genomic position (int)
            - 'af': population allele frequency (float in [0,1])
            - 'cfDNA_ref_reads': number of reference allele reads (int >= 0)
            - 'cfDNA_alt_reads': number of alternative allele reads (int >= 0)
                 f_min (float, optional): Minimum fetal fraction to search. Defaults to 0.001.
         f_max (float, optional): Maximum fetal fraction to search. Defaults to 0.5.
         f_step (float, optional): Step size for grid search. Defaults to 0.001.
         ncpus (int, optional): Number of CPU cores to use. Defaults to cpu_count().
    
    Returns:
        Tuple[float, Dict[float, float]]: A tuple containing:
            - best_ff: Estimated fetal fraction that maximizes log-likelihood
            - log_likelihoods: Dictionary mapping FF values to their log-likelihoods
    
    Raises:
        ValueError: If input DataFrame is missing required columns or contains invalid data
        ValueError: If search parameters are invalid (f_min >= f_max, negative values, etc.)
    """
    # Input validation
    required_columns = {'chr', 'pos', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads'}
    if not required_columns.issubset(background_chr_df.columns):
        missing_cols = required_columns - set(background_chr_df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not (0 <= f_min < f_max <= 1):
        raise ValueError(f"Invalid search range: f_min={f_min}, f_max={f_max}")
    
    if f_step <= 0:
        raise ValueError(f"Step size must be positive: f_step={f_step}")
    
    if len(background_chr_df) == 0:
        raise ValueError("Input DataFrame is empty")
    
    # Helper functions for numerical stability
    def _log_binomial_coeff(n: int, k: int) -> float:
        """Compute log(C(n,k)) using log-gamma function for numerical stability."""
        if k < 0 or k > n or n < 0:
            return float('-inf')
        if k == 0 or k == n:
            return 0.0
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    
    def _log_binomial_pmf(k: int, n: int, p: float) -> float:
        """Compute log-probability mass function of Binomial(n, p) at k."""
        if n < 0 or k < 0 or k > n:
            return float('-inf')
        if p <= 0.0:
            return 0.0 if k == 0 else float('-inf')
        if p >= 1.0:
            return 0.0 if k == n else float('-inf')
        
        log_coeff = _log_binomial_coeff(n, k)
        return log_coeff + k * math.log(p) + (n - k) * math.log(1 - p)
    
    def _log_sum_exp(log_values: np.ndarray) -> float:
        """Compute log(sum(exp(x_i))) for numerical stability."""
        if len(log_values) == 0:
            return float('-inf')
        max_val = np.max(log_values)
        if max_val == float('-inf'):
            return float('-inf')
        return max_val + math.log(np.sum(np.exp(log_values - max_val)))
    
    # Data preprocessing and validation
    df_clean = background_chr_df.copy()
    
    # Validate allele frequencies
    invalid_af = (df_clean['af'] < 0) | (df_clean['af'] > 1)
    if invalid_af.any():
        console.print(f"[yellow]Warning: Removing {invalid_af.sum()} SNPs with invalid allele frequencies[/yellow]")
        df_clean = df_clean[~invalid_af]
    
    # Validate read counts
    invalid_reads = (df_clean['cfDNA_ref_reads'] < 0) | (df_clean['cfDNA_alt_reads'] < 0)
    if invalid_reads.any():
        console.print(f"[yellow]Warning: Removing {invalid_reads.sum()} SNPs with negative read counts[/yellow]")
        df_clean = df_clean[~invalid_reads]
    
    # Filter out SNPs with zero coverage
    df_clean['total_reads'] = df_clean['cfDNA_ref_reads'] + df_clean['cfDNA_alt_reads']
    zero_coverage = df_clean['total_reads'] == 0
    if zero_coverage.any():
        console.print(f"[yellow]Warning: Removing {zero_coverage.sum()} SNPs with zero coverage[/yellow]")
        df_clean = df_clean[df_clean['total_reads'] > 0]
    
    if len(df_clean) == 0:
        raise ValueError("No valid SNPs remaining after filtering")
    
    # Convert to numpy arrays for performance
    p_arr = df_clean['af'].to_numpy(dtype=np.float64)
    ref_arr = df_clean['cfDNA_ref_reads'].to_numpy(dtype=np.int32)
    alt_arr = df_clean['cfDNA_alt_reads'].to_numpy(dtype=np.int32)
    n_arr = ref_arr + alt_arr
    
    num_snps = len(p_arr)
    console.print(f"[green]Processing {num_snps} SNPs for fetal fraction estimation[/green]")
    
    # Pre-compute genotype-specific parameters
    # Maternal alt-allele fractions: 0/0 -> 0.0, 0/1 -> 0.5, 1/1 -> 1.0
    maternal_alt_frac = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    
    # Hardy-Weinberg priors and fetal alt-allele fractions
    prior_G = np.zeros((num_snps, 3), dtype=np.float64)
    fetal_alt_frac = np.zeros((num_snps, 3), dtype=np.float64)
    
    # Vectorized computation of priors and fetal fractions
    prior_G[:, 0] = (1 - p_arr) ** 2        # P(mother = 0/0)
    prior_G[:, 1] = 2 * p_arr * (1 - p_arr)  # P(mother = 0/1)
    prior_G[:, 2] = p_arr ** 2              # P(mother = 1/1)
    
    fetal_alt_frac[:, 0] = 0.5 * p_arr               # E[fetal_alt | mother=0/0]
    fetal_alt_frac[:, 1] = 0.25 + 0.5 * p_arr       # E[fetal_alt | mother=0/1]
    fetal_alt_frac[:, 2] = 0.5 + 0.5 * p_arr        # E[fetal_alt | mother=1/1]
    
    # Compute log priors for numerical stability
    with np.errstate(divide='ignore'):
        log_prior_G = np.log(prior_G)
    
    # Grid search over fetal fraction values
    f_values = np.arange(f_min, f_max + f_step, f_step)
    log_likelihoods = {}
    
    console.print(f"[cyan]Using {ncpus} CPU cores for parallel processing[/cyan]")
    
    # Prepare shared data for worker processes
    shared_data = (p_arr, n_arr, alt_arr, log_prior_G, maternal_alt_frac, fetal_alt_frac)
    
    # Split FF values into chunks for parallel processing
    chunk_size = max(1, len(f_values) // ncpus)
    f_value_chunks = [f_values[i:i + chunk_size] for i in range(0, len(f_values), chunk_size)]
    
    # Prepare arguments for worker processes
    worker_args = [(chunk, shared_data) for chunk in f_value_chunks]
    
    # Progress bar for fetal fraction estimation
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Estimating fetal fraction...", total=len(f_value_chunks))
        
        # Use multiprocessing for parallel computation
        if ncpus == 1:
            # Single-threaded execution for debugging or small datasets
            for args in worker_args:
                chunk_results = _compute_ff_likelihood_chunk(args)
                log_likelihoods.update(chunk_results)
                progress.advance(task)
        else:
            # Multi-threaded execution
            with Pool(processes=ncpus) as pool:
                chunk_results_list = []
                
                # Submit all jobs
                async_results = [pool.apply_async(_compute_ff_likelihood_chunk, (args,)) for args in worker_args]
                
                # Collect results as they complete
                for async_result in async_results:
                    chunk_results = async_result.get()
                    log_likelihoods.update(chunk_results)
                    progress.advance(task)
    
    # Find the fetal fraction with maximum likelihood
    best_ff = max(log_likelihoods, key=log_likelihoods.get)
    
    console.print(f"[green]✓ Estimated fetal fraction: {best_ff:.3f}[/green]")
    
    return best_ff, log_likelihoods


def LR_calculator(input_df: pd.DataFrame, fetal_fraction: float) -> float:
    """
    Calculate the likelihood ratio (LR) of trisomy vs. disomy for a target chromosome.
    
    This function implements a comprehensive statistical model that:
    1. Considers all possible maternal and paternal genotype combinations
    2. Models fetal genotype distributions under both disomy and trisomy hypotheses
    3. Accounts for mixture effects due to fetal fraction in maternal plasma
    4. Computes likelihood ratio as L_trisomy / L_disomy
    
    Args:
        input_df (pd.DataFrame): DataFrame with target chromosome SNP data containing:
            - 'chr': chromosome identifier (string or int)
            - 'pos': genomic position (int)
            - 'af': population allele frequency (float in [0,1])
            - 'cfDNA_ref_reads': number of reference allele reads (int >= 0)
            - 'cfDNA_alt_reads': number of alternative allele reads (int >= 0)
        fetal_fraction (float): Estimated fetal fraction in maternal plasma (0 < ff < 1)
    
    Returns:
        float: Likelihood ratio LR = L_trisomy / L_disomy. 
               Returns np.inf if L_disomy = 0, and 0.0 if L_trisomy = 0.
    
    Raises:
        ValueError: If input DataFrame is missing required columns
        ValueError: If fetal_fraction is not in valid range (0, 1)
        ValueError: If no valid SNPs are found
    """
    # Input validation
    required_cols = {'chr', 'pos', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads'}
    if not required_cols.issubset(input_df.columns):
        missing = required_cols - set(input_df.columns)
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")
    
    if not (0.0 < fetal_fraction < 1.0):
        raise ValueError(f"fetal_fraction must be between 0 and 1 (exclusive): {fetal_fraction}")
    
    if len(input_df) == 0:
        raise ValueError("No SNPs found in input DataFrame")
    
    # Data cleaning and validation
    df_clean = input_df.copy()
    
    # Remove SNPs with invalid data
    invalid_mask = (
        (df_clean['af'] < 0) | (df_clean['af'] > 1) |
        (df_clean['cfDNA_ref_reads'] < 0) | (df_clean['cfDNA_alt_reads'] < 0) |
        ((df_clean['cfDNA_ref_reads'] + df_clean['cfDNA_alt_reads']) == 0)
    )
    
    if invalid_mask.any():
        console.print(f"[yellow]Warning: Removing {invalid_mask.sum()} invalid SNPs[/yellow]")
        df_clean = df_clean[~invalid_mask]
    
    if len(df_clean) == 0:
        raise ValueError("No valid SNPs remaining after filtering")
    
    # Helper functions for genotype modeling
    def genotype_priors(af: float) -> Dict[str, float]:
        """Compute Hardy-Weinberg genotype priors for a given allele frequency."""
        return {
            '0/0': (1.0 - af) ** 2,
            '0/1': 2.0 * af * (1.0 - af),
            '1/1': af ** 2
        }
    
    def get_fetal_prob_disomy(gm: str, gp: str) -> Dict[str, float]:
        """
        Compute fetal genotype probabilities under disomy given parental genotypes.
        
        Args:
            gm: Maternal genotype ('0/0', '0/1', or '1/1')
            gp: Paternal genotype ('0/0', '0/1', or '1/1')
            
        Returns:
            Dictionary mapping fetal genotypes to their probabilities
        """
        def allele_probs_from_genotype(gt: str) -> Dict[int, float]:
            """Extract allele probabilities from genotype string."""
            a1, a2 = map(int, gt.split('/'))
            counts = {0: 0, 1: 0}
            counts[a1] += 1
            counts[a2] += 1
            return {0: counts[0] / 2.0, 1: counts[1] / 2.0}
        
        pm = allele_probs_from_genotype(gm)
        pp = allele_probs_from_genotype(gp)
        
        fetal_probs = {'0/0': 0.0, '0/1': 0.0, '1/1': 0.0}
        
        # Enumerate all maternal-paternal allele combinations
        for mat_allele, p_mat in pm.items():
            for pat_allele, p_pat in pp.items():
                prob_combo = p_mat * p_pat
                fetal_genotype = f"{min(mat_allele, pat_allele)}/{max(mat_allele, pat_allele)}"
                fetal_probs[fetal_genotype] += prob_combo
        
        return fetal_probs
    
    def get_fetal_prob_trisomy(gm: str, gp: str) -> Dict[int, float]:
        """
        Compute fetal ALT-allele dosage probabilities under trisomy.
        
        Models maternal nondisjunction by drawing 2 alleles from mother
        and 1 allele from father, returning dosage distribution.
        
        Args:
            gm: Maternal genotype ('0/0', '0/1', or '1/1')
            gp: Paternal genotype ('0/0', '0/1', or '1/1')
            
        Returns:
            Dictionary mapping ALT-allele dosages (0,1,2,3) to probabilities
        """
        def allele_probs(gt: str) -> Dict[int, float]:
            """Extract single-allele probabilities from genotype."""
            a1, a2 = map(int, gt.split('/'))
            counts = {0: 0, 1: 0}
            counts[a1] += 1
            counts[a2] += 1
            return {0: counts[0] / 2.0, 1: counts[1] / 2.0}
        
        pm = allele_probs(gm)
        pp = allele_probs(gp)
        
        # Maternal contribution: 2 alleles drawn with replacement
        p_mat_alt = pm[1]
        maternal_dosage_probs = {
            0: (1 - p_mat_alt) ** 2,
            1: 2 * p_mat_alt * (1 - p_mat_alt),
            2: p_mat_alt ** 2
        }
        
        # Paternal contribution: 1 allele
        paternal_alt_prob = pp[1]
        
        # Total fetal dosage = maternal_dosage + paternal_contribution
        fetal_dosage_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for k_m in [0, 1, 2]:
            for c_p in [0, 1]:
                d = k_m + c_p
                prob = maternal_dosage_probs[k_m] * (paternal_alt_prob if c_p == 1 else (1 - paternal_alt_prob))
                fetal_dosage_probs[d] += prob
        
        return fetal_dosage_probs
    
    def log_binomial_pmf(k: int, n: int, p: float) -> float:
        """Compute log-probability mass function of Binomial(n, p) at k."""
        if p < 0.0 or p > 1.0 or n < 0 or k < 0 or k > n:
            return float('-inf')
        
        if p == 0.0:
            return 0.0 if k == 0 else float('-inf')
        if p == 1.0:
            return 0.0 if k == n else float('-inf')
        
        # Use log-gamma for numerical stability
        log_comb = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
        return log_comb + k * math.log(p) + (n - k) * math.log(1.0 - p)
    
    def logsumexp(log_values: np.ndarray) -> float:
        """Numerically stable log-sum-exp computation."""
        if len(log_values) == 0:
            return float('-inf')
        max_val = np.max(log_values)
        if max_val == float('-inf'):
            return float('-inf')
        return max_val + math.log(np.sum(np.exp(log_values - max_val)))
    
    # Main likelihood computation
    logL_disomy = 0.0
    logL_trisomy = 0.0
    genotype_labels = ['0/0', '0/1', '1/1']
    
    console.print(f"[green]Calculating likelihood ratio for {len(df_clean)} SNPs[/green]")
    
    # Process each SNP with progress tracking
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Computing likelihood ratios...", total=len(df_clean))
        
        for idx, snp in df_clean.iterrows():
            # Extract observed data
            ref_count = int(snp['cfDNA_ref_reads'])
            alt_count = int(snp['cfDNA_alt_reads'])
            depth = ref_count + alt_count
            af = float(snp['af'])
            
            # Compute genotype priors
            geno_priors = genotype_priors(af)
            
            # Collect log-likelihood terms for disomy
            log_terms_disomy = []
            log_terms_trisomy = []
            
            # Enumerate all parental genotype combinations
            for gm in genotype_labels:
                p_gm = geno_priors[gm]
                if p_gm <= 0.0:
                    continue
                log_p_gm = math.log(p_gm)
                
                for gp in genotype_labels:
                    p_gp = geno_priors[gp]
                    if p_gp <= 0.0:
                        continue
                    log_p_gp = math.log(p_gp)
                    
                    # Disomy calculation
                    fetal_dist_dis = get_fetal_prob_disomy(gm, gp)
                    d_maternal = sum(int(allele) for allele in gm.split('/'))
                    
                    for gf, p_gf in fetal_dist_dis.items():
                        if p_gf <= 0.0:
                            continue
                        log_p_gf = math.log(p_gf)
                        
                        d_fetal = sum(int(allele) for allele in gf.split('/'))
                        p_alt_exp = ((1.0 - fetal_fraction) * (d_maternal / 2.0) + 
                                   fetal_fraction * (d_fetal / 2.0))
                        
                        log_p_obs = log_binomial_pmf(alt_count, depth, p_alt_exp)
                        if log_p_obs != float('-inf'):
                            log_terms_disomy.append(log_p_gm + log_p_gp + log_p_gf + log_p_obs)
                    
                    # Trisomy calculation
                    fetal_dist_tri = get_fetal_prob_trisomy(gm, gp)
                    
                    for d_fetal, p_fd in fetal_dist_tri.items():
                        if p_fd <= 0.0:
                            continue
                        log_p_fd = math.log(p_fd)
                        
                        p_alt_exp_tri = ((1.0 - 1.5 * fetal_fraction) * (d_maternal / 2.0) + 
                                       1.5 * fetal_fraction * (d_fetal / 3.0))
                        
                        log_p_obs_tri = log_binomial_pmf(alt_count, depth, p_alt_exp_tri)
                        if log_p_obs_tri != float('-inf'):
                            log_terms_trisomy.append(log_p_gm + log_p_gp + log_p_fd + log_p_obs_tri)
            
            # Compute SNP-level log-likelihoods
            if log_terms_disomy:
                logL_snp_disomy = logsumexp(np.array(log_terms_disomy))
                logL_disomy += logL_snp_disomy
            else:
                return 0.0  # No valid disomy combinations
            
            if log_terms_trisomy:
                logL_snp_trisomy = logsumexp(np.array(log_terms_trisomy))
                logL_trisomy += logL_snp_trisomy
            else:
                return 0.0  # No valid trisomy combinations
            
            progress.advance(task)
    
    # Compute final likelihood ratio
    delta_logL = logL_trisomy - logL_disomy
    
    # Handle numerical overflow/underflow
    if delta_logL > 700:
        return float('inf')
    elif delta_logL < -700:
        return 0.0
    else:
        return math.exp(delta_logL)


@click.command()
@click.option(
    '--input-path', '-i',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to input TSV.GZ file containing SNP data'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    required=True,
    help='Output directory for results'
)
@click.option(
    '--ff-min',
    type=click.FloatRange(0.0, 1.0),
    default=0.001,
    help='Minimum fetal fraction for estimation (default: 0.001)'
)
@click.option(
    '--ff-max',
    type=click.FloatRange(0.0, 1.0),
    default=0.3,
    help='Maximum fetal fraction for estimation (default: 0.3)'
)
@click.option(
    '--ff-step',
    type=click.FloatRange(0.0001, 0.1),
    default=0.001,
    help='Step size for fetal fraction grid search (default: 0.001)'
)
@click.option(
    '--chromosomes',
    default='1-22',
    help='Chromosomes to analyze (e.g., "1-22", "1,2,3", or "21"). Default: 1-22'
)
@click.option(
    '--ncpus',
    type=click.IntRange(1, cpu_count()),
    default=cpu_count(),
    help=f'Number of CPU cores to use for parallel processing (default: {cpu_count()})'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def main(
    input_path: Path,
    output_dir: Path,
    ff_min: float,
    ff_max: float,
    ff_step: float,
    chromosomes: str,
    ncpus: int,
    verbose: bool
) -> None:
    """
    Fetal Fraction Estimation and Likelihood Ratio Calculator.
    
    This tool analyzes cell-free DNA sequencing data to estimate fetal fraction
    and calculate likelihood ratios for trisomy detection across chromosomes.
    
         The input file should be a TSV.GZ file with columns:
     chr, pos, af, cfDNA_ref_reads, cfDNA_alt_reads
     
     Results are saved as TSV files with fetal fraction estimates and likelihood ratios.
     
     Multi-threading is used by default to speed up fetal fraction estimation.
     Use --ncpus to control the number of CPU cores used.
    """
    # Configure console output
    if verbose:
        console.print("[blue]Verbose mode enabled[/blue]")
    
    try:
        # Validate and parse chromosome specification
        target_chromosomes = parse_chromosome_list(chromosomes)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Display startup information
        console.print(Panel.fit(
             f"[bold green]Fetal Fraction & Likelihood Ratio Calculator[/bold green]\n"
             f"Input: {input_path}\n"
             f"Output: {output_dir}\n"
             f"Chromosomes: {', '.join(map(str, target_chromosomes))}\n"
             f"FF Range: {ff_min:.3f} - {ff_max:.3f} (step: {ff_step:.3f})\n"
             f"CPU Cores: {ncpus}",
             title="Configuration"
         ))
        
        # Generate output filename based on input
        input_filename = input_path.stem.replace('.tsv', '')
        output_path = output_dir / f'{input_filename}_lr.tsv'
        
        # Load and validate input data
        console.print("[cyan]Loading input data...[/cyan]")
        df = load_and_validate_data(input_path)
        
        console.print(f"[green]✓ Loaded {len(df)} SNPs from {df['chr'].nunique()} chromosomes[/green]")
        
        # Initialize results storage
        results = {}
        
        # Process each target chromosome
        for target_chr in target_chromosomes:
            chr_name = f"chr{target_chr}"
            console.print(f"\n[bold cyan]Processing chromosome {target_chr}[/bold cyan]")
            
            # Filter data for target chromosome
            target_data = df[df['chr'] == chr_name]
            background_data = df[df['chr'] != chr_name]
            
            if len(target_data) == 0:
                console.print(f"[yellow]Warning: No SNPs found for {chr_name}, skipping[/yellow]")
                continue
            
            if len(background_data) == 0:
                console.print(f"[red]Error: No background SNPs available for {chr_name}[/red]")
                continue
            
            console.print(f"Target SNPs: {len(target_data)}, Background SNPs: {len(background_data)}")
            
            try:
                # Estimate fetal fraction using background chromosomes
                console.print(f"[cyan]Estimating fetal fraction for {chr_name}...[/cyan]")
                est_ff, ff_likelihoods = estimate_fetal_fraction(
                    background_data,
                    f_min=ff_min,
                    f_max=ff_max,
                    f_step=ff_step,
                    ncpus=ncpus
                )
                
                # Calculate likelihood ratio for target chromosome
                console.print(f"[cyan]Calculating likelihood ratio for {chr_name}...[/cyan]")
                lr = LR_calculator(target_data, est_ff)
                
                # Store results
                results[f'{chr_name}_lr'] = lr
                results[f'{chr_name}_ff'] = est_ff
                
                console.print(f"[green]✓ {chr_name}: FF = {est_ff:.3f}, LR = {lr:.2e}[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ Error processing {chr_name}: {str(e)}[/red]")
                if verbose:
                    console.print_exception()
                continue
        
        # Save results
        if results:
            console.print(f"\n[cyan]Saving results to {output_path}...[/cyan]")
            results_df = pd.DataFrame([results])
            results_df.to_csv(output_path, sep='\t', index=False)
            
            # Display summary table
            display_results_summary(results, target_chromosomes)
            console.print(f"[bold green]✓ Analysis complete! Results saved to {output_path}[/bold green]")
        else:
            console.print("[red]✗ No results generated - check input data and parameters[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]✗ Fatal error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def parse_chromosome_list(chr_spec: str) -> list:
    """
    Parse chromosome specification string into list of chromosome numbers.
    
    Args:
        chr_spec: String like "1-22", "1,2,3", or "21"
    
    Returns:
        List of chromosome numbers (integers)
    
    Raises:
        ValueError: If chromosome specification is invalid
    """
    try:
        if '-' in chr_spec:
            # Range specification (e.g., "1-22")
            start, end = map(int, chr_spec.split('-'))
            return list(range(start, end + 1))
        elif ',' in chr_spec:
            # Comma-separated list (e.g., "1,2,3")
            return [int(x.strip()) for x in chr_spec.split(',')]
        else:
            # Single chromosome (e.g., "21")
            return [int(chr_spec)]
    except ValueError:
        raise ValueError(f"Invalid chromosome specification: {chr_spec}")


def load_and_validate_data(input_path: Path) -> pd.DataFrame:
    """
    Load and validate input SNP data from TSV.GZ file.
    
    Args:
        input_path: Path to input file
    
    Returns:
        Validated pandas DataFrame
    
    Raises:
        ValueError: If data validation fails
        FileNotFoundError: If input file doesn't exist
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        # Load data with appropriate compression detection
        df = pd.read_csv(input_path, sep='\t', compression='gzip' if input_path.suffix == '.gz' else None)
    except Exception as e:
        raise ValueError(f"Failed to load input file: {str(e)}")
    
    # Validate required columns
    required_columns = {'chr', 'pos', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Basic data validation
    if len(df) == 0:
        raise ValueError("Input file is empty")
    
    # Ensure proper data types
    df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
    df['af'] = pd.to_numeric(df['af'], errors='coerce')
    df['cfDNA_ref_reads'] = pd.to_numeric(df['cfDNA_ref_reads'], errors='coerce', downcast='integer')
    df['cfDNA_alt_reads'] = pd.to_numeric(df['cfDNA_alt_reads'], errors='coerce', downcast='integer')
    
    # Remove rows with invalid data
    invalid_mask = (
        df['af'].isna() | df['cfDNA_ref_reads'].isna() | df['cfDNA_alt_reads'].isna() |
        (df['af'] < 0) | (df['af'] > 1) |
        (df['cfDNA_ref_reads'] < 0) | (df['cfDNA_alt_reads'] < 0)
    )
    
    if invalid_mask.any():
        console.print(f"[yellow]Warning: Removing {invalid_mask.sum()} rows with invalid data[/yellow]")
        df = df[~invalid_mask]
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after filtering")
    
    return df


def display_results_summary(results: Dict, target_chromosomes: list) -> None:
    """
    Display a formatted summary table of results.
    
    Args:
        results: Dictionary containing LR and FF results
        target_chromosomes: List of analyzed chromosomes
    """
    table = Table(title="Analysis Results Summary")
    table.add_column("Chromosome", justify="center", style="cyan")
    table.add_column("Fetal Fraction", justify="right", style="green")
    table.add_column("Likelihood Ratio", justify="right", style="yellow")
    table.add_column("Interpretation", justify="center", style="magenta")
    
    for chr_num in target_chromosomes:
        chr_name = f"chr{chr_num}"
        ff_key = f"{chr_name}_ff"
        lr_key = f"{chr_name}_lr"
        
        if ff_key in results and lr_key in results:
            ff_val = results[ff_key]
            lr_val = results[lr_key]
            
            # Simple interpretation logic
            if lr_val > 10:
                interpretation = "Strong Evidence"
            elif lr_val > 1:
                interpretation = "Moderate Evidence"
            elif lr_val == 1:
                interpretation = "No Evidence"
            else:
                interpretation = "Against Trisomy"
            
            table.add_row(
                str(chr_num),
                f"{ff_val:.3f}",
                f"{lr_val:.2e}" if lr_val != float('inf') else "∞",
                interpretation
            )
    
    console.print(table)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set
    
    main()