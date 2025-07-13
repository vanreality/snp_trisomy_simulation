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
    ncpus: int = cpu_count(),
    ref_col: str = 'cfDNA_ref_reads',
    alt_col: str = 'cfDNA_alt_reads'
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
         ref_col (str, optional): Column name for reference reads. Defaults to 'cfDNA_ref_reads'.
         alt_col (str, optional): Column name for alternative reads. Defaults to 'cfDNA_alt_reads'.
    
    Returns:
        Tuple[float, Dict[float, float]]: A tuple containing:
            - best_ff: Estimated fetal fraction that maximizes log-likelihood
            - log_likelihoods: Dictionary mapping FF values to their log-likelihoods
    
    Raises:
        ValueError: If input DataFrame is missing required columns or contains invalid data
        ValueError: If search parameters are invalid (f_min >= f_max, negative values, etc.)
    """
    # Input validation
    required_columns = {'chr', 'pos', 'af', ref_col, alt_col}
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
    invalid_reads = (df_clean[ref_col] < 0) | (df_clean[alt_col] < 0)
    if invalid_reads.any():
        console.print(f"[yellow]Warning: Removing {invalid_reads.sum()} SNPs with negative read counts[/yellow]")
        df_clean = df_clean[~invalid_reads]
    
    # Filter out SNPs with zero coverage
    df_clean['total_reads'] = df_clean[ref_col] + df_clean[alt_col]
    zero_coverage = df_clean['total_reads'] == 0
    if zero_coverage.any():
        console.print(f"[yellow]Warning: Removing {zero_coverage.sum()} SNPs with zero coverage[/yellow]")
        df_clean = df_clean[df_clean['total_reads'] > 0]
    
    if len(df_clean) == 0:
        raise ValueError("No valid SNPs remaining after filtering")
    
    # Convert to numpy arrays for performance
    p_arr = df_clean['af'].to_numpy(dtype=np.float64)
    ref_arr = df_clean[ref_col].to_numpy(dtype=np.int32)
    alt_arr = df_clean[alt_col].to_numpy(dtype=np.int32)
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

 
def LR_calculator(
    input_df: pd.DataFrame, 
    fetal_fraction: float,
    ref_col: str = 'cfDNA_ref_reads',
    alt_col: str = 'cfDNA_alt_reads'
) -> float:
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
             - ref_col: number of reference allele reads (int >= 0)
             - alt_col: number of alternative allele reads (int >= 0)
        fetal_fraction (float): Estimated fetal fraction in maternal plasma (0 < ff < 1)
        ref_col (str, optional): Column name for reference reads. Defaults to 'cfDNA_ref_reads'.
        alt_col (str, optional): Column name for alternative reads. Defaults to 'cfDNA_alt_reads'.
    
    Returns:
        float: Likelihood ratio LR = L_trisomy / L_disomy. 
               Returns np.inf if L_disomy = 0, and 0.0 if L_trisomy = 0.
    
    Raises:
        ValueError: If input DataFrame is missing required columns
        ValueError: If fetal_fraction is not in valid range (0, 1)
        ValueError: If no valid SNPs are found
    """
    # Input validation
    required_cols = {'chr', 'pos', 'af', ref_col, alt_col}
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
        (df_clean[ref_col] < 0) | (df_clean[alt_col] < 0) |
        ((df_clean[ref_col] + df_clean[alt_col]) == 0)
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
            ref_count = int(snp[ref_col])
            alt_count = int(snp[alt_col])
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


def estimate_fetal_fraction_with_maternal_reads(
    background_chr_df: pd.DataFrame,
    f_min: float = 0.001,
    f_max: float = 0.5,
    f_step: float = 0.001,
    ncpus: int = cpu_count(),
    ref_col: str = 'cfDNA_ref_reads',
    alt_col: str = 'cfDNA_alt_reads',
    maternal_ref_col: str = 'maternal_ref_reads',
    maternal_alt_col: str = 'maternal_alt_reads'
) -> Tuple[float, Dict[float, float]]:
    """
    Estimate fetal fraction (FF) from maternal cfDNA SNP read counts using maximum likelihood.
    Incorporates maternal WBC genotyping to define prior maternal genotypes before computing
    the mixture likelihood over fetal fraction.

    Args:
        background_chr_df (pd.DataFrame): DataFrame containing SNP data with columns:
            - 'chr', 'pos', 'af', cfDNA ref and alt read counts, and maternal WBC ref/alt reads
        f_min (float): Minimum fetal fraction to search.
        f_max (float): Maximum fetal fraction to search.
        f_step (float): Step size for grid search.
        ncpus (int): Number of CPU cores for parallel processing.
        ref_col (str): Column name for cfDNA reference reads.
        alt_col (str): Column name for cfDNA alternative reads.
        maternal_ref_col (str): Column name for maternal WBC reference reads.
        maternal_alt_col (str): Column name for maternal WBC alternative reads.

    Returns:
        best_ff (float): Estimated fetal fraction.
        log_likelihoods (Dict[float, float]): Mapping of FF values to log-likelihoods.
    """
    # Input validation
    required_columns = {'chr', 'pos', 'af', ref_col, alt_col, maternal_ref_col, maternal_alt_col}
    if not required_columns.issubset(background_chr_df.columns):
        missing = required_columns - set(background_chr_df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    if not (0 <= f_min < f_max <= 1):
        raise ValueError(f"Invalid fetal fraction range: [{f_min}, {f_max}]")
    if f_step <= 0:
        raise ValueError(f"Step size must be positive: {f_step}")
    if background_chr_df.empty:
        raise ValueError("Input DataFrame is empty")

    # Copy and clean data
    df = background_chr_df.copy()
    # Filter invalid allele frequencies
    bad_af = (df['af'] < 0) | (df['af'] > 1)
    if bad_af.any():
        console.print(f"[yellow]Warning: Removing {bad_af.sum()} SNPs with invalid AF[/yellow]")
        df = df[~bad_af]
    # Filter invalid read counts
    bad_reads = (df[ref_col] < 0) | (df[alt_col] < 0)
    if bad_reads.any():
        console.print(f"[yellow]Warning: Removing {bad_reads.sum()} SNPs with negative cfDNA reads[/yellow]")
        df = df[~bad_reads]
    # Remove zero-coverage cfDNA SNPs
    df['total_cfDNA'] = df[ref_col] + df[alt_col]
    zero_cov = df['total_cfDNA'] == 0
    if zero_cov.any():
        console.print(f"[yellow]Warning: Removing {zero_cov.sum()} SNPs with zero cfDNA coverage[/yellow]")
        df = df[df['total_cfDNA'] > 0]
    if df.empty:
        raise ValueError("No valid SNPs after filtering")

    # Extract arrays
    p_arr = df['af'].to_numpy(dtype=np.float64)
    ref_arr = df[ref_col].to_numpy(dtype=np.int32)
    alt_arr = df[alt_col].to_numpy(dtype=np.int32)
    n_arr = ref_arr + alt_arr
    num_snps = len(p_arr)
    console.print(f"[green]Processing {num_snps} SNPs...[/green]")

    # Maternal WBC genotyping: derive genotype at each SNP by maximum-likelihood
    m_ref = df[maternal_ref_col].to_numpy(dtype=np.int32)
    m_alt = df[maternal_alt_col].to_numpy(dtype=np.int32)
    m_n = m_ref + m_alt
    # Possible maternal alt allele fractions for genotypes 0/0, 0/1, 1/1
    maternal_alt_frac = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    # Compute log-likelihoods of maternal counts under each genotype
    maternal_loglik = np.zeros((num_snps, 3), dtype=np.float64)
    for g in range(3):
        # Vectorized per-SNP log binomial
        maternal_loglik[:, g] = [
            _log_binomial_pmf_global(a, n, maternal_alt_frac[g])
            for a, n in zip(m_alt, m_n)
        ]
    # Assign maternal genotype as the one with highest log-likelihood
    maternal_gt = np.argmax(maternal_loglik, axis=1)
    # Build one-hot prior matrix from maternal calls
    prior_G = np.zeros((num_snps, 3), dtype=np.float64)
    prior_G[np.arange(num_snps), maternal_gt] = 1.0
    # Log-priors (0 for called genotype, -inf elsewhere)
    with np.errstate(divide='ignore'):
        log_prior_G = np.log(prior_G)

    # Compute expected fetal alt fractions conditional on maternal genotype
    # E[fetal_alt_frac | maternal genotype]
    fetal_alt_frac = np.zeros((num_snps, 3), dtype=np.float64)
    # Based on population AF p_arr
    fetal_alt_frac[:, 0] = 0.5 * p_arr             # mother 0/0
    fetal_alt_frac[:, 1] = 0.25 + 0.5 * p_arr       # mother 0/1
    fetal_alt_frac[:, 2] = 0.5 + 0.5 * p_arr        # mother 1/1

    # Prepare grid of fetal fractions to evaluate
    f_values = np.arange(f_min, f_max + f_step, f_step)
    log_likelihoods: Dict[float, float] = {}
    console.print(f"[cyan]Using {ncpus} CPU cores for grid search from {f_min} to {f_max}[/cyan]")

    # Shared data tuple
    shared_data = (p_arr, n_arr, alt_arr, log_prior_G, maternal_alt_frac, fetal_alt_frac)
    
    # Split FF values into chunks for parallel processing
    chunk_size = max(1, len(f_values) // ncpus)
    f_value_chunks = [f_values[i:i + chunk_size] for i in range(0, len(f_values), chunk_size)]
    
    # Prepare arguments for worker processes
    worker_args = [(chunk, shared_data) for chunk in f_value_chunks]

    # Progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Estimating fetal fraction...", total=len(f_value_chunks))
        if ncpus == 1:
            for args in worker_args:
                results = _compute_ff_likelihood_chunk(args)
                log_likelihoods.update(results)
                progress.advance(task)
        else:
            with Pool(processes=ncpus) as pool:
                asyncs = [pool.apply_async(_compute_ff_likelihood_chunk, (args,)) for args in worker_args]
                for a in asyncs:
                    res = a.get()
                    log_likelihoods.update(res)
                    progress.advance(task)

    # Select best FF
    best_ff = max(log_likelihoods, key=log_likelihoods.get)
    console.print(f"[green]✓ Estimated fetal fraction: {best_ff:.3f}[/green]")
    return best_ff, log_likelihoods


def LR_calculator_with_maternal_reads(
    input_df: pd.DataFrame,
    fetal_fraction: float,
    ref_col: str = 'cfDNA_ref_reads',
    alt_col: str = 'cfDNA_alt_reads',
    maternal_ref_col: str = 'maternal_ref_reads',
    maternal_alt_col: str = 'maternal_alt_reads'
) -> float:
    """
    Calculate the likelihood ratio (LR) of trisomy vs. disomy for a target chromosome,
    incorporating maternal WBC genotype information.

    This function first determines the most likely maternal genotype at each SNP using
    read counts from maternal WBCs. It then calculates the likelihood of observed
    cfDNA reads under both disomy and trisomy hypotheses, conditioned on the inferred
    maternal genotype and marginalizing over paternal genotypes based on population
    allele frequencies.

    Args:
        input_df (pd.DataFrame): DataFrame with SNP data including cfDNA and maternal reads.
            Required columns: 'chr', 'pos', 'af', cfDNA reads, and maternal WBC reads.
        fetal_fraction (float): Estimated fetal fraction in maternal plasma (0 < ff < 1).
        ref_col (str, optional): Column name for cfDNA reference reads.
        alt_col (str, optional): Column name for cfDNA alternative reads.
        maternal_ref_col (str, optional): Column for maternal WBC reference reads.
        maternal_alt_col (str, optional): Column for maternal WBC alternative reads.

    Returns:
        float: The likelihood ratio LR = L_trisomy / L_disomy. Returns np.inf if
               the disomy likelihood is zero, and 0.0 if the trisomy likelihood is zero.

    Raises:
        ValueError: If input data is invalid, fetal fraction is out of range,
                    or required columns are missing.
    """
    # Input validation
    required_cols = {'chr', 'pos', 'af', ref_col, alt_col, maternal_ref_col, maternal_alt_col}
    if not required_cols.issubset(input_df.columns):
        missing = required_cols - set(input_df.columns)
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    if not (0.0 < fetal_fraction < 1.0):
        raise ValueError(f"fetal_fraction must be between 0 and 1 (exclusive): {fetal_fraction}")

    if input_df.empty:
        raise ValueError("No SNPs found in input DataFrame")

    # Data cleaning and validation
    df_clean = input_df.copy()
    invalid_mask = (
        (df_clean['af'] < 0) | (df_clean['af'] > 1) |
        (df_clean[ref_col] < 0) | (df_clean[alt_col] < 0) |
        ((df_clean[ref_col] + df_clean[alt_col]) == 0) |
        (df_clean[maternal_ref_col] < 0) | (df_clean[maternal_alt_col] < 0)
    )

    if invalid_mask.any():
        console.print(f"[yellow]Warning: Removing {invalid_mask.sum()} invalid SNPs[/yellow]")
        df_clean = df_clean[~invalid_mask]

    if df_clean.empty:
        raise ValueError("No valid SNPs remaining after filtering")

    # Helper functions for genotype modeling (copied from LR_calculator for encapsulation)
    def genotype_priors(af: float) -> Dict[str, float]:
        """Compute Hardy-Weinberg genotype priors for a given allele frequency."""
        return {
            '0/0': (1.0 - af) ** 2,
            '0/1': 2.0 * af * (1.0 - af),
            '1/1': af ** 2
        }

    def get_fetal_prob_disomy(gm: str, gp: str) -> Dict[str, float]:
        """Compute fetal genotype probabilities under disomy given parental genotypes."""
        def allele_probs_from_genotype(gt: str) -> Dict[int, float]:
            a1, a2 = map(int, gt.split('/'))
            counts = {0: 0, 1: 0}
            counts[a1] += 1
            counts[a2] += 1
            return {0: counts[0] / 2.0, 1: counts[1] / 2.0}
        
        pm = allele_probs_from_genotype(gm)
        pp = allele_probs_from_genotype(gp)
        fetal_probs = {'0/0': 0.0, '0/1': 0.0, '1/1': 0.0}
        
        for mat_allele, p_mat in pm.items():
            for pat_allele, p_pat in pp.items():
                prob_combo = p_mat * p_pat
                fetal_genotype = f"{min(mat_allele, pat_allele)}/{max(mat_allele, pat_allele)}"
                fetal_probs[fetal_genotype] += prob_combo
        return fetal_probs

    def get_fetal_prob_trisomy(gm: str, gp: str) -> Dict[int, float]:
        """Compute fetal ALT-allele dosage probabilities under trisomy."""
        def allele_probs(gt: str) -> Dict[int, float]:
            a1, a2 = map(int, gt.split('/'))
            counts = {0: 0, 1: 0}
            counts[a1] += 1
            counts[a2] += 1
            return {0: counts[0] / 2.0, 1: counts[1] / 2.0}

        pm = allele_probs(gm)
        pp = allele_probs(gp)
        
        p_mat_alt = pm[1]
        maternal_dosage_probs = {
            0: (1 - p_mat_alt) ** 2,
            1: 2 * p_mat_alt * (1 - p_mat_alt),
            2: p_mat_alt ** 2
        }
        
        paternal_alt_prob = pp[1]
        fetal_dosage_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for k_m, p_m in maternal_dosage_probs.items():
            for k_p in [0, 1]:
                d = k_m + k_p
                prob = p_m * (paternal_alt_prob if k_p == 1 else (1 - paternal_alt_prob))
                fetal_dosage_probs[d] += prob
        return fetal_dosage_probs
    
    # Infer maternal genotype from WBC reads
    m_ref = df_clean[maternal_ref_col].to_numpy(dtype=np.int32)
    m_alt = df_clean[maternal_alt_col].to_numpy(dtype=np.int32)
    m_n = m_ref + m_alt
    
    maternal_alt_frac_options = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    maternal_loglik = np.zeros((len(df_clean), 3), dtype=np.float64)
    
    for g in range(3):
        p = maternal_alt_frac_options[g]
        log_pmfs = [_log_binomial_pmf_global(k, n, p) for k, n in zip(m_alt, m_n)]
        maternal_loglik[:, g] = log_pmfs
    
    maternal_genotypes_idx = np.argmax(maternal_loglik, axis=1)
    genotype_labels = ['0/0', '0/1', '1/1']
    df_clean['maternal_genotype'] = [genotype_labels[i] for i in maternal_genotypes_idx]

    # Main likelihood computation
    logL_disomy = 0.0
    logL_trisomy = 0.0
    
    console.print(f"[green]Calculating likelihood ratio for {len(df_clean)} SNPs with maternal genotypes[/green]")
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Computing likelihood ratios...", total=len(df_clean))
        
        for idx, snp in df_clean.iterrows():
            ref_count = int(snp[ref_col])
            alt_count = int(snp[alt_col])
            depth = ref_count + alt_count
            af = float(snp['af'])
            gm = snp['maternal_genotype']  # Use inferred maternal genotype

            paternal_geno_priors = genotype_priors(af)
            
            log_terms_disomy = []
            log_terms_trisomy = []
            
            # Marginalize over paternal genotypes
            for gp in genotype_labels:
                p_gp = paternal_geno_priors[gp]
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
                    log_p_obs = _log_binomial_pmf_global(alt_count, depth, p_alt_exp)
                    if log_p_obs > float('-inf'):
                        log_terms_disomy.append(log_p_gp + log_p_gf + log_p_obs)

                # Trisomy calculation
                fetal_dist_tri = get_fetal_prob_trisomy(gm, gp)
                
                for d_fetal, p_fd in fetal_dist_tri.items():
                    if p_fd <= 0.0:
                        continue
                    log_p_fd = math.log(p_fd)
                    p_alt_exp_tri = ((1.0 - 1.5 * fetal_fraction) * (d_maternal / 2.0) +
                                     1.5 * fetal_fraction * (d_fetal / 3.0))
                    log_p_obs_tri = _log_binomial_pmf_global(alt_count, depth, p_alt_exp_tri)
                    if log_p_obs_tri > float('-inf'):
                        log_terms_trisomy.append(log_p_gp + log_p_fd + log_p_obs_tri)

            # Compute SNP-level log-likelihoods
            if log_terms_disomy:
                logL_disomy += _log_sum_exp_global(np.array(log_terms_disomy))
            
            if log_terms_trisomy:
                logL_trisomy += _log_sum_exp_global(np.array(log_terms_trisomy))
            
            progress.advance(task)

    # Compute final likelihood ratio
    delta_logL = logL_trisomy - logL_disomy
    
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
    '--mode',
    type=click.Choice(['cfDNA', 'cfDNA+WBC', 'cfDNA+model'], case_sensitive=False),
    default='cfDNA',
    help='Analysis mode: cfDNA (standard), cfDNA+WBC (with maternal WBC), cfDNA+model (with modeled maternal reads)'
)
@click.option(
    '--cfdna-ref-col',
    default='cfDNA_ref_reads',
    help='Column name for cfDNA reference reads (default: cfDNA_ref_reads)'
)
@click.option(
    '--cfdna-alt-col',
    default='cfDNA_alt_reads',
    help='Column name for cfDNA alternative reads (default: cfDNA_alt_reads)'
)
@click.option(
    '--wbc-ref-col',
    default='maternal_ref_reads',
    help='Column name for maternal WBC reference reads (default: maternal_ref_reads)'
)
@click.option(
    '--wbc-alt-col',
    default='maternal_alt_reads',
    help='Column name for maternal WBC alternative reads (default: maternal_alt_reads)'
)
@click.option(
    '--model-ref-col',
    default='fetal_ref_reads_from_model',
    help='Column name for modeled fetal reference reads (default: fetal_ref_reads_from_model)'
)
@click.option(
    '--model-alt-col',
    default='fetal_alt_reads_from_model',
    help='Column name for modeled fetal alternative reads (default: fetal_alt_reads_from_model)'
)
@click.option(
    '--min-raw-depth',
    type=click.IntRange(0, None),
    default=0,
    help='Minimum raw depth filter for cfDNA reads (default: 0)'
)
@click.option(
    '--min-model-depth',
    type=click.IntRange(0, None),
    default=0,
    help='Minimum model depth filter for model filtered reads (default: 0)'
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
    mode: str,
    cfdna_ref_col: str,
    cfdna_alt_col: str,
    wbc_ref_col: str,
    wbc_alt_col: str,
    model_ref_col: str,
    model_alt_col: str,
    min_raw_depth: int,
    min_model_depth: int,
    ncpus: int,
    verbose: bool
) -> None:
    """
    Fetal Fraction Estimation and Likelihood Ratio Calculator.
    
    This tool analyzes cell-free DNA sequencing data to estimate fetal fraction
    and calculate likelihood ratios for trisomy detection across chromosomes.
    
    The input file should be a TSV.GZ file with columns:
    chr, pos, af, and read count columns (names configurable via CLI options)
     
    Supports three analysis modes:
    - cfDNA: Standard cell-free DNA analysis
    - cfDNA+WBC: Analysis with maternal white blood cell data
    - cfDNA+model: Analysis with modeled maternal reads
    
    Depth filtering options:
    - min-raw-depth: Filter SNPs by minimum cfDNA read depth
    - min-model-depth: Filter SNPs by minimum model filtered read depth (cfDNA+model mode only)
    
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
        
        # Determine which columns to display based on mode
        mode_info = f"Mode: {mode}"
        if mode == 'cfDNA':
            column_info = f"cfDNA columns: {cfdna_ref_col}, {cfdna_alt_col}"
        elif mode == 'cfDNA+WBC':
            column_info = f"cfDNA columns: {cfdna_ref_col}, {cfdna_alt_col}\nWBC columns: {wbc_ref_col}, {wbc_alt_col}"
        elif mode == 'cfDNA+model':
            column_info = f"cfDNA columns: {cfdna_ref_col}, {cfdna_alt_col}\nModel columns: {model_ref_col}, {model_alt_col}"
        
        # Build depth filter info
        depth_filter_info = f"Raw depth filter: ≥{min_raw_depth}"
        if mode == 'cfDNA+model':
            depth_filter_info += f"\nModel depth filter: ≥{min_model_depth}"
        
        # Display startup information
        console.print(Panel.fit(
            f"[bold green]Fetal Fraction & Likelihood Ratio Calculator[/bold green]\n"
            f"Input: {input_path}\n"
            f"Output: {output_dir}\n"
            f"{mode_info}\n"
            f"{column_info}\n"
            f"Chromosomes: {', '.join(map(str, target_chromosomes))}\n"
            f"FF Range: {ff_min:.3f} - {ff_max:.3f} (step: {ff_step:.3f})\n"
            f"{depth_filter_info}\n"
            f"CPU Cores: {ncpus}",
            title="Configuration"
        ))
        
        # Generate output filename based on input
        input_filename = input_path.stem.replace('.tsv', '')
        output_path = output_dir / f'{input_filename}_lr.tsv'
        
        # Load and validate input data
        console.print("[cyan]Loading input data...[/cyan]")
        df = load_and_validate_data(
            input_path,
            mode=mode,
            cfdna_ref_col=cfdna_ref_col,
            cfdna_alt_col=cfdna_alt_col,
            wbc_ref_col=wbc_ref_col,
            wbc_alt_col=wbc_alt_col,
            model_ref_col=model_ref_col,
            model_alt_col=model_alt_col,
            min_raw_depth=min_raw_depth,
            min_model_depth=min_model_depth
        )
        
        console.print(f"[green]✓ Loaded {len(df)} SNPs from {df['chr'].nunique()} chromosomes[/green]")
        
        # Initialize results storage
        results_list = []
        
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
                if mode == 'cfDNA':
                    est_ff, _ = estimate_fetal_fraction(
                        background_data,
                        f_min=ff_min,
                        f_max=ff_max,
                        f_step=ff_step,
                        ncpus=ncpus,
                        ref_col=cfdna_ref_col,
                        alt_col=cfdna_alt_col
                    )
                    
                    # Calculate likelihood ratio for target chromosome
                    console.print(f"[cyan]Calculating likelihood ratio for {chr_name}...[/cyan]")
                    lr = LR_calculator(target_data, 
                                       est_ff, 
                                       ref_col=cfdna_ref_col, 
                                       alt_col=cfdna_alt_col)
                    
                    # Store results
                    results_list.append({
                        'Chrom': chr_name,
                        'LR': lr,
                        'Fetal Fraction': est_ff
                    })
                    
                    console.print(f"[green]✓ {chr_name}: FF = {est_ff:.3f}, LR = {lr:.2e}[/green]")

                elif mode == 'cfDNA+WBC':
                    est_ff, _ = estimate_fetal_fraction_with_maternal_reads(
                        background_data,
                        f_min=ff_min,
                        f_max=ff_max,
                        f_step=ff_step,
                        ncpus=ncpus,
                        ref_col=cfdna_ref_col,
                        alt_col=cfdna_alt_col,
                        maternal_ref_col=wbc_ref_col,
                        maternal_alt_col=wbc_alt_col
                    )

                    # Calculate likelihood ratio for target chromosome
                    console.print(f"[cyan]Calculating likelihood ratio for {chr_name}...[/cyan]")
                    lr = LR_calculator_with_maternal_reads(
                        target_data, 
                        est_ff, 
                        ref_col=cfdna_ref_col, 
                        alt_col=cfdna_alt_col, 
                        maternal_ref_col=wbc_ref_col, 
                        maternal_alt_col=wbc_alt_col
                    )
                    # Store results
                    results_list.append({
                        'Chrom': chr_name,
                        'LR': lr,
                        'Fetal Fraction': est_ff
                    })
                    
                    console.print(f"[green]✓ {chr_name}: FF = {est_ff:.3f}, LR = {lr:.2e}[/green]")

                elif mode == 'cfDNA+model':
                    est_ff, _ = estimate_fetal_fraction(
                        background_data,
                        f_min=ff_min,
                        f_max=ff_max,
                        f_step=ff_step,
                        ncpus=ncpus,
                        ref_col=model_ref_col,
                        alt_col=model_alt_col
                    )

                    # Calculate likelihood ratio for target chromosome
                    console.print(f"[cyan]Calculating likelihood ratio for {chr_name}...[/cyan]")
                    lr = LR_calculator(
                        target_data, 
                        est_ff, 
                        ref_col=model_ref_col, 
                        alt_col=model_alt_col
                    )
                    
                    # Store results
                    results_list.append({
                        'Chrom': chr_name,
                        'LR': lr,
                        'Fetal Fraction': est_ff
                    })
                    
                    console.print(f"[green]✓ {chr_name}: FF = {est_ff:.3f}, LR = {lr:.2e}[/green]")

            except Exception as e:
                console.print(f"[red]✗ Error processing {chr_name}: {str(e)}[/red]")
                if verbose:
                    console.print_exception()
                continue
        
        # Save results
        if results_list:
            console.print(f"\n[cyan]Saving results to {output_path}...[/cyan]")
            results_df = pd.DataFrame(results_list, columns=['Chrom', 'LR', 'Fetal Fraction'])
            results_df.to_csv(output_path, sep='\t', index=False)
            
            # Display summary table
            display_results_summary(results_df)
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


def load_and_validate_data(
    input_path: Path, 
    mode: str = 'cfDNA',
    cfdna_ref_col: str = 'cfDNA_ref_reads',
    cfdna_alt_col: str = 'cfDNA_alt_reads',
    wbc_ref_col: str = 'maternal_ref_reads',
    wbc_alt_col: str = 'maternal_alt_reads',
    model_ref_col: str = 'maternal_ref_reads_from_model',
    model_alt_col: str = 'maternal_alt_reads_from_model',
    min_raw_depth: int = 0,
    min_model_depth: int = 0
) -> pd.DataFrame:
    """
    Load and validate input SNP data from TSV.GZ file.
    
    This function validates the presence of required columns based on the analysis mode
    and performs data type validation and cleaning. It also applies depth filters.
    
    Args:
        input_path (Path): Path to input file
        mode (str): Analysis mode ('cfDNA', 'cfDNA+WBC', or 'cfDNA+model')
        cfdna_ref_col (str): Column name for cfDNA reference reads
        cfdna_alt_col (str): Column name for cfDNA alternative reads
        wbc_ref_col (str): Column name for maternal WBC reference reads
        wbc_alt_col (str): Column name for maternal WBC alternative reads
        model_ref_col (str): Column name for modeled maternal reference reads
        model_alt_col (str): Column name for modeled maternal alternative reads
        min_raw_depth (int): Minimum depth for cfDNA reads (default: 0)
        min_model_depth (int): Minimum depth for modeled reads (default: 0)
    
    Returns:
        pd.DataFrame: Validated pandas DataFrame with proper data types
    
    Raises:
        ValueError: If data validation fails or required columns are missing
        FileNotFoundError: If input file doesn't exist
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        # Load data with appropriate compression detection
        df = pd.read_csv(input_path, sep='\t', compression='gzip' if input_path.suffix == '.gz' else None)
    except Exception as e:
        raise ValueError(f"Failed to load input file: {str(e)}")
    
    # Determine required columns based on mode
    base_columns = {'chr', 'pos', 'af'}
    required_columns = base_columns.copy()
    read_columns = []
    
    if mode == 'cfDNA':
        required_columns.update({cfdna_ref_col, cfdna_alt_col})
        read_columns = [cfdna_ref_col, cfdna_alt_col]
        console.print(f"[cyan]Validating cfDNA mode with columns: {cfdna_ref_col}, {cfdna_alt_col}[/cyan]")
    elif mode == 'cfDNA+WBC':
        required_columns.update({cfdna_ref_col, cfdna_alt_col, wbc_ref_col, wbc_alt_col})
        read_columns = [cfdna_ref_col, cfdna_alt_col, wbc_ref_col, wbc_alt_col]
        console.print(f"[cyan]Validating cfDNA+WBC mode with columns: {', '.join(read_columns)}[/cyan]")
    elif mode == 'cfDNA+model':
        required_columns.update({cfdna_ref_col, cfdna_alt_col, model_ref_col, model_alt_col})
        read_columns = [cfdna_ref_col, cfdna_alt_col, model_ref_col, model_alt_col]
        console.print(f"[cyan]Validating cfDNA+model mode with columns: {', '.join(read_columns)}[/cyan]")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: cfDNA, cfDNA+WBC, cfDNA+model")
    
    # Check for required columns
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns for mode '{mode}': {missing}")
    
    # Basic data validation
    if len(df) == 0:
        raise ValueError("Input file is empty")
    
    # Ensure proper data types for base columns
    df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
    df['af'] = pd.to_numeric(df['af'], errors='coerce')
    
    # Ensure proper data types for read count columns
    for col in read_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
    
    # Build invalid mask for base columns
    invalid_mask = (
        df['af'].isna() | (df['af'] < 0) | (df['af'] > 1)
    )
    
    # Add read count validation to invalid mask
    for col in read_columns:
        invalid_mask |= (df[col].isna() | (df[col] < 0))
    
    if invalid_mask.any():
        console.print(f"[yellow]Warning: Removing {invalid_mask.sum()} rows with invalid data[/yellow]")
        df = df[~invalid_mask]
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after filtering")
    
    # Apply depth filters
    initial_count = len(df)
    
    # Filter by minimum raw depth (cfDNA reads)
    if min_raw_depth > 0:
        df['cfdna_total_depth'] = df[cfdna_ref_col] + df[cfdna_alt_col]
        df = df[df['cfdna_total_depth'] >= min_raw_depth]
        remaining_count = len(df)
        percentage = (remaining_count / initial_count) * 100 if initial_count > 0 else 0
        console.print(f"[cyan]Raw depth filter (≥{min_raw_depth}): {remaining_count:,} SNPs remaining ({percentage:.1f}%)[/cyan]")
        df = df.drop(columns=['cfdna_total_depth']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No SNPs remaining after raw depth filtering")
    
    # Filter by minimum model depth (only for cfDNA+model mode)
    if min_model_depth > 0 and mode == 'cfDNA+model':
        current_count = len(df)
        df['model_total_depth'] = df[model_ref_col] + df[model_alt_col]
        df = df[df['model_total_depth'] >= min_model_depth]
        remaining_count = len(df)
        percentage = (remaining_count / current_count) * 100 if current_count > 0 else 0
        console.print(f"[cyan]Model depth filter (≥{min_model_depth}): {remaining_count:,} SNPs remaining ({percentage:.1f}%)[/cyan]")
        df = df.drop(columns=['model_total_depth']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No SNPs remaining after model depth filtering")
    elif min_model_depth > 0 and mode != 'cfDNA+model':
        console.print(f"[yellow]Warning: Model depth filter ignored for mode '{mode}' (only applies to 'cfDNA+model')[/yellow]")
    
    return df


def display_results_summary(results_df: pd.DataFrame) -> None:
    """
    Display a formatted summary table of results.
    
    Args:
        results_df: DataFrame containing LR and FF results with columns 'Chrom', 'LR', 'Fetal Fraction'.
    """
    table = Table(title="Analysis Results Summary")
    table.add_column("Chromosome", justify="center", style="cyan")
    table.add_column("Fetal Fraction", justify="right", style="green")
    table.add_column("Likelihood Ratio", justify="right", style="yellow")
    table.add_column("Interpretation", justify="center", style="magenta")
    
    for _, row in results_df.iterrows():
        chr_name = row['Chrom']
        ff_val = row['Fetal Fraction']
        lr_val = row['LR']
        
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
            str(chr_name.replace("chr", "")),
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