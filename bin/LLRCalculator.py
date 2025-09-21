#!/usr/bin/env python3
"""
Log Likelihood Ratio Calculator for Aneuploidy Detection

This module provides the LLRCalculator class for calculating log likelihood ratios
for trisomy vs. disomy detection using different analysis modes.
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, Tuple
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# Initialize rich console for beautiful output
console = Console()


# Global helper functions for numerical stability
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


def _log_beta_fn(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _alpha_beta_from_p_rho(p: float, rho: float, eps: float = 1e-12):
    p = min(max(p, eps), 1.0 - eps)
    rho = max(min(rho, 1.0 - eps), 0.0)
    if rho <= 0.0:
        return None, None
    theta = (1.0 - rho) / rho
    alpha = max(p * theta, eps)
    beta  = max((1.0 - p) * theta, eps)
    return alpha, beta


def _log_betabinom_pmf_global(k: int, n: int, p: float, rho: float) -> float:
    if n < 0 or k < 0 or k > n:
        return float('-inf')
    alpha, beta = _alpha_beta_from_p_rho(p, rho)
    if alpha is None:
        return _log_binomial_pmf_global(k, n, p)
    log_coeff = _log_binomial_coeff_global(n, k)
    return log_coeff + _log_beta_fn(k + alpha, n - k + beta) - _log_beta_fn(alpha, beta)


def _log_sum_exp_global(log_values: np.ndarray) -> float:
    """Compute log(sum(exp(x_i))) for numerical stability."""
    if len(log_values) == 0:
        return float('-inf')
    max_val = np.max(log_values)
    if max_val == float('-inf'):
        return float('-inf')
    return max_val + math.log(np.sum(np.exp(log_values - max_val)))


def _get_trisomy_factor_distribution(n_points: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate discretized trisomy factor distribution N(1.5, 0.5^2).
    
    Args:
        n_points (int): Number of discretization points. Defaults to 21.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (factor_values, log_probabilities)
            - factor_values: Array of factor values
            - log_probabilities: Log probabilities for each factor value
    """
    # Parameters for the trisomy factor distribution
    mean = 1.5
    std = 0.5
    
    # Create discretization points covering 3 standard deviations on each side
    min_factor = max(0.1, mean - 3 * std)  # Ensure positive factor
    max_factor = mean + 3 * std
    
    factor_values = np.linspace(min_factor, max_factor, n_points)
    
    # Compute log probabilities (unnormalized is fine since we'll normalize)
    log_probs = -0.5 * ((factor_values - mean) / std) ** 2
    
    # Normalize to get proper log probabilities
    log_probs = log_probs - _log_sum_exp_global(log_probs)
    
    return factor_values, log_probs


def _get_disomy_factor_distribution(n_points: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate discretized disomy factor distribution N(1.0, 0.5^2).
    
    Args:
        n_points (int): Number of discretization points. Defaults to 21.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (factor_values, log_probabilities)
            - factor_values: Array of factor values
            - log_probabilities: Log probabilities for each factor value
    """
    # Parameters for the disomy factor distribution
    mean = 1.0
    std = 0.5
    
    # Create discretization points covering 3 standard deviations on each side
    min_factor = max(0.1, mean - 3 * std)  # Ensure positive factor
    max_factor = mean + 3 * std
    
    factor_values = np.linspace(min_factor, max_factor, n_points)
    
    # Compute log probabilities (unnormalized is fine since we'll normalize)
    log_probs = -0.5 * ((factor_values - mean) / std) ** 2
    
    # Normalize to get proper log probabilities
    log_probs = log_probs - _log_sum_exp_global(log_probs)
    
    return factor_values, log_probs


class LLRCalculator:
    """
    Log Likelihood Ratio Calculator class for trisomy detection.
    
    This class provides methods to calculate log likelihood ratios for
    trisomy vs. disomy detection using different analysis modes.
    """
    
    def __init__(self, mode: str = 'cfDNA', factor_modeling: bool = False, beta_binomial: bool = False):
        """
        Initialize the Log Likelihood Ratio Calculator.
        
        Args:
            mode (str): Analysis mode, options: 'cfDNA', 'cfDNA+WBC', 'cfDNA+model', or 'cfDNA+model+mGT'.
                       'cfDNA' uses only cell-free DNA reads.
                       'cfDNA+WBC' incorporates maternal white blood cell genotyping.
                       'cfDNA+model' uses modeled fetal reads for calculation.
                       'cfDNA+model+mGT' uses modeled reads with maternal genotyping filtering.
            factor_modeling (bool): Whether to use probabilistic modeling for read release factors.
                                  If True, uses normal distributions for disomy (N(1, 0.5²)) and trisomy (N(1.55, 0.6²)).
                                  If False, uses fixed factors (1.0 for disomy, 1.5 for trisomy).
        
        Raises:
            ValueError: If mode is not one of the supported modes.
        """
        valid_modes = ['cfDNA', 'cfDNA+WBC', 'cfDNA+model', 'cfDNA+model+mGT']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: {valid_modes}")
        
        self.mode = mode
        self.allelic_bias = 0.47747748
        self.allelic_theo = 0.5
        self.factor_modeling = factor_modeling
        self.beta_binomial = beta_binomial
        self.rho = 1e-4

        if factor_modeling:
            console.print(f"[green]LLRCalculator initialized in {mode} mode with probabilistic factor modeling[/green]")
        else:
            console.print(f"[green]LLRCalculator initialized in {mode} mode with fixed factors[/green]")

        if beta_binomial:
            console.print(f"[green]LLRCalculator initialized in {mode} mode with beta-binomial distribution[/green]")
        else:
            console.print(f"[green]LLRCalculator initialized in {mode} mode with binomial distribution[/green]")
    
    def calculate(
        self,
        input_df: pd.DataFrame, 
        fetal_fraction: float,
        ref_col: str = 'cfDNA_ref_reads',
        alt_col: str = 'cfDNA_alt_reads',
        maternal_ref_col: str = 'maternal_ref_reads',
        maternal_alt_col: str = 'maternal_alt_reads'
    ) -> float:
        """
        Calculate the log likelihood ratio based on the configured analysis mode.
        
        Args:
            input_df (pd.DataFrame): DataFrame with target chromosome SNP data.
            fetal_fraction (float): Estimated fetal fraction in maternal plasma.
            ref_col (str, optional): Column name for reference reads.
            alt_col (str, optional): Column name for alternative reads.
            maternal_ref_col (str, optional): Column for maternal WBC reference reads.
            maternal_alt_col (str, optional): Column for maternal WBC alternative reads.
        
        Returns:
            float: Log likelihood ratio (log LR) = log(L_trisomy) - log(L_disomy).
        
        Raises:
            ValueError: If input data is invalid or analysis mode is unsupported.
        """
        if self.mode == 'cfDNA':
            return self._calculate_lr_cfdna(
                input_df=input_df,
                fetal_fraction=fetal_fraction,
                ref_col=ref_col,
                alt_col=alt_col
            )
        elif self.mode == 'cfDNA+WBC':
            return self._calculate_lr_with_maternal_reads(
                input_df=input_df,
                fetal_fraction=fetal_fraction,
                ref_col=ref_col,
                alt_col=alt_col,
                maternal_ref_col=maternal_ref_col,
                maternal_alt_col=maternal_alt_col
            )
        elif self.mode == 'cfDNA+model':
            # Use model reads directly for LLR calculation (like cfDNA mode)
            return self._calculate_lr_cfdna(
                input_df=input_df,
                fetal_fraction=fetal_fraction,
                ref_col=ref_col,
                alt_col=alt_col
            )
        elif self.mode == 'cfDNA+model+mGT':
            # Use model reads with maternal genotyping filtering
            return self._calculate_lr_with_model_and_maternal_genotyping(
                input_df=input_df,
                fetal_fraction=fetal_fraction,
                ref_col=ref_col,
                alt_col=alt_col,
                maternal_ref_col=maternal_ref_col,
                maternal_alt_col=maternal_alt_col
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _allelice_bias_adjust(
        self,
        p_alt_exp: float
    ) -> float:
        def logit(p): 
            p = np.clip(p, 1e-10, 1-1e-10)
            return np.log(p / (1 - p))
        def sigmoid(x): 
            return 1 / (1 + np.exp(-x))

        delta = logit(self.allelic_bias) - logit(self.allelic_theo)
        return sigmoid(logit(p_alt_exp) + delta)

    def _calculate_lr_cfdna(
        self,
        input_df: pd.DataFrame, 
        fetal_fraction: float,
        ref_col: str = 'cfDNA_ref_reads',
        alt_col: str = 'cfDNA_alt_reads'
    ) -> float:
        """
        Calculate the log likelihood ratio (log LR) of trisomy vs. disomy for a target chromosome.
        
        This function implements a comprehensive statistical model that:
        1. Considers all possible maternal and paternal genotype combinations
        2. Models fetal genotype distributions under both disomy and trisomy hypotheses
        3. Accounts for mixture effects due to fetal fraction in maternal plasma
        4. Computes log likelihood ratio as log(L_trisomy) - log(L_disomy)
        
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
            float: Log likelihood ratio (log LR) = log(L_trisomy) - log(L_disomy).
                   Positive values indicate evidence for trisomy, negative values indicate evidence for disomy.
        
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

        # Main likelihood computation
        logL_disomy = 0.0
        logL_trisomy = 0.0
        genotype_labels = ['0/0', '0/1', '1/1']
        
        console.print(f"[green]Calculating log likelihood ratio for {len(df_clean)} SNPs[/green]")
        
        # Process each SNP with progress tracking
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Computing log likelihood ratios...", total=len(df_clean))
            
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
                            
                            if self.factor_modeling:
                                # Use probabilistic disomy factor distribution
                                disomy_factor_values, disomy_log_factor_probs = _get_disomy_factor_distribution()
                                
                                # Integrate over disomy factor distribution
                                log_obs_terms_disomy = []
                                for factor, log_factor_prob in zip(disomy_factor_values, disomy_log_factor_probs):
                                    p_alt_exp = ((1.0 - factor * fetal_fraction) * (d_maternal / 2.0) + 
                                               factor * fetal_fraction * (d_fetal / 2.0))
                                    
                                    # Apply allelic bias
                                    b = self.allelic_bias
                                    p_alt_exp = (p_alt_exp * b) / (p_alt_exp * b + (1 - p_alt_exp))
                                    
                                    log_p_obs = _log_binomial_pmf_global(alt_count, depth, p_alt_exp)
                                    if log_p_obs != float('-inf'):
                                        log_obs_terms_disomy.append(log_factor_prob + log_p_obs)
                                
                                # Combine all factor-weighted observations for disomy
                                if log_obs_terms_disomy:
                                    log_integrated_obs_disomy = _log_sum_exp_global(np.array(log_obs_terms_disomy))
                                    log_terms_disomy.append(log_p_gm + log_p_gp + log_p_gf + log_integrated_obs_disomy)
                            else:
                                # Use fixed disomy factor (1.0)
                                p_alt_exp = (((1.0 - fetal_fraction) * d_maternal + fetal_fraction * d_fetal) / 2.0)

                                if self.beta_binomial:
                                    log_p_obs = _log_betabinom_pmf_global(alt_count, depth, p_alt_exp, self.rho)
                                else:
                                    log_p_obs = _log_binomial_pmf_global(alt_count, depth, p_alt_exp)

                                if log_p_obs != float('-inf'):
                                    log_terms_disomy.append(log_p_gm + log_p_gp + log_p_gf + log_p_obs)
                        
                        # Trisomy calculation
                        fetal_dist_tri = get_fetal_prob_trisomy(gm, gp)
                        
                        for d_fetal, p_fd in fetal_dist_tri.items():
                            if p_fd <= 0.0:
                                continue
                            log_p_fd = math.log(p_fd)
                            
                            if self.factor_modeling:
                                # Use probabilistic trisomy factor distribution
                                factor_values, log_factor_probs = _get_trisomy_factor_distribution()
                                
                                # Integrate over trisomy factor distribution
                                log_obs_terms = []
                                for factor, log_factor_prob in zip(factor_values, log_factor_probs):
                                    p_alt_exp_tri = ((1.0 - factor * fetal_fraction) * (d_maternal / 2.0) + 
                                                   factor * fetal_fraction * (d_fetal / 3.0))
                                    
                                    # Apply allelic bias
                                    b = self.allelic_bias
                                    p_alt_exp_tri = (p_alt_exp_tri * b) / (p_alt_exp_tri * b + (1 - p_alt_exp_tri))
                                    
                                    log_p_obs_tri = _log_binomial_pmf_global(alt_count, depth, p_alt_exp_tri)
                                    if log_p_obs_tri != float('-inf'):
                                        log_obs_terms.append(log_factor_prob + log_p_obs_tri)
                                
                                # Combine all factor-weighted observations
                                if log_obs_terms:
                                    log_integrated_obs = _log_sum_exp_global(np.array(log_obs_terms))
                                    log_terms_trisomy.append(log_p_gm + log_p_gp + log_p_fd + log_integrated_obs)
                            else:
                                # Use fixed trisomy factor (1.5)
                                p_alt_exp_tri = ((1.0 - 1.5 * fetal_fraction) * (d_maternal / 2.0) + 
                                            1.5 * fetal_fraction * (d_fetal / 3.0))
                                # p_alt_exp_tri = ((1 - fetal_fraction) * d_maternal + fetal_fraction * d_fetal) / (2 * (1 - fetal_fraction) + 3 * fetal_fraction)

                                if self.beta_binomial:
                                    log_p_obs_tri = _log_betabinom_pmf_global(alt_count, depth, p_alt_exp_tri, self.rho)
                                else:
                                    log_p_obs_tri = _log_binomial_pmf_global(alt_count, depth, p_alt_exp_tri)

                                if log_p_obs_tri != float('-inf'):
                                    log_terms_trisomy.append(log_p_gm + log_p_gp + log_p_fd + log_p_obs_tri)
                
                # Compute SNP-level log-likelihoods
                if log_terms_disomy:
                    logL_snp_disomy = _log_sum_exp_global(np.array(log_terms_disomy))
                    logL_disomy += logL_snp_disomy
                else:
                    return 0.0  # No valid disomy combinations
                
                if log_terms_trisomy:
                    logL_snp_trisomy = _log_sum_exp_global(np.array(log_terms_trisomy))
                    logL_trisomy += logL_snp_trisomy
                else:
                    return 0.0  # No valid trisomy combinations
                
                progress.advance(task)
        
        # Compute final log likelihood ratio
        delta_logL = logL_trisomy - logL_disomy
        
        # Return log likelihood ratio directly (no need for overflow handling since we're not exponentiating)
        return delta_logL

    def _calculate_lr_with_maternal_reads(
        self,
        input_df: pd.DataFrame,
        fetal_fraction: float,
        ref_col: str = 'cfDNA_ref_reads',
        alt_col: str = 'cfDNA_alt_reads',
        maternal_ref_col: str = 'maternal_ref_reads',
        maternal_alt_col: str = 'maternal_alt_reads'
    ) -> float:
        """
        Calculate the log likelihood ratio (log LR) of trisomy vs. disomy for a target chromosome,
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
            float: The log likelihood ratio (log LR) = log(L_trisomy) - log(L_disomy).
                   Positive values indicate evidence for trisomy, negative values indicate evidence for disomy.

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
        
        console.print(f"[green]Calculating log likelihood ratio for {len(df_clean)} SNPs with maternal genotypes[/green]")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Computing log likelihood ratios...", total=len(df_clean))
            
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
                        
                        if self.factor_modeling:
                            # Use probabilistic disomy factor distribution
                            disomy_factor_values, disomy_log_factor_probs = _get_disomy_factor_distribution()
                            
                            # Integrate over disomy factor distribution
                            log_obs_terms_disomy = []
                            for factor, log_factor_prob in zip(disomy_factor_values, disomy_log_factor_probs):
                                p_alt_exp = ((1.0 - factor * fetal_fraction) * (d_maternal / 2.0) +
                                             factor * fetal_fraction * (d_fetal / 2.0))
                                log_p_obs = _log_binomial_pmf_global(alt_count, depth, p_alt_exp)
                                if log_p_obs > float('-inf'):
                                    log_obs_terms_disomy.append(log_factor_prob + log_p_obs)
                            
                            # Combine all factor-weighted observations for disomy
                            if log_obs_terms_disomy:
                                log_integrated_obs_disomy = _log_sum_exp_global(np.array(log_obs_terms_disomy))
                                log_terms_disomy.append(log_p_gp + log_p_gf + log_integrated_obs_disomy)
                        else:
                            # Use fixed disomy factor (1.0)
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
                        
                        if self.factor_modeling:
                            # Use probabilistic trisomy factor distribution
                            factor_values, log_factor_probs = _get_trisomy_factor_distribution()
                            
                            # Integrate over trisomy factor distribution
                            log_obs_terms = []
                            for factor, log_factor_prob in zip(factor_values, log_factor_probs):
                                p_alt_exp_tri = ((1.0 - factor * fetal_fraction) * (d_maternal / 2.0) +
                                                 factor * fetal_fraction * (d_fetal / 3.0))
                                
                                log_p_obs_tri = _log_binomial_pmf_global(alt_count, depth, p_alt_exp_tri)
                                if log_p_obs_tri > float('-inf'):
                                    log_obs_terms.append(log_factor_prob + log_p_obs_tri)
                            
                            # Combine all factor-weighted observations
                            if log_obs_terms:
                                log_integrated_obs = _log_sum_exp_global(np.array(log_obs_terms))
                                log_terms_trisomy.append(log_p_gp + log_p_fd + log_integrated_obs)
                        else:
                            # Use fixed trisomy factor (1.5)
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

        # Compute final log likelihood ratio
        delta_logL = logL_trisomy - logL_disomy
        
        # Return log likelihood ratio directly (no need for overflow handling since we're not exponentiating)
        return delta_logL

    def _calculate_lr_with_model_and_maternal_genotyping(
        self,
        input_df: pd.DataFrame,
        fetal_fraction: float,
        ref_col: str = 'cfDNA_ref_reads',
        alt_col: str = 'cfDNA_alt_reads',
        maternal_ref_col: str = 'maternal_ref_reads',
        maternal_alt_col: str = 'maternal_alt_reads'
    ) -> float:
        """
        Calculate the log likelihood ratio (log LR) of trisomy vs. disomy for a target chromosome,
        using modeled fetal reads with maternal genotyping filtering.

        This function:
        1. Uses modeled fetal reads for LLR calculation
        2. Genotypes maternal reads (calculated as cfDNA - model reads)
        3. Filters to only homozygous maternal sites (0/0 and 1/1) for calculation
        4. Performs LLR calculation on the filtered homozygous SNPs

        Args:
            input_df (pd.DataFrame): DataFrame with SNP data including cfDNA and model reads.
                Required columns: 'chr', 'pos', 'af', cfDNA reads, and model reads.
                Maternal reads will be calculated as cfDNA - model reads.
            fetal_fraction (float): Estimated fetal fraction in maternal plasma (0 < ff < 1).
            ref_col (str, optional): Column name for cfDNA reference reads.
            alt_col (str, optional): Column name for cfDNA alternative reads.
            maternal_ref_col (str, optional): Column for calculated maternal reference reads.
            maternal_alt_col (str, optional): Column for calculated maternal alternative reads.

        Returns:
            float: The log likelihood ratio (log LR) = log(L_trisomy) - log(L_disomy).
                   Positive values indicate evidence for trisomy, negative values indicate evidence for disomy.

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

        # Genotype maternal reads to filter for homozygous sites
        m_ref = df_clean[maternal_ref_col].to_numpy(dtype=np.int32)
        m_alt = df_clean[maternal_alt_col].to_numpy(dtype=np.int32)
        m_n = m_ref + m_alt
        
        # Avoid division by zero
        valid_coverage = m_n > 0
        if not valid_coverage.any():
            console.print("[yellow]Warning: No maternal reads with coverage > 0[/yellow]")
            return 0.0
        
        # Calculate maternal alt allele fraction
        maternal_alt_fraction = np.zeros_like(m_n, dtype=np.float64)
        maternal_alt_fraction[valid_coverage] = m_alt[valid_coverage] / m_n[valid_coverage]
        
        # Genotype maternal reads using simple thresholds
        # 0/0: alt_fraction < 0.2, 0/1: 0.2 <= alt_fraction <= 0.8, 1/1: alt_fraction > 0.8
        maternal_genotype_mask = np.zeros(len(df_clean), dtype=bool)
        
        # Keep only homozygous sites (0/0 or 1/1)
        homozygous_00 = (maternal_alt_fraction < 0.2) & valid_coverage
        homozygous_11 = (maternal_alt_fraction > 0.8) & valid_coverage
        maternal_genotype_mask = homozygous_00 | homozygous_11

        # Keep only heterozygous sites (0/1)
        # heterozygous_01 = (maternal_alt_fraction >= 0.2) & (maternal_alt_fraction <= 0.8) & valid_coverage
        # maternal_genotype_mask = heterozygous_01
        
        # Filter to homozygous sites only
        df_filtered = df_clean[maternal_genotype_mask].reset_index(drop=True)
        
        if len(df_filtered) == 0:
            console.print("[yellow]Warning: No homozygous maternal sites found for analysis[/yellow]")
            return 0.0
        
        console.print(f"[cyan]Filtered to {len(df_filtered)} homozygous maternal sites out of {len(df_clean)} total SNPs[/cyan]")
        
        # Helper functions for genotype modeling (same as cfDNA mode)
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
        
        # Main likelihood computation using filtered homozygous SNPs
        logL_disomy = 0.0
        logL_trisomy = 0.0
        genotype_labels = ['0/0', '0/1', '1/1']
        
        console.print(f"[green]Calculating log likelihood ratio for {len(df_filtered)} homozygous SNPs[/green]")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Computing log likelihood ratios (homozygous sites)...", total=len(df_filtered))
            
            for idx, snp in df_filtered.iterrows():
                ref_count = int(snp[ref_col])
                alt_count = int(snp[alt_col])
                depth = ref_count + alt_count
                af = float(snp['af'])

                geno_priors = genotype_priors(af)
                
                log_terms_disomy = []
                log_terms_trisomy = []
                
                for gm in genotype_labels:
                    p_gm = geno_priors[gm]
                    if p_gm <= 0.0:
                        continue
                    log_p_gm = math.log(p_gm)
                    
                    # Marginalize over paternal genotypes
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
                            
                            if self.factor_modeling:
                                # Use probabilistic disomy factor distribution
                                disomy_factor_values, disomy_log_factor_probs = _get_disomy_factor_distribution()
                                
                                # Integrate over disomy factor distribution
                                log_obs_terms_disomy = []
                                for factor, log_factor_prob in zip(disomy_factor_values, disomy_log_factor_probs):
                                    p_alt_exp = ((1.0 - factor * fetal_fraction) * (d_maternal / 2.0) +
                                                factor * fetal_fraction * (d_fetal / 2.0))
                                    log_p_obs = _log_binomial_pmf_global(alt_count, depth, p_alt_exp)
                                    if log_p_obs > float('-inf'):
                                        log_obs_terms_disomy.append(log_factor_prob + log_p_obs)
                                
                                # Combine all factor-weighted observations for disomy
                                if log_obs_terms_disomy:
                                    log_integrated_obs_disomy = _log_sum_exp_global(np.array(log_obs_terms_disomy))
                                    log_terms_disomy.append(log_p_gp + log_p_gf + log_integrated_obs_disomy)
                            else:
                                # Use fixed disomy factor (1.0)
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
                            
                            if self.factor_modeling:
                                # Use probabilistic trisomy factor distribution
                                factor_values, log_factor_probs = _get_trisomy_factor_distribution()
                                
                                # Integrate over trisomy factor distribution
                                log_obs_terms = []
                                for factor, log_factor_prob in zip(factor_values, log_factor_probs):
                                    p_alt_exp_tri = ((1.0 - factor * fetal_fraction) * (d_maternal / 2.0) +
                                                    factor * fetal_fraction * (d_fetal / 3.0))
                                    log_p_obs_tri = _log_binomial_pmf_global(alt_count, depth, p_alt_exp_tri)
                                    if log_p_obs_tri > float('-inf'):
                                        log_obs_terms.append(log_factor_prob + log_p_obs_tri)
                                
                                # Combine all factor-weighted observations
                                if log_obs_terms:
                                    log_integrated_obs = _log_sum_exp_global(np.array(log_obs_terms))
                                    log_terms_trisomy.append(log_p_gp + log_p_fd + log_integrated_obs)
                            else:
                                # Use fixed trisomy factor (1.5)
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

        # Compute final log likelihood ratio
        delta_logL = logL_trisomy - logL_disomy
        
        return delta_logL


def load_and_validate_data(
    input_path, 
    mode: str = 'cfDNA',
    cfdna_ref_col: str = 'cfDNA_ref_reads',
    cfdna_alt_col: str = 'cfDNA_alt_reads',
    wbc_ref_col: str = 'maternal_ref_reads',
    wbc_alt_col: str = 'maternal_alt_reads',
    model_ref_col: str = 'fetal_ref_reads_from_model',
    model_alt_col: str = 'fetal_alt_reads_from_model',
    min_raw_depth: int = 0,
    min_model_depth: int = 0
) -> pd.DataFrame:
    """
    Load and validate input SNP data from TSV.GZ file.
    
    This function validates the presence of required columns based on the analysis mode
    and performs data type validation and cleaning. It also applies depth filters.
    
    Args:
        input_path: Path to input file
        mode (str): Analysis mode ('cfDNA', 'cfDNA+WBC', or 'cfDNA+model')
        cfdna_ref_col (str): Column name for cfDNA reference reads
        cfdna_alt_col (str): Column name for cfDNA alternative reads
        wbc_ref_col (str): Column name for maternal WBC reference reads
        wbc_alt_col (str): Column name for maternal WBC alternative reads
        model_ref_col (str): Column name for modeled fetal reference reads
        model_alt_col (str): Column name for modeled fetal alternative reads
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
    elif mode == 'cfDNA+model+mGT':
        required_columns.update({cfdna_ref_col, cfdna_alt_col, model_ref_col, model_alt_col})
        read_columns = [cfdna_ref_col, cfdna_alt_col, model_ref_col, model_alt_col]
        console.print(f"[cyan]Validating cfDNA+model+mGT mode with columns: {', '.join(read_columns)}[/cyan]")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: cfDNA, cfDNA+WBC, cfDNA+model, cfDNA+model+mGT")
    
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
    elif min_model_depth > 0 and mode not in ['cfDNA+model', 'cfDNA+model+mGT']:
        console.print(f"[yellow]Warning: Model depth filter ignored for mode '{mode}' (only applies to 'cfDNA+model' modes)[/yellow]")
    
    # Calculate maternal reads for cfDNA+model+mGT mode
    if mode == 'cfDNA+model+mGT':
        console.print("[cyan]Calculating maternal reads as cfDNA - model reads[/cyan]")
        
        # Calculate maternal reads by subtracting model reads from cfDNA reads
        df[wbc_ref_col] = df[cfdna_ref_col] - df[model_ref_col]
        df[wbc_alt_col] = df[cfdna_alt_col] - df[model_alt_col]
        
        # Handle edge cases where subtraction results in negative values
        negative_ref = df[wbc_ref_col] < 0
        negative_alt = df[wbc_alt_col] < 0
        
        if negative_ref.any() or negative_alt.any():
            negative_count = (negative_ref | negative_alt).sum()
            console.print(f"[yellow]Warning: {negative_count} SNPs have negative maternal reads (model > cfDNA), setting to 0[/yellow]")
            df.loc[negative_ref, wbc_ref_col] = 0
            df.loc[negative_alt, wbc_alt_col] = 0
        
        # Filter out SNPs where maternal coverage is zero after calculation
        df['maternal_total_reads'] = df[wbc_ref_col] + df[wbc_alt_col]
        zero_maternal_coverage = df['maternal_total_reads'] == 0
        
        if zero_maternal_coverage.any():
            console.print(f"[yellow]Warning: Removing {zero_maternal_coverage.sum()} SNPs with zero maternal coverage[/yellow]")
            df = df[~zero_maternal_coverage]
        
        # Clean up temporary column
        df = df.drop(columns=['maternal_total_reads']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No SNPs remaining after calculating maternal reads")
        
        console.print(f"[green]Successfully calculated maternal reads for {len(df)} SNPs[/green]")
    
    return df


import click
from pathlib import Path


@click.command()
@click.option(
    '--input-path', '-i',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to input TSV.GZ file containing SNP data'
)
@click.option(
    '--fetal-fraction', '-f',
    type=click.FloatRange(0.0, 1.0),
    required=True,
    help='Fetal fraction value for LLR calculation (0.0-1.0)'
)
@click.option(
    '--chromosome', '-c',
    type=str,
    default='chr21',
    help='Target chromosome for LLR calculation (e.g., "chr21", "21"). Default: chr21'
)
@click.option(
    '--mode',
    type=click.Choice(['cfDNA', 'cfDNA+WBC', 'cfDNA+model', 'cfDNA+model+mGT'], case_sensitive=False),
    default='cfDNA',
    help='Analysis mode: cfDNA (standard), cfDNA+WBC (with maternal WBC), cfDNA+model (with modeled maternal reads), cfDNA+model+mGT (with modeled reads and maternal genotyping)'
)
@click.option(
    '--factor-modeling',
    is_flag=True,
    default=False,
    help='Use probabilistic modeling for read release factors (disomy: N(1, 0.5²), trisomy: N(1.55, 0.6²)) instead of fixed factors (default: False)'
)
@click.option(
    '--beta-binomial',
    is_flag=True,
    default=False,
    help='Use beta-binomial distribution for allelic distribution (default: False)'
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
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def main(
    input_path: Path,
    fetal_fraction: float,
    chromosome: str,
    mode: str,
    factor_modeling: bool,
    beta_binomial: bool,
    cfdna_ref_col: str,
    cfdna_alt_col: str,
    wbc_ref_col: str,
    wbc_alt_col: str,
    model_ref_col: str,
    model_alt_col: str,
    min_raw_depth: int,
    min_model_depth: int,
    verbose: bool
) -> None:
    """
    Log Likelihood Ratio Calculator
    
    This tool calculates log likelihood ratios for trisomy vs. disomy detection
    from cell-free DNA sequencing data.
    
    The input file should be a TSV.GZ file with columns:
    chr, pos, af, and read count columns (names configurable via CLI options)
     
    Supports four analysis modes:
    - cfDNA: Standard cell-free DNA analysis
    - cfDNA+WBC: Analysis with maternal white blood cell data
    - cfDNA+model: Analysis with modeled fetal reads
    - cfDNA+model+mGT: Analysis with modeled fetal reads and maternal genotyping filtering
    
    Returns the log likelihood ratio to stdout.
    """
    # Configure console output
    if verbose:
        console.print("[blue]Verbose mode enabled[/blue]")
    
    try:
        # Normalize chromosome name
        if not chromosome.startswith('chr'):
            chromosome = f"chr{chromosome}"
        
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
        
        # Filter data for target chromosome
        target_data = df[df['chr'] == chromosome]
        
        if len(target_data) == 0:
            raise ValueError(f"No SNPs found for chromosome {chromosome}")
        
        console.print(f"[cyan]Found {len(target_data)} SNPs for {chromosome}[/cyan]")
        
        # Initialize LLRCalculator
        llr_calculator = LLRCalculator(mode=mode, factor_modeling=factor_modeling, beta_binomial=beta_binomial)
        
        # Calculate log likelihood ratio
        console.print(f"[cyan]Calculating log likelihood ratio for {chromosome}...[/cyan]")
        
        if mode == 'cfDNA':
            llr = llr_calculator.calculate(
                target_data,
                fetal_fraction,
                ref_col=cfdna_ref_col,
                alt_col=cfdna_alt_col
            )
        elif mode == 'cfDNA+WBC':
            llr = llr_calculator.calculate(
                target_data,
                fetal_fraction,
                ref_col=cfdna_ref_col,
                alt_col=cfdna_alt_col,
                maternal_ref_col=wbc_ref_col,
                maternal_alt_col=wbc_alt_col
            )
        elif mode == 'cfDNA+model':
            llr = llr_calculator.calculate(
                target_data,
                fetal_fraction,
                ref_col=model_ref_col,
                alt_col=model_alt_col
            )
        elif mode == 'cfDNA+model+mGT':
            llr = llr_calculator.calculate(
                target_data,
                fetal_fraction,
                ref_col=model_ref_col,
                alt_col=model_alt_col,
                maternal_ref_col=wbc_ref_col,
                maternal_alt_col=wbc_alt_col
            )
        
        # Output the log likelihood ratio to stdout
        print(f"{llr:.6f}")
        
        if verbose:
            console.print(f"[bold green]✓ Log likelihood ratio calculation complete: {llr:.6f}[/bold green]")
            if llr > 0:
                console.print(f"[green]Positive LLR suggests evidence for trisomy[/green]")
            elif llr < 0:
                console.print(f"[blue]Negative LLR suggests evidence for disomy[/blue]")
            else:
                console.print(f"[yellow]Zero LLR suggests no evidence either way[/yellow]")
        
    except Exception as e:
        console.print(f"[red]✗ Error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()
