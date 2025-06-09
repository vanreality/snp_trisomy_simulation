"""
SNP Sequencing Data Simulator for cfDNA from Pregnant Women

This script simulates SNP sequencing data of cell-free DNA (cfDNA) from pregnant women 
who carry either triploid or diploid fetuses. It models the complex interactions between 
maternal and fetal DNA in circulating blood samples, accounting for various factors such as 
fetal fraction, sequencing depth, and classifier accuracy.

The simulation supports:
- Diploid (normal) fetal genotypes
- Triploid (trisomy) fetal genotypes for specific chromosomes
- Realistic sequencing depth variation (Poisson-distributed)
- Machine learning classifier filtering effects
- Parallel processing for efficiency

Author: Generated with AI assistance
License: MIT
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import logging
from functools import partial

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from numpy.random import default_rng, Generator
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeRemainingColumn
)
from tqdm import tqdm
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Initialize console for rich output
console = Console()

# Initialize random number generator with a fixed seed for reproducibility
RANDOM_SEED = 42
rng = default_rng(RANDOM_SEED)


class SNPSimulationError(Exception):
    """Custom exception for SNP simulation errors."""
    pass


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.
    
    Args:
        verbose: If True, enable debug logging. Otherwise, use info level.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('snp_simulation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_inputs(
    n_repeats: int,
    model_accuracy: float,
    min_depth: int,
    max_depth: int,
    num_depth: int,
    min_ff: float,
    max_ff: float,
    num_ff: int,
    potential_snp_path: Path
) -> None:
    """
    Validate all input parameters to ensure they are within acceptable ranges.
    
    Args:
        n_repeats: Number of simulation repeats
        model_accuracy: Model accuracy (0-1)
        min_depth: Minimum sequencing depth
        max_depth: Maximum sequencing depth
        num_depth: Number of depth points
        min_ff: Minimum fetal fraction
        max_ff: Maximum fetal fraction
        num_ff: Number of fetal fraction points
        potential_snp_path: Path to SNP data file
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    if n_repeats < 1:
        raise ValidationError("Number of repeats must be at least 1")
    
    if not 0 <= model_accuracy <= 1:
        raise ValidationError("Model accuracy must be between 0 and 1")
    
    if min_depth < 1 or max_depth < min_depth:
        raise ValidationError("Invalid depth range: min_depth must be >= 1 and max_depth >= min_depth")
    
    if num_depth < 1:
        raise ValidationError("Number of depth points must be at least 1")
    
    if not 0 <= min_ff <= 1 or not 0 <= max_ff <= 1 or min_ff > max_ff:
        raise ValidationError("Invalid fetal fraction range: values must be between 0 and 1, min_ff <= max_ff")
    
    if num_ff < 1:
        raise ValidationError("Number of fetal fraction points must be at least 1")
    
    if not potential_snp_path.exists():
        raise ValidationError(f"SNP data file not found: {potential_snp_path}")


def extract_AF(row: pd.Series) -> float:
    """
    Parse the AF (allele frequency) field from the INFO column.
    
    Args:
        row: Pandas Series containing variant information with 'info' column
        
    Returns:
        Allele frequency as float
        
    Raises:
        SNPSimulationError: If AF field cannot be parsed
    """
    try:
        info = row['info']
        if 'AF=' not in info:
            raise SNPSimulationError(f"AF field not found in INFO column: {info}")
        
        af_str = info.split('AF=')[1].split(';')[0]
        af = float(af_str)
        
        if not 0 <= af <= 1:
            raise SNPSimulationError(f"Invalid allele frequency: {af}")
            
        return af
    except (IndexError, ValueError) as e:
        raise SNPSimulationError(f"Failed to parse AF from INFO: {info}") from e


def generate_genotype(af: float, rng_instance: Optional[Generator] = None) -> int:
    """
    Generate a genotype based on allele frequency using Hardy-Weinberg equilibrium.
    
    Args:
        af: Allele frequency of the alternative allele (0-1)
        rng_instance: Optional random number generator instance
        
    Returns:
        Genotype code (0 = homozygous reference, 1 = heterozygous, 2 = homozygous alternative)
        
    Raises:
        ValidationError: If allele frequency is invalid
    """
    if not 0 <= af <= 1:
        raise ValidationError(f"Allele frequency must be between 0 and 1, got: {af}")
    
    if rng_instance is None:
        rng_instance = rng
    
    # Hardy-Weinberg equilibrium probabilities
    p_hom_ref = (1 - af) ** 2
    p_het = 2 * af * (1 - af)
    p_hom_alt = af ** 2
    
    # Ensure probabilities sum to 1 (handle floating point precision)
    total = p_hom_ref + p_het + p_hom_alt
    if abs(total - 1.0) > 1e-10:
        p_hom_ref /= total
        p_het /= total
        p_hom_alt /= total
    
    return rng_instance.choice([0, 1, 2], p=[p_hom_ref, p_het, p_hom_alt])


def get_disomy_fetal_genotype(
    maternal_gt: int, 
    paternal_gt: int, 
    rng_instance: Optional[Generator] = None
) -> int:
    """
    Determine the fetal genotype for diploid inheritance given parental genotypes.
    
    This function models normal Mendelian inheritance where the fetus receives
    one allele from each parent.
    
    Args:
        maternal_gt: Maternal genotype (0, 1, or 2)
        paternal_gt: Paternal genotype (0, 1, or 2)
        rng_instance: Optional random number generator instance
        
    Returns:
        Fetal genotype (0, 1, or 2)
        
    Raises:
        ValidationError: If genotypes are invalid
    """
    if maternal_gt not in [0, 1, 2] or paternal_gt not in [0, 1, 2]:
        raise ValidationError(f"Invalid genotypes: maternal={maternal_gt}, paternal={paternal_gt}")
    
    if rng_instance is None:
        rng_instance = rng
    
    # Mapping for more readable code
    genotype_map = {
        0: "RR",  # Homozygous reference
        1: "RA",  # Heterozygous
        2: "AA"   # Homozygous alternative
    }
    
    # Determine possible offspring genotypes based on Mendelian inheritance
    if maternal_gt == 0:  # Maternal RR
        if paternal_gt == 0:    # Paternal RR → offspring RR
            return 0
        elif paternal_gt == 1:  # Paternal RA → offspring RR or RA
            return rng_instance.choice([0, 1])
        else:  # paternal_gt == 2, Paternal AA → offspring RA
            return 1
    elif maternal_gt == 1:  # Maternal RA
        if paternal_gt == 0:    # Paternal RR → offspring RR or RA
            return rng_instance.choice([0, 1])
        elif paternal_gt == 1:  # Paternal RA → offspring RR, RA, or AA (1:2:1)
            return rng_instance.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
        else:  # paternal_gt == 2, Paternal AA → offspring RA or AA
            return rng_instance.choice([1, 2])
    else:  # maternal_gt == 2, Maternal AA
        if paternal_gt == 0:    # Paternal RR → offspring RA
            return 1
        elif paternal_gt == 1:  # Paternal RA → offspring RA or AA
            return rng_instance.choice([1, 2])
        else:  # paternal_gt == 2, Paternal AA → offspring AA
            return 2


def get_trisomy_fetal_genotype(
    maternal_gt: int, 
    paternal_gt: int, 
    rng_instance: Optional[Generator] = None
) -> int:
    """
    Determine the fetal genotype for trisomy inheritance given parental genotypes.
    
    This function models trisomy where the fetus receives an extra chromosome,
    randomly inherited from either mother or father. The genotype is encoded
    as the count of alternate alleles (0-3).
    
    Args:
        maternal_gt: Maternal genotype (0, 1, or 2)
        paternal_gt: Paternal genotype (0, 1, or 2)
        rng_instance: Optional random number generator instance
        
    Returns:
        Trisomy fetal genotype (0-3 alternate alleles)
        
    Raises:
        ValidationError: If genotypes are invalid
    """
    if maternal_gt not in [0, 1, 2] or paternal_gt not in [0, 1, 2]:
        raise ValidationError(f"Invalid genotypes: maternal={maternal_gt}, paternal={paternal_gt}")
    
    if rng_instance is None:
        rng_instance = rng
    
    # First get a normal diploid genotype
    disomy_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt, rng_instance)
    disomy_alt_count = disomy_gt  # Convert genotype code to alt allele count
    
    # Randomly choose the origin of the extra chromosome
    origin = rng_instance.choice(['maternal', 'paternal'])
    
    # Sample one allele from the chosen parent
    if origin == 'maternal':
        if maternal_gt == 0:        # RR → always contributes R
            extra_alt = 0
        elif maternal_gt == 1:      # RA → 50% R, 50% A
            extra_alt = rng_instance.choice([0, 1])
        else:  # maternal_gt == 2   # AA → always contributes A
            extra_alt = 1
    else:  # origin == 'paternal'
        if paternal_gt == 0:        # RR → always contributes R
            extra_alt = 0
        elif paternal_gt == 1:      # RA → 50% R, 50% A
            extra_alt = rng_instance.choice([0, 1])
        else:  # paternal_gt == 2   # AA → always contributes A
            extra_alt = 1
    
    # Total alternate allele count for trisomy (0-3)
    trisomy_alt_count = disomy_alt_count + extra_alt
    return min(trisomy_alt_count, 3)  # Ensure max value is 3


def simulate_disomy_sequencing(
    depth: int, 
    fetal_fraction: float, 
    maternal_gt: int, 
    fetal_gt: int,
    rng_instance: Optional[Generator] = None
) -> Tuple[int, int, int, int, int, int]:
    """
    Simulate sequencing reads for diploid inheritance scenario.
    
    This function models the sequencing process where maternal and fetal cfDNA
    are mixed in the sample according to the fetal fraction.
    
    Args:
        depth: Total sequencing depth for this SNP
        fetal_fraction: Proportion of fetal cfDNA (0-1)
        maternal_gt: Maternal genotype (0, 1, 2)
        fetal_gt: Fetal genotype (0, 1, 2)
        rng_instance: Optional random number generator instance
        
    Returns:
        Tuple of (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, 
                 fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads)
                 
    Raises:
        ValidationError: If parameters are invalid
    """
    if depth < 0:
        raise ValidationError(f"Depth must be non-negative, got: {depth}")
    if not 0 <= fetal_fraction <= 1:
        raise ValidationError(f"Fetal fraction must be between 0 and 1, got: {fetal_fraction}")
    if maternal_gt not in [0, 1, 2] or fetal_gt not in [0, 1, 2]:
        raise ValidationError(f"Invalid genotypes: maternal={maternal_gt}, fetal={fetal_gt}")
    
    if rng_instance is None:
        rng_instance = rng
    
    # Handle edge case of zero depth
    if depth == 0:
        return 0, 0, 0, 0, 0, 0
    
    # Simulate the number of fetal vs maternal reads
    fetal_reads = rng_instance.binomial(depth, fetal_fraction)
    maternal_reads = depth - fetal_reads
    
    # Simulate maternal allele counts
    maternal_alt_prob = maternal_gt / 2.0
    maternal_alt_reads = rng_instance.binomial(maternal_reads, maternal_alt_prob)
    maternal_ref_reads = maternal_reads - maternal_alt_reads
    
    # Simulate fetal allele counts
    fetal_alt_prob = fetal_gt / 2.0
    fetal_alt_reads = rng_instance.binomial(fetal_reads, fetal_alt_prob)
    fetal_ref_reads = fetal_reads - fetal_alt_reads
    
    # Calculate combined cfDNA allele counts
    cfDNA_alt_reads = maternal_alt_reads + fetal_alt_reads
    cfDNA_ref_reads = maternal_ref_reads + fetal_ref_reads
    
    return (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, 
            fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads)


def simulate_trisomy_sequencing(
    depth: int, 
    fetal_fraction: float, 
    maternal_gt: int, 
    fetal_gt: int,
    rng_instance: Optional[Generator] = None
) -> Tuple[int, int, int, int, int, int]:
    """
    Simulate sequencing reads for trisomy inheritance scenario.
    
    This function models sequencing for trisomy cases where the fetus has
    three copies of a chromosome instead of two.
    
    Args:
        depth: Total sequencing depth for this SNP
        fetal_fraction: Proportion of fetal cfDNA (0-1)
        maternal_gt: Maternal genotype (0, 1, 2)
        fetal_gt: Fetal genotype (0, 1, 2, 3 for trisomy)
        rng_instance: Optional random number generator instance
        
    Returns:
        Tuple of (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, 
                 fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads)
                 
    Raises:
        ValidationError: If parameters are invalid
    """
    if depth < 0:
        raise ValidationError(f"Depth must be non-negative, got: {depth}")
    if not 0 <= fetal_fraction <= 1:
        raise ValidationError(f"Fetal fraction must be between 0 and 1, got: {fetal_fraction}")
    if maternal_gt not in [0, 1, 2] or fetal_gt not in [0, 1, 2, 3]:
        raise ValidationError(f"Invalid genotypes: maternal={maternal_gt}, fetal={fetal_gt}")
    
    if rng_instance is None:
        rng_instance = rng
    
    # Handle edge case of zero depth
    if depth == 0:
        return 0, 0, 0, 0, 0, 0
    
    # For trisomy, we increase the expected fetal reads by 50% to account for extra chromosome
    adjusted_fetal_fraction = min(fetal_fraction * 1.5, 1.0)
    fetal_reads = rng_instance.binomial(depth, adjusted_fetal_fraction)
    maternal_reads = depth - fetal_reads
    
    # Simulate maternal allele counts (normal diploid)
    maternal_alt_prob = maternal_gt / 2.0
    maternal_alt_reads = rng_instance.binomial(maternal_reads, maternal_alt_prob)
    maternal_ref_reads = maternal_reads - maternal_alt_reads
    
    # Simulate fetal allele counts (trisomy - 3 chromosomes)
    fetal_alt_prob = fetal_gt / 3.0
    fetal_alt_reads = rng_instance.binomial(fetal_reads, fetal_alt_prob)
    fetal_ref_reads = fetal_reads - fetal_alt_reads
    
    # Calculate combined cfDNA allele counts
    cfDNA_alt_reads = maternal_alt_reads + fetal_alt_reads
    cfDNA_ref_reads = maternal_ref_reads + fetal_ref_reads
    
    return (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, 
            fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads)


def classifier_filtering(
    model_accuracy: float,
    maternal_ref_reads: int,
    maternal_alt_reads: int,
    fetal_ref_reads: int,
    fetal_alt_reads: int,
    rng_instance: Optional[Generator] = None
) -> Tuple[int, int, int, int]:
    """
    Simulate the effect of a machine learning classifier that attempts to separate
    maternal from fetal reads.
    
    The classifier has a specified accuracy rate, leading to both correctly
    identified fetal reads (true positives) and misclassified maternal reads
    (false positives).
    
    Args:
        model_accuracy: Accuracy of the classifier (0-1)
        maternal_ref_reads: Number of maternal reference reads
        maternal_alt_reads: Number of maternal alternate reads
        fetal_ref_reads: Number of fetal reference reads
        fetal_alt_reads: Number of fetal alternate reads
        rng_instance: Optional random number generator instance
        
    Returns:
        Tuple of (correct_fetal_ref, correct_fetal_alt, misclassified_maternal_ref,
                 misclassified_maternal_alt)
                 
    Raises:
        ValidationError: If parameters are invalid
    """
    if not 0 <= model_accuracy <= 1:
        raise ValidationError(f"Model accuracy must be between 0 and 1, got: {model_accuracy}")
    
    for reads in [maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads]:
        if reads < 0:
            raise ValidationError(f"Read counts must be non-negative, got: {reads}")
    
    if rng_instance is None:
        rng_instance = rng
    
    # True positives: correctly identified fetal reads
    correct_fetal_ref = rng_instance.binomial(fetal_ref_reads, model_accuracy)
    correct_fetal_alt = rng_instance.binomial(fetal_alt_reads, model_accuracy)
    
    # False positives: maternal reads misclassified as fetal
    misclassified_maternal_ref = rng_instance.binomial(maternal_ref_reads, 1 - model_accuracy)
    misclassified_maternal_alt = rng_instance.binomial(maternal_alt_reads, 1 - model_accuracy)
    
    return (correct_fetal_ref, correct_fetal_alt, 
            misclassified_maternal_ref, misclassified_maternal_alt)


def run_single_disomy_simulation_set(
    variants_df: pd.DataFrame,
    depth_lambda: int,
    fetal_fraction_value: float,
    model_accuracy: float,
    progress_callback: Optional[callable] = None
) -> pd.DataFrame:
    """
    Run disomy simulation for a set of variants with given parameters.
    
    Args:
        variants_df: DataFrame containing variant information with 'af' column
        depth_lambda: Poisson lambda parameter for sequencing depth
        fetal_fraction_value: Fraction of fetal cfDNA
        model_accuracy: Accuracy of the classifier model
        progress_callback: Optional callback function for progress updates
        
    Returns:
        DataFrame containing simulation results for each variant
        
    Raises:
        ValidationError: If input parameters are invalid
        SNPSimulationError: If simulation fails
    """
    if variants_df.empty:
        console.print("[yellow]Warning: Empty variants DataFrame provided[/yellow]")
        return pd.DataFrame()
    
    required_columns = ['chr', 'pos', 'ref', 'alt', 'af']
    missing_columns = [col for col in required_columns if col not in variants_df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")
    
    if depth_lambda <= 0:
        raise ValidationError(f"Depth lambda must be positive, got: {depth_lambda}")
    
    records = []
    local_rng = default_rng()  # Local RNG for thread safety
    
    for idx, (_, row) in enumerate(variants_df.iterrows()):
        try:
            af = row['af']
            if not 0 <= af <= 1:
                console.print(f"[yellow]Warning: Invalid AF value {af} for variant at {row['chr']}:{row['pos']}, skipping[/yellow]")
                continue
            
            # Generate parental genotypes
            maternal_gt = generate_genotype(af, local_rng)
            paternal_gt = generate_genotype(af, local_rng)
            fetal_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt, local_rng)
            
            # Draw sequencing depth from Poisson distribution
            current_depth = local_rng.poisson(depth_lambda)
            if current_depth == 0:
                continue  # Skip variants with zero depth
            
            # Simulate sequencing
            (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, 
             fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads) = simulate_disomy_sequencing(
                current_depth, fetal_fraction_value, maternal_gt, fetal_gt, local_rng
            )
            
            # Apply classifier filtering
            (filtered_fetal_ref, filtered_fetal_alt, 
             misclassified_maternal_ref, misclassified_maternal_alt) = classifier_filtering(
                model_accuracy, maternal_ref_reads, maternal_alt_reads, 
                fetal_ref_reads, fetal_alt_reads, local_rng
            )
            
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
            
            # Update progress if callback provided
            if progress_callback and idx % 100 == 0:
                progress_callback(idx)
                
        except Exception as e:
            console.print(f"[red]Error processing variant at {row['chr']}:{row['pos']}: {e}[/red]")
            continue
    
    if not records:
        console.print("[yellow]Warning: No valid simulation results generated[/yellow]")
        return pd.DataFrame()
    
    return pd.DataFrame(records)


def run_single_trisomy_simulation_set(
    variants_df: pd.DataFrame,
    depth_lambda: int,
    fetal_fraction_value: float,
    model_accuracy: float,
    trisomy_chr: str,
    progress_callback: Optional[callable] = None
) -> pd.DataFrame:
    """
    Run trisomy simulation for a set of variants with given parameters.
    
    Args:
        variants_df: DataFrame containing variant information with 'af' column
        depth_lambda: Poisson lambda parameter for sequencing depth
        fetal_fraction_value: Fraction of fetal cfDNA
        model_accuracy: Accuracy of the classifier model
        trisomy_chr: Chromosome to simulate trisomy for
        progress_callback: Optional callback function for progress updates
        
    Returns:
        DataFrame containing simulation results for each variant
        
    Raises:
        ValidationError: If input parameters are invalid
        SNPSimulationError: If simulation fails
    """
    if variants_df.empty:
        console.print("[yellow]Warning: Empty variants DataFrame provided[/yellow]")
        return pd.DataFrame()
    
    required_columns = ['chr', 'pos', 'ref', 'alt', 'af']
    missing_columns = [col for col in required_columns if col not in variants_df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")
    
    if depth_lambda <= 0:
        raise ValidationError(f"Depth lambda must be positive, got: {depth_lambda}")
    
    records = []
    local_rng = default_rng()  # Local RNG for thread safety
    
    for idx, (_, row) in enumerate(variants_df.iterrows()):
        try:
            af = row['af']
            if not 0 <= af <= 1:
                console.print(f"[yellow]Warning: Invalid AF value {af} for variant at {row['chr']}:{row['pos']}, skipping[/yellow]")
                continue
            
            # Generate parental genotypes
            maternal_gt = generate_genotype(af, local_rng)
            paternal_gt = generate_genotype(af, local_rng)
            
            # Determine fetal genotype based on chromosome
            if row['chr'] == trisomy_chr:
                fetal_gt = get_trisomy_fetal_genotype(maternal_gt, paternal_gt, local_rng)
                is_trisomy = True
            else:
                fetal_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt, local_rng)
                is_trisomy = False
            
            # Draw sequencing depth from Poisson distribution
            current_depth = local_rng.poisson(depth_lambda)
            if current_depth == 0:
                continue  # Skip variants with zero depth
            
            # Simulate sequencing based on ploidy
            if is_trisomy:
                (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, 
                 fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads) = simulate_trisomy_sequencing(
                    current_depth, fetal_fraction_value, maternal_gt, fetal_gt, local_rng
                )
            else:
                (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, 
                 fetal_alt_reads, cfDNA_ref_reads, cfDNA_alt_reads) = simulate_disomy_sequencing(
                    current_depth, fetal_fraction_value, maternal_gt, fetal_gt, local_rng
                )
            
            # Apply classifier filtering
            (filtered_fetal_ref, filtered_fetal_alt, 
             misclassified_maternal_ref, misclassified_maternal_alt) = classifier_filtering(
                model_accuracy, maternal_ref_reads, maternal_alt_reads, 
                fetal_ref_reads, fetal_alt_reads, local_rng
            )
            
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
                'is_trisomy_chr': is_trisomy,
            })
            
            # Update progress if callback provided
            if progress_callback and idx % 100 == 0:
                progress_callback(idx)
                
        except Exception as e:
            console.print(f"[red]Error processing variant at {row['chr']}:{row['pos']}: {e}[/red]")
            continue
    
    if not records:
        console.print("[yellow]Warning: No valid simulation results generated[/yellow]")
        return pd.DataFrame()
    
    return pd.DataFrame(records)


def run_simulation(args: Tuple) -> None:
    """
    Wrapper function for parallel execution of simulations.
    
    Args:
        args: Tuple containing (depth, ff, repeat_idx, trisomy_chr, potential_df, 
              model_accuracy, output_dir)
    """
    depth, ff, repeat_idx, trisomy_chr, potential_df, model_accuracy, output_dir = args
    
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run disomy simulation
        disomy_simulated_df = run_single_disomy_simulation_set(
            potential_df, depth, ff, model_accuracy
        )
        
        if not disomy_simulated_df.empty:
            disomy_output_path = Path(output_dir) / f"disomy_{depth}_{ff:.3f}_{repeat_idx}.tsv"
            disomy_simulated_df.to_csv(disomy_output_path, sep='\t', index=False, header=True, compression='gzip')
        
        # Run trisomy simulation
        trisomy_simulated_df = run_single_trisomy_simulation_set(
            potential_df, depth, ff, model_accuracy, trisomy_chr
        )
        
        if not trisomy_simulated_df.empty:
            trisomy_output_path = Path(output_dir) / f"trisomy_{depth}_{ff:.3f}_{repeat_idx}.tsv"
            trisomy_simulated_df.to_csv(trisomy_output_path, sep='\t', index=False, header=True, compression='gzip')
            
    except Exception as e:
        console.print(f"[red]Error in simulation for depth={depth}, ff={ff:.3f}, repeat={repeat_idx}: {e}[/red]")


def load_snp_data(file_path: Path) -> pd.DataFrame:
    """
    Load and validate SNP data from file.
    
    Args:
        file_path: Path to the SNP data file
        
    Returns:
        Validated DataFrame with AF column added
        
    Raises:
        SNPSimulationError: If file cannot be loaded or is invalid
    """
    try:
        # Try to load the file
        potential_df = pd.read_csv(
            file_path,
            sep='\t',
            names=['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
            dtype={'chr': str, 'pos': int, 'id': str, 'ref': str, 'alt': str}
        )
        
        if potential_df.empty:
            raise SNPSimulationError("SNP data file is empty")
        
        # Extract allele frequencies
        console.print(f"[blue]Extracting allele frequencies from {len(potential_df)} variants...[/blue]")
        
        # Apply AF extraction with error handling
        af_values = []
        invalid_count = 0
        
        for idx, row in potential_df.iterrows():
            try:
                af = extract_AF(row)
                af_values.append(af)
            except SNPSimulationError:
                af_values.append(np.nan)
                invalid_count += 1
        
        potential_df['af'] = af_values
        
        # Remove rows with invalid AF values
        before_filter = len(potential_df)
        potential_df = potential_df.dropna(subset=['af'])
        after_filter = len(potential_df)
        
        if invalid_count > 0:
            console.print(f"[yellow]Removed {invalid_count} variants with invalid AF values[/yellow]")
        
        if potential_df.empty:
            raise SNPSimulationError("No valid variants remaining after AF extraction")
        
        console.print(f"[green]Successfully loaded {after_filter} valid variants[/green]")
        
        # Display summary statistics
        af_stats = potential_df['af'].describe()
        table = Table(title="Allele Frequency Statistics")
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", style="green")
        
        for stat, value in af_stats.items():
            if isinstance(value, float):
                table.add_row(stat, f"{value:.3f}")
            else:
                table.add_row(stat, str(value))
        
        console.print(table)
        
        return potential_df
        
    except FileNotFoundError:
        raise SNPSimulationError(f"SNP data file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise SNPSimulationError(f"SNP data file is empty: {file_path}")
    except Exception as e:
        raise SNPSimulationError(f"Failed to load SNP data: {e}") from e


@click.command()
@click.option(
    '--n_repeats', 
    default=1, 
    type=click.IntRange(min=1),
    help='Number of repetitions per parameter combination'
)
@click.option(
    '--model_accuracy', 
    default=0.81, 
    type=click.FloatRange(min=0.0, max=1.0),
    help='Model accuracy for simulations (0.0-1.0)'
)
@click.option(
    '--trisomy_chr', 
    default='chr16', 
    type=str,
    help='Chromosome to simulate trisomy for (e.g., chr16, chr18, chr21)'
)
@click.option(
    '--min_depth', 
    default=50, 
    type=click.IntRange(min=1),
    help='Minimum sequencing depth'
)
@click.option(
    '--max_depth', 
    default=150, 
    type=click.IntRange(min=1),
    help='Maximum sequencing depth'
)
@click.option(
    '--num_depth', 
    default=11, 
    type=click.IntRange(min=1),
    help='Number of depth points'
)
@click.option(
    '--min_ff', 
    default=0.005, 
    type=click.FloatRange(min=0.0, max=1.0),
    help='Minimum fetal fraction (0.0-1.0)'
)
@click.option(
    '--max_ff', 
    default=0.05, 
    type=click.FloatRange(min=0.0, max=1.0),
    help='Maximum fetal fraction (0.0-1.0)'
)
@click.option(
    '--num_ff', 
    default=10, 
    type=click.IntRange(min=1),
    help='Number of fetal fraction points'
)
@click.option(
    '--potential_snp_path', 
    default='filtered_senddmr_igtc_ChinaMAP.tsv',
    type=click.Path(exists=True, path_type=Path),
    help='Path to variant data TSV file'
)
@click.option(
    '--output_dir', 
    default='results',
    type=click.Path(path_type=Path),
    help='Output directory for results'
)
@click.option(
    '--verbose', 
    is_flag=True,
    help='Enable verbose logging'
)
@click.option(
    '--n_processes',
    default=None,
    type=click.IntRange(min=1),
    help='Number of processes for parallel execution (default: CPU count - 1)'
)
def main(
    n_repeats: int,
    model_accuracy: float,
    trisomy_chr: str,
    min_depth: int,
    max_depth: int,
    num_depth: int,
    min_ff: float,
    max_ff: float,
    num_ff: int,
    potential_snp_path: Path,
    output_dir: Path,
    verbose: bool,
    n_processes: Optional[int]
) -> None:
    """
    SNP Sequencing Data Simulator for cfDNA Analysis
    
    This tool simulates SNP sequencing data from cell-free DNA (cfDNA) samples
    of pregnant women carrying either diploid or triploid fetuses. It models
    the complex mixture of maternal and fetal DNA in blood samples.
    
    The simulation accounts for:
    - Varying sequencing depths (Poisson-distributed)
    - Different fetal fractions
    - Machine learning classifier accuracy effects
    - Both diploid and trisomy scenarios
    
    Results are saved as TSV files for downstream analysis.
    """
    # Setup logging and console output
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]SNP Sequencing Data Simulator[/bold blue]\n"
        "[dim]Simulating cfDNA from pregnant women with diploid/triploid fetuses[/dim]",
        border_style="blue"
    ))
    
    try:
        # Validate inputs
        validate_inputs(
            n_repeats, model_accuracy, min_depth, max_depth, num_depth,
            min_ff, max_ff, num_ff, potential_snp_path
        )
        
        # Load and validate SNP data
        console.print("[blue]Loading SNP data...[/blue]")
        potential_df = load_snp_data(potential_snp_path)
        
        # Generate parameter ranges
        depth_range = np.linspace(min_depth, max_depth, num=num_depth, dtype=int)
        ff_range = np.linspace(min_ff, max_ff, num=num_ff)
        
        # Display simulation parameters
        params_table = Table(title="Simulation Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="green")
        
        params_table.add_row("Repeats per combination", str(n_repeats))
        params_table.add_row("Model accuracy", f"{model_accuracy:.3f}")
        params_table.add_row("Trisomy chromosome", trisomy_chr)
        params_table.add_row("Depth range", f"{min_depth}-{max_depth} ({num_depth} points)")
        params_table.add_row("Fetal fraction range", f"{min_ff:.3f}-{max_ff:.3f} ({num_ff} points)")
        params_table.add_row("Total combinations", str(len(depth_range) * len(ff_range) * n_repeats))
        params_table.add_row("Output directory", str(output_dir))
        
        console.print(params_table)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Generate all parameter combinations
        param_combinations = [
            (depth, ff, repeat_idx, trisomy_chr, potential_df, model_accuracy, str(output_dir))
            for depth in depth_range
            for ff in ff_range
            for repeat_idx in range(n_repeats)
        ]
        
        total_combinations = len(param_combinations)
        console.print(f"[blue]Starting simulation of {total_combinations} parameter combinations...[/blue]")
        
        # Determine number of processes
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)
        
        console.print(f"[blue]Using {n_processes} parallel processes[/blue]")
        
        # Run simulations with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "Running simulations...", 
                total=total_combinations
            )
            
            # Use multiprocessing for parallel execution
            with Pool(processes=n_processes) as pool:
                # Submit all jobs
                results = []
                for args in param_combinations:
                    result = pool.apply_async(run_simulation, (args,))
                    results.append(result)
                
                # Monitor progress
                completed = 0
                while completed < total_combinations:
                    completed = sum(1 for r in results if r.ready())
                    progress.update(task, completed=completed)
                    
                    # Small delay to prevent excessive CPU usage
                    if completed < total_combinations:
                        import time
                        time.sleep(0.1)
                
                # Ensure all processes complete
                pool.close()
                pool.join()
        
        # Summary
        console.print(Panel.fit(
            f"[bold green]Simulation completed successfully![/bold green]\n"
            f"[dim]Generated {total_combinations} simulation files in {output_dir}[/dim]",
            border_style="green"
        ))
        
        logger.info(f"Simulation completed. Output saved to: {output_dir}")
        
    except ValidationError as e:
        console.print(f"[bold red]Validation Error:[/bold red] {e}")
        sys.exit(1)
    except SNPSimulationError as e:
        console.print(f"[bold red]Simulation Error:[/bold red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        logger.exception("Unexpected error occurred")
        sys.exit(1)


if __name__ == '__main__':
    main()