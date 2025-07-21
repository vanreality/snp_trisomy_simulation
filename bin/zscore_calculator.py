#!/usr/bin/env python3
"""
Z-score Calculator for SNP-based NIPT Analysis

This script calculates z-scores for fetal fraction estimation by processing
a single pileup file with genotype data and reference parameters (mu, sigma)
to compute normalized z-scores across chromosomes.

The script loads pileup data, overlaps with genotype information, calculates
fetal reads percentages per chromosome, and normalizes using reference parameters
to produce z-scores for NIPT analysis.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def load_pileup_file(pileup_path: Path) -> pd.DataFrame:
    """
    Load a pileup file from gzip or plain text format.
    
    Args:
        pileup_path: Path to the pileup file (.tsv or .tsv.gz)
        
    Returns:
        DataFrame with pileup data containing columns:
        chr, pos, ref, alt, af, cfDNA_ref_reads, cfDNA_alt_reads, current_depth
        
    Raises:
        FileNotFoundError: If the pileup file doesn't exist
        ValueError: If the file format is invalid or missing required columns
    """
    if not pileup_path.exists():
        raise FileNotFoundError(f"Pileup file not found: {pileup_path}")
    
    try:
        # Determine if file is gzipped
        if pileup_path.suffix == '.gz':
            df = pd.read_csv(pileup_path, sep='\t', compression='gzip')
        else:
            df = pd.read_csv(pileup_path, sep='\t')
        
        # Validate required columns
        required_columns = ['chr', 'pos', 'ref', 'alt', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in {pileup_path}: {missing_columns}")
        
        # Convert numeric columns to appropriate types
        numeric_columns = ['pos', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid data
        initial_rows = len(df)
        df = df.dropna(subset=numeric_columns)
        
        if len(df) < initial_rows:
            console.print(f"[yellow]Warning:[/yellow] Removed {initial_rows - len(df)} rows with invalid numeric data from {pileup_path.name}")
        
        console.print(f"[green]✓[/green] Loaded {len(df)} SNP sites from pileup file")
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading pileup file {pileup_path}: {str(e)}")


def load_genotype_file(genotype_path: Path, genotype_type: str) -> pd.DataFrame:
    """
    Load genotype file with chromosome, position, and genotype information.
    
    Args:
        genotype_path: Path to the genotype TSV file
        genotype_type: Type of genotype file ('maternal' or 'fetal') for logging
        
    Returns:
        DataFrame with genotype data containing columns: chr, pos, genotype
        
    Raises:
        FileNotFoundError: If the genotype file doesn't exist
        ValueError: If the file format is invalid or missing required columns
    """
    if not genotype_path.exists():
        raise FileNotFoundError(f"{genotype_type.capitalize()} genotype file not found: {genotype_path}")
    
    try:
        df = pd.read_csv(genotype_path, sep='\t', usecols=['chr', 'pos', 'genotype'])
        
        # Validate required columns
        required_columns = ['chr', 'pos', 'genotype']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in {genotype_type} genotype file: {missing_columns}")
        
        # Convert position to numeric
        df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
        
        # Remove rows with invalid data
        initial_rows = len(df)
        df = df.dropna()
        
        if len(df) < initial_rows:
            console.print(f"[yellow]Warning:[/yellow] Removed {initial_rows - len(df)} rows with invalid data from {genotype_type} genotype file")
        
        console.print(f"[green]✓[/green] Loaded {len(df)} {genotype_type} genotype records")
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading {genotype_type} genotype file: {str(e)}")


def load_reference_file(reference_path: Path) -> pd.DataFrame:
    """
    Load reference file with chromosome-specific mu and sigma values.
    
    Args:
        reference_path: Path to the reference TSV file
        
    Returns:
        DataFrame with reference data containing columns: chr, mu, sigma
        
    Raises:
        FileNotFoundError: If the reference file doesn't exist
        ValueError: If the file format is invalid or missing required columns
    """
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    
    try:
        df = pd.read_csv(reference_path, sep='\t')
        
        # Validate required columns
        required_columns = ['chr', 'mu', 'sigma']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in reference file: {missing_columns}")
        
        # Convert numeric columns to appropriate types
        numeric_columns = ['mu', 'sigma']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid data
        initial_rows = len(df)
        df = df.dropna()
        
        if len(df) < initial_rows:
            console.print(f"[yellow]Warning:[/yellow] Removed {initial_rows - len(df)} rows with invalid data from reference file")
        
        # Check for zero sigma values which would cause division by zero
        zero_sigma = df[df['sigma'] == 0]
        if not zero_sigma.empty:
            console.print(f"[red]Error:[/red] Found chromosomes with zero sigma values: {list(zero_sigma['chr'])}")
            raise ValueError("Cannot calculate z-scores with zero sigma values")
        
        console.print(f"[green]✓[/green] Loaded reference parameters for {len(df)} chromosomes")
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading reference file: {str(e)}")


def filter_snps_by_genotypes(maternal_genotype: pd.DataFrame, fetal_genotype: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SNPs to only include those with maternal 'AA' and fetal 'Aa' genotypes.
    
    Args:
        maternal_genotype: DataFrame with maternal genotype data
        fetal_genotype: DataFrame with fetal genotype data
        
    Returns:
        DataFrame with filtered SNPs containing chr and pos columns
        
    Raises:
        ValueError: If no valid SNPs are found after filtering
    """
    # Filter maternal genotype for 'AA' only
    maternal_aa = maternal_genotype[maternal_genotype['genotype'] == 'AA'][['chr', 'pos']]
    
    # Filter fetal genotype for 'Aa' only
    fetal_aa = fetal_genotype[fetal_genotype['genotype'] == 'Aa'][['chr', 'pos']]
    
    if maternal_aa.empty:
        raise ValueError("No maternal 'AA' genotype sites found")
    
    if fetal_aa.empty:
        raise ValueError("No fetal 'Aa' genotype sites found")
    
    # Find intersection of maternal AA and fetal Aa sites
    filtered_snps = pd.merge(maternal_aa, fetal_aa, on=['chr', 'pos'], how='inner')
    
    if filtered_snps.empty:
        raise ValueError("No overlapping SNP sites found with maternal 'AA' and fetal 'Aa' genotypes")
    
    console.print(f"[green]✓[/green] Found {len(filtered_snps)} SNPs with maternal 'AA' and fetal 'Aa' genotypes")
    return filtered_snps


def calculate_fetal_reads_percentage(pileup_data: pd.DataFrame, filtered_snps: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate fetal reads percentage for each chromosome.
    
    Since we only use SNPs with maternal 'AA' and fetal 'Aa' genotypes,
    the fetal reads percentage is simply: p_fetal = sum(cfDNA_alt_reads) / sum(current_depth)
    
    Args:
        pileup_data: DataFrame with pileup information
        filtered_snps: DataFrame with filtered SNP sites (chr, pos)
        
    Returns:
        Dictionary mapping chromosome to fetal reads percentage
        
    Raises:
        ValueError: If no overlapping SNP sites are found
    """
    # Merge pileup and filtered SNP data on chr and pos
    merged_data = pd.merge(pileup_data, filtered_snps, on=['chr', 'pos'], how='inner')
    
    if merged_data.empty:
        raise ValueError("No overlapping SNP sites found between pileup and filtered SNP data")
    
    console.print(f"[green]✓[/green] Found {len(merged_data)} overlapping SNP sites")
    
    chromosome_percentages = {}
    
    # Group by chromosome for processing
    for chr_name, chr_group in merged_data.groupby('chr'):
        # Filter out rows with zero depth to avoid division by zero
        chr_group = chr_group[chr_group['current_depth'] > 0]
        
        if chr_group.empty:
            console.print(f"[yellow]Warning:[/yellow] No valid depth data for chromosome {chr_name}")
            continue
        
        # Calculate fetal reads percentage: p_fetal = sum(cfDNA_alt_reads) / sum(current_depth)
        total_alt_reads = chr_group['cfDNA_alt_reads'].sum()
        total_depth = chr_group['current_depth'].sum()
        
        if total_depth > 0:
            p_fetal_reads = total_alt_reads / total_depth
            chromosome_percentages[chr_name] = p_fetal_reads
            console.print(f"[dim]Chromosome {chr_name}: p_fetal = {p_fetal_reads:.6f} (sites: {len(chr_group)})[/dim]")
        else:
            console.print(f"[yellow]Warning:[/yellow] No valid reads data for chromosome {chr_name}")
    
    return chromosome_percentages


def calculate_zscores(chromosome_percentages: Dict[str, float], reference_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate z-scores for each chromosome using reference parameters.
    
    Args:
        chromosome_percentages: Dictionary mapping chromosomes to fetal reads percentages
        reference_data: DataFrame with reference mu and sigma values
        
    Returns:
        DataFrame with columns: chr, p_fetal, zscore
        
    Raises:
        ValueError: If no valid z-scores can be calculated
    """
    if not chromosome_percentages:
        raise ValueError("No chromosome data available for z-score calculation")
    
    # Convert reference data to dictionary for faster lookup
    reference_dict = {}
    for _, row in reference_data.iterrows():
        reference_dict[row['chr']] = {'mu': row['mu'], 'sigma': row['sigma']}
    
    results = []
    
    for chr_name, p_fetal in chromosome_percentages.items():
        if chr_name not in reference_dict:
            console.print(f"[yellow]Warning:[/yellow] No reference data for chromosome {chr_name}")
            continue
        
        ref_params = reference_dict[chr_name]
        mu = ref_params['mu']
        sigma = ref_params['sigma']
        
        # Calculate z-score: (observed - mean) / standard_deviation
        zscore = (p_fetal - mu) / sigma
        
        results.append({
            'chr': chr_name,
            'p_fetal': p_fetal,
            'zscore': zscore
        })
        
        console.print(f"[dim]Chromosome {chr_name}: z-score = {zscore:.6f}[/dim]")
    
    if not results:
        raise ValueError("No valid z-scores could be calculated")
    
    return pd.DataFrame(results)


def display_results_summary(results_df: pd.DataFrame) -> None:
    """
    Display a formatted summary of the z-score calculation results.
    
    Args:
        results_df: DataFrame containing the calculated z-scores and p_fetal values
    """
    table = Table(title="Z-score Calculation Results", box=box.ROUNDED)
    table.add_column("Chromosome", style="cyan", no_wrap=True)
    table.add_column("P_fetal", style="blue", justify="right")
    table.add_column("Z-score", style="green", justify="right")
    table.add_column("Interpretation", style="yellow", justify="center")
    
    for _, row in results_df.iterrows():
        zscore = row['zscore']
        p_fetal = row['p_fetal']
        
        # Add interpretation based on z-score magnitude
        if abs(zscore) < 1.96:
            interpretation = "Normal"
            style = "green"
        elif abs(zscore) < 3.0:
            interpretation = "Moderate"
            style = "yellow"
        else:
            interpretation = "High"
            style = "red"
        
        table.add_row(
            str(row['chr']),
            f"{p_fetal:.6f}",
            f"{zscore:.6f}",
            f"[{style}]{interpretation}[/{style}]"
        )
    
    console.print(table)


@click.command()
@click.option('--input-pileup', '-i', 
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to pileup file (.tsv or .tsv.gz format)')
@click.option('--maternal-genotype', '-m',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to maternal genotype TSV file with chr, pos, genotype columns')
@click.option('--fetal-genotype', '-f',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to fetal genotype TSV file with chr, pos, genotype columns')
@click.option('--reference', '-r',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to reference TSV file with chr, mu, sigma columns')
@click.option('--output', '-o',
              type=str,
              required=True,
              help='Output prefix for the z-score results file')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output for debugging')
def main(input_pileup: Path, maternal_genotype: Path, fetal_genotype: Path, reference: Path, output: str, verbose: bool) -> None:
    """
    Calculate z-scores for fetal fraction estimation from pileup data.
    
    This tool processes a single pileup sample with maternal and fetal genotype information
    and reference parameters to calculate normalized z-scores per chromosome for NIPT analysis.
    
    Only SNPs that have maternal 'AA' genotype AND fetal 'Aa' genotype are used for analysis.
    The fetal reads percentage is calculated as: p_fetal = sum(cfDNA_alt_reads) / sum(current_depth)
    
    The tool performs the following steps:
    1. Loads pileup data and filters SNPs with maternal 'AA' and fetal 'Aa' genotypes
    2. Calculates fetal reads percentage per chromosome
    3. Normalizes using reference parameters (mu, sigma) to compute z-scores
    4. Outputs results to {output}_zscore.tsv
    
    The output file contains three columns:
    - chr: Chromosome name
    - p_fetal: Calculated fetal reads percentage
    - zscore: Calculated z-score value
    
    Examples:
        # Basic usage
        python zscore_calculator.py -i sample.tsv.gz -m maternal.tsv -f fetal.tsv -r reference.tsv -o sample_results
        
        # Enable verbose output
        python zscore_calculator.py -i sample.tsv.gz -m maternal.tsv -f fetal.tsv -r reference.tsv -o sample_results -v
    """
    try:
        # Display header
        console.print(Panel.fit(
            "[bold blue]Z-score Calculator[/bold blue]\n"
            "Calculating z-scores for NIPT fetal fraction analysis",
            border_style="blue"
        ))
        
        if verbose:
            console.print(f"[dim]Input pileup: {input_pileup}[/dim]")
            console.print(f"[dim]Maternal genotype file: {maternal_genotype}[/dim]")
            console.print(f"[dim]Fetal genotype file: {fetal_genotype}[/dim]")
            console.print(f"[dim]Reference file: {reference}[/dim]")
            console.print(f"[dim]Output prefix: {output}[/dim]")
        
        # Prepare output filename
        output_file = Path(f"{output}_zscore.tsv")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Step 1: Load all input files
            task_id = progress.add_task("Loading input files...", total=4)
            
            console.print("\n[bold]Step 1:[/bold] Loading input files...")
            
            # Load maternal genotype data
            progress.update(task_id, description="Loading maternal genotype data...")
            maternal_genotype_data = load_genotype_file(maternal_genotype, "maternal")
            progress.update(task_id, advance=1)
            
            # Load fetal genotype data
            progress.update(task_id, description="Loading fetal genotype data...")
            fetal_genotype_data = load_genotype_file(fetal_genotype, "fetal")
            progress.update(task_id, advance=1)
            
            # Load reference data
            progress.update(task_id, description="Loading reference data...")
            reference_data = load_reference_file(reference)
            progress.update(task_id, advance=1)
            
            # Load pileup data
            progress.update(task_id, description="Loading pileup data...")
            pileup_data = load_pileup_file(input_pileup)
            progress.update(task_id, advance=1)
            
            if verbose:
                maternal_counts = maternal_genotype_data['genotype'].value_counts()
                fetal_counts = fetal_genotype_data['genotype'].value_counts()
                console.print(f"[dim]Maternal genotype distribution: {dict(maternal_counts)}[/dim]")
                console.print(f"[dim]Fetal genotype distribution: {dict(fetal_counts)}[/dim]")
                ref_chromosomes = set(reference_data['chr'])
                console.print(f"[dim]Reference chromosomes: {sorted(ref_chromosomes)}[/dim]")
            
            # Step 2: Filter SNPs by genotype criteria
            task_id = progress.add_task("Filtering SNPs by genotype criteria...", total=1)
            console.print("\n[bold]Step 2:[/bold] Filtering SNPs by genotype criteria...")
            
            filtered_snps = filter_snps_by_genotypes(maternal_genotype_data, fetal_genotype_data)
            progress.update(task_id, advance=1)
            
            if verbose:
                chromosomes_with_snps = set(filtered_snps['chr'])
                console.print(f"[dim]Chromosomes with valid SNPs: {sorted(chromosomes_with_snps)}[/dim]")
            
            # Step 3: Calculate fetal reads percentages
            task_id = progress.add_task("Calculating fetal reads percentages...", total=1)
            console.print("\n[bold]Step 3:[/bold] Calculating fetal reads percentages...")
            
            chromosome_percentages = calculate_fetal_reads_percentage(pileup_data, filtered_snps)
            progress.update(task_id, advance=1)
            
            console.print(f"[green]✓[/green] Calculated percentages for {len(chromosome_percentages)} chromosomes")
            
            # Step 4: Calculate z-scores
            task_id = progress.add_task("Calculating z-scores...", total=1)
            console.print("\n[bold]Step 4:[/bold] Calculating z-scores...")
            
            results_df = calculate_zscores(chromosome_percentages, reference_data)
            progress.update(task_id, advance=1)
            
            console.print(f"[green]✓[/green] Calculated z-scores for {len(results_df)} chromosomes")
            
            # Step 5: Save results
            task_id = progress.add_task("Saving results...", total=1)
            console.print(f"\n[bold]Step 5:[/bold] Saving results to {output_file}...")
            
            results_df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
            progress.update(task_id, advance=1)
            
            console.print(f"[green]✓[/green] Results saved successfully")
        
        # Display summary
        console.print("\n[bold]Results Summary:[/bold]")
        display_results_summary(results_df)
        
        # Show statistical summary
        if verbose:
            zscore_stats = results_df['zscore'].describe()
            console.print(f"\n[bold]Z-score Statistics:[/bold]")
            console.print(f"[dim]Mean: {zscore_stats['mean']:.6f}[/dim]")
            console.print(f"[dim]Std: {zscore_stats['std']:.6f}[/dim]")
            console.print(f"[dim]Min: {zscore_stats['min']:.6f}[/dim]")
            console.print(f"[dim]Max: {zscore_stats['max']:.6f}[/dim]")
        
        console.print(f"\n[green]✨ Z-score calculation completed successfully![/green]")
        console.print(f"[green]Output written to: {output_file}[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[red]❌ Process interrupted by user[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Error: {str(e)}[/red]")
        if verbose:
            console.print_exception(show_locals=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
