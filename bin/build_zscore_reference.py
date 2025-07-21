#!/usr/bin/env python3
"""
Z-score Reference Builder for SNP-based NIPT Analysis

This script calculates mu and sigma values for z-score normalization by processing
pileup files and genotype data to compute fetal reads percentages across chromosomes.

The script processes multiple pileup samples, overlaps them with genotype information,
calculates fetal reads percentages per chromosome, and outputs statistical parameters
for z-score reference building.
"""

import sys
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import click
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def load_pileup_file(pileup_path: Path) -> pd.DataFrame:
    """
    Load a pileup file from either gzip or plain text format.
    
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
        else:
            console.print(f"[yellow]Warning:[/yellow] No valid reads data for chromosome {chr_name}")
    
    return chromosome_percentages


def process_pileup_files(pileup_list_path: Path, filtered_snps: pd.DataFrame, 
                        progress: Progress, task_id: TaskID) -> Dict[str, List[float]]:
    """
    Process all pileup files and calculate fetal reads percentages.
    
    Args:
        pileup_list_path: Path to file containing list of pileup files
        filtered_snps: DataFrame with filtered SNP sites (chr, pos)
        progress: Rich progress bar instance
        task_id: Task ID for progress tracking
        
    Returns:
        Dictionary mapping chromosome names to lists of fetal reads percentages
        
    Raises:
        FileNotFoundError: If pileup list file doesn't exist
        ValueError: If no valid pileup files are found
    """
    if not pileup_list_path.exists():
        raise FileNotFoundError(f"Pileup list file not found: {pileup_list_path}")
    
    # Read pileup file paths
    with open(pileup_list_path, 'r') as f:
        pileup_paths = [Path(line.strip()) for line in f if line.strip()]
    
    if not pileup_paths:
        raise ValueError("No pileup files found in the list")
    
    console.print(f"[green]Found {len(pileup_paths)} pileup files to process[/green]")
    
    # Dictionary to store fetal reads percentages for each chromosome across all samples
    chromosome_percentages = {}
    
    progress.update(task_id, total=len(pileup_paths))
    
    # Process each pileup file
    for i, pileup_path in enumerate(pileup_paths):
        progress.update(task_id, description=f"Processing {pileup_path.name}")
        
        try:
            # Load pileup data
            pileup_data = load_pileup_file(pileup_path)
            
            # Calculate fetal reads percentage for this sample
            sample_percentages = calculate_fetal_reads_percentage(pileup_data, filtered_snps)
            
            # Add to chromosome collections
            for chr_name, percentage in sample_percentages.items():
                if chr_name not in chromosome_percentages:
                    chromosome_percentages[chr_name] = []
                chromosome_percentages[chr_name].append(percentage)
            
        except Exception as e:
            console.print(f"[red]Error processing {pileup_path}: {str(e)}[/red]")
            continue
        
        progress.update(task_id, advance=1)
    
    return chromosome_percentages


def calculate_zscore_parameters(chromosome_percentages: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Calculate mu and sigma for each chromosome from fetal reads percentages.
    
    Args:
        chromosome_percentages: Dictionary mapping chromosomes to percentage lists
        
    Returns:
        DataFrame with columns: chr, mu, sigma
        
    Raises:
        ValueError: If no valid data is available for calculation
    """
    if not chromosome_percentages:
        raise ValueError("No chromosome data available for z-score calculation")
    
    results = []
    
    for chr_name, percentages in chromosome_percentages.items():
        if len(percentages) < 2:
            console.print(f"[yellow]Warning:[/yellow] Insufficient data for chromosome {chr_name} (need ≥2 samples)")
            continue
        
        # Calculate mu (mean) and sigma (standard deviation)
        percentages_array = np.array(percentages)
        mu = np.mean(percentages_array)
        sigma = np.std(percentages_array, ddof=1)  # Use sample standard deviation
        
        results.append({
            'chr': chr_name,
            'mu': mu,
            'sigma': sigma
        })
    
    if not results:
        raise ValueError("No valid z-score parameters could be calculated")
    
    return pd.DataFrame(results)


def display_results_summary(results_df: pd.DataFrame) -> None:
    """
    Display a formatted summary of the z-score calculation results.
    
    Args:
        results_df: DataFrame containing the calculated z-score parameters
    """
    table = Table(title="Z-score Reference Parameters", box=box.ROUNDED)
    table.add_column("Chromosome", style="cyan", no_wrap=True)
    table.add_column("Mu (μ)", style="green", justify="right")
    table.add_column("Sigma (σ)", style="yellow", justify="right")
    
    for _, row in results_df.iterrows():
        table.add_row(
            str(row['chr']),
            f"{row['mu']:.6f}",
            f"{row['sigma']:.6f}"
        )
    
    console.print(table)


@click.command()
@click.option('--input-pileup-list', '-i', 
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to text file containing list of pileup files (.tsv.gz format)')
@click.option('--maternal-genotype', '-m',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to maternal genotype TSV file with chr, pos, genotype columns')
@click.option('--fetal-genotype', '-f',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Path to fetal genotype TSV file with chr, pos, genotype columns')
@click.option('--output', '-o',
              type=click.Path(path_type=Path),
              default='zscore_reference.tsv',
              help='Output file path for z-score reference parameters (default: zscore_reference.tsv)')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output for debugging')
def main(input_pileup_list: Path, maternal_genotype: Path, fetal_genotype: Path, output: Path, verbose: bool) -> None:
    """
    Build z-score reference parameters from pileup files and genotype data.
    
    This tool processes multiple pileup samples, filters SNPs with maternal 'AA' and 
    fetal 'Aa' genotypes, calculates fetal reads percentages per chromosome, and outputs 
    statistical parameters (mu and sigma) for z-score normalization in NIPT analysis.
    
    Only SNPs that have maternal 'AA' genotype AND fetal 'Aa' genotype are used for analysis.
    The fetal reads percentage is calculated as: p_fetal = sum(cfDNA_alt_reads) / sum(current_depth)
    
    The output file contains three columns:
    - chr: Chromosome name
    - mu: Mean fetal reads percentage
    - sigma: Standard deviation of fetal reads percentage
    
    Examples:
        # Basic usage
        python build_zscore_reference.py -i pileup_list.txt -m maternal.tsv -f fetal.tsv
        
        # Specify custom output file
        python build_zscore_reference.py -i pileup_list.txt -m maternal.tsv -f fetal.tsv -o my_reference.tsv
        
        # Enable verbose output
        python build_zscore_reference.py -i pileup_list.txt -m maternal.tsv -f fetal.tsv -v
    """
    try:
        # Display header
        console.print(Panel.fit(
            "[bold blue]Z-score Reference Builder[/bold blue]\n"
            "Calculating mu and sigma for NIPT z-score normalization",
            border_style="blue"
        ))
        
        if verbose:
            console.print(f"[dim]Input pileup list: {input_pileup_list}[/dim]")
            console.print(f"[dim]Maternal genotype file: {maternal_genotype}[/dim]")
            console.print(f"[dim]Fetal genotype file: {fetal_genotype}[/dim]")
            console.print(f"[dim]Output file: {output}[/dim]")
        
        # Load genotype data
        console.print("\n[bold]Step 1:[/bold] Loading genotype data...")
        maternal_genotype_data = load_genotype_file(maternal_genotype, "maternal")
        fetal_genotype_data = load_genotype_file(fetal_genotype, "fetal")
        
        if verbose:
            maternal_counts = maternal_genotype_data['genotype'].value_counts()
            fetal_counts = fetal_genotype_data['genotype'].value_counts()
            console.print(f"[dim]Maternal genotype distribution: {dict(maternal_counts)}[/dim]")
            console.print(f"[dim]Fetal genotype distribution: {dict(fetal_counts)}[/dim]")
        
        # Filter SNPs by genotype criteria
        console.print("\n[bold]Step 2:[/bold] Filtering SNPs by genotype criteria...")
        filtered_snps = filter_snps_by_genotypes(maternal_genotype_data, fetal_genotype_data)
        
        if verbose:
            chromosomes_with_snps = set(filtered_snps['chr'])
            console.print(f"[dim]Chromosomes with valid SNPs: {sorted(chromosomes_with_snps)}[/dim]")
        
        # Process pileup files
        console.print("\n[bold]Step 3:[/bold] Processing pileup files...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task_id = progress.add_task("Processing pileup files...", total=None)
            chromosome_percentages = process_pileup_files(input_pileup_list, filtered_snps, progress, task_id)
        
        console.print(f"[green]✓[/green] Processed data for {len(chromosome_percentages)} chromosomes")
        
        if verbose:
            for chr_name, percentages in chromosome_percentages.items():
                console.print(f"[dim]Chromosome {chr_name}: {len(percentages)} samples[/dim]")
        
        # Calculate z-score parameters
        console.print("\n[bold]Step 4:[/bold] Calculating z-score parameters...")
        results_df = calculate_zscore_parameters(chromosome_percentages)
        console.print(f"[green]✓[/green] Calculated parameters for {len(results_df)} chromosomes")
        
        # Save results
        console.print(f"\n[bold]Step 5:[/bold] Saving results to {output}...")
        results_df.to_csv(output, sep='\t', index=False, float_format='%.6f')
        console.print(f"[green]✓[/green] Results saved successfully")
        
        # Display summary
        console.print("\n[bold]Results Summary:[/bold]")
        display_results_summary(results_df)
        
        console.print(f"\n[green]✨ Z-score reference building completed successfully![/green]")
        console.print(f"[green]Output written to: {output}[/green]")
        
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
