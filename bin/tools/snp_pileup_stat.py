#!/usr/bin/env python3
"""
SNP Pileup Statistics Calculator

This script analyzes BAM files and SNP pileup files to generate comprehensive
statistics including depth coverage, VAF distributions, and informative SNP counts.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import click
import pandas as pd
import pysam
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box

console = Console()
progress_lock = Lock()


def calculate_bam_mean_depth(bam_path: Path) -> float:
    """
    Calculate mean depth of covered regions in a BAM file.
    
    Args:
        bam_path: Path to the BAM file.
        
    Returns:
        Mean depth of covered regions (depth > 0). Returns 0.0 if no coverage.
        
    Raises:
        FileNotFoundError: If BAM file doesn't exist.
        ValueError: If BAM file is invalid or corrupted.
    """
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")
    
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            total_depth = 0
            covered_positions = 0
            
            # Iterate through all references (chromosomes)
            for ref in bam.references:
                # Use pileup to calculate depth at each position
                for pileup_column in bam.pileup(ref, truncate=True, max_depth=100000):
                    depth = pileup_column.nsegments
                    if depth > 0:
                        total_depth += depth
                        covered_positions += 1
            
            if covered_positions == 0:
                return 0.0
            
            return total_depth / covered_positions
            
    except Exception as e:
        raise ValueError(f"Error processing BAM file {bam_path}: {str(e)}")


def calculate_pileup_statistics(pileup_path: Path) -> Dict[str, float]:
    """
    Calculate comprehensive statistics from a SNP pileup file.
    
    Args:
        pileup_path: Path to the pileup TSV file (can be gzipped).
        
    Returns:
        Dictionary containing the following statistics:
        - covered_snp_ratio: Ratio of SNPs with non-zero depth
        - covered_snp_mean_depth: Mean depth of covered SNPs
        - snp_gt_60x: Count of SNPs with depth > 60
        - snp_gt_100x: Count of SNPs with depth > 100
        - snp_gt_200x: Count of SNPs with depth > 200
        - snp_vaf_eq_0: Count of SNPs with VAF = 0
        - snp_vaf_0_to_0.2: Count of SNPs with 0 < VAF < 0.2
        - snp_vaf_0.2_to_0.8: Count of SNPs with 0.2 <= VAF <= 0.8
        - snp_vaf_0.8_to_1: Count of SNPs with 0.8 < VAF < 1
        - snp_vaf_eq_1: Count of SNPs with VAF = 1
        - informative_snp_count: Count of informative SNPs (0 < VAF < 0.2 or 0.8 < VAF < 1)
        
    Raises:
        FileNotFoundError: If pileup file doesn't exist.
        ValueError: If pileup file is missing required columns.
    """
    if not pileup_path.exists():
        raise FileNotFoundError(f"Pileup file not found: {pileup_path}")
    
    try:
        # Read pileup file (automatically handles gzip if .gz extension)
        df = pd.read_csv(pileup_path, sep='\t', compression='gzip' if str(pileup_path).endswith('.gz') else None)
        
        # Verify required columns exist
        required_columns = ['cfDNA_alt_reads', 'current_depth', 'fetal_current_depth_from_model', 'maternal_current_depth']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate VAF (handle division by zero)
        df['vaf'] = 0.0
        mask_nonzero_depth = df['current_depth'] > 0
        df.loc[mask_nonzero_depth, 'vaf'] = (
            df.loc[mask_nonzero_depth, 'cfDNA_alt_reads'] / df.loc[mask_nonzero_depth, 'current_depth']
        )
        
        total_snps = len(df)
        covered_snps = mask_nonzero_depth.sum()
        
        # Calculate statistics
        stats = {
            'covered_snp_ratio': covered_snps / total_snps if total_snps > 0 else 0.0,
            'covered_snp_mean_depth': df.loc[mask_nonzero_depth, 'current_depth'].mean() if covered_snps > 0 else 0.0,
            'fetal_covered_snp_mean_depth': df.loc[mask_nonzero_depth, 'fetal_current_depth_from_model'].mean() if covered_snps > 0 else 0.0,
            'maternal_covered_snp_mean_depth': df.loc[mask_nonzero_depth, 'maternal_current_depth'].mean() if covered_snps > 0 else 0.0,
            'snp_gt_60x': (df['current_depth'] > 60).sum(),
            'snp_gt_100x': (df['current_depth'] > 100).sum(),
            'snp_gt_200x': (df['current_depth'] > 200).sum(),
            'snp_vaf_eq_0': (df['vaf'] == 0.0).sum(),
            'snp_vaf_0_to_0.2': ((df['vaf'] > 0.0) & (df['vaf'] < 0.2)).sum(),
            'snp_vaf_0.2_to_0.8': ((df['vaf'] >= 0.2) & (df['vaf'] <= 0.8)).sum(),
            'snp_vaf_0.8_to_1': ((df['vaf'] > 0.8) & (df['vaf'] < 1.0)).sum(),
            'snp_vaf_eq_1': (df['vaf'] == 1.0).sum(),
        }
        
        # Calculate informative SNP count (0 < VAF < 0.2 or 0.8 < VAF < 1)
        stats['informative_snp_count'] = (
            (((df['vaf'] > 0.0) & (df['vaf'] < 0.2)) | ((df['vaf'] > 0.8) & (df['vaf'] < 1.0))) &
            (df['current_depth'] > 100)
        ).sum()
        
        return stats
        
    except Exception as e:
        raise ValueError(f"Error processing pileup file {pileup_path}: {str(e)}")


def extract_sample_name(filename: str, suffix: str) -> str:
    """
    Extract sample name from filename by removing suffix.
    
    Args:
        filename: The filename to process.
        suffix: The suffix to remove (e.g., "_target.bam").
        
    Returns:
        Sample name with suffix removed.
        
    Examples:
        >>> extract_sample_name("SAMPLE001_target.bam", "_target.bam")
        'SAMPLE001'
    """
    if filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename


def find_samples(input_dir: Path) -> List[str]:
    """
    Find all unique sample names from BAM and pileup files.
    
    Args:
        input_dir: Root directory containing subdirectories with data files.
        
    Returns:
        Sorted list of unique sample names found across all subdirectories.
        
    Raises:
        FileNotFoundError: If required subdirectories don't exist.
    """
    samples = set()
    
    # Check target BAM files
    target_dir = input_dir / "samtools_merge_target"
    if target_dir.exists():
        for bam_file in target_dir.glob("*_target.bam"):
            sample = extract_sample_name(bam_file.name, "_target.bam")
            samples.add(sample)
    
    # Check background BAM files
    background_dir = input_dir / "samtools_merge_background"
    if background_dir.exists():
        for bam_file in background_dir.glob("*_background.bam"):
            sample = extract_sample_name(bam_file.name, "_background.bam")
            samples.add(sample)
    
    # Check pileup files
    pileup_dir = input_dir / "merge_pileup_hard_filter"
    if pileup_dir.exists():
        for pileup_file in pileup_dir.glob("*_pileup.tsv.gz"):
            sample = extract_sample_name(pileup_file.name, "_pileup.tsv.gz")
            samples.add(sample)
    
    return sorted(list(samples))


def process_sample(
    sample: str,
    input_dir: Path,
    bam_depth_stat: bool = True,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None
) -> Dict[str, any]:
    """
    Process a single sample to calculate all statistics.
    
    Thread-safe function that can be called in parallel.
    
    Args:
        sample: Sample name to process.
        input_dir: Root directory containing subdirectories with data files.
        bam_depth_stat: Whether to calculate BAM depth statistics (default: True).
        progress: Rich Progress object for updating progress (optional).
        task_id: Task ID for progress tracking (optional).
        
    Returns:
        Dictionary containing all statistics for the sample.
        
    Raises:
        Exception: Any error during processing is caught and logged.
    """
    result = {'sample': sample}
    
    def update_progress(description: str):
        """Thread-safe progress update."""
        if progress is not None and task_id is not None:
            with progress_lock:
                progress.update(task_id, description=description)
    
    def print_warning(message: str):
        """Thread-safe console print."""
        with progress_lock:
            console.print(message)
    
    try:
        # Process target BAM (optional based on bam_depth_stat flag)
        if bam_depth_stat:
            target_bam = input_dir / "samtools_merge_target" / f"{sample}_target.bam"
            if target_bam.exists():
                update_progress(f"[cyan]Processing {sample} - target BAM")
                result['target_mean_depth'] = calculate_bam_mean_depth(target_bam)
            else:
                print_warning(f"[yellow]Warning: Target BAM not found for {sample}")
                result['target_mean_depth'] = float('nan')
            
            # Process background BAM
            background_bam = input_dir / "samtools_merge_background" / f"{sample}_background.bam"
            if background_bam.exists():
                update_progress(f"[cyan]Processing {sample} - background BAM")
                result['background_mean_depth'] = calculate_bam_mean_depth(background_bam)
            else:
                print_warning(f"[yellow]Warning: Background BAM not found for {sample}")
                result['background_mean_depth'] = float('nan')
        
        # Process pileup file
        pileup_file = input_dir / "merge_pileup_hard_filter" / f"{sample}_pileup.tsv.gz"
        if pileup_file.exists():
            update_progress(f"[cyan]Processing {sample} - pileup")
            pileup_stats = calculate_pileup_statistics(pileup_file)
            result.update(pileup_stats)
        else:
            print_warning(f"[yellow]Warning: Pileup file not found for {sample}")
            # Set all pileup stats to NaN
            pileup_keys = [
                'covered_snp_ratio', 'covered_snp_mean_depth', 
                'fetal_covered_snp_mean_depth', 'maternal_covered_snp_mean_depth',
                'snp_gt_60x', 'snp_gt_100x', 'snp_gt_200x',
                'snp_vaf_eq_0', 'snp_vaf_0_to_0.2', 'snp_vaf_0.2_to_0.8',
                'snp_vaf_0.8_to_1', 'snp_vaf_eq_1', 'informative_snp_count'
            ]
            for key in pileup_keys:
                result[key] = float('nan')
        
        update_progress(f"[green]✓ Completed {sample}")
        
    except Exception as e:
        print_warning(f"[red]Error processing sample {sample}: {str(e)}")
        # Fill with NaN values on error
        for key in ['target_mean_depth', 'background_mean_depth', 'covered_snp_ratio',
                    'covered_snp_mean_depth', 'fetal_covered_snp_mean_depth', 'maternal_covered_snp_mean_depth',
                    'snp_gt_60x', 'snp_gt_100x', 'snp_gt_200x',
                    'snp_vaf_eq_0', 'snp_vaf_0_to_0.2', 'snp_vaf_0.2_to_0.8',
                    'snp_vaf_0.8_to_1', 'snp_vaf_eq_1', 'informative_snp_count']:
            if key not in result:
                result[key] = float('nan')
    
    return result


def display_summary_table(df: pd.DataFrame) -> None:
    """
    Display a summary table of statistics using Rich.
    
    Args:
        df: DataFrame containing all sample statistics.
    """
    table = Table(title="SNP Pileup Statistics Summary", box=box.ROUNDED)
    
    table.add_column("Statistic", style="cyan", no_wrap=True)
    table.add_column("Mean", style="magenta")
    table.add_column("Median", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="red")
    
    # Select numeric columns for summary
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) > 0:
            table.add_row(
                col,
                f"{values.mean():.2f}",
                f"{values.median():.2f}",
                f"{values.min():.2f}",
                f"{values.max():.2f}"
            )
    
    console.print("\n")
    console.print(table)


@click.command()
@click.option(
    '--input_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Input directory containing samtools_merge_target, samtools_merge_background, and merge_pileup_hard_filter subdirectories.'
)
@click.option(
    '--output_prefix',
    type=str,
    required=True,
    help='Prefix for output TSV file. Output will be saved as {prefix}.tsv'
)
@click.option(
    '--ncpus',
    type=int,
    default=1,
    show_default=True,
    help='Number of CPU threads to use for parallel processing.'
)
@click.option(
    '--bam_depth_stat',
    type=bool,
    default=True,
    show_default=True,
    help='Whether to calculate BAM depth statistics (target_mean_depth and background_mean_depth). Set to False to skip BAM processing.'
)
def main(input_dir: Path, output_prefix: str, ncpus: int, bam_depth_stat: bool) -> None:
    """
    Calculate comprehensive statistics from BAM and SNP pileup files.
    
    This script processes BAM files to calculate mean depth coverage and pileup files
    to calculate SNP statistics including VAF distributions and informative SNP counts.
    Supports parallel processing with multiple CPU threads for improved performance.
    
    Args:
        input_dir: Directory containing subdirectories with BAM and pileup files.
        output_prefix: Prefix for output TSV file.
        ncpus: Number of CPU threads to use for parallel processing.
        bam_depth_stat: Whether to calculate BAM depth statistics (default: True).
        
    Examples:
        $ python snp_pileup_stat.py --input_dir /path/to/data --output_prefix results
        $ python snp_pileup_stat.py --input_dir /path/to/data --output_prefix results --ncpus 8
        $ python snp_pileup_stat.py --input_dir /path/to/data --output_prefix results --bam_depth_stat False
    """
    console.rule("[bold blue]SNP Pileup Statistics Calculator")
    console.print(f"[cyan]Input directory: {input_dir}")
    console.print(f"[cyan]Output prefix: {output_prefix}")
    console.print(f"[cyan]CPU threads: {ncpus}")
    console.print(f"[cyan]BAM depth statistics: {bam_depth_stat}")
    
    # Validate input directory structure
    required_dirs = [
        "samtools_merge_target",
        "samtools_merge_background",
        "merge_pileup_hard_filter"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = input_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        console.print(f"[yellow]Warning: Missing subdirectories: {', '.join(missing_dirs)}")
        console.print("[yellow]Processing will continue with available data.")
    
    # Find all samples
    console.print("\n[bold]Finding samples...")
    samples = find_samples(input_dir)
    
    if not samples:
        console.print("[red]Error: No samples found in input directory!")
        sys.exit(1)
    
    console.print(f"[green]Found {len(samples)} samples")
    
    # Validate ncpus value
    if ncpus < 1:
        console.print("[red]Error: --ncpus must be at least 1")
        sys.exit(1)
    
    # Adjust ncpus if it exceeds the number of samples
    effective_ncpus = min(ncpus, len(samples))
    if effective_ncpus < ncpus:
        console.print(f"[yellow]Note: Using {effective_ncpus} threads (limited by number of samples)")
    
    # Process all samples with multi-threading
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing samples...", total=len(samples))
        
        if ncpus == 1:
            # Single-threaded processing for ncpus=1
            for sample in samples:
                result = process_sample(sample, input_dir, bam_depth_stat, progress, task)
                results.append(result)
                progress.advance(task)
        else:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=effective_ncpus) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(process_sample, sample, input_dir, bam_depth_stat, progress, task): sample
                    for sample in samples
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        with progress_lock:
                            console.print(f"[red]Failed to process sample {sample}: {str(e)}")
                        # Add a failed result with NaN values
                        failed_result = {'sample': sample}
                        for key in ['target_mean_depth', 'background_mean_depth', 'covered_snp_ratio',
                                    'covered_snp_mean_depth', 'fetal_covered_snp_mean_depth', 'maternal_covered_snp_mean_depth',
                                    'snp_gt_60x', 'snp_gt_100x', 'snp_gt_200x',
                                    'snp_vaf_eq_0', 'snp_vaf_0_to_0.2', 'snp_vaf_0.2_to_0.8',
                                    'snp_vaf_0.8_to_1', 'snp_vaf_eq_1', 'informative_snp_count']:
                            failed_result[key] = float('nan')
                        results.append(failed_result)
                    finally:
                        progress.advance(task)
    
    # Create DataFrame with results
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = ['sample']
    
    # Add BAM depth columns only if bam_depth_stat is True
    if bam_depth_stat:
        column_order.extend(['target_mean_depth', 'background_mean_depth'])
    
    # Add remaining columns
    column_order.extend([
        'covered_snp_ratio',
        'covered_snp_mean_depth',
        'fetal_covered_snp_mean_depth',
        'maternal_covered_snp_mean_depth',
        'snp_gt_60x',
        'snp_gt_100x',
        'snp_gt_200x',
        'snp_vaf_eq_0',
        'snp_vaf_0_to_0.2',
        'snp_vaf_0.2_to_0.8',
        'snp_vaf_0.8_to_1',
        'snp_vaf_eq_1',
        'informative_snp_count'
    ])
    
    # Only include columns that exist in the dataframe
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Save to TSV
    output_file = f"{output_prefix}.tsv"
    df.to_csv(output_file, sep='\t', index=False, float_format='%.4f')
    
    console.print(f"\n[green]✓ Results saved to: {output_file}")
    console.print(f"[green]✓ Processed {len(results)} samples successfully")
    
    # Display summary statistics
    display_summary_table(df)
    
    console.rule("[bold green]Processing Complete")


if __name__ == '__main__':
    main()

