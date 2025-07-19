#!/usr/bin/env python3
"""
BAM to Pileup Converter

This script processes VCF files and known variant sites to generate pileup data
for cfDNA analysis. It extracts allele depths from VCF files and merges them
with known variant information to create comprehensive pileup statistics.
"""

import sys
import pandas as pd
import re
from pathlib import Path
from typing import Optional, Tuple

import click
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich import print as rprint

# Initialize rich console for output formatting
console = Console()


def extract_allele_frequency(info_field: str) -> Optional[float]:
    """
    Extract the AF (Allele Frequency) value from a VCF info field string.

    Args:
        info_field (str): The info field string from the variant record containing
                         semicolon-separated key=value pairs.

    Returns:
        Optional[float]: The extracted AF value, or None if not found or conversion fails.

    Examples:
        >>> extract_allele_frequency("AC=2;AF=0.00040;AN=5008")
        0.0004
        >>> extract_allele_frequency("AC=2;AN=5008")
        None
    """
    af_pattern = r"AF=([^;]+)"
    af_match = re.search(af_pattern, info_field)
    if af_match:
        try:
            return float(af_match.group(1))
        except ValueError:
            console.print(f"[yellow]Warning: Could not convert AF value '{af_match.group(1)}' to float[/yellow]")
            return None
    return None


def extract_reference_alternate_counts(variant_row: pd.Series) -> pd.Series:
    """
    Extract reference and alternate allele counts from a merged variant record.
    
    Handles special base conversion cases for bisulfite sequencing:
    - C->T conversions: T depths are added to reference C counts
    - G->A conversions: A depths are added to reference G counts

    Args:
        variant_row (pd.Series): A row containing:
            - 'reference_allele': reference base (single character)
            - 'alternate_alleles_vcf': comma-separated list of alternate alleles from VCF
            - 'alternate_allele_target': the target alternate allele to extract
            - 'allele_depths': comma-separated depths (first is ref, then one per alt)

    Returns:
        pd.Series: Contains 'reference_count' and 'alternate_count' fields.

    Examples:
        For a C>A variant with depths "59,34,1,0" and alternates "A,C,<*>":
        - reference_count = 59 (base C count)
        - alternate_count = 34 (A count)
    """
    # Parse comma-separated alternate alleles and depths
    alternate_alleles_list = variant_row['alternate_alleles_vcf'].split(',')  # e.g. ['A','C','<*>']
    depth_values = [int(depth) for depth in variant_row['allele_depths'].split(',')]  # e.g. [59,34,1,0]

    # Initialize reference count with the first depth value (reference allele)
    reference_count = depth_values[0]

    # Handle bisulfite sequencing conversion artifacts:
    # Add T depths to C reference counts (C->T conversion)
    if variant_row['reference_allele'] == 'C' and 'T' in alternate_alleles_list:
        t_allele_index = alternate_alleles_list.index('T') + 1  # +1 because first depth is ref
        reference_count += depth_values[t_allele_index]
    
    # Add A depths to G reference counts (G->A conversion)
    if variant_row['reference_allele'] == 'G' and 'A' in alternate_alleles_list:
        a_allele_index = alternate_alleles_list.index('A') + 1  # +1 because first depth is ref
        reference_count += depth_values[a_allele_index]

    # Extract depth for the target alternate allele
    alternate_count = 0
    if variant_row['alternate_allele_target'] in alternate_alleles_list:
        target_allele_index = alternate_alleles_list.index(variant_row['alternate_allele_target']) + 1
        alternate_count = depth_values[target_allele_index]

    return pd.Series({
        'reference_count': reference_count,
        'alternate_count': alternate_count
    })


def load_vcf_data(vcf_file_path: Path, progress: Progress, task_id: TaskID) -> pd.DataFrame:
    """
    Load and process VCF file data.

    Args:
        vcf_file_path (Path): Path to the input VCF file (can be gzipped).
        progress (Progress): Rich progress bar instance.
        task_id (TaskID): Task ID for progress tracking.

    Returns:
        pd.DataFrame: Processed VCF data with extracted allele depths.

    Raises:
        FileNotFoundError: If the VCF file doesn't exist.
        pd.errors.EmptyDataError: If the VCF file is empty.
    """
    if not vcf_file_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_file_path}")
    
    progress.update(task_id, description="Loading VCF file...")
    
    try:
        # Read VCF file, skipping header lines and selecting relevant columns
        vcf_data = pd.read_csv(
            vcf_file_path, 
            sep='\t', 
            compression='gzip' if vcf_file_path.suffix == '.gz' else None,
            comment='#', 
            usecols=[0, 1, 3, 4, 9], 
            names=['chromosome', 'position', 'reference_allele', 'alternate_alleles_vcf', 'sample_info']
        )
        
        if vcf_data.empty:
            raise pd.errors.EmptyDataError("VCF file contains no data")
        
        # Extract allele depths (AD) from the sample info field (last colon-separated value)
        vcf_data['allele_depths'] = vcf_data['sample_info'].apply(lambda x: x.split(':')[-1])
        
        progress.update(task_id, advance=50)
        console.print(f"[green]✓[/green] Loaded {len(vcf_data):,} variants from VCF file")
        
        return vcf_data
        
    except Exception as e:
        console.print(f"[red]Error loading VCF file: {e}[/red]")
        raise


def load_known_sites_data(known_sites_file_path: Path, progress: Progress, task_id: TaskID) -> pd.DataFrame:
    """
    Load and process known variant sites data.

    Args:
        known_sites_file_path (Path): Path to the known sites TSV file.
        progress (Progress): Rich progress bar instance.
        task_id (TaskID): Task ID for progress tracking.

    Returns:
        pd.DataFrame: Processed known sites data with extracted allele frequencies.

    Raises:
        FileNotFoundError: If the known sites file doesn't exist.
        pd.errors.EmptyDataError: If the known sites file is empty.
    """
    if not known_sites_file_path.exists():
        raise FileNotFoundError(f"Known sites file not found: {known_sites_file_path}")
    
    progress.update(task_id, description="Loading known sites file...")
    
    try:
        # Read known sites file, selecting relevant columns
        known_sites_data = pd.read_csv(
            known_sites_file_path, 
            sep='\t', 
            usecols=[0, 1, 3, 4, 7], 
            names=['chromosome', 'position', 'reference_allele', 'alternate_allele_target', 'info_field'], 
            comment='#'
        )
        
        if known_sites_data.empty:
            raise pd.errors.EmptyDataError("Known sites file contains no data")
        
        # Extract allele frequencies from info field
        known_sites_data['allele_frequency'] = known_sites_data['info_field'].apply(extract_allele_frequency)
        
        progress.update(task_id, advance=50)
        console.print(f"[green]✓[/green] Loaded {len(known_sites_data):,} known variant sites")
        
        return known_sites_data
        
    except Exception as e:
        console.print(f"[red]Error loading known sites file: {e}[/red]")
        raise


def merge_variant_data(vcf_data: pd.DataFrame, known_sites_data: pd.DataFrame, 
                      progress: Progress, task_id: TaskID) -> pd.DataFrame:
    """
    Merge VCF data with known sites data on chromosome, position, and reference allele.

    Args:
        vcf_data (pd.DataFrame): Processed VCF data.
        known_sites_data (pd.DataFrame): Processed known sites data.
        progress (Progress): Rich progress bar instance.
        task_id (TaskID): Task ID for progress tracking.

    Returns:
        pd.DataFrame: Merged dataset ready for pileup analysis.

    Raises:
        ValueError: If no variants remain after merging.
    """
    progress.update(task_id, description="Merging VCF and known sites data...")
    
    # Merge on chromosome, position, and reference allele
    merged_variants = pd.merge(
        vcf_data, 
        known_sites_data, 
        on=['chromosome', 'position', 'reference_allele']
    )
    
    if merged_variants.empty:
        raise ValueError("No overlapping variants found between VCF and known sites files")
    
    progress.update(task_id, advance=30)
    console.print(f"[green]✓[/green] Merged data: {len(merged_variants):,} overlapping variants")
    
    return merged_variants


def process_pileup_data(merged_variants: pd.DataFrame, progress: Progress, task_id: TaskID) -> pd.DataFrame:
    """
    Process merged variant data to generate pileup statistics.

    Args:
        merged_variants (pd.DataFrame): Merged VCF and known sites data.
        progress (Progress): Rich progress bar instance.
        task_id (TaskID): Task ID for progress tracking.

    Returns:
        pd.DataFrame: Final pileup data with reference/alternate counts and depth.
    """
    progress.update(task_id, description="Processing pileup data...")
    
    # Apply reference/alternate count extraction to each variant
    allele_counts = merged_variants.apply(extract_reference_alternate_counts, axis=1)
    
    # Combine position information with count data
    pileup_result = pd.concat([
        merged_variants[['chromosome', 'position', 'reference_allele', 'alternate_allele_target']].rename(
            columns={'alternate_allele_target': 'alternate_allele'}
        ),
        allele_counts
    ], axis=1)
    
    # Rename columns to match expected output format
    pileup_result = pileup_result.rename(columns={
        'reference_count': 'cfDNA_ref_reads',
        'alternate_count': 'cfDNA_alt_reads'
    })
    
    # Calculate total sequencing depth
    pileup_result['current_depth'] = pileup_result['cfDNA_ref_reads'] + pileup_result['cfDNA_alt_reads']
    
    progress.update(task_id, advance=70)
    console.print(f"[green]✓[/green] Generated pileup data for {len(pileup_result):,} variants")
    
    return pileup_result


def save_pileup_output(pileup_data: pd.DataFrame, output_prefix: str, 
                      progress: Progress, task_id: TaskID) -> Path:
    """
    Save pileup data to a compressed TSV file.

    Args:
        pileup_data (pd.DataFrame): Processed pileup data.
        output_prefix (str): Output file prefix.
        progress (Progress): Rich progress bar instance.
        task_id (TaskID): Task ID for progress tracking.

    Returns:
        Path: Path to the saved output file.
    """
    output_file_path = Path(f"{output_prefix}_pileup.tsv.gz")
    
    progress.update(task_id, description="Saving pileup data...")
    
    try:
        pileup_data.to_csv(output_file_path, sep='\t', compression='gzip', index=False)
        progress.update(task_id, advance=100)
        console.print(f"[green]✓[/green] Pileup data saved to: {output_file_path}")
        return output_file_path
        
    except Exception as e:
        console.print(f"[red]Error saving output file: {e}[/red]")
        raise


@click.command()
@click.option(
    '--input-vcf',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to input VCF file (can be gzipped)'
)
@click.option(
    '--known-sites',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to known variant sites TSV file'
)
@click.option(
    '--output',
    required=True,
    type=str,
    help='Output file prefix (will create {prefix}_pileup.tsv.gz)'
)
def main(input_vcf: Path, known_sites: Path, output: str) -> None:
    """
    Convert VCF files to pileup format for cfDNA analysis.
    
    This tool processes VCF files containing variant calls and merges them with
    known variant sites to generate comprehensive pileup statistics suitable for
    cell-free DNA analysis workflows.
    
    The output includes reference and alternate allele read counts, with special
    handling for bisulfite sequencing artifacts (C->T and G->A conversions).
    """
    console.print("\n[bold blue]BAM to Pileup Converter[/bold blue]")
    console.print("="*50)
    
    # Display input parameters
    params_table = Table(title="Input Parameters", show_header=True, header_style="bold magenta")
    params_table.add_column("Parameter", style="cyan", no_wrap=True)
    params_table.add_column("Value", style="white")
    
    params_table.add_row("Input VCF", str(input_vcf))
    params_table.add_row("Known Sites", str(known_sites))
    params_table.add_row("Output Prefix", output)
    
    console.print(params_table)
    console.print()
    
    try:
        with Progress(console=console) as progress:
            # Create progress tasks
            vcf_task = progress.add_task("Loading VCF data...", total=100)
            sites_task = progress.add_task("Loading known sites...", total=100)
            merge_task = progress.add_task("Merging datasets...", total=100)
            process_task = progress.add_task("Processing pileup...", total=100)
            save_task = progress.add_task("Saving output...", total=100)
            
            # Load input files
            vcf_data = load_vcf_data(input_vcf, progress, vcf_task)
            known_sites_data = load_known_sites_data(known_sites, progress, sites_task)
            
            # Merge datasets
            merged_variants = merge_variant_data(vcf_data, known_sites_data, progress, merge_task)
            
            # Process pileup data
            pileup_result = process_pileup_data(merged_variants, progress, process_task)
            
            # Save output
            output_file = save_pileup_output(pileup_result, output, progress, save_task)
            
        # Display summary statistics
        summary_table = Table(title="Processing Summary", show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Count", style="white", justify="right")
        
        summary_table.add_row("Input VCF variants", f"{len(vcf_data):,}")
        summary_table.add_row("Known variant sites", f"{len(known_sites_data):,}")
        summary_table.add_row("Overlapping variants", f"{len(merged_variants):,}")
        summary_table.add_row("Final pileup entries", f"{len(pileup_result):,}")
        summary_table.add_row("Mean depth", f"{pileup_result['current_depth'].mean():.1f}")
        
        console.print(summary_table)
        console.print(f"\n[bold green]✓ Processing completed successfully![/bold green]")
        console.print(f"Output file: [cyan]{output_file}[/cyan]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during processing:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()