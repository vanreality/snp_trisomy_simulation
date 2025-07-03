#!/usr/bin/env python3
"""
VCF to Pileup Converter

This script processes VCF files and potential SNPs files to extract and merge
allele frequency and read count information for downstream analysis.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import Optional, Union

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Initialize rich console
console = Console()

def extract_AF(x: pd.Series) -> float:
    """Parse the AF (allele frequency) field out of the INFO column.
    
    Args:
        x: A pandas Series containing the row data with 'info' column.
        
    Returns:
        The allele frequency as a float value.
        
    Raises:
        ValueError: If AF field is not found or cannot be parsed.
        IndexError: If INFO column format is unexpected.
    """
    try:
        if pd.isna(x['info']) or not isinstance(x['info'], str):
            raise ValueError("INFO column is missing or not a string")
        
        if 'AF=' not in x['info']:
            raise ValueError("AF field not found in INFO column")
            
        AF = x['info'].split('AF=')[1].split(';')[0]
        return float(AF)
    except (ValueError, IndexError) as e:
        console.print(f"[red]Error parsing AF from INFO: {x['info']} - {e}[/red]")
        return np.nan

def extract_ref_read(x: pd.Series) -> int:
    """Extract reference read count from sample columns based on format.
    
    Args:
        x: A pandas Series containing the row data with format and sample columns.
        
    Returns:
        The reference read count as an integer.
        
    Raises:
        ValueError: If format is unrecognized or data cannot be parsed.
        IndexError: If expected fields are missing.
    """
    try:
        if pd.isna(x['format']) or not isinstance(x['format'], str):
            raise ValueError("FORMAT column is missing or not a string")
            
        if x['format'] == 'GT:GQ:DP:AD:VAF:PL':
            if pd.isna(x['gatk']) or not isinstance(x['gatk'], str):
                raise ValueError("GATK sample data is missing")
            ref_read = int(x['gatk'].split(':')[3].split(',')[0])
        else:
            if pd.isna(x['deepvariant']) or not isinstance(x['deepvariant'], str):
                raise ValueError("DeepVariant sample data is missing")
            ref_read = int(x['deepvariant'].split(':')[1].split(',')[0])
        return ref_read
    except (ValueError, IndexError) as e:
        console.print(f"[red]Error extracting reference reads: {e}[/red]")
        return 0

def extract_alt_read(x: pd.Series) -> int:
    """Extract alternate read count from sample columns based on format.
    
    Args:
        x: A pandas Series containing the row data with format and sample columns.
        
    Returns:
        The alternate read count as an integer.
        
    Raises:
        ValueError: If format is unrecognized or data cannot be parsed.
        IndexError: If expected fields are missing.
    """
    try:
        if pd.isna(x['format']) or not isinstance(x['format'], str):
            raise ValueError("FORMAT column is missing or not a string")
            
        if x['format'] == 'GT:GQ:DP:AD:VAF:PL':
            if pd.isna(x['gatk']) or not isinstance(x['gatk'], str):
                raise ValueError("GATK sample data is missing")
            alt_read = int(x['gatk'].split(':')[3].split(',')[1])
        else:
            if pd.isna(x['deepvariant']) or not isinstance(x['deepvariant'], str):
                raise ValueError("DeepVariant sample data is missing")
            alt_read = int(x['deepvariant'].split(':')[1].split(',')[1])
        return alt_read
    except (ValueError, IndexError) as e:
        console.print(f"[red]Error extracting alternate reads: {e}[/red]")
        return 0

def read_potential_snps(potential_snps_file: Union[str, Path]) -> pd.DataFrame:
    """Read and process potential SNPs file.
    
    Args:
        potential_snps_file: Path to the potential SNPs file (tab-separated).
        
    Returns:
        A pandas DataFrame containing SNP information with allele frequencies.
        
    Raises:
        FileNotFoundError: If the input file doesn't exist.
        pd.errors.EmptyDataError: If the file is empty.
        Exception: For other file reading errors.
    """
    potential_snps_file = Path(potential_snps_file)
    
    if not potential_snps_file.exists():
        raise FileNotFoundError(f"Potential SNPs file not found: {potential_snps_file}")
    
    if potential_snps_file.stat().st_size == 0:
        raise pd.errors.EmptyDataError("Potential SNPs file is empty")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Reading potential SNPs file...", total=None)
            
            df = pd.read_csv(
                potential_snps_file,
                sep='\t',
                names=['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info']
            )
            
            if df.empty:
                raise pd.errors.EmptyDataError("No data found in potential SNPs file")
            
            progress.update(task, description="Extracting allele frequencies...")
            df['af'] = df.apply(extract_AF, axis=1)
            
            # Remove rows where AF extraction failed
            initial_count = len(df)
            df = df.dropna(subset=['af'])
            final_count = len(df)
            
            if final_count < initial_count:
                console.print(f"[yellow]Warning: Dropped {initial_count - final_count} rows due to AF parsing errors[/yellow]")
        
        console.print(f"[green]Successfully loaded {len(df)} potential SNPs[/green]")
        return df
        
    except Exception as e:
        console.print(f"[red]Error reading potential SNPs file: {e}[/red]")
        raise

def read_vcf(vcf_file: Union[str, Path], potential_snps_df: pd.DataFrame) -> pd.DataFrame:
    """Read VCF file and merge with potential SNPs data.
    
    Args:
        vcf_file: Path to the VCF file.
        potential_snps_df: DataFrame containing potential SNPs information.
        
    Returns:
        A merged DataFrame containing SNP and read count information.
        
    Raises:
        FileNotFoundError: If the VCF file doesn't exist.
        pd.errors.EmptyDataError: If the file is empty.
        Exception: For other file reading errors.
    """
    vcf_file = Path(vcf_file)
    
    if not vcf_file.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_file}")
    
    if vcf_file.stat().st_size == 0:
        raise pd.errors.EmptyDataError("VCF file is empty")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            read_task = progress.add_task("Reading VCF file...", total=None)
            
            vcf_df = pd.read_csv(
                vcf_file,
                sep='\t',
                comment='#',
                header=None,
                names=['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info', 'format', 'gatk', 'deepvariant']
            )
            
            if vcf_df.empty:
                raise pd.errors.EmptyDataError("No data found in VCF file")
            
            progress.update(read_task, description=f"Processing {len(vcf_df)} VCF records...")
            
            # Filter for PASS variants
            initial_count = len(vcf_df)
            vcf_df = vcf_df[vcf_df['filter'] == 'PASS']
            pass_count = len(vcf_df)
            
            if pass_count == 0:
                console.print("[yellow]Warning: No PASS variants found in VCF file[/yellow]")
                return pd.DataFrame()
            
            console.print(f"[blue]Filtered to {pass_count} PASS variants from {initial_count} total[/blue]")
            
            progress.update(read_task, description="Extracting read counts...")
            
            # Extract read counts with error handling
            vcf_df['cfDNA_ref_reads'] = vcf_df.apply(extract_ref_read, axis=1)
            vcf_df['cfDNA_alt_reads'] = vcf_df.apply(extract_alt_read, axis=1)
            vcf_df['current_depth'] = vcf_df['cfDNA_ref_reads'] + vcf_df['cfDNA_alt_reads']
            
            progress.update(read_task, description="Merging with potential SNPs...")
            
            # Merge with potential SNPs
            output_df = pd.merge(potential_snps_df, vcf_df, on=['chr', 'pos'], how='left')
            output_df = output_df[['chr', 'pos', 'ref', 'alt', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth']]
            
            # Calculate coverage statistics
            covered_snps = output_df.dropna(subset=['current_depth'])
            coverage_rate = len(covered_snps) / len(output_df) * 100 if len(output_df) > 0 else 0
            
        console.print(f"[green]Successfully processed VCF file[/green]")
        console.print(f"[blue]Coverage: {len(covered_snps)}/{len(output_df)} SNPs ({coverage_rate:.1f}%)[/blue]")
        
        return output_df
        
    except Exception as e:
        console.print(f"[red]Error reading VCF file: {e}[/red]")
        raise

def validate_inputs(potential_snps_file: Union[str, Path], vcf_file: Union[str, Path]) -> None:
    """Validate input file paths and accessibility.
    
    Args:
        potential_snps_file: Path to the potential SNPs file.
        vcf_file: Path to the VCF file.
        
    Raises:
        click.ClickException: If validation fails.
    """
    potential_snps_path = Path(potential_snps_file)
    vcf_path = Path(vcf_file)
    
    # Check file existence
    if not potential_snps_path.exists():
        raise click.ClickException(f"Potential SNPs file not found: {potential_snps_path}")
    
    if not vcf_path.exists():
        raise click.ClickException(f"VCF file not found: {vcf_path}")
    
    # Check file readability
    if not os.access(potential_snps_path, os.R_OK):
        raise click.ClickException(f"Cannot read potential SNPs file: {potential_snps_path}")
    
    if not os.access(vcf_path, os.R_OK):
        raise click.ClickException(f"Cannot read VCF file: {vcf_path}")
    
    console.print("[green]✓ Input file validation passed[/green]")

def display_summary(output_df: pd.DataFrame) -> None:
    """Display a summary table of the processing results.
    
    Args:
        output_df: The final merged DataFrame to summarize.
    """
    if output_df.empty:
        console.print("[yellow]No data to summarize[/yellow]")
        return
    
    # Calculate statistics
    total_snps = len(output_df)
    covered_snps = len(output_df.dropna(subset=['current_depth']))
    uncovered_snps = total_snps - covered_snps
    
    if covered_snps > 0:
        avg_depth = output_df['current_depth'].mean()
        median_depth = output_df['current_depth'].median()
        max_depth = output_df['current_depth'].max()
        min_depth = output_df['current_depth'].min()
    else:
        avg_depth = median_depth = max_depth = min_depth = 0
    
    # Create summary table
    table = Table(title="Processing Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    table.add_row("Total SNPs", str(total_snps))
    table.add_row("Covered SNPs", str(covered_snps))
    table.add_row("Uncovered SNPs", str(uncovered_snps))
    table.add_row("Coverage Rate", f"{covered_snps/total_snps*100:.1f}%" if total_snps > 0 else "0%")
    
    if covered_snps > 0:
        table.add_row("Average Depth", f"{avg_depth:.1f}")
        table.add_row("Median Depth", f"{median_depth:.1f}")
        table.add_row("Min Depth", f"{min_depth}")
        table.add_row("Max Depth", f"{max_depth}")
    
    console.print(table)

@click.command()
@click.option(
    '--potential-snps-file',
    required=True,
    type=click.Path(exists=True, readable=True, path_type=Path),
    help='Path to the potential SNPs file (tab-separated format).'
)
@click.option(
    '--vcf-file',
    required=True,
    type=click.Path(exists=True, readable=True, path_type=Path),
    help='Path to the VCF file to process.'
)
@click.option(
    '--output-file',
    type=click.Path(path_type=Path),
    help='Output file path for the merged results (optional, defaults to stdout).'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose output.'
)
def main(
    potential_snps_file: Path,
    vcf_file: Path,
    output_file: Optional[Path],
    verbose: bool
) -> None:
    """
    Convert VCF file to pileup format by merging with potential SNPs.
    
    This tool processes VCF files and potential SNPs files to extract and merge
    allele frequency and read count information for downstream analysis.
    
    Args:
        potential_snps_file: Path to the potential SNPs file.
        vcf_file: Path to the VCF file.
        output_file: Optional output file path.
        verbose: Enable verbose output.
    """
    # Display header
    console.print(Panel.fit(
        "[bold blue]VCF to Pileup Converter[/bold blue]\n"
        "Processing VCF files and potential SNPs for analysis",
        border_style="blue"
    ))
    
    try:
        # Validate inputs
        if verbose:
            console.print("[blue]Validating input files...[/blue]")
        validate_inputs(potential_snps_file, vcf_file)
        
        # Read potential SNPs file
        if verbose:
            console.print(f"[blue]Reading potential SNPs from: {potential_snps_file}[/blue]")
        potential_snps_df = read_potential_snps(potential_snps_file)
        
        # Read and process VCF file
        if verbose:
            console.print(f"[blue]Processing VCF file: {vcf_file}[/blue]")
        output_df = read_vcf(vcf_file, potential_snps_df)
        
        # Display summary
        if verbose:
            display_summary(output_df)
        
        # Output results
        if output_file:
            try:
                output_df.to_csv(output_file, sep='\t', index=False, na_rep='NA', compression='gzip')
                console.print(f"[green]Results saved to: {output_file}[/green]")
            except Exception as e:
                console.print(f"[red]Error saving to file: {e}[/red]")
                raise click.ClickException(f"Failed to save output file: {e}")
        else:
            # Output to stdout
            output_df.to_csv(sys.stdout, sep='\t', index=False, na_rep='NA')
        
        console.print("[green]✓ Processing completed successfully[/green]")
        
    except click.ClickException:
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise click.ClickException(f"Processing failed: {e}")

if __name__ == '__main__':
    main()


    
