#!/usr/bin/env python3
"""
Filter pileup files based on Variant Allele Frequency (VAF) thresholds.

This script processes a single pileup file, computes various VAF metrics,
and filters variants based on specified VAF conditions.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Initialize rich console for formatted output
console = Console()


def validate_input_file(input_path: Path) -> None:
    """
    Validate that the input file exists and has the correct format.

    Args:
        input_path: Path to the input pileup file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input file has an invalid extension.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check for valid extensions (gzipped or plain TSV)
    valid_extensions = {'.tsv.gz', '.tsv'}
    if not any(str(input_path).endswith(ext) for ext in valid_extensions):
        raise ValueError(
            f"Invalid file extension. Expected .tsv or .tsv.gz, got: {input_path.suffix}"
        )


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains all required columns.

    Only cfDNA columns are strictly required. Fetal and maternal columns
    are optional and will be processed if present.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    # Only cfDNA columns are required
    required_columns = [
        'cfDNA_alt_reads',
        'current_depth'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )


def compute_vaf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variant Allele Frequency (VAF) for different categories.

    This function calculates:
    - raw_vaf: VAF from cfDNA reads (always computed)
    - target_vaf: VAF from fetal model reads (computed if columns exist)
    - background_vaf: VAF from maternal reads (computed if columns exist)

    Args:
        df: Input DataFrame containing read count columns.

    Returns:
        DataFrame with added VAF columns.

    Note:
        Division by zero is handled by replacing inf/nan values with 0.
        Fetal and maternal VAF are only computed if the corresponding columns exist.
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Compute VAF with division by zero handling
    with pd.option_context('mode.use_inf_as_na', True):
        # raw_vaf: cfDNA alternative reads / current depth (always required)
        result_df['raw_vaf'] = result_df['cfDNA_alt_reads'] / result_df['current_depth']
        result_df['raw_vaf'] = result_df['raw_vaf'].fillna(0)
        
        # target_vaf: fetal alternative reads / fetal depth (optional)
        if 'fetal_alt_reads_from_model' in result_df.columns and \
           'fetal_current_depth_from_model' in result_df.columns:
            result_df['target_vaf'] = (
                result_df['fetal_alt_reads_from_model'] / 
                result_df['fetal_current_depth_from_model']
            )
            result_df['target_vaf'] = result_df['target_vaf'].fillna(0)
        
        # background_vaf: maternal alternative reads / maternal depth (optional)
        if 'maternal_alt_reads' in result_df.columns and \
           'maternal_current_depth' in result_df.columns:
            result_df['background_vaf'] = (
                result_df['maternal_alt_reads'] / 
                result_df['maternal_current_depth']
            )
            result_df['background_vaf'] = result_df['background_vaf'].fillna(0)
    
    return result_df


def filter_by_vaf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter variants based on VAF thresholds.

    Keeps variants where raw_vaf is either:
    - Between 0 and 0.2 (low frequency variants)
    - Between 0.8 and 1.0 (high frequency variants)

    Args:
        df: Input DataFrame with VAF columns.

    Returns:
        Filtered DataFrame containing only variants meeting the VAF criteria.
    """
    # Filter VAF: keep variants with raw_vaf in (0, 0.2) or (0.8, 1.0)
    condition_1 = ((df['raw_vaf'] > 0) & (df['raw_vaf'] < 0.2))
    condition_2 = ((df['raw_vaf'] > 0.8) & (df['raw_vaf'] < 1))
    
    filtered_df = df[condition_1 | condition_2].copy()
    
    return filtered_df


def process_pileup_file(input_path: Path) -> pd.DataFrame:
    """
    Process a single pileup file: load, compute VAF, and filter.

    Args:
        input_path: Path to the input pileup file.

    Returns:
        Filtered DataFrame with VAF metrics.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file format or required columns are invalid.
        Exception: For any other processing errors.
    """
    # Validate input file
    validate_input_file(input_path)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Read the pileup file
        task1 = progress.add_task("[cyan]Reading pileup file...", total=1)
        
        try:
            # Auto-detect compression based on file extension
            compression = 'gzip' if str(input_path).endswith('.gz') else None
            df = pd.read_csv(input_path, sep='\t', compression=compression)
            progress.update(task1, completed=1)
            
            if df.empty:
                console.print("[yellow]Warning: Input file is empty[/yellow]")
                return pd.DataFrame()
            
            console.print(f"[green]✓[/green] Loaded {len(df):,} variants")
            
        except pd.errors.EmptyDataError:
            console.print("[yellow]Warning: Input file contains no data[/yellow]")
            return pd.DataFrame()
        except Exception as e:
            raise Exception(f"Error reading pileup file: {str(e)}")
        
        # Validate required columns
        task2 = progress.add_task("[cyan]Validating columns...", total=1)
        validate_required_columns(df)
        progress.update(task2, completed=1)
        console.print("[green]✓[/green] Required columns present")
        
        # Check for optional columns
        has_fetal = 'fetal_alt_reads_from_model' in df.columns and \
                    'fetal_current_depth_from_model' in df.columns
        has_maternal = 'maternal_alt_reads' in df.columns and \
                       'maternal_current_depth' in df.columns
        
        if has_fetal:
            console.print("[green]✓[/green] Fetal columns detected")
        if has_maternal:
            console.print("[green]✓[/green] Maternal columns detected")
        
        # Compute VAF
        task3 = progress.add_task("[cyan]Computing VAF metrics...", total=1)
        df = compute_vaf(df)
        progress.update(task3, completed=1)
        
        # Report which VAF metrics were computed
        vaf_computed = ["raw_vaf"]
        if has_fetal:
            vaf_computed.append("target_vaf")
        if has_maternal:
            vaf_computed.append("background_vaf")
        console.print(f"[green]✓[/green] VAF metrics computed: {', '.join(vaf_computed)}")
        
        # Filter by VAF
        task4 = progress.add_task("[cyan]Filtering variants...", total=1)
        filtered_df = filter_by_vaf(df)
        progress.update(task4, completed=1)
        console.print(
            f"[green]✓[/green] Filtered to {len(filtered_df):,} variants "
            f"({len(filtered_df)/len(df)*100:.2f}% retained)"
        )
    
    return filtered_df


def get_output_filename(input_path: Path) -> str:
    """
    Generate output filename based on input filename.

    Extracts the sample name from the input filename and creates
    an output filename with '_filtered_pileup.tsv' suffix.

    Args:
        input_path: Path to the input file.

    Returns:
        Output filename string.

    Examples:
        >>> get_output_filename(Path("sample123_pileup.tsv.gz"))
        'sample123_filtered_pileup.tsv'
        >>> get_output_filename(Path("test_pileup.tsv"))
        'test_filtered_pileup.tsv'
    """
    # Remove .tsv.gz or .tsv extension
    filename = input_path.name
    
    # Handle both .tsv.gz and .tsv extensions
    if filename.endswith('.tsv.gz'):
        base_name = filename[:-7]  # Remove .tsv.gz
    elif filename.endswith('.tsv'):
        base_name = filename[:-4]  # Remove .tsv
    else:
        base_name = filename
    
    # Remove _pileup suffix if present
    if base_name.endswith('_pileup'):
        base_name = base_name[:-7]
    
    # Create output filename
    return f"{base_name}_filtered_pileup.tsv"


@click.command()
@click.option(
    '--input-path',
    '-i',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to the input pileup file (TSV or TSV.GZ format).'
)
@click.option(
    '--output-dir',
    '-o',
    required=True,
    type=click.Path(path_type=Path),
    help='Directory where the filtered output file will be saved.'
)
def main(input_path: Path, output_dir: Path) -> None:
    """
    Filter pileup files based on Variant Allele Frequency (VAF) thresholds.

    This script processes a pileup file, computes VAF metrics (raw_vaf is always
    computed; target_vaf and background_vaf are computed if fetal/maternal columns
    exist), and filters variants where raw_vaf is in the ranges (0, 0.2) or (0.8, 1.0).

    Required columns:
        - cfDNA_alt_reads
        - current_depth

    Optional columns (for additional VAF metrics):
        - fetal_alt_reads_from_model, fetal_current_depth_from_model
        - maternal_alt_reads, maternal_current_depth

    Args:
        input_path: Path to the input pileup file.
        output_dir: Directory for the output filtered pileup file.

    Examples:
        $ python filter_pileup.py \\
            --input-path sample_pileup.tsv.gz \\
            --output-dir ./output/
    """
    console.print("[bold blue]Pileup Filter[/bold blue]", style="bold")
    console.print("=" * 60)
    console.print(f"Input file: {input_path}")
    console.print(f"Output directory: {output_dir}")
    console.print("=" * 60)
    console.print()
    
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the pileup file
        filtered_df = process_pileup_file(input_path)
        
        # Handle empty result
        if filtered_df.empty:
            console.print("[yellow]No variants passed the filtering criteria[/yellow]")
            # Still create an empty output file
            output_filename = get_output_filename(input_path)
            output_path = output_dir / output_filename
            filtered_df.to_csv(output_path, sep='\t', index=False)
            console.print(f"[yellow]Empty output file created: {output_path}[/yellow]")
            return
        
        # Save filtered results
        output_filename = get_output_filename(input_path)
        output_path = output_dir / output_filename
        
        console.print(f"\n[cyan]Saving filtered results...[/cyan]")
        filtered_df.to_csv(output_path, sep='\t', index=False)
        
        console.print()
        console.print("[bold green]✓ Success![/bold green]")
        console.print(f"Output file: {output_path}")
        console.print(f"Output size: {len(filtered_df):,} variants")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}", err=True)
        sys.exit(1)
    except ValueError as e:
        console.print(f"[bold red]Validation Error:[/bold red] {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()