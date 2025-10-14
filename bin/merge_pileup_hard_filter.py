#!/usr/bin/env python3
"""
Pileup Hard Filter Merging Script

This script merges target and background pileup TSV files by combining them on common
genomic positions. It renames columns appropriately and calculates combined cfDNA reads.

The target pileup represents fetal signal, and the background represents maternal signal.
The final output combines both signals into a single pileup file.

Author: SNP Simulate Pipeline
"""

import sys
import gzip
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Initialize rich console
console = Console()

# Define column specifications
REQUIRED_COLUMNS = ['chr', 'pos', 'ref', 'alt', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth']
MERGE_COLUMNS = ['chr', 'pos', 'ref', 'alt', 'af']
TARGET_RENAME_MAP = {
    'cfDNA_ref_reads': 'fetal_ref_reads_from_model',
    'cfDNA_alt_reads': 'fetal_alt_reads_from_model',
    'current_depth': 'fetal_current_depth_from_model'
}
BACKGROUND_RENAME_MAP = {
    'cfDNA_ref_reads': 'maternal_ref_reads',
    'cfDNA_alt_reads': 'maternal_alt_reads',
    'current_depth': 'maternal_current_depth'
}


def validate_file_exists(file_path: str) -> Path:
    """
    Validate that a file exists and is readable.
    
    Args:
        file_path (str): Path to the file to validate.
        
    Returns:
        Path: Validated Path object.
        
    Raises:
        click.ClickException: If file doesn't exist or isn't readable.
        
    Examples:
        >>> path = validate_file_exists("data/sample.tsv.gz")
    """
    path = Path(file_path)
    if not path.exists():
        raise click.ClickException(f"Input file does not exist: {file_path}")
    if not path.is_file():
        raise click.ClickException(f"Path is not a file: {file_path}")
    if not path.stat().st_size > 0:
        raise click.ClickException(f"Input file is empty: {file_path}")
    return path


def read_pileup_file(file_path: Path, file_type: str = "pileup") -> pd.DataFrame:
    """
    Read a pileup TSV file and validate its structure.
    
    This function reads both compressed (.gz) and uncompressed TSV files,
    validates the presence of required columns, and ensures data integrity.
    
    Args:
        file_path (Path): Path to the pileup file.
        file_type (str): Type of file for error messages (e.g., "target", "background").
        
    Returns:
        pd.DataFrame: Loaded pileup data with validated structure.
        
    Raises:
        click.ClickException: If file format is invalid or required columns are missing.
        
    Examples:
        >>> df = read_pileup_file(Path("sample.tsv.gz"), "target")
    """
    try:
        # Read the file - handle both .gz and regular .tsv files
        if str(file_path).endswith('.gz'):
            df = pd.read_csv(file_path, sep='\t', compression='gzip')
        else:
            df = pd.read_csv(file_path, sep='\t')
            
        # Validate required columns
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            raise click.ClickException(
                f"{file_type.capitalize()} file {file_path.name} is missing required columns: {missing_required}"
            )
            
        # Ensure merge columns have no missing values
        merge_missing = df[MERGE_COLUMNS].isnull().any()
        if merge_missing.any():
            problematic_cols = merge_missing[merge_missing].index.tolist()
            raise click.ClickException(
                f"{file_type.capitalize()} file {file_path.name} has missing values in merge columns: {problematic_cols}"
            )
        
        # Validate that numeric columns contain numeric data
        numeric_cols = ['cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth']
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Check if any values became NaN (were non-numeric)
                if df[col].isnull().any():
                    console.print(f"[yellow]Warning: {file_type.capitalize()} file contains non-numeric values in column '{col}'. These will be replaced with 0.[/yellow]")
                # Fill NaN with 0
                df[col] = df[col].fillna(0)
        
        # Ensure chromosome is string type for proper merging
        df['chr'] = df['chr'].astype(str)
        
        return df
        
    except pd.errors.EmptyDataError:
        raise click.ClickException(f"{file_type.capitalize()} file {file_path.name} is empty or contains no valid data")
    except pd.errors.ParserError as e:
        raise click.ClickException(f"Error parsing {file_type} file {file_path.name}: {str(e)}")
    except click.ClickException:
        # Re-raise click exceptions as-is
        raise
    except Exception as e:
        raise click.ClickException(f"Unexpected error reading {file_type} file {file_path.name}: {str(e)}")


def merge_pileup_files(target_df: pd.DataFrame, background_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge target and background pileup dataframes with proper renaming and calculations.
    
    This function performs the following operations:
    1. Renames target columns to fetal_* names
    2. Renames background columns to maternal_* names
    3. Merges on genomic position (chr, pos, ref, alt, af)
    4. Calculates combined cfDNA_* columns by summing fetal and maternal reads
    5. Fills missing values with 0
    
    Args:
        target_df (pd.DataFrame): Target pileup dataframe (fetal signal).
        background_df (pd.DataFrame): Background pileup dataframe (maternal signal).
        
    Returns:
        pd.DataFrame: Merged dataframe with all columns properly renamed and calculated.
        
    Raises:
        click.ClickException: If merging fails due to incompatible data.
        
    Examples:
        >>> merged = merge_pileup_files(target_df, background_df)
    """
    try:
        # Create copies to avoid modifying original dataframes
        target = target_df.copy()
        background = background_df.copy()
        
        # Rename columns in target dataframe
        target = target.rename(columns=TARGET_RENAME_MAP)
        
        # Rename columns in background dataframe
        background = background.rename(columns=BACKGROUND_RENAME_MAP)
        
        # Merge on the first five columns (chr, pos, ref, alt, af)
        merged_df = target.merge(
            background,
            on=MERGE_COLUMNS,
            how='outer',
            suffixes=('_target', '_background')
        )
        
        # Fill NaN values with 0 for fetal columns
        for col in TARGET_RENAME_MAP.values():
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)
        
        # Fill NaN values with 0 for maternal columns
        for col in BACKGROUND_RENAME_MAP.values():
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)
        
        # Calculate combined cfDNA columns by adding fetal and maternal
        merged_df['cfDNA_ref_reads'] = (
            merged_df['fetal_ref_reads_from_model'].fillna(0) + 
            merged_df['maternal_ref_reads'].fillna(0)
        )
        merged_df['cfDNA_alt_reads'] = (
            merged_df['fetal_alt_reads_from_model'].fillna(0) + 
            merged_df['maternal_alt_reads'].fillna(0)
        )
        merged_df['current_depth'] = (
            merged_df['fetal_current_depth_from_model'].fillna(0) + 
            merged_df['maternal_current_depth'].fillna(0)
        )
        
        # Sort by chromosome and position for consistent output
        # Handle chromosome sorting properly (chr1, chr2, ..., chr10, not chr1, chr10, chr2)
        try:
            # Try to sort chromosomes numerically if possible
            merged_df['chr_sort'] = merged_df['chr'].str.replace('chr', '').str.replace('X', '23').str.replace('Y', '24').str.replace('M', '25').str.replace('MT', '25')
            merged_df['chr_sort'] = pd.to_numeric(merged_df['chr_sort'], errors='coerce')
            merged_df = merged_df.sort_values(['chr_sort', 'pos']).drop(columns=['chr_sort']).reset_index(drop=True)
        except:
            # Fall back to simple alphabetical sorting if numeric sorting fails
            merged_df = merged_df.sort_values(['chr', 'pos']).reset_index(drop=True)
        
        return merged_df
        
    except Exception as e:
        raise click.ClickException(f"Error during merge operation: {str(e)}")


def write_output_file(df: pd.DataFrame, output_prefix: str) -> Path:
    """
    Write the merged dataframe to a compressed TSV file.
    
    Args:
        df (pd.DataFrame): Merged dataframe to write.
        output_prefix (str): Output file prefix. The final filename will be {prefix}.tsv.gz.
        
    Returns:
        Path: Path to the created output file.
        
    Raises:
        click.ClickException: If writing fails.
        
    Examples:
        >>> output_path = write_output_file(merged_df, "merged_pileup")
    """
    try:
        # Create output filename
        output_path = Path(f"{output_prefix}.tsv.gz")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to compressed TSV
        df.to_csv(output_path, sep='\t', compression='gzip', index=False)
        
        return output_path
        
    except Exception as e:
        raise click.ClickException(f"Error writing output file: {str(e)}")


def display_merge_summary(
    target_file: str,
    background_file: str,
    output_file: Path,
    target_rows: int,
    background_rows: int,
    merged_rows: int,
    merged_df: pd.DataFrame
) -> None:
    """
    Display a comprehensive summary table of the merge operation.
    
    Args:
        target_file (str): Name of target input file.
        background_file (str): Name of background input file.
        output_file (Path): Path to output file.
        target_rows (int): Number of rows in target file.
        background_rows (int): Number of rows in background file.
        merged_rows (int): Number of rows in merged output.
        merged_df (pd.DataFrame): Final merged dataframe for statistics.
        
    Examples:
        >>> display_merge_summary("target.tsv.gz", "bg.tsv.gz", Path("out.tsv.gz"), 1000, 1200, 1500, df)
    """
    # Create summary table
    table = Table(title="Pileup Merge Summary", show_header=True, header_style="bold magenta")
    table.add_column("File Type", style="cyan", width=20)
    table.add_column("File Name", style="blue")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Status", style="yellow")
    
    table.add_row("Target (Fetal)", Path(target_file).name, str(target_rows), "✓ Processed")
    table.add_row("Background (Maternal)", Path(background_file).name, str(background_rows), "✓ Processed")
    table.add_row("", "", "", "")
    table.add_row("MERGED OUTPUT", output_file.name, str(merged_rows), "✓ Complete", style="bold green")
    
    console.print(table)
    
    # Display statistics
    stats_table = Table(title="Output Statistics", show_header=True, header_style="bold cyan")
    stats_table.add_column("Column", style="cyan")
    stats_table.add_column("Mean", justify="right", style="green")
    stats_table.add_column("Max", justify="right", style="yellow")
    stats_table.add_column("Total", justify="right", style="blue")
    
    numeric_cols = [
        'fetal_ref_reads_from_model',
        'fetal_alt_reads_from_model',
        'maternal_ref_reads',
        'maternal_alt_reads',
        'cfDNA_ref_reads',
        'cfDNA_alt_reads',
        'current_depth'
    ]
    
    for col in numeric_cols:
        if col in merged_df.columns:
            mean_val = merged_df[col].mean()
            max_val = merged_df[col].max()
            total_val = merged_df[col].sum()
            stats_table.add_row(
                col,
                f"{mean_val:.2f}",
                f"{int(max_val)}",
                f"{int(total_val)}"
            )
    
    console.print("\n")
    console.print(stats_table)
    
    # Display column information
    console.print(f"\n[bold green]Output columns:[/bold green]")
    console.print(f"  • Merge keys: {', '.join(MERGE_COLUMNS)}")
    console.print(f"  • Fetal columns: {', '.join(TARGET_RENAME_MAP.values())}")
    console.print(f"  • Maternal columns: {', '.join(BACKGROUND_RENAME_MAP.values())}")
    console.print(f"  • Combined columns: cfDNA_ref_reads, cfDNA_alt_reads, current_depth")


@click.command()
@click.option(
    '--target_pileup',
    required=True,
    type=click.Path(exists=True),
    help='Target pileup TSV file (fetal signal) with columns: chr, pos, ref, alt, af, cfDNA_ref_reads, cfDNA_alt_reads, current_depth'
)
@click.option(
    '--background_pileup',
    required=True,
    type=click.Path(exists=True),
    help='Background pileup TSV file (maternal signal) with the same format as target'
)
@click.option(
    '--output',
    required=True,
    help='Output file prefix. The output file will be saved as {prefix}.tsv.gz'
)
def main(target_pileup: str, background_pileup: str, output: str) -> None:
    """
    Merge target and background pileup files into a combined output.
    
    This script merges two pileup TSV files by combining them on common genomic
    positions (chr, pos, ref, alt, af). The target pileup represents fetal signal
    and the background represents maternal signal. The output contains:
    
    - Original merge columns (chr, pos, ref, alt, af)
    - Renamed target columns as fetal_*_from_model
    - Renamed background columns as maternal_*
    - Calculated cfDNA_* columns (sum of fetal and maternal)
    
    Args:
        target_pileup (str): Path to target pileup file (fetal signal).
        background_pileup (str): Path to background pileup file (maternal signal).
        output (str): Output file prefix for the merged data.
        
    Examples:
        python merge_pileup_hard_filter.py \\
            --target_pileup target.tsv.gz \\
            --background_pileup background.tsv.gz \\
            --output merged_output
            
    Raises:
        click.ClickException: If any validation or processing step fails.
    """
    console.print(Panel.fit("🧬 Pileup Hard Filter Merger", style="bold blue"))
    
    # Validate input files
    console.print("\n[bold]Validating input files...[/bold]")
    try:
        target_path = validate_file_exists(target_pileup)
        console.print(f"✓ Target file validated: {target_path.name}")
        
        background_path = validate_file_exists(background_pileup)
        console.print(f"✓ Background file validated: {background_path.name}")
    except click.ClickException as e:
        console.print(f"✗ [red]{e}[/red]")
        sys.exit(1)
    
    # Validate output path
    output_file = Path(f"{output}.tsv.gz")
    if output_file.exists():
        console.print(f"[yellow]Warning: Output file {output_file} already exists and will be overwritten.[/yellow]")
    
    # Process files with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing...", total=4)
        
        # Read target file
        progress.update(task, description="Reading target pileup file...")
        try:
            target_df = read_pileup_file(target_path, "target")
            target_rows = len(target_df)
            console.print(f"✓ Loaded target file: {target_rows:,} rows")
        except click.ClickException as e:
            console.print(f"✗ [red]Error loading target file: {e}[/red]")
            sys.exit(1)
        progress.advance(task)
        
        # Read background file
        progress.update(task, description="Reading background pileup file...")
        try:
            background_df = read_pileup_file(background_path, "background")
            background_rows = len(background_df)
            console.print(f"✓ Loaded background file: {background_rows:,} rows")
        except click.ClickException as e:
            console.print(f"✗ [red]Error loading background file: {e}[/red]")
            sys.exit(1)
        progress.advance(task)
        
        # Merge dataframes
        progress.update(task, description="Merging pileup files...")
        try:
            merged_df = merge_pileup_files(target_df, background_df)
            merged_rows = len(merged_df)
            console.print(f"✓ Merged into {merged_rows:,} rows")
        except click.ClickException as e:
            console.print(f"✗ [red]Error during merge: {e}[/red]")
            sys.exit(1)
        progress.advance(task)
        
        # Write output
        progress.update(task, description=f"Writing output to {output_file.name}...")
        try:
            output_path = write_output_file(merged_df, output)
            console.print(f"✓ Output written to {output_path}")
        except click.ClickException as e:
            console.print(f"✗ [red]Error writing output: {e}[/red]")
            sys.exit(1)
        progress.advance(task)
    
    # Display summary
    console.print("\n")
    display_merge_summary(
        target_pileup,
        background_pileup,
        output_path,
        target_rows,
        background_rows,
        merged_rows,
        merged_df
    )
    
    console.print("\n[bold green]✨ Merge completed successfully![/bold green]")


if __name__ == '__main__':
    main()

