#!/usr/bin/env python3
"""
Pileup Merging Script

This script merges multiple pileup TSV.gz files by combining data on common genomic positions
and summing numeric columns. It handles both required and optional columns with proper
error handling and progress monitoring.

Author: SNP Simulate Pipeline
"""

import sys
import gzip
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
NUMERIC_COLUMNS = ['cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth', 'fetal_ref_reads_from_model', 'fetal_alt_reads_from_model', 'fetal_current_depth_from_model', 'maternal_ref_reads', 'maternal_alt_reads', 'maternal_current_depth']


def validate_file_exists(file_path: str) -> Path:
    """
    Validate that a file exists and is readable.
    
    Args:
        file_path (str): Path to the file to validate.
        
    Returns:
        Path: Validated Path object.
        
    Raises:
        click.ClickException: If file doesn't exist or isn't readable.
    """
    path = Path(file_path)
    if not path.exists():
        raise click.ClickException(f"Input file does not exist: {file_path}")
    if not path.is_file():
        raise click.ClickException(f"Path is not a file: {file_path}")
    if not path.stat().st_size > 0:
        raise click.ClickException(f"Input file is empty: {file_path}")
    return path


def read_pileup_file(file_path: Path) -> pd.DataFrame:
    """
    Read a pileup TSV.gz file and validate its structure.
    
    Args:
        file_path (Path): Path to the pileup file.
        
    Returns:
        pd.DataFrame: Loaded pileup data.
        
    Raises:
        click.ClickException: If file format is invalid or required columns are missing.
    """
    try:
        # Read the file - handle both .gz and regular .tsv files
        if file_path.suffix.lower() == '.gz':
            df = pd.read_csv(file_path, sep='\t', compression='gzip')
        else:
            df = pd.read_csv(file_path, sep='\t')
            
        # Validate required columns
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            raise click.ClickException(
                f"File {file_path.name} is missing required columns: {missing_required}"
            )
            
        # Ensure merge columns have no missing values
        merge_missing = df[MERGE_COLUMNS].isnull().any()
        if merge_missing.any():
            problematic_cols = merge_missing[merge_missing].index.tolist()
            raise click.ClickException(
                f"File {file_path.name} has missing values in merge columns: {problematic_cols}"
            )
            
        # Convert numeric columns to appropriate types, filling missing values with 0
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        return df
        
    except pd.errors.EmptyDataError:
        raise click.ClickException(f"File {file_path.name} is empty or contains no valid data")
    except pd.errors.ParserError as e:
        raise click.ClickException(f"Error parsing file {file_path.name}: {str(e)}")
    except Exception as e:
        raise click.ClickException(f"Unexpected error reading file {file_path.name}: {str(e)}")


def merge_pileup_dataframes(dataframes: List[pd.DataFrame], file_names: List[str]) -> pd.DataFrame:
    """
    Merge multiple pileup dataframes on merge columns and sum numeric columns.
    
    Args:
        dataframes (List[pd.DataFrame]): List of pileup dataframes to merge.
        file_names (List[str]): Names of files for logging purposes.
        
    Returns:
        pd.DataFrame: Merged dataframe with summed numeric columns.
        
    Raises:
        click.ClickException: If merging fails due to incompatible data.
    """
    if not dataframes:
        raise click.ClickException("No valid dataframes to merge")
        
    if len(dataframes) == 1:
        console.print("[yellow]Warning: Only one file provided, returning as-is[/yellow]")
        return dataframes[0]
    
    try:
        # Start with the first dataframe
        merged_df = dataframes[0].copy()
        
        # Progressively merge with each subsequent dataframe
        for i, df in enumerate(dataframes[1:], 1):
            # Get common columns between current merged_df and new df
            common_numeric_cols = [col for col in NUMERIC_COLUMNS 
                                 if col in merged_df.columns and col in df.columns]
            
            # Merge on the merge columns
            merged_df = merged_df.merge(
                df, 
                on=MERGE_COLUMNS, 
                how='outer', 
                suffixes=('', f'_file{i+1}')
            )
            
            # Sum the numeric columns
            for col in common_numeric_cols:
                if f"{col}_file{i+1}" in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna(0) + merged_df[f"{col}_file{i+1}"].fillna(0)
                    merged_df = merged_df.drop(columns=[f"{col}_file{i+1}"])
            
            # Handle columns that exist in new df but not in merged_df
            new_cols = [col for col in NUMERIC_COLUMNS 
                       if col in df.columns and col not in merged_df.columns]
            for col in new_cols:
                if f"{col}_file{i+1}" in merged_df.columns:
                    merged_df[col] = merged_df[f"{col}_file{i+1}"].fillna(0)
                    merged_df = merged_df.drop(columns=[f"{col}_file{i+1}"])
        
        # Fill any remaining NaN values with 0 for numeric columns
        for col in NUMERIC_COLUMNS:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)
        
        # Sort by chromosome and position for consistent output
        merged_df = merged_df.sort_values(['chr', 'pos']).reset_index(drop=True)
        
        return merged_df
        
    except Exception as e:
        raise click.ClickException(f"Error during merge operation: {str(e)}")


def write_output_file(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write the merged dataframe to a TSV.gz file.
    
    Args:
        df (pd.DataFrame): Merged dataframe to write.
        output_path (Path): Path where to write the output file.
        
    Raises:
        click.ClickException: If writing fails.
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to compressed TSV
        df.to_csv(output_path, sep='\t', compression='gzip', index=False)
        
    except Exception as e:
        raise click.ClickException(f"Error writing output file {output_path}: {str(e)}")


def display_merge_summary(input_files: List[str], output_file: str, 
                         merged_df: pd.DataFrame, original_counts: List[int]) -> None:
    """
    Display a summary table of the merge operation.
    
    Args:
        input_files (List[str]): List of input file names.
        output_file (str): Output file name.
        merged_df (pd.DataFrame): Final merged dataframe.
        original_counts (List[int]): Row counts from original files.
    """
    # Create summary table
    table = Table(title="Pileup Merge Summary", show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Original Rows", justify="right", style="green")
    table.add_column("Status", style="yellow")
    
    for i, (file_name, count) in enumerate(zip(input_files, original_counts)):
        table.add_row(Path(file_name).name, str(count), "✓ Processed")
    
    table.add_row("", "", "")
    table.add_row("MERGED OUTPUT", str(len(merged_df)), "✓ Complete", style="bold green")
    table.add_row("Output File", output_file, "", style="bold blue")
    
    console.print(table)
    
    # Display column information
    numeric_cols_in_output = [col for col in NUMERIC_COLUMNS if col in merged_df.columns]
    if numeric_cols_in_output:
        console.print(f"\n[bold green]Summed columns:[/bold green] {', '.join(numeric_cols_in_output)}")


@click.command()
@click.option(
    '--inputs',
    required=True,
    help='Space-separated list of input TSV.gz pileup files to merge'
)
@click.option(
    '--output',
    required=True,
    help='Output file name for the merged pileup data (will be compressed as TSV.gz)'
)
def main(inputs: str, output: str) -> None:
    """
    Merge multiple pileup TSV.gz files into a single output file.
    
    This script merges pileup files by combining data on common genomic positions
    (chr, pos, ref, alt, af) and summing numeric columns. Required columns are
    validated, and optional columns are handled gracefully.
    
    Args:
        inputs (str): Space-separated input file paths.
        output (str): Output file path for merged data.
        
    Examples:
        merge_pileups.py --inputs "file1.tsv.gz file2.tsv.gz file3.tsv.gz" --output merged.tsv.gz
    """
    console.print(Panel.fit("🧬 Pileup File Merger", style="bold blue"))
    
    # Parse input files
    input_files = inputs.strip().split()
    if not input_files:
        raise click.ClickException("No input files provided")
    
    console.print(f"[bold]Found {len(input_files)} input files to merge[/bold]")
    
    # Validate input files
    valid_files = []
    for file_path in input_files:
        try:
            validated_path = validate_file_exists(file_path)
            valid_files.append(validated_path)
            console.print(f"✓ Validated: {validated_path.name}")
        except click.ClickException as e:
            console.print(f"✗ [red]{e}[/red]")
            sys.exit(1)
    
    # Validate output path
    output_path = Path(output)
    if output_path.exists():
        if not click.confirm(f"Output file {output} exists. Overwrite?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            sys.exit(0)
    
    # Process files with progress tracking
    dataframes = []
    original_counts = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Read files
        read_task = progress.add_task("Reading input files...", total=len(valid_files))
        
        for file_path in valid_files:
            progress.update(read_task, description=f"Reading {file_path.name}...")
            
            try:
                df = read_pileup_file(file_path)
                dataframes.append(df)
                original_counts.append(len(df))
                console.print(f"✓ Loaded {file_path.name}: {len(df)} rows")
                
            except click.ClickException as e:
                console.print(f"✗ [red]Error loading {file_path.name}: {e}[/red]")
                sys.exit(1)
                
            progress.advance(read_task)
        
        # Merge dataframes
        progress.update(read_task, description="Merging dataframes...")
        try:
            merged_df = merge_pileup_dataframes(dataframes, [f.name for f in valid_files])
            console.print(f"✓ Merged into {len(merged_df)} rows")
        except click.ClickException as e:
            console.print(f"✗ [red]Error during merge: {e}[/red]")
            sys.exit(1)
        
        # Write output
        progress.update(read_task, description=f"Writing output to {output_path.name}...")
        try:
            write_output_file(merged_df, output_path)
            console.print(f"✓ Output written to {output}")
        except click.ClickException as e:
            console.print(f"✗ [red]Error writing output: {e}[/red]")
            sys.exit(1)
    
    # Display summary
    display_merge_summary(input_files, output, merged_df, original_counts)
    
    console.print("\n[bold green]✨ Merge completed successfully![/bold green]")


if __name__ == '__main__':
    main()
