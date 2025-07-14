import pandas as pd
import click
import sys
from pathlib import Path
from typing import List, Optional
import re
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Initialize rich console for beautiful output
console = Console()


def parse_sample_name(filepath: Path) -> str:
    """
    Parse sample name from lr_calculator output filename.
    
    Expects filename format: ${sample}_pileup_lr.tsv
    
    Args:
        filepath (Path): Path to the lr_calculator output file
    
    Returns:
        str: Sample name extracted from filename
    
    Raises:
        ValueError: If filename doesn't match expected pattern
    
    Examples:
        >>> parse_sample_name(Path("sample1_pileup_lr.tsv"))
        'sample1'
        >>> parse_sample_name(Path("complex_sample_name_pileup_lr.tsv"))
        'complex_sample_name'
    """
    filename = filepath.name
    
    # Expected pattern: ${sample}_pileup_lr.tsv
    pattern = r'^(.+)_pileup_lr\.tsv$'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Filename doesn't match expected pattern '{{sample}}_pileup_lr.tsv': {filename}")
    
    return match.group(1)


def load_lr_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single lr_calculator output file and validate its format.
    
    Args:
        filepath (Path): Path to the lr_calculator output file
    
    Returns:
        pd.DataFrame: DataFrame with lr_calculator results
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or missing expected columns
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, sep='\t')
    except Exception as e:
        raise ValueError(f"Failed to read file {filepath}: {str(e)}")
    
    # Validate expected columns
    expected_cols = {'Chrom', 'LR', 'Fetal Fraction'}
    if not expected_cols.issubset(df.columns):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Missing expected columns in {filepath}: {missing}")
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(df['LR']):
        console.print(f"[yellow]Warning: Converting LR column to numeric in {filepath.name}[/yellow]")
        df['LR'] = pd.to_numeric(df['LR'], errors='coerce')
    
    if not pd.api.types.is_numeric_dtype(df['Fetal Fraction']):
        console.print(f"[yellow]Warning: Converting Fetal Fraction column to numeric in {filepath.name}[/yellow]")
        df['Fetal Fraction'] = pd.to_numeric(df['Fetal Fraction'], errors='coerce')
    
    return df


def merge_lr_files(input_files: List[Path]) -> pd.DataFrame:
    """
    Merge multiple lr_calculator output files into a single DataFrame.
    
    This function processes each input file, extracts the sample name from the filename,
    loads the data, renames columns to standard format, and adds a sample identifier.
    
    Args:
        input_files (List[Path]): List of paths to lr_calculator output files
    
    Returns:
        pd.DataFrame: Merged DataFrame with columns ['chr', 'lr', 'ff', 'sample']
    
    Raises:
        ValueError: If no valid files are provided or processing fails
    """
    if not input_files:
        raise ValueError("No input files provided")
    
    merged_data = []
    failed_files = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing files...", total=len(input_files))
        
        for filepath in input_files:
            try:
                # Parse sample name from filename
                sample_name = parse_sample_name(filepath)
                
                # Load the file
                df = load_lr_file(filepath)
                
                # Rename columns to standard format
                df = df.rename(columns={
                    'Chrom': 'chr',
                    'LR': 'lr',
                    'Fetal Fraction': 'ff'
                })
                
                # Add sample column
                df['sample'] = sample_name
                
                # Reorder columns for consistency
                df = df[['chr', 'lr', 'ff', 'sample']]
                
                merged_data.append(df)
                
                console.print(f"[green]✓ Processed {filepath.name}: {len(df)} records from sample '{sample_name}'[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ Error processing {filepath.name}: {str(e)}[/red]")
                failed_files.append(filepath.name)
                continue
            
            progress.advance(task)
    
    if not merged_data:
        raise ValueError("No valid files were processed successfully")
    
    if failed_files:
        console.print(f"[yellow]Warning: {len(failed_files)} files failed to process: {', '.join(failed_files)}[/yellow]")
    
    # Concatenate all DataFrames
    console.print("[cyan]Merging all data...[/cyan]")
    merged_df = pd.concat(merged_data, ignore_index=True)
    
    return merged_df


def validate_merged_data(df: pd.DataFrame) -> None:
    """
    Validate the merged DataFrame for consistency and completeness.
    
    Performs comprehensive validation including column presence, data types,
    missing values, and logical consistency checks.
    
    Args:
        df (pd.DataFrame): Merged DataFrame to validate
    
    Raises:
        ValueError: If validation fails
    """
    # Check for required columns
    required_cols = {'chr', 'lr', 'ff', 'sample'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in merged data: {missing}")
    
    # Check for empty data
    if len(df) == 0:
        raise ValueError("Merged data is empty")
    
    # Check for missing values in critical columns
    if df['chr'].isna().any():
        raise ValueError("Missing chromosome values found in merged data")
    
    if df['sample'].isna().any():
        raise ValueError("Missing sample values found in merged data")
    
    # Validate data types and ranges
    numeric_cols = ['lr', 'ff']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            console.print(f"[yellow]Warning: Column '{col}' is not numeric, attempting conversion[/yellow]")
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                raise ValueError(f"Failed to convert column '{col}' to numeric: {str(e)}")
    
    # Check for reasonable value ranges
    if df['ff'].min() < 0 or df['ff'].max() > 1:
        console.print(f"[yellow]Warning: Fetal fraction values outside expected range [0,1]: {df['ff'].min():.3f} - {df['ff'].max():.3f}[/yellow]")
    
    if df['lr'].min() < 0:
        console.print(f"[yellow]Warning: Negative likelihood ratio values found: {df['lr'].min():.2e}[/yellow]")
    
    # Check for duplicate entries
    duplicate_mask = df.duplicated(subset=['chr', 'sample'])
    if duplicate_mask.any():
        n_duplicates = duplicate_mask.sum()
        console.print(f"[yellow]Warning: Found {n_duplicates} duplicate chr-sample combinations[/yellow]")
    
    console.print(f"[green]✓ Validation passed: {len(df)} records from {df['sample'].nunique()} samples[/green]")


def display_summary(df: pd.DataFrame) -> None:
    """
    Display a comprehensive summary table of the merged data.
    
    Shows overall statistics and per-sample summary if the number of samples
    is reasonable for display.
    
    Args:
        df (pd.DataFrame): Merged DataFrame to summarize
    """
    # Create summary by sample
    summary_stats = df.groupby('sample').agg({
        'chr': 'count',
        'lr': ['mean', 'std', 'min', 'max'],
        'ff': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.rename(columns={'chr_count': 'n_chromosomes'})
    
    # Display overall summary
    unique_chrs = sorted(df['chr'].unique())
    chr_display = ', '.join(unique_chrs) if len(unique_chrs) <= 10 else f"{len(unique_chrs)} chromosomes"
    
    console.print(Panel.fit(
        f"[bold green]Merge Summary[/bold green]\n"
        f"Total Records: {len(df):,}\n"
        f"Unique Samples: {df['sample'].nunique()}\n"
        f"Chromosomes: {chr_display}\n"
        f"LR Range: {df['lr'].min():.2e} - {df['lr'].max():.2e}\n"
        f"FF Range: {df['ff'].min():.3f} - {df['ff'].max():.3f}\n"
        f"Mean FF: {df['ff'].mean():.3f} ± {df['ff'].std():.3f}",
        title="Results Overview"
    ))
    
    # Display per-sample summary if reasonable number of samples
    if df['sample'].nunique() <= 25:
        table = Table(title="Per-Sample Summary")
        table.add_column("Sample", justify="left", style="cyan")
        table.add_column("Chromosomes", justify="right", style="green")
        table.add_column("Mean LR", justify="right", style="yellow")
        table.add_column("Mean FF", justify="right", style="magenta")
        table.add_column("FF Range", justify="right", style="blue")
        
        for sample in sorted(df['sample'].unique()):
            sample_data = df[df['sample'] == sample]
            table.add_row(
                sample,
                str(len(sample_data)),
                f"{sample_data['lr'].mean():.2e}",
                f"{sample_data['ff'].mean():.3f}",
                f"{sample_data['ff'].min():.3f}-{sample_data['ff'].max():.3f}"
            )
        
        console.print(table)
    else:
        console.print(f"[cyan]Per-sample summary omitted ({df['sample'].nunique()} samples > 25)[/cyan]")


@click.command()
@click.option(
    '--input-files',
    required=True,
    help='Space-separated list of lr_calculator output files to merge'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    default='merged_lr_output.tsv.gz',
    help='Output filename (default: merged_lr_output.tsv.gz)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output and detailed error reporting'
)
@click.option(
    '--force',
    is_flag=True,
    help='Overwrite output file if it already exists'
)
def main(input_files: str, output: Path, verbose: bool, force: bool) -> None:
    """
    Merge Likelihood Ratio Calculator Output Files.
    
    This script merges multiple lr_calculator output files from different samples
    into a single consolidated TSV file for downstream analysis.
    
    Input files are expected to be named as: ${sample}_pileup_lr.tsv
    Each file should contain columns: ['Chrom', 'LR', 'Fetal Fraction']
    
    Output will have columns: ['chr', 'lr', 'ff', 'sample']
    
    The script performs comprehensive validation and provides detailed progress
    reporting and error handling.
    """
    try:
        # Check if output file exists and handle accordingly
        if output.exists() and not force:
            console.print(f"[red]✗ Output file already exists: {output}[/red]")
            console.print("[yellow]Use --force to overwrite existing output[/yellow]")
            sys.exit(1)
        
        # Parse input file paths
        file_paths = [Path(f.strip()) for f in input_files.split()]
        
        # Validate input files
        valid_files = []
        for filepath in file_paths:
            if filepath.exists():
                valid_files.append(filepath)
            else:
                console.print(f"[yellow]Warning: File not found, skipping: {filepath}[/yellow]")
        
        if not valid_files:
            console.print("[red]✗ No valid input files found[/red]")
            sys.exit(1)
        
        # Display startup information
        console.print(Panel.fit(
            f"[bold green]LR Output Merger[/bold green]\n"
            f"Input Files: {len(valid_files)}\n"
            f"Output: {output}\n"
            f"Verbose: {verbose}\n"
            f"Force Overwrite: {force}",
            title="Configuration"
        ))
        
        # Merge files
        console.print("[cyan]Starting merge process...[/cyan]")
        merged_df = merge_lr_files(valid_files)
        
        # Validate merged data
        console.print("[cyan]Validating merged data...[/cyan]")
        validate_merged_data(merged_df)
        
        # Display summary
        display_summary(merged_df)
        
        # Save output
        console.print(f"[cyan]Saving merged data to {output}...[/cyan]")
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as compressed TSV
        merged_df.to_csv(output, sep='\t', index=False, compression='gzip')
        
        # Verify output file was created
        if output.exists():
            file_size = output.stat().st_size
            console.print(f"[bold green]✓ Successfully merged {len(merged_df)} records to {output} ({file_size:,} bytes)[/bold green]")
        else:
            raise ValueError("Output file was not created successfully")
        
    except KeyboardInterrupt:
        console.print("[red]✗ Operation cancelled by user[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Fatal error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
