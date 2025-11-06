#!/usr/bin/env python3
"""
Merge multiple text files containing genomic data.

This script merges multiple text files with columns 'name', 'prob_class_1', and 'mTcount',
optionally filters by mTcount threshold, and outputs a merged file with 'name' and 'prob_class_1'.
"""

import sys
from pathlib import Path
from typing import List

import click
import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Initialize rich console for output
console = Console()


def validate_input_files(input_files: List[Path]) -> None:
    """
    Validate that all input files exist and are readable.
    
    Args:
        input_files: List of Path objects pointing to input files.
        
    Raises:
        FileNotFoundError: If any input file does not exist.
        PermissionError: If any input file is not readable.
    """
    for file_path in input_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")


def read_txt_file(file_path: Path, required_columns: List[str]) -> pl.DataFrame:
    """
    Read a single text file into a Polars DataFrame.
    
    Args:
        file_path: Path to the text file to read.
        required_columns: List of column names that must be present.
        
    Returns:
        A Polars DataFrame containing the data from the file.
        
    Raises:
        ValueError: If required columns are missing from the file.
        Exception: If the file cannot be read or parsed.
    """
    try:
        # Read the file with tab separator (common for genomic data)
        df = pl.read_csv(
            file_path,
            separator="\t",
            has_header=True,
            null_values=["NA", "na", "N/A", ""]
        )
        
        # Validate required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"File {file_path} is missing required columns: {', '.join(missing_columns)}"
            )
        
        # Select only the required columns to save memory
        df = df.select(required_columns)
        
        return df
        
    except Exception as e:
        console.print(f"[bold red]Error reading file {file_path}:[/bold red] {str(e)}")
        raise


def merge_txt_files(
    input_files: List[Path],
    ncpgs: int = 0
) -> pl.DataFrame:
    """
    Merge multiple text files and apply filtering.
    
    Args:
        input_files: List of Path objects pointing to input files.
        ncpgs: Minimum mTcount threshold for filtering (default: 0, no filtering).
        
    Returns:
        A merged and filtered Polars DataFrame with 'name' and 'prob_class_1' columns.
        
    Raises:
        ValueError: If no valid data remains after filtering.
    """
    required_columns = ["name", "prob_class_1", "mTcount"]
    dataframes = []
    
    # Read all input files with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Reading input files...",
            total=len(input_files)
        )
        
        for file_path in input_files:
            try:
                df = read_txt_file(file_path, required_columns)
                dataframes.append(df)
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Reading input files... ({file_path.name})"
                )
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/bold yellow] Skipping {file_path}: {str(e)}")
                progress.update(task, advance=1)
    
    # Check if we have any valid dataframes
    if not dataframes:
        raise ValueError("No valid input files could be read")
    
    console.print(f"[green]✓[/green] Successfully read {len(dataframes)} file(s)")
    
    # Merge all dataframes
    console.print("[cyan]Merging dataframes...[/cyan]")
    merged_df = pl.concat(dataframes, how="vertical")
    
    initial_rows = len(merged_df)
    console.print(f"[green]✓[/green] Merged data contains {initial_rows:,} rows")
    
    # Apply filtering if ncpgs > 0
    if ncpgs > 0:
        console.print(f"[cyan]Applying filter: mTcount >= {ncpgs}...[/cyan]")
        merged_df = merged_df.filter(pl.col("mTcount") >= ncpgs)
        filtered_rows = len(merged_df)
        removed_rows = initial_rows - filtered_rows
        console.print(
            f"[green]✓[/green] Filtered data contains {filtered_rows:,} rows "
            f"({removed_rows:,} rows removed)"
        )
        
        if filtered_rows == 0:
            raise ValueError(
                f"No data remains after filtering with mTcount >= {ncpgs}. "
                f"Consider using a lower threshold."
            )
    else:
        console.print("[yellow]ℹ[/yellow] No filtering applied (ncpgs = 0)")
    
    # Select only the output columns
    output_df = merged_df.select(["name", "prob_class_1"])
    
    return output_df


def write_output(df: pl.DataFrame, output_path: Path) -> None:
    """
    Write the merged DataFrame to an output file.
    
    Args:
        df: Polars DataFrame to write.
        output_path: Path where the output file will be written.
        
    Raises:
        PermissionError: If the output file cannot be written.
        Exception: If an error occurs during writing.
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file with tab separator
        df.write_csv(
            output_path,
            separator="\t",
            include_header=True
        )
        
        console.print(f"[green]✓[/green] Output written to: {output_path}")
        console.print(f"[green]✓[/green] Output contains {len(df):,} rows")
        
    except Exception as e:
        console.print(f"[bold red]Error writing output file:[/bold red] {str(e)}")
        raise


@click.command()
@click.option(
    "--inputs",
    required=True,
    type=str,
    help="Space-separated list of input text files to merge."
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output file path for the merged data."
)
@click.option(
    "--ncpgs",
    default=0,
    type=int,
    help="Minimum mTcount threshold for filtering (default: 0, no filtering)."
)
def main(inputs: str, output: str, ncpgs: int) -> None:
    """
    Merge multiple text files containing genomic data.
    
    This script reads multiple text files with columns 'name', 'prob_class_1', and 'mTcount',
    merges them together, applies optional filtering based on mTcount threshold,
    and outputs a file with 'name' and 'prob_class_1' columns.
    
    Args:
        inputs: Space-separated string of input file paths.
        output: Path to the output file.
        ncpgs: Minimum mTcount threshold for filtering (0 means no filtering).
        
    Examples:
        # Merge files without filtering
        python merge_txt.py --inputs "file1.txt file2.txt file3.txt" --output merged.txt
        
        # Merge files with filtering (mTcount >= 5)
        python merge_txt.py --inputs "file1.txt file2.txt" --output merged.txt --ncpgs 5
    """
    console.print("[bold cyan]Text File Merger[/bold cyan]")
    console.print("=" * 60)
    
    try:
        # Parse input file paths
        input_file_paths = [Path(f.strip()) for f in inputs.split() if f.strip()]
        
        if not input_file_paths:
            raise ValueError("No input files provided")
        
        console.print(f"[cyan]Input files:[/cyan] {len(input_file_paths)}")
        console.print(f"[cyan]Output file:[/cyan] {output}")
        console.print(f"[cyan]mTcount threshold:[/cyan] {ncpgs}")
        console.print("=" * 60)
        
        # Validate input files
        console.print("[cyan]Validating input files...[/cyan]")
        validate_input_files(input_file_paths)
        console.print(f"[green]✓[/green] All {len(input_file_paths)} input file(s) validated")
        
        # Merge the files
        merged_df = merge_txt_files(input_file_paths, ncpgs)
        
        # Write output
        output_path = Path(output)
        console.print("[cyan]Writing output file...[/cyan]")
        write_output(merged_df, output_path)
        
        console.print("=" * 60)
        console.print("[bold green]✓ Process completed successfully![/bold green]")
        
    except Exception as e:
        console.print("=" * 60)
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        console.print("[bold red]Process failed![/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

