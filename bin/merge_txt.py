#!/usr/bin/env python3
"""
Merge multiple text files containing genomic data.

This script merges multiple text files with columns 'name', 'prob_class_1', and 'mTcount',
optionally filters by mTcount threshold, and outputs a merged file with 'name' and 'prob_class_1'.
"""

import gc
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


def read_txt_file_lazy(file_path: Path, required_columns: List[str], ncpgs: int = 0) -> pl.LazyFrame:
    """
    Read a single text file into a Polars LazyFrame for memory-efficient processing.
    
    Args:
        file_path: Path to the text file to read.
        required_columns: List of column names that must be present.
        ncpgs: Minimum mTcount threshold for filtering (applied lazily).
        
    Returns:
        A Polars LazyFrame containing the data from the file with filters applied.
        
    Raises:
        ValueError: If required columns are missing from the file.
        Exception: If the file cannot be read or parsed.
    """
    try:
        # Use scan_csv for lazy reading - doesn't load data into memory immediately
        lazy_df = pl.scan_csv(
            file_path,
            separator="\t",
            has_header=True,
            null_values=["NA", "na", "N/A", ""],
            low_memory=True,  # Optimize for memory usage over speed
        )
        
        # Validate required columns exist (this requires a small schema check)
        schema_columns = lazy_df.collect_schema().names()
        missing_columns = [col for col in required_columns if col not in schema_columns]
        if missing_columns:
            raise ValueError(
                f"File {file_path} is missing required columns: {', '.join(missing_columns)}"
            )
        
        # Apply filter early if ncpgs > 0 (lazy operation, very efficient)
        if ncpgs > 0:
            lazy_df = lazy_df.filter(pl.col("mTcount") >= ncpgs)
        
        # Select only the required columns to save memory (lazy operation)
        lazy_df = lazy_df.select(required_columns)
        
        return lazy_df
        
    except Exception as e:
        console.print(f"[bold red]Error reading file {file_path}:[/bold red] {str(e)}")
        raise


def merge_txt_files(
    input_files: List[Path],
    ncpgs: int = 0
) -> pl.DataFrame:
    """
    Merge multiple text files and apply filtering using lazy evaluation for memory efficiency.
    
    Args:
        input_files: List of Path objects pointing to input files.
        ncpgs: Minimum mTcount threshold for filtering (default: 0, no filtering).
        
    Returns:
        A merged and filtered Polars DataFrame with 'name' and 'prob_class_1' columns.
        
    Raises:
        ValueError: If no valid data remains after filtering.
    """
    required_columns = ["name", "prob_class_1", "mTcount"]
    lazy_frames = []
    
    # Build lazy frames for all input files with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Preparing input files...",
            total=len(input_files)
        )
        
        for file_path in input_files:
            try:
                # Create lazy frame (no data loaded into memory yet)
                lazy_df = read_txt_file_lazy(file_path, required_columns, ncpgs)
                lazy_frames.append(lazy_df)
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Preparing input files... ({file_path.name})"
                )
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/bold yellow] Skipping {file_path}: {str(e)}")
                progress.update(task, advance=1)
    
    # Check if we have any valid lazy frames
    if not lazy_frames:
        raise ValueError("No valid input files could be read")
    
    console.print(f"[green]✓[/green] Successfully prepared {len(lazy_frames)} file(s)")
    
    # Merge all lazy frames (still lazy, no data loaded)
    console.print("[cyan]Merging and processing data (this may take a while for large files)...[/cyan]")
    
    # Concatenate lazy frames
    merged_lazy = pl.concat(lazy_frames, how="vertical")
    
    # Select only the output columns we need (lazy operation)
    output_lazy = merged_lazy.select(["name", "prob_class_1"])
    
    # Now collect the data (this is where the actual computation happens)
    # Use streaming mode for better memory efficiency with large datasets
    try:
        output_df = output_lazy.collect(streaming=True)
    except Exception as e:
        # Fallback to non-streaming if streaming fails
        console.print("[yellow]⚠[/yellow] Streaming mode failed, falling back to standard collection...")
        output_df = output_lazy.collect()
    
    # Force garbage collection to free up memory
    gc.collect()
    
    output_rows = len(output_df)
    console.print(f"[green]✓[/green] Processed data contains {output_rows:,} rows")
    
    if ncpgs > 0:
        console.print(f"[green]✓[/green] Filter applied: mTcount >= {ncpgs}")
    else:
        console.print("[yellow]ℹ[/yellow] No filtering applied (ncpgs = 0)")
    
    if output_rows == 0:
        raise ValueError(
            f"No data remains after processing. "
            f"Check your input files and filter threshold (ncpgs={ncpgs})."
        )
    
    return output_df


def write_output(df: pl.DataFrame, output_path: Path) -> None:
    """
    Write the merged DataFrame to an output file with memory-efficient batching.
    
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
        
        console.print("[cyan]Writing output file...[/cyan]")
        
        # Write to file with tab separator
        df.write_csv(
            output_path,
            separator="\t",
            include_header=True
        )
        
        console.print(f"[green]✓[/green] Output written to: {output_path}")
        console.print(f"[green]✓[/green] Output contains {len(df):,} rows")
        
        # Force garbage collection after writing
        gc.collect()
        
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
        write_output(merged_df, output_path)
        
        # Final cleanup
        del merged_df
        gc.collect()
        
        console.print("=" * 60)
        console.print("[bold green]✓ Process completed successfully![/bold green]")
        
    except Exception as e:
        console.print("=" * 60)
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        console.print("[bold red]Process failed![/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

