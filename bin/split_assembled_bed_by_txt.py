#!/usr/bin/env python3
"""Split assembled BED file by probability threshold from txt file.

This script reads a txt file containing read names and their classification probabilities,
then splits a BED file into target and background files based on a probability threshold.
"""

import gzip
import sys
from pathlib import Path
from typing import Dict, Tuple

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Initialize rich console
console = Console()


def read_probabilities(txt_file: Path) -> Dict[str, float]:
    """Read probability data from txt file.
    
    Reads the txt file and creates a dictionary mapping read names to their
    prob_class_1 values for efficient lookup during BED file processing.
    
    Args:
        txt_file: Path to the input txt file containing 'name' and 'prob_class_1' columns.
    
    Returns:
        Dictionary mapping read names (str) to probability values (float).
    
    Raises:
        FileNotFoundError: If the txt file does not exist.
        KeyError: If required columns are missing from the txt file.
        ValueError: If probability values cannot be converted to float.
    """
    if not txt_file.exists():
        raise FileNotFoundError(f"Input txt file not found: {txt_file}")
    
    console.print(f"[cyan]Reading probabilities from:[/cyan] {txt_file}")
    
    try:
        # Read only required columns for memory efficiency
        df = pd.read_csv(
            txt_file,
            sep='\t',
            usecols=['name', 'prob_class_1'],
            dtype={'name': str, 'prob_class_1': float}
        )
        # Drop duplicate 'name' rows, keep the first occurrence
        df = df.drop_duplicates(subset=['name'], keep='first')
    except ValueError as e:
        if 'name' not in str(e) and 'prob_class_1' not in str(e):
            raise
        raise KeyError(
            "Required columns 'name' and 'prob_class_1' not found in txt file. "
            f"Available columns should include these fields."
        ) from e
    
    # Check for missing values
    if df['name'].isna().any():
        console.print("[yellow]Warning: Found missing values in 'name' column, removing them[/yellow]")
        df = df.dropna(subset=['name'])
    
    if df['prob_class_1'].isna().any():
        console.print("[yellow]Warning: Found missing values in 'prob_class_1' column, removing them[/yellow]")
        df = df.dropna(subset=['prob_class_1'])
    
    # Convert to dictionary for O(1) lookup
    prob_dict = df.set_index('name')['prob_class_1'].to_dict()
    
    console.print(f"[green]✓[/green] Loaded {len(prob_dict):,} read probabilities")
    
    return prob_dict


def split_bed_file(
    bed_file: Path,
    prob_dict: Dict[str, float],
    threshold: float,
    output_prefix: str
) -> Tuple[int, int, int]:
    """Split BED file into target and background files based on probability threshold.
    
    Reads a gzip-compressed BED file and splits it into two files:
    - target.bed: reads with prob_class_1 > threshold
    - background.bed: reads with prob_class_1 <= threshold
    
    Reads not found in the probability dictionary are skipped with a warning.
    
    Args:
        bed_file: Path to the input gzip-compressed BED file.
        prob_dict: Dictionary mapping read names to probability values.
        threshold: Probability threshold for splitting (reads > threshold go to target).
        output_prefix: Prefix for output files.
    
    Returns:
        Tuple of (target_count, background_count, skipped_count).
    
    Raises:
        FileNotFoundError: If the BED file does not exist.
        OSError: If there are issues reading/writing files.
    """
    if not bed_file.exists():
        raise FileNotFoundError(f"Input BED file not found: {bed_file}")
    
    console.print(f"[cyan]Splitting BED file:[/cyan] {bed_file}")
    console.print(f"[cyan]Threshold:[/cyan] {threshold}")
    
    # Output file paths
    target_bed = Path(f"{output_prefix}_target.bed")
    background_bed = Path(f"{output_prefix}_background.bed")
    
    target_count = 0
    background_count = 0
    skipped_count = 0
    total_lines = 0
    
    try:
        # Open input and output files
        with gzip.open(bed_file, 'rt') as bed_in, \
             open(target_bed, 'w') as target_out, \
             open(background_bed, 'w') as background_out:
            
            # Use rich progress bar for visual feedback
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                # We don't know the total lines, so we'll use indeterminate progress
                task = progress.add_task("[cyan]Processing BED records...", total=None)
                
                for line in bed_in:
                    total_lines += 1
                    
                    # Update progress every 10000 lines
                    if total_lines % 10000 == 0:
                        progress.update(
                            task,
                            description=f"[cyan]Processing BED records... "
                            f"({total_lines:,} processed, "
                            f"{target_count:,} target, "
                            f"{background_count:,} background)"
                        )
                    
                    # Skip empty lines
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse BED line - 4th column (index 3) contains read name
                    fields = line.split('\t')
                    if len(fields) < 4:
                        console.print(
                            f"[yellow]Warning: Skipping malformed BED line (< 4 fields) at line {total_lines}[/yellow]"
                        )
                        skipped_count += 1
                        continue
                    
                    read_name = fields[3]
                    
                    # Check if read name exists in probability dictionary
                    if read_name not in prob_dict:
                        skipped_count += 1
                        continue
                    
                    prob = prob_dict[read_name]
                    
                    # Split based on threshold
                    if prob > threshold:
                        target_out.write(line + '\n')
                        target_count += 1
                    else:
                        background_out.write(line + '\n')
                        background_count += 1
                
                # Final progress update
                progress.update(
                    task,
                    description=f"[green]✓ Completed processing {total_lines:,} BED records"
                )
    
    except Exception as e:
        # Clean up partial output files on error
        console.print(f"[red]Error during BED file processing: {e}[/red]")
        if target_bed.exists():
            target_bed.unlink()
        if background_bed.exists():
            background_bed.unlink()
        raise
    
    console.print(f"[green]✓[/green] Target reads (prob > {threshold}): {target_count:,}")
    console.print(f"[green]✓[/green] Background reads (prob ≤ {threshold}): {background_count:,}")
    
    if skipped_count > 0:
        console.print(
            f"[yellow]⚠[/yellow] Skipped reads (not in txt or malformed): {skipped_count:,}"
        )
    
    return target_count, background_count, skipped_count


@click.command()
@click.option(
    '--input_txt',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Input txt file with header containing "name" and "prob_class_1" columns'
)
@click.option(
    '--input_bed',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Input BED file in gzip compressed format (4th column contains read names)'
)
@click.option(
    '--threshold',
    default=0.5,
    type=float,
    show_default=True,
    help='Probability threshold for splitting (reads with prob > threshold go to target)'
)
@click.option(
    '--output_prefix',
    required=True,
    type=str,
    help='Prefix for output files (will create {prefix}_target.bed and {prefix}_background.bed)'
)
def main(input_txt: Path, input_bed: Path, threshold: float, output_prefix: str):
    """Split assembled BED file by probability threshold from txt file.
    
    This script reads a txt file containing read names and their classification
    probabilities, then splits a gzip-compressed BED file into target and background
    files based on the specified probability threshold.
    
    Example:
        python split_assembled_bam_by_txt.py \\
            --input_txt reads.txt \\
            --input_bed reads.bed.gz \\
            --threshold 0.5 \\
            --output_prefix sample_001
    """
    console.rule("[bold blue]BED File Splitter[/bold blue]")
    
    try:
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            console.print("[red]Error: Threshold must be between 0.0 and 1.0[/red]")
            sys.exit(1)
        
        # Step 1: Read probabilities from txt file
        prob_dict = read_probabilities(input_txt)
        
        if not prob_dict:
            console.print("[red]Error: No valid probabilities found in txt file[/red]")
            sys.exit(1)
        
        # Step 2: Split BED file based on probabilities
        target_count, background_count, skipped_count = split_bed_file(
            input_bed,
            prob_dict,
            threshold,
            output_prefix
        )
        
        # Step 3: Summary
        console.rule("[bold green]Summary[/bold green]")
        total_classified = target_count + background_count
        
        if total_classified > 0:
            target_pct = (target_count / total_classified) * 100
            background_pct = (background_count / total_classified) * 100
            
            console.print(f"Total classified reads: {total_classified:,}")
            console.print(f"  • Target: {target_count:,} ({target_pct:.2f}%)")
            console.print(f"  • Background: {background_count:,} ({background_pct:.2f}%)")
            
            if skipped_count > 0:
                console.print(f"Skipped reads: {skipped_count:,}")
        else:
            console.print("[yellow]Warning: No reads were classified[/yellow]")
        
        console.print("\n[bold green]✓ Processing completed successfully![/bold green]")
        console.print(f"Output files:")
        console.print(f"  • {output_prefix}_target.bed")
        console.print(f"  • {output_prefix}_background.bed")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if console.is_terminal:
            console.print_exception()
        sys.exit(1)


if __name__ == '__main__':
    main()
