#!/usr/bin/env python3
"""
BED File Generator from TSV Variants

This script processes a TSV file containing variant information and splits it into
three BED files based on variant types:
- half_depth_ct.bed: G/A variants
- half_depth_ga.bed: C/T variants  
- full_depth.bed: all other variants

The script converts 1-based TSV positions to 0-based BED coordinates.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, TextIO, Tuple

import click
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table


# Initialize rich console for output formatting
console = Console()


class VariantClassifier:
    """Classifies variants based on reference and alternate alleles."""
    
    @staticmethod
    def classify_variant(ref: str, alt: str) -> str:
        """
        Classify a variant based on reference and alternate alleles.
        
        Args:
            ref: Reference allele (single nucleotide)
            alt: Alternate allele (single nucleotide)
            
        Returns:
            str: Variant classification ('ga', 'ct', or 'other')
            
        Examples:
            >>> VariantClassifier.classify_variant('G', 'A')
            'ga'
            >>> VariantClassifier.classify_variant('C', 'T') 
            'ct'
            >>> VariantClassifier.classify_variant('A', 'T')
            'other'
        """
        # Normalize to uppercase for consistent comparison
        ref_upper = ref.upper()
        alt_upper = alt.upper()
        
        # Check G/A variants (bidirectional)
        if (ref_upper == "G" and alt_upper == "A") or (ref_upper == "A" and alt_upper == "G"):
            return "ga"
        
        # Check C/T variants (bidirectional)
        elif (ref_upper == "C" and alt_upper == "T") or (ref_upper == "T" and alt_upper == "C"):
            return "ct"
        
        # All other variants
        else:
            return "other"


class BEDWriter:
    """Handles writing BED format files with proper coordinate conversion."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize BED writer with output directory.
        
        Args:
            output_dir: Directory where BED files will be written
        """
        self.output_dir = output_dir
        self.file_handles: Dict[str, TextIO] = {}
        self.counts: Dict[str, int] = {"ga": 0, "ct": 0, "other": 0}
        
        # Define output filenames based on variant types
        self.filenames = {
            "ga": "half_depth_ct.bed",  # G/A variants go to half_depth_ct.bed
            "ct": "half_depth_ga.bed",  # C/T variants go to half_depth_ga.bed  
            "other": "full_depth.bed"   # Other variants go to full_depth.bed
        }
        
    def __enter__(self):
        """Context manager entry - open all output files."""
        try:
            for variant_type, filename in self.filenames.items():
                filepath = self.output_dir / filename
                self.file_handles[variant_type] = open(filepath, 'w', newline='')
            return self
        except IOError as e:
            console.print(f"[red]Error opening output files: {e}[/red]")
            self.close_all()
            raise
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all output files."""
        self.close_all()
        
    def close_all(self):
        """Close all open file handles."""
        for handle in self.file_handles.values():
            if handle and not handle.closed:
                handle.close()
        self.file_handles.clear()
        
    def write_variant(self, chromosome: str, position: int, variant_type: str):
        """
        Write a variant to the appropriate BED file.
        
        Args:
            chromosome: Chromosome name
            position: 1-based position from TSV
            variant_type: Variant classification ('ga', 'ct', or 'other')
            
        Raises:
            ValueError: If variant_type is not recognized
            IOError: If writing to file fails
        """
        if variant_type not in self.file_handles:
            raise ValueError(f"Unknown variant type: {variant_type}")
            
        # Convert 1-based TSV position to 0-based BED start coordinate
        bed_start = position - 1
        bed_end = position
        
        # Write BED format: chr\tstart\tend
        bed_line = f"{chromosome}\t{bed_start}\t{bed_end}\n"
        
        try:
            self.file_handles[variant_type].write(bed_line)
            self.counts[variant_type] += 1
        except IOError as e:
            console.print(f"[red]Error writing to BED file: {e}[/red]")
            raise
            
    def get_counts(self) -> Dict[str, int]:
        """
        Get variant counts for each category.
        
        Returns:
            Dict mapping variant types to their counts
        """
        return self.counts.copy()


def validate_input_file(input_file: Path) -> None:
    """
    Validate that the input file exists and is readable.
    
    Args:
        input_file: Path to input TSV file
        
    Raises:
        click.ClickException: If file validation fails
    """
    if not input_file.exists():
        raise click.ClickException(f"Input file does not exist: {input_file}")
        
    if not input_file.is_file():
        raise click.ClickException(f"Input path is not a file: {input_file}")
        
    if not os.access(input_file, os.R_OK):
        raise click.ClickException(f"Input file is not readable: {input_file}")


def validate_output_directory(output_dir: Path) -> None:
    """
    Validate and create output directory if needed.
    
    Args:
        output_dir: Path to output directory
        
    Raises:
        click.ClickException: If directory validation fails
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise click.ClickException(f"Cannot create output directory {output_dir}: {e}")
        
    if not os.access(output_dir, os.W_OK):
        raise click.ClickException(f"Output directory is not writable: {output_dir}")


def parse_tsv_line(line: str, line_number: int) -> Tuple[str, int, str, str]:
    """
    Parse a single TSV line and extract required fields.
    
    Args:
        line: TSV line to parse
        line_number: Line number for error reporting
        
    Returns:
        Tuple of (chromosome, position, ref_allele, alt_allele)
        
    Raises:
        ValueError: If line format is invalid
    """
    fields = line.strip().split('\t')
    
    if len(fields) < 5:
        raise ValueError(f"Line {line_number}: Expected at least 5 columns, got {len(fields)}")
    
    chromosome = fields[0].strip()
    position_str = fields[1].strip()
    ref_allele = fields[3].strip()
    alt_allele = fields[4].strip()
    
    # Validate chromosome
    if not chromosome:
        raise ValueError(f"Line {line_number}: Empty chromosome")
    
    # Validate and convert position
    try:
        position = int(position_str)
        if position < 1:
            raise ValueError(f"Line {line_number}: Position must be positive, got {position}")
    except ValueError:
        raise ValueError(f"Line {line_number}: Invalid position '{position_str}'")
    
    # Validate alleles
    if not ref_allele or not alt_allele:
        raise ValueError(f"Line {line_number}: Empty allele(s)")
    
    # Check for single nucleotide alleles
    if len(ref_allele) != 1 or len(alt_allele) != 1:
        console.print(f"[yellow]Warning line {line_number}: Multi-nucleotide variant {ref_allele}→{alt_allele}[/yellow]")
    
    return chromosome, position, ref_allele, alt_allele


def count_lines(file_path: Path) -> int:
    """
    Count total lines in file for progress tracking.
    
    Args:
        file_path: Path to file
        
    Returns:
        Number of lines in file
    """
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except IOError:
        # If we can't count lines, return -1 to indicate unknown
        return -1


@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='.',
    help='Output directory for BED files (default: current directory)'
)
@click.option(
    '--skip-errors', '-s',
    is_flag=True,
    help='Skip malformed lines instead of stopping'
)
def main(input_file: Path, output_dir: Path, skip_errors: bool):
    """
    Split TSV variant file into three BED files based on variant types.
    
    Reads a TSV file with variant information and creates three BED files:
    - half_depth_ct.bed: G/A variants  
    - half_depth_ga.bed: C/T variants
    - full_depth.bed: all other variants
    
    INPUT_FILE: Path to input TSV file (no header expected)
    
    TSV format expected:
    Column 1: chromosome
    Column 2: position (1-based)
    Column 4: reference allele
    Column 5: alternate allele
    """
    console.print("[bold blue]BED File Generator[/bold blue]")
    console.print(f"Input file: {input_file}")
    console.print(f"Output directory: {output_dir}")
    
    # Validate inputs
    validate_input_file(input_file)
    validate_output_directory(output_dir)
    
    # Count total lines for progress tracking
    total_lines = count_lines(input_file)
    
    # Initialize classifier and writer
    classifier = VariantClassifier()
    
    # Process the TSV file
    with BEDWriter(output_dir) as bed_writer:
        with Progress() as progress:
            if total_lines > 0:
                task = progress.add_task("[green]Processing variants...", total=total_lines)
            else:
                task = progress.add_task("[green]Processing variants...", total=None)
            
            error_count = 0
            processed_count = 0
            
            try:
                with open(input_file, 'r') as f:
                    for line_number, line in enumerate(f, 1):
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        try:
                            # Parse TSV line
                            chromosome, position, ref_allele, alt_allele = parse_tsv_line(line, line_number)
                            
                            # Classify variant
                            variant_type = classifier.classify_variant(ref_allele, alt_allele)
                            
                            # Write to appropriate BED file
                            bed_writer.write_variant(chromosome, position, variant_type)
                            
                            processed_count += 1
                            
                        except ValueError as e:
                            error_count += 1
                            if skip_errors:
                                console.print(f"[yellow]Skipping line {line_number}: {e}[/yellow]")
                            else:
                                console.print(f"[red]Error: {e}[/red]")
                                raise click.ClickException(f"Processing failed at line {line_number}")
                        
                        # Update progress
                        if total_lines > 0:
                            progress.update(task, advance=1)
                        else:
                            progress.update(task, completed=line_number)
                            
            except IOError as e:
                console.print(f"[red]Error reading input file: {e}[/red]")
                raise click.ClickException("File reading failed")
    
    # Display summary
    counts = bed_writer.get_counts()
    
    console.print("\n[bold green]Processing Complete![/bold green]")
    
    # Create summary table
    table = Table(title="Variant Processing Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Output File", style="magenta")
    table.add_column("Count", justify="right", style="green")
    
    table.add_row("G/A variants", "half_depth_ct.bed", str(counts["ga"]))
    table.add_row("C/T variants", "half_depth_ga.bed", str(counts["ct"]))
    table.add_row("Other variants", "full_depth.bed", str(counts["other"]))
    table.add_row("Total processed", "", str(processed_count), style="bold")
    
    console.print(table)
    
    if error_count > 0:
        console.print(f"\n[yellow]Warnings: {error_count} lines had errors[/yellow]")
    
    console.print(f"\n[green]Output files written to: {output_dir}[/green]")


if __name__ == "__main__":
    main()
