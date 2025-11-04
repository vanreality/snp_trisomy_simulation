#!/usr/bin/env python3
"""
Unclassified BAM Statistics Calculator

This script analyzes BAM files to calculate read overlap statistics with BED regions.
For each sample, it determines how many reads are completely within, partially overlapping,
or outside BED regions. For paired-end reads, it also tracks mate pair relationships
relative to BED regions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict

import click
import pandas as pd
import pysam
from intervaltree import IntervalTree, Interval
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box

console = Console()
progress_lock = Lock()


class BEDRegionIndex:
    """
    Efficient index structure for querying BED regions using IntervalTree.
    
    Uses IntervalTree for O(log n) overlap and containment queries.
    Each chromosome has its own IntervalTree for efficient querying.
    """
    
    def __init__(self):
        """Initialize empty region index."""
        self.trees: Dict[str, IntervalTree] = {}
        self.region_count: int = 0
    
    def add_region(self, chrom: str, start: int, end: int) -> None:
        """
        Add a region to the index.
        
        Args:
            chrom: Chromosome name.
            start: Start position (0-based, inclusive).
            end: End position (exclusive).
        """
        if chrom not in self.trees:
            self.trees[chrom] = IntervalTree()
        
        # IntervalTree uses half-open intervals [start, end)
        self.trees[chrom].addi(start, end)
        self.region_count += 1
    
    def is_completely_within(self, chrom: str, start: int, end: int) -> bool:
        """
        Check if a region is completely within any BED region.
        
        A region is completely within if there exists a BED interval that
        fully contains it (BED.start <= start and end <= BED.end).
        
        Args:
            chrom: Chromosome name.
            start: Start position.
            end: End position.
            
        Returns:
            True if completely within any BED region, False otherwise.
        """
        if chrom not in self.trees:
            return False
        
        # Find all overlapping intervals
        overlapping = self.trees[chrom].overlap(start, end)
        
        # Check if any interval completely contains this region
        for interval in overlapping:
            if interval.begin <= start and end <= interval.end:
                return True
        
        return False
    
    def has_overlap(self, chrom: str, start: int, end: int) -> bool:
        """
        Check if a region overlaps with any BED region.
        
        Args:
            chrom: Chromosome name.
            start: Start position.
            end: End position.
            
        Returns:
            True if overlaps with any BED region, False otherwise.
        """
        if chrom not in self.trees:
            return False
        
        # IntervalTree.overlap() returns a set of overlapping intervals
        # Returns empty set if no overlap
        return bool(self.trees[chrom].overlap(start, end))
    
    def get_region_count(self) -> int:
        """
        Get total number of regions in the index.
        
        Returns:
            Total number of BED regions.
        """
        return self.region_count


def load_bed_file(bed_path: Path) -> BEDRegionIndex:
    """
    Load BED file and create an indexed region structure using IntervalTree.
    
    Args:
        bed_path: Path to the BED file.
        
    Returns:
        BEDRegionIndex containing all regions from the BED file.
        
    Raises:
        FileNotFoundError: If BED file doesn't exist.
        ValueError: If BED file format is invalid.
        
    Examples:
        >>> bed_index = load_bed_file(Path("regions.bed"))
        >>> bed_index.has_overlap("chr1", 1000, 2000)
        True
    """
    if not bed_path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_path}")
    
    bed_index = BEDRegionIndex()
    line_count = 0
    
    try:
        with open(bed_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue
                
                fields = line.split('\t')
                
                # BED format requires at least 3 fields: chrom, start, end
                if len(fields) < 3:
                    raise ValueError(f"Invalid BED format at line {line_num}: expected at least 3 fields, got {len(fields)}")
                
                chrom = fields[0]
                try:
                    start = int(fields[1])
                    end = int(fields[2])
                except ValueError:
                    raise ValueError(f"Invalid BED format at line {line_num}: start and end must be integers")
                
                if start < 0 or end < 0:
                    raise ValueError(f"Invalid BED format at line {line_num}: start and end must be non-negative")
                
                if start >= end:
                    raise ValueError(f"Invalid BED format at line {line_num}: start must be less than end")
                
                # Add region directly to IntervalTree-based index
                bed_index.add_region(chrom, start, end)
                line_count += 1
        
        if line_count == 0:
            raise ValueError("BED file is empty or contains no valid regions")
        
        return bed_index
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error parsing BED file {bed_path}: {str(e)}")


@dataclass
class ReadClassification:
    """
    Classification results for reads in a BAM file.
    
    Attributes:
        total_reads: Total number of reads processed.
        within_region: Reads completely within BED regions.
        partial_overlap: Reads partially overlapping BED regions.
        outside_region: Reads not overlapping any BED regions.
        pairs_one_in_one_out: Pairs with one mate in/overlapping and one outside.
        pairs_both_out: Pairs with both mates outside all regions.
        total_pairs: Total number of read pairs processed.
    """
    total_reads: int = 0
    within_region: int = 0
    partial_overlap: int = 0
    outside_region: int = 0
    pairs_one_in_one_out: int = 0
    pairs_both_out: int = 0
    total_pairs: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert classification results to dictionary with counts and proportions.
        
        Returns:
            Dictionary containing all statistics including proportions.
        """
        result = {
            'total_reads': self.total_reads,
            'within_region_count': self.within_region,
            'partial_overlap_count': self.partial_overlap,
            'outside_region_count': self.outside_region,
        }
        
        # Calculate proportions (avoid division by zero)
        if self.total_reads > 0:
            result['within_region_proportion'] = self.within_region / self.total_reads
            result['partial_overlap_proportion'] = self.partial_overlap / self.total_reads
            result['outside_region_proportion'] = self.outside_region / self.total_reads
        else:
            result['within_region_proportion'] = 0.0
            result['partial_overlap_proportion'] = 0.0
            result['outside_region_proportion'] = 0.0
        
        # Add pair statistics
        result['total_pairs'] = self.total_pairs
        result['pairs_one_in_one_out_count'] = self.pairs_one_in_one_out
        result['pairs_both_out_count'] = self.pairs_both_out
        
        # Calculate pair proportions
        if self.total_pairs > 0:
            result['pairs_one_in_one_out_proportion'] = self.pairs_one_in_one_out / self.total_pairs
            result['pairs_both_out_proportion'] = self.pairs_both_out / self.total_pairs
        else:
            result['pairs_one_in_one_out_proportion'] = 0.0
            result['pairs_both_out_proportion'] = 0.0
        
        return result


def classify_read(
    read: pysam.AlignedSegment,
    bed_index: BEDRegionIndex
) -> str:
    """
    Classify a single read relative to BED regions.
    
    Args:
        read: Aligned read from BAM file.
        bed_index: Index of BED regions for querying.
        
    Returns:
        Classification string: 'within', 'partial', or 'outside'.
    """
    # Skip unmapped reads
    if read.is_unmapped:
        return 'outside'
    
    chrom = read.reference_name
    start = read.reference_start
    end = read.reference_end
    
    # Check if completely within any region
    if bed_index.is_completely_within(chrom, start, end):
        return 'within'
    
    # Check if partially overlaps any region
    if bed_index.has_overlap(chrom, start, end):
        return 'partial'
    
    # Otherwise, it's outside all regions
    return 'outside'


def analyze_bam_file(
    bam_path: Path,
    bed_index: BEDRegionIndex
) -> ReadClassification:
    """
    Analyze a BAM file and classify reads relative to BED regions.
    
    Args:
        bam_path: Path to the BAM file.
        bed_index: Index of BED regions for querying.
        
    Returns:
        ReadClassification object containing all statistics.
        
    Raises:
        FileNotFoundError: If BAM file doesn't exist.
        ValueError: If BAM file is invalid or corrupted.
    """
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")
    
    classification = ReadClassification()
    
    # Track read pairs to avoid double-counting
    # Store read name -> classification mapping
    read_pairs: Dict[str, List[str]] = defaultdict(list)
    
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch():
                # Skip secondary and supplementary alignments
                if read.is_secondary or read.is_supplementary:
                    continue
                
                classification.total_reads += 1
                
                # Classify the read
                read_class = classify_read(read, bed_index)
                
                if read_class == 'within':
                    classification.within_region += 1
                elif read_class == 'partial':
                    classification.partial_overlap += 1
                else:
                    classification.outside_region += 1
                
                # Track paired reads for mate pair analysis
                if read.is_paired:
                    read_pairs[read.query_name].append(read_class)
            
            # Analyze read pairs
            for read_name, classifications in read_pairs.items():
                # Only process complete pairs (both mates present)
                if len(classifications) == 2:
                    classification.total_pairs += 1
                    
                    mate1_class = classifications[0]
                    mate2_class = classifications[1]
                    
                    # Check if one is in/overlapping and one is outside
                    mate1_in_or_overlap = mate1_class in ['within', 'partial']
                    mate2_in_or_overlap = mate2_class in ['within', 'partial']
                    
                    if (mate1_in_or_overlap and not mate2_in_or_overlap) or \
                       (mate2_in_or_overlap and not mate1_in_or_overlap):
                        classification.pairs_one_in_one_out += 1
                    elif not mate1_in_or_overlap and not mate2_in_or_overlap:
                        classification.pairs_both_out += 1
        
        return classification
        
    except Exception as e:
        raise ValueError(f"Error processing BAM file {bam_path}: {str(e)}")


def find_unclassified_bam_files(input_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all unclassified BAM files in the input directory.
    
    Args:
        input_dir: Root directory containing samtools_merge_unclassified subdirectory.
        
    Returns:
        List of tuples (sample_name, bam_path) for all found BAM files.
        
    Raises:
        FileNotFoundError: If the samtools_merge_unclassified directory doesn't exist.
    """
    unclassified_dir = input_dir / "samtools_merge_unclassified"
    
    if not unclassified_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {unclassified_dir}\n"
            f"Expected to find 'samtools_merge_unclassified' subdirectory in {input_dir}"
        )
    
    bam_files = []
    
    for bam_file in unclassified_dir.glob("*_unclassified.bam"):
        # Extract sample name by removing suffix
        sample_name = bam_file.name.replace("_unclassified.bam", "")
        bam_files.append((sample_name, bam_file))
    
    return sorted(bam_files, key=lambda x: x[0])


def process_sample(
    sample_name: str,
    bam_path: Path,
    bed_index: BEDRegionIndex,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None
) -> Dict[str, any]:
    """
    Process a single sample BAM file.
    
    Thread-safe function that can be called in parallel.
    
    Args:
        sample_name: Name of the sample.
        bam_path: Path to the BAM file.
        bed_index: Index of BED regions for querying.
        progress: Rich Progress object for updating progress (optional).
        task_id: Task ID for progress tracking (optional).
        
    Returns:
        Dictionary containing sample name and all statistics.
        
    Raises:
        Exception: Any error during processing is caught and logged.
    """
    result = {'sample': sample_name}
    
    def update_progress(description: str):
        """Thread-safe progress update."""
        if progress is not None and task_id is not None:
            with progress_lock:
                progress.update(task_id, description=description)
    
    def print_warning(message: str):
        """Thread-safe console print."""
        with progress_lock:
            console.print(message)
    
    try:
        update_progress(f"[cyan]Processing {sample_name}")
        
        # Analyze the BAM file
        classification = analyze_bam_file(bam_path, bed_index)
        
        # Convert to dictionary and add to result
        result.update(classification.to_dict())
        
        update_progress(f"[green]✓ Completed {sample_name}")
        
    except Exception as e:
        print_warning(f"[red]Error processing sample {sample_name}: {str(e)}")
        
        # Fill with default values on error
        result.update({
            'total_reads': 0,
            'within_region_count': 0,
            'partial_overlap_count': 0,
            'outside_region_count': 0,
            'within_region_proportion': float('nan'),
            'partial_overlap_proportion': float('nan'),
            'outside_region_proportion': float('nan'),
            'total_pairs': 0,
            'pairs_one_in_one_out_count': 0,
            'pairs_both_out_count': 0,
            'pairs_one_in_one_out_proportion': float('nan'),
            'pairs_both_out_proportion': float('nan'),
        })
    
    return result


def display_summary_table(df: pd.DataFrame) -> None:
    """
    Display a summary table of statistics using Rich.
    
    Args:
        df: DataFrame containing all sample statistics.
    """
    table = Table(title="Unclassified BAM Statistics Summary", box=box.ROUNDED)
    
    table.add_column("Statistic", style="cyan", no_wrap=True)
    table.add_column("Mean", style="magenta")
    table.add_column("Median", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="red")
    
    # Select numeric columns for summary (exclude counts, show only proportions)
    summary_cols = [
        'total_reads',
        'within_region_proportion',
        'partial_overlap_proportion',
        'outside_region_proportion',
        'total_pairs',
        'pairs_one_in_one_out_proportion',
        'pairs_both_out_proportion'
    ]
    
    for col in summary_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                table.add_row(
                    col,
                    f"{values.mean():.4f}",
                    f"{values.median():.4f}",
                    f"{values.min():.4f}",
                    f"{values.max():.4f}"
                )
    
    console.print("\n")
    console.print(table)


@click.command()
@click.option(
    '--input_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Input directory containing samtools_merge_unclassified subdirectory with *_unclassified.bam files.'
)
@click.option(
    '--input_bed',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help='BED file defining genomic regions of interest.'
)
@click.option(
    '--output_prefix',
    type=str,
    required=True,
    help='Prefix for output file. Output will be saved as {prefix}_report.tsv'
)
@click.option(
    '--ncpus',
    type=int,
    default=1,
    show_default=True,
    help='Number of CPU threads to use for parallel processing.'
)
def main(input_dir: Path, input_bed: Path, output_prefix: str, ncpus: int) -> None:
    """
    Analyze unclassified BAM files for read overlap with BED regions.
    
    This script processes BAM files to determine how reads overlap with genomic
    regions defined in a BED file. It calculates:
    - Reads completely within BED regions
    - Reads partially overlapping BED regions
    - Reads outside all BED regions
    - Paired-end statistics (one mate in/one out, both mates out)
    
    Supports parallel processing with multiple CPU threads for improved performance.
    
    Args:
        input_dir: Directory containing samtools_merge_unclassified subdirectory.
        input_bed: BED file defining genomic regions of interest.
        output_prefix: Prefix for output TSV file.
        ncpus: Number of CPU threads to use for parallel processing.
        
    Examples:
        $ python unclassified_bam_stat.py --input_dir /path/to/data --input_bed regions.bed --output_prefix results
        $ python unclassified_bam_stat.py --input_dir /path/to/data --input_bed regions.bed --output_prefix results --ncpus 8
    """
    console.rule("[bold blue]Unclassified BAM Statistics Calculator")
    console.print(f"[cyan]Input directory: {input_dir}")
    console.print(f"[cyan]Input BED file: {input_bed}")
    console.print(f"[cyan]Output prefix: {output_prefix}")
    console.print(f"[cyan]CPU threads: {ncpus}")
    
    # Validate ncpus value
    if ncpus < 1:
        console.print("[red]Error: --ncpus must be at least 1")
        sys.exit(1)
    
    # Load BED file
    console.print("\n[bold]Loading BED file...")
    try:
        bed_index = load_bed_file(input_bed)
        console.print(f"[green]✓ Loaded {bed_index.get_region_count()} BED regions")
    except Exception as e:
        console.print(f"[red]Error loading BED file: {str(e)}")
        sys.exit(1)
    
    # Find all BAM files
    console.print("\n[bold]Finding BAM files...")
    try:
        bam_files = find_unclassified_bam_files(input_dir)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {str(e)}")
        sys.exit(1)
    
    if not bam_files:
        console.print("[red]Error: No *_unclassified.bam files found!")
        sys.exit(1)
    
    console.print(f"[green]Found {len(bam_files)} BAM files")
    
    # Adjust ncpus if it exceeds the number of samples
    effective_ncpus = min(ncpus, len(bam_files))
    if effective_ncpus < ncpus:
        console.print(f"[yellow]Note: Using {effective_ncpus} threads (limited by number of samples)")
    
    # Process all samples with multi-threading
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing BAM files...", total=len(bam_files))
        
        if ncpus == 1:
            # Single-threaded processing for ncpus=1
            for sample_name, bam_path in bam_files:
                result = process_sample(sample_name, bam_path, bed_index, progress, task)
                results.append(result)
                progress.advance(task)
        else:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=effective_ncpus) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(process_sample, sample_name, bam_path, bed_index, progress, task): sample_name
                    for sample_name, bam_path in bam_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_sample):
                    sample_name = future_to_sample[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        with progress_lock:
                            console.print(f"[red]Failed to process sample {sample_name}: {str(e)}")
                        
                        # Add a failed result with default values
                        failed_result = {
                            'sample': sample_name,
                            'total_reads': 0,
                            'within_region_count': 0,
                            'partial_overlap_count': 0,
                            'outside_region_count': 0,
                            'within_region_proportion': float('nan'),
                            'partial_overlap_proportion': float('nan'),
                            'outside_region_proportion': float('nan'),
                            'total_pairs': 0,
                            'pairs_one_in_one_out_count': 0,
                            'pairs_both_out_count': 0,
                            'pairs_one_in_one_out_proportion': float('nan'),
                            'pairs_both_out_proportion': float('nan'),
                        }
                        results.append(failed_result)
                    finally:
                        progress.advance(task)
    
    # Create DataFrame with results
    df = pd.DataFrame(results)
    
    # Define column order for output
    column_order = [
        'sample',
        'total_reads',
        'within_region_count',
        'within_region_proportion',
        'partial_overlap_count',
        'partial_overlap_proportion',
        'outside_region_count',
        'outside_region_proportion',
        'total_pairs',
        'pairs_one_in_one_out_count',
        'pairs_one_in_one_out_proportion',
        'pairs_both_out_count',
        'pairs_both_out_proportion'
    ]
    
    # Only include columns that exist in the dataframe
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Save to TSV
    output_file = f"{output_prefix}_report.tsv"
    df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
    
    console.print(f"\n[green]✓ Results saved to: {output_file}")
    console.print(f"[green]✓ Processed {len(results)} samples successfully")
    
    # Display summary statistics
    display_summary_table(df)
    
    console.rule("[bold green]Processing Complete")


if __name__ == '__main__':
    main()

