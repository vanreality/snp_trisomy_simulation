'''
Convert BED format alignment table to BAM file with multi-threading support.

Copyright (c) 2025-10-23 by LiMingyang, YiLab, Peking University.

Author: Li Mingyang (limingyang200101@gmail.com)

Institute: AAIS, Peking University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''
from rich import print, pretty
from rich.traceback import install
from rich.progress import Progress, BarColumn, TimeRemainingColumn, MofNCompleteColumn
pretty.install()
install(show_locals=True)
import re
import sys
import click
import polars as pl
import pysam
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import tempfile
import os
import multiprocessing
import queue
from threading import Thread

# Prepare BAM header information
HEADER_DICT = OrderedDict([
    ("HD", {"VN": "1.6", "SO": "coordinate"}),
    ("SQ", [
        {"LN": 248956422, "SN": "chr1"},
        {"LN": 242193529, "SN": "chr2"},
        {"LN": 198295559, "SN": "chr3"},
        {"LN": 190214555, "SN": "chr4"},
        {"LN": 181538259, "SN": "chr5"},
        {"LN": 170805979, "SN": "chr6"},
        {"LN": 159345973, "SN": "chr7"},
        {"LN": 145138636, "SN": "chr8"},
        {"LN": 138394717, "SN": "chr9"},
        {"LN": 133797422, "SN": "chr10"},
        {"LN": 135086622, "SN": "chr11"},
        {"LN": 133275309, "SN": "chr12"},
        {"LN": 114364328, "SN": "chr13"},
        {"LN": 107043718, "SN": "chr14"},
        {"LN": 101991189, "SN": "chr15"},
        {"LN": 90338345, "SN": "chr16"},
        {"LN": 83257441, "SN": "chr17"},
        {"LN": 80373285, "SN": "chr18"},
        {"LN": 58617616, "SN": "chr19"},
        {"LN": 64444167, "SN": "chr20"},
        {"LN": 46709983, "SN": "chr21"},
        {"LN": 50818468, "SN": "chr22"},
        {"LN": 156040895, "SN": "chrX"},
        {"LN": 57227415, "SN": "chrY"},
        {"LN": 16569, "SN": "chrM"}
    ]),
    ("RG", [{"ID": "SAMPLE", "SM": "SAMPLE"}])
])

# Create chromosome name to index mapping
CHROM_TO_TID = {sq["SN"]: tid for tid, sq in enumerate(HEADER_DICT["SQ"])}

def create_aligned_segment(row: Dict) -> pysam.AlignedSegment:
    """Create a single AlignedSegment object from a row dict"""
    seg = pysam.AlignedSegment()
    
    # Set basic attributes
    seg.query_name = row["read_name"]
    seg.query_sequence = row["seq"]
    seg.flag = row["flag"]
    
    # Set chromosome position
    chrom = row["chrom"]
    if chrom not in CHROM_TO_TID:
        return None
        
    seg.reference_id = CHROM_TO_TID[chrom]
    seg.reference_start = row["start"]
    
    # Set quality values and CIGAR
    seg.query_qualities = pysam.qualitystring_to_array(row["qual"])
    seg.cigar = [(0, len(row["seq"]))]  # Perfect match
    
    # Set other required fields
    seg.mapping_quality = 255  # Mapping quality unknown
    seg.next_reference_id = -1  # No paired read
    seg.next_reference_start = -1
    seg.template_length = 0  # Insert size
    
    # Set read group
    seg.set_tag("RG", HEADER_DICT["RG"][0]["ID"])
    
    return seg


def process_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Process a dataframe chunk: sequence conversion and quality generation"""
    return df.with_columns(
        # Sequence conversion: C→T, M→C
        pl.col("seq").str.replace_all('C','T',literal=True).str.replace_all('M','C',literal=True),
        
        # Set flag value
        pl.lit(99).alias("flag"),
        
        # Create quality string: all set to 'I' (Phred40)
        pl.col("seq").map_elements(
            lambda s: "I" * len(s), return_dtype=pl.String
        ).alias("qual"),
    )


def process_chunk_to_bam(chunk_df: pl.DataFrame, output_path: str) -> tuple:
    """Process data chunk and write to temporary BAM file"""
    skipped = 0
    written = 0
    
    try:
        # Process the chunk
        chunk_df = process_dataframe(chunk_df)
        
        with pysam.AlignmentFile(output_path, "wb", header=dict(HEADER_DICT)) as bam:
            for row in chunk_df.rows(named=True):
                seg = create_aligned_segment(row)
                if seg is None:
                    skipped += 1
                    continue
                bam.write(seg)
                written += 1
        return (output_path, written, skipped)
    except Exception as e:
        click.echo(f"Error processing chunk: {str(e)}", err=True)
        return (output_path, 0, 0)


def merge_bam_files(bam_files: List[str], output_bam: str):
    """Merge multiple BAM files into one"""
    if len(bam_files) == 1:
        # Only one file, just rename
        os.rename(bam_files[0], output_bam)
    else:
        # Use pysam merge
        pysam.merge("-f", output_bam, *bam_files)
        # Clean up temporary files
        for bam_file in bam_files:
            try:
                os.remove(bam_file)
            except:
                pass


def count_lines_fast(file_path: str) -> int:
    """Fast line counting for progress tracking"""
    try:
        with open(file_path, 'rb') as f:
            return sum(1 for _ in f)
    except:
        return None


def process_bed_to_bam(input_path, output_bam, n_threads=4, chunk_size=50000):
    """Convert BED format table to BAM file (multi-threaded, memory-efficient version)"""
    
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        
        # Count total lines for progress tracking
        count_task = progress.add_task("[cyan]Counting records...", total=1)
        total_lines = count_lines_fast(input_path)
        progress.update(count_task, completed=1)
        
        if total_lines:
            click.echo(f"Total records: {total_lines:,}, using {n_threads} threads")
            click.echo(f"Batch size: {chunk_size:,} records per chunk")
        else:
            click.echo(f"Processing file with {n_threads} threads (batch size: {chunk_size:,})")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="bam_temp_")
        temp_bam_files = []
        
        # Statistics
        total_written = 0
        total_skipped = 0
        chunk_index = 0
        
        # Create batched reader for memory-efficient processing
        reader = pl.read_csv_batched(
            input_path,
            separator="\t",
            has_header=False,
            new_columns=[
                "chrom", "start", "end", "read_name", 
                "flag_str", "dot", "seq", "seq_len"
            ],
            batch_size=chunk_size
        )
        
        # Estimate total chunks
        estimated_chunks = (total_lines // chunk_size + 1) if total_lines else 100
        write_task = progress.add_task(
            "[green]Processing and writing BAM...", 
            total=estimated_chunks
        )
        
        try:
            # Process batches with thread pool
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = []
                
                # Read and submit batches
                while True:
                    try:
                        batch = reader.next_batches(1)
                        if not batch:
                            break
                        
                        chunk_df = batch[0]
                        if chunk_df.height == 0:
                            break
                        
                        temp_bam = os.path.join(temp_dir, f"chunk_{chunk_index:05d}.bam")
                        future = executor.submit(process_chunk_to_bam, chunk_df, temp_bam)
                        futures.append(future)
                        chunk_index += 1
                        
                        # Process completed futures to avoid memory buildup
                        if len(futures) >= n_threads * 2:
                            completed = []
                            for future in as_completed(futures):
                                bam_path, written, skipped = future.result()
                                temp_bam_files.append(bam_path)
                                total_written += written
                                total_skipped += skipped
                                progress.update(write_task, advance=1)
                                completed.append(future)
                            
                            # Remove completed futures
                            futures = [f for f in futures if f not in completed]
                        
                    except StopIteration:
                        break
                    except Exception as e:
                        click.echo(f"Error reading batch: {str(e)}", err=True)
                        break
                
                # Process remaining futures
                for future in as_completed(futures):
                    bam_path, written, skipped = future.result()
                    temp_bam_files.append(bam_path)
                    total_written += written
                    total_skipped += skipped
                    progress.update(write_task, advance=1)
            
            # Update progress bar to actual chunk count
            progress.update(write_task, completed=chunk_index, total=chunk_index)
            
            if total_skipped > 0:
                click.echo(f"Warning: Skipped {total_skipped:,} records with unknown chromosomes", err=True)
            
            # Merge BAM files
            if len(temp_bam_files) > 0:
                merge_task = progress.add_task("[magenta]Merging BAM files...", total=1)
                # Sort by filename to maintain order
                temp_bam_files.sort()
                merge_bam_files(temp_bam_files, output_bam)
                progress.update(merge_task, completed=1)
                
                click.echo(f"[green]✓[/green] Successfully converted {total_written:,} records to {output_bam}")
            else:
                click.echo("Error: No BAM files were generated", err=True)
                sys.exit(1)
                
        except Exception as e:
            click.echo(f"Processing failed: {str(e)}", err=True)
            sys.exit(1)
        finally:
            # Clean up temporary directory
            try:
                os.rmdir(temp_dir)
            except:
                pass

@click.command()
@click.option("--input_bed", type=click.Path(exists=True), required=True, help="Input BED file path (8 columns)")
@click.option("--output_bam", type=click.Path(), required=True, help="Output BAM file path")
@click.option("-t", "--threads", default=None, type=int, help="Number of parallel threads (default: auto-detect all available CPUs)")
@click.option("-c", "--chunk-size", default=50000, type=int, help="Number of records per chunk (default: 50000)")
def cli(input_bed, output_bam, threads, chunk_size):
    """Convert BED format alignment table to BAM file (multi-threaded, memory-efficient version)
    
    This tool uses streaming processing to handle large files without loading 
    everything into memory at once. It automatically detects and uses all available 
    CPU cores for parallel processing.
    
    Examples:
        # Auto-detect CPUs, default chunk size
        python assembled_to_bam.py --input_bed input.bed --output_bam output.bam
        
        # Use 8 threads with 100K records per chunk
        python assembled_to_bam.py --input_bed input.bed --output_bam output.bam -t 8 -c 100000
    """
    # Auto-detect CPU count
    if threads is None:
        threads = multiprocessing.cpu_count()
        click.echo(f"Auto-detected {threads} CPU cores")
    
    process_bed_to_bam(input_bed, output_bam, n_threads=threads, chunk_size=chunk_size)

if __name__ == "__main__":
    cli()