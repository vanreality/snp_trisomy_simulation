#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamed BED-like TSV -> BAM converter (low-memory, robust)
- Single writer thread + bounded queue provides hard backpressure.
- Reads input line-by-line (csv module), no DataFrame kept in memory.
- Writes directly to a single BAM (no temporary BAMs, no merge).
"""

from __future__ import annotations
import os
import sys
import csv
import click
import tempfile
from typing import Optional, Tuple
from collections import OrderedDict
from threading import Thread
from queue import Queue, Empty

import pysam
from rich import print, pretty
from rich.traceback import install
from rich.progress import Progress, BarColumn, TimeRemainingColumn, MofNCompleteColumn

pretty.install()
install(show_locals=False)

# ---------------------------
# BAM header and chromosome map
# ---------------------------
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
CHROM_TO_TID = {sq["SN"]: tid for tid, sq in enumerate(HEADER_DICT["SQ"])}

# ---------------------------
# Core helpers
# ---------------------------
def _transform_seq(seq: str) -> str:
    """Apply sequence conversion: C->T then M->C (order matters)."""
    # NOTE: order is important to preserve original intention
    return seq.replace("C", "T").replace("M", "C")

def _convert_seq(seq: str) -> str:
    """Return the complementary strand of a DNA sequence."""
    # Convert to uppercase and define complement mapping
    seq = seq.upper()
    complement_map = str.maketrans('ACTGN', 'TGACN')
    
    # Reverse and complement the sequence
    return seq.translate(complement_map)[::-1]

def _row_to_segment(chrom: str, start: int, read_name: str, flag_str: str,  seq: str) -> Optional[pysam.AlignedSegment]:
    """Create a pysam AlignedSegment from minimal fields."""
    tid = CHROM_TO_TID.get(chrom)
    if tid is None:
        return None

    seq2 = _transform_seq(seq)
    qual = "I" * len(seq2)

    seg = pysam.AlignedSegment()
    seg.query_name = read_name
    # seg.query_sequence = seq2 if '99' in flag_str else _convert_seq(seq2)
    seg.query_sequence = seq2
    seg.flag = 0  if '99' in flag_str else 16
    seg.reference_id = tid
    seg.reference_start = int(start)
    seg.mapping_quality = 255
    seg.cigar = [(0, len(seq2))]  # M of full length
    seg.next_reference_id = -1
    seg.next_reference_start = -1
    seg.template_length = 0
    seg.query_qualities = pysam.qualitystring_to_array(qual)
    seg.set_tag("RG", HEADER_DICT["RG"][0]["ID"])
    return seg

# ---------------------------
# Writer thread
# ---------------------------
def _writer_worker(
    q: Queue,
    result_q: Queue,
    out_bam_path: str,
    progress_task: int,
    progress: Progress,
    progress_step: int = 10_000
):
    """
    Single writer that consumes (chrom, start, read_name, seq) tuples and writes to one BAM.
    Puts (written, skipped) into result_q when done.
    """
    written = 0
    skipped = 0

    try:
        with pysam.AlignmentFile(out_bam_path, "wb", header=dict(HEADER_DICT)) as bam:
            while True:
                item = q.get()
                if item is None:  # sentinel marks the end
                    q.task_done()
                    break
                chrom, start, read_name, flag_str, seq = item
                seg = _row_to_segment(chrom, start, read_name, flag_str, seq)
                if seg is None:
                    skipped += 1
                else:
                    bam.write(seg)
                    written += 1
                    # Throttle UI updates to reduce overhead
                    if progress_task is not None and (written % progress_step == 0):
                        progress.update(progress_task, advance=progress_step)
                q.task_done()
    except Exception as e:
        # Propagate failure via result queue
        result_q.put(("error", str(e)))
        return

    result_q.put(("ok", (written, skipped)))

# ---------------------------
# Counting utility (optional)
# ---------------------------
def _count_lines(path: str) -> Optional[int]:
    """Count lines in binary mode; return None if fails (e.g., pipe)."""
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None

# ---------------------------
# Main streamed converter
# ---------------------------
def process_bed_to_bam_stream(
    input_path: str,
    output_bam: str,
    queue_size: int = 50_000,
    do_count: bool = True,
    csv_dialect: str = "excel-tab"
):
    """
    Fully streaming BED-like TSV -> BAM converter with bounded memory.
    Input columns (no header):
        0: chrom
        1: start
        2: end
        3: read_name
        4: flag_str
        5: dot      (unused)
        6: seq
        7: seq_len  (unused)
    """

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_bam)) or "."
    os.makedirs(out_dir, exist_ok=True)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:

        # Optional counting pass for nicer ETA; can be disabled to save IO
        total_lines = None
        if do_count:
            cnt_task = progress.add_task("[cyan]Counting records...", total=1)
            total_lines = _count_lines(input_path)
            progress.update(cnt_task, completed=1)

        write_task = progress.add_task(
            "[green]Writing BAM...",
            total=total_lines if total_lines else None
        )

        q: Queue = Queue(maxsize=max(1, queue_size))
        result_q: Queue = Queue(maxsize=1)

        # Start writer thread (daemon so it won't block interpreter exit on fatal error)
        writer = Thread(
            target=_writer_worker,
            args=(q, result_q, output_bam, write_task, progress),
            daemon=True
        )
        writer.start()

        # Producer: scan file line-by-line and feed queue
        produced = 0
        skipped_input = 0

        # Use csv.reader with tab delimiter; robust and memory-light.
        with open(input_path, "r", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                # Validate row shape and extract minimal fields
                # Skip malformed lines to keep pipeline running
                try:
                    chrom = row[0]
                    start = int(row[1])
                    read_name = row[3]
                    flag_str = row[4]
                    seq = row[6]
                except Exception:
                    skipped_input += 1
                    continue

                # This put() will block when queue is full -> hard backpressure
                q.put((chrom, start, read_name, flag_str, seq))
                produced += 1

                # If we know total, keep progress in sync with production
                if total_lines and (produced % 10_000 == 0):
                    progress.update(write_task, completed=produced)

        # Signal writer to finish and wait for all tasks drained
        q.put(None)
        q.join()

        # Collect writer results
        status, payload = result_q.get()
        if status == "error":
            raise RuntimeError(f"BAM writing failed: {payload}")

        written, skipped_unknown_chrom = payload

        # Finalize progress bar
        if total_lines:
            progress.update(write_task, completed=produced, total=produced)

        # Summary
        if skipped_input > 0:
            click.echo(f"[yellow]Note[/yellow]: skipped malformed input lines: {skipped_input:,}", err=True)
        if skipped_unknown_chrom > 0:
            click.echo(f"[yellow]Note[/yellow]: skipped reads with unknown chromosomes: {skipped_unknown_chrom:,}", err=True)

        click.echo(f"[green]✓[/green] Wrote BAM to [bold]{output_bam}[/bold]. "
                   f"Input rows: {produced:,}, written: {written:,}.")

# ---------------------------
# CLI
# ---------------------------
@click.command()
@click.option("--input_bed", type=click.Path(exists=True), required=True,
              help="Input BED-like TSV path (8 columns, tab-separated, no header)")
@click.option("--output_bam", type=click.Path(), required=True,
              help="Output BAM file path")
@click.option("--queue-size", "-q", type=int, default=50_000,
              help="Bounded queue size for backpressure (default: 50,000)")
@click.option("--no-count", is_flag=True, default=True,
              help="Skip the initial line counting pass to save I/O")
def cli(input_bed: str, output_bam: str, queue_size: int, no_count: bool):
    """
    Convert a BED-like alignment table to BAM using a single-writer streaming pipeline.
    Example:
        python assembled_to_bam_stream.py --input_bed input.bed --output_bam output.bam
        python assembled_to_bam_stream.py -q 200000 --no-count --input_bed input.bed --output_bam output.bam
    """
    try:
        process_bed_to_bam_stream(
            input_bed,
            output_bam,
            queue_size=queue_size,
            do_count=not no_count
        )
    except Exception as e:
        click.echo(f"[red]Error[/red]: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()