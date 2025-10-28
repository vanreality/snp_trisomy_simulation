#!/usr/bin/env python3
"""
Hard Filter BAM to Pileup Converter for Bisulfite Sequencing

This script processes BAM files from bisulfite methylation sequencing to generate
simple pileup data at targeted SNP sites. It performs basic read counting without 
probability weighting, reporting raw reference and alternate allele counts.

The script handles bisulfite conversion chemistry artifacts by filtering reads based
on their alignment strand and SNP type. Bisulfite conversion introduces:
- C→T changes on forward strand reads (flags 0/99/147)
- G→A changes on reverse strand reads (flags 16/83/163)

Therefore, for accurate SNP calling:
- C↔T SNPs: only count reverse strand reads (flags 16/83/163)
- G↔A SNPs: only count forward strand reads (flags 0/99/147)
- Other SNPs: count all strand reads

The script also applies additional methylation chemistry base classification rules
and quality filtering to produce accurate pileup statistics for bisulfite sequencing
analysis workflows.
"""

import sys
import gzip
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import click
import pandas as pd
import pysam
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Initialize rich console for output formatting
console = Console()


@dataclass
class SNPSite:
    """
    Data class representing a single SNP site.
    
    Attributes:
        chr (str): Chromosome name
        pos (int): 1-based genomic position
        ref (str): Reference allele (single nucleotide)
        alt (str): Alternate allele (single nucleotide)
        af (float): Allele frequency from INFO field
    """
    chr: str
    pos: int
    ref: str
    alt: str
    af: float


@dataclass
class PileupCounts:
    """
    Data class for storing raw pileup counts.
    
    Attributes:
        cfDNA_ref_reads (int): Raw unweighted reference allele count
        cfDNA_alt_reads (int): Raw unweighted alternate allele count
    """
    cfDNA_ref_reads: int = 0
    cfDNA_alt_reads: int = 0
    
    @property
    def current_depth(self) -> int:
        """Total raw unweighted depth."""
        return self.cfDNA_ref_reads + self.cfDNA_alt_reads


def parse_known_sites(sites_file: Path, progress: Progress, task_id: TaskID) -> List[SNPSite]:
    """
    Parse known sites file and extract SNP information using pandas for efficiency.
    
    Args:
        sites_file (Path): Path to the known sites TSV file
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        List[SNPSite]: List of parsed SNP sites
        
    Raises:
        FileNotFoundError: If sites file doesn't exist
        ValueError: If file format is invalid
    """
    if not sites_file.exists():
        raise FileNotFoundError(f"Known sites file not found: {sites_file}")
    
    progress.update(task_id, description="Parsing known sites...")
    
    try:
        # Read known sites file using pandas, selecting only required columns
        sites_data = pd.read_csv(
            sites_file, 
            sep='\t', 
            comment='#',
            usecols=[0, 1, 3, 4, 7],
            names=['chr', 'pos', 'ref', 'alt', 'info_field'],
            dtype={'chr': str, 'pos': int, 'ref': str, 'alt': str, 'info_field': str}
        )
        
        progress.update(task_id, advance=25)
        
        # Convert bases to uppercase
        sites_data['ref'] = sites_data['ref'].str.upper()
        sites_data['alt'] = sites_data['alt'].str.upper()
        
        # Filter to single-nucleotide variants only
        single_nuc_mask = (sites_data['ref'].str.len() == 1) & (sites_data['alt'].str.len() == 1)
        sites_data = sites_data[single_nuc_mask]
        
        progress.update(task_id, advance=25)
        
        # Extract AF from INFO field using vectorized operations
        af_pattern = r'AF=([^;]+)'
        sites_data['af_match'] = sites_data['info_field'].str.extract(af_pattern, expand=False)
        
        # Filter out rows without AF values
        sites_data = sites_data.dropna(subset=['af_match'])
        
        # Convert AF to float, dropping invalid values
        sites_data['af'] = pd.to_numeric(sites_data['af_match'], errors='coerce')
        sites_data = sites_data.dropna(subset=['af'])
        
        progress.update(task_id, advance=25)
        
        # Convert to list of SNPSite objects
        sites = [
            SNPSite(row['chr'], row['pos'], row['ref'], row['alt'], row['af'])
            for _, row in sites_data.iterrows()
        ]
        
        progress.update(task_id, advance=25)
        console.print(f"[green]✓[/green] Parsed {len(sites):,} single-nucleotide SNP sites")
        
        return sites
        
    except Exception as e:
        console.print(f"[red]Error parsing known sites file: {e}[/red]")
        raise


def get_allowed_flags_for_snp(ref: str, alt: str) -> Optional[set]:
    """
    Determine which SAM flags are allowed for a given SNP type in bisulfite sequencing.
    
    Bisulfite conversion chemistry introduces strand-specific artifacts:
    - Forward strand reads (flags 0/99/147): C→T conversion occurs
    - Reverse strand reads (flags 16/83/163): G→A conversion occurs (complement of C→T)
    
    To avoid counting chemistry-induced variations as SNPs:
    - For C↔T SNPs: only use reverse strand reads (flags 16/83/163)
    - For G↔A SNPs: only use forward strand reads (flags 0/99/147)
    - For other SNPs: use all reads (no flag restriction)
    
    SAM Flag Reference:
    - 0: Unmapped or single-end forward strand
    - 16: Reverse strand
    - 99: First in pair, forward strand, mate reverse
    - 147: Second in pair, reverse strand (but represents forward original strand)
    - 83: First in pair, reverse strand, mate forward
    - 163: Second in pair, forward strand (but represents reverse original strand)
    
    Args:
        ref (str): Reference allele (single nucleotide, uppercase)
        alt (str): Alternate allele (single nucleotide, uppercase)
        
    Returns:
        Optional[set]: Set of allowed SAM flags, or None if all flags are allowed
        
    Examples:
        >>> get_allowed_flags_for_snp('C', 'T')
        {16, 83, 163}
        >>> get_allowed_flags_for_snp('G', 'A')
        {0, 99, 147}
        >>> get_allowed_flags_for_snp('A', 'G')
        {0, 99, 147}
        >>> get_allowed_flags_for_snp('A', 'C')
        None
    """
    ref = ref.upper()
    alt = alt.upper()
    
    # Forward strand flags (original forward strand in bisulfite sequencing)
    forward_flags = {0, 99, 147}
    
    # Reverse strand flags (original reverse strand in bisulfite sequencing)
    reverse_flags = {16, 83, 163}
    
    # Check for C↔T SNPs (affected by forward strand bisulfite conversion)
    if (ref == 'C' and alt == 'T') or (ref == 'T' and alt == 'C'):
        return reverse_flags  # Only use reverse strand reads
    
    # Check for G↔A SNPs (affected by reverse strand bisulfite conversion)
    elif (ref == 'G' and alt == 'A') or (ref == 'A' and alt == 'G'):
        return forward_flags  # Only use forward strand reads
    
    # For all other SNP types, no flag restriction needed
    else:
        return None  # Allow all flags


def classify_base_with_methylation(base: str, ref: str, alt: str) -> Optional[str]:
    """
    Classify a sequenced base as REF or ALT considering methylation chemistry.
    
    Methylation chemistry rules:
    - If ref=C and alt≠T: count both C and T as REF
    - If ref=G and alt≠A: count both G and A as REF  
    - If alt=C and ref≠T: count both C and T as ALT
    - If alt=G and ref≠A: count both G and A as ALT
    - Otherwise: strict matching
    
    Args:
        base (str): Sequenced base (uppercase)
        ref (str): Reference allele (uppercase)
        alt (str): Alternate allele (uppercase)
        
    Returns:
        Optional[str]: 'REF', 'ALT', or None if base doesn't match either
    """
    base = base.upper()
    ref = ref.upper()
    alt = alt.upper()
    
    # Check REF classification with methylation rules
    if ref == 'C' and alt != 'T':
        if base in ['C', 'T']:
            return 'REF'
    elif ref == 'G' and alt != 'A':
        if base in ['G', 'A']:
            return 'REF'
    elif base == ref:
        return 'REF'
    
    # Check ALT classification with methylation rules
    if alt == 'C' and ref != 'T':
        if base in ['C', 'T']:
            return 'ALT'
    elif alt == 'G' and ref != 'A':
        if base in ['G', 'A']:
            return 'ALT'
    elif base == alt:
        return 'ALT'
    
    return None


def process_pileup_site(bam_file: pysam.AlignmentFile, site: SNPSite, 
                       min_mapq: int, min_bq: int) -> PileupCounts:
    """
    Process a single SNP site to compute raw pileup counts with bisulfite-aware flag filtering.
    
    For bisulfite sequencing data, reads are filtered by SAM flags based on SNP type
    to avoid counting chemistry-induced base changes:
    - C↔T SNPs: only count reverse strand reads (flags 16/83/163)
    - G↔A SNPs: only count forward strand reads (flags 0/99/147)
    - Other SNPs: count all strand reads
    
    Args:
        bam_file (pysam.AlignmentFile): Opened BAM file
        site (SNPSite): SNP site to process
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        
    Returns:
        PileupCounts: Raw counts for this site
        
    Raises:
        Warning: Prints warning if error occurs during processing, but continues
    """
    counts = PileupCounts()
    processed_templates = set()  # Track processed template names to avoid double-counting
    
    # Determine which SAM flags are allowed for this SNP type
    allowed_flags = get_allowed_flags_for_snp(site.ref, site.alt)
    
    try:
        # Get pileup column at the specified position
        for pileup_column in bam_file.pileup(
            site.chr, 
            site.pos - 1,  # Convert to 0-based for pysam
            site.pos,
            stepper='nofilter',  # We'll do our own filtering
            ignore_overlaps=True,  # Avoid double-counting overlapping mates
            ignore_orphans=True,
            min_base_quality=min_bq,
            min_mapping_quality=min_mapq
        ):
            if pileup_column.pos != site.pos - 1:  # Check correct position (0-based)
                continue
            
            for pileup_read in pileup_column.pileups:
                read = pileup_read.alignment
                
                # Skip if read doesn't meet quality criteria
                if read.mapping_quality < min_mapq:
                    continue
                if read.is_duplicate or read.is_secondary or read.is_supplementary:
                    continue
                if read.is_unmapped:
                    continue
                
                # Apply bisulfite-specific flag filtering based on SNP type
                # Extract the core SAM flag (excluding paired-end flags)
                read_flag = read.flag

                # Apply flag filtering if this SNP type requires it
                if allowed_flags is not None:
                    if read_flag not in allowed_flags:
                        continue  # Skip reads with disallowed flags for this SNP type
                
                # Skip deletions, reference skips, and insertions
                if pileup_read.is_del or pileup_read.is_refskip:
                    continue
                
                # Get the query base at this position
                if pileup_read.query_position is None:
                    continue
                
                query_base = read.query_sequence[pileup_read.query_position]
                base_quality = read.query_qualities[pileup_read.query_position]
                
                if base_quality < min_bq:
                    continue
                
                # Avoid double-counting paired reads (use template name)
                template_name = read.query_name
                template_key = (template_name, site.chr, site.pos)
                if template_key in processed_templates:
                    continue
                processed_templates.add(template_key)
                
                # Classify base according to methylation chemistry rules
                classification = classify_base_with_methylation(query_base, site.ref, site.alt)
                if classification is None:
                    continue
                
                # Update counts based on classification
                if classification == 'REF':
                    counts.cfDNA_ref_reads += 1
                elif classification == 'ALT':
                    counts.cfDNA_alt_reads += 1
            
            break  # We only need the column at our target position
            
    except Exception as e:
        console.print(f"[yellow]Warning: Error processing site {site.chr}:{site.pos}: {e}[/yellow]")
    
    return counts


def process_sites_chunk(bam_path: Path, sites_chunk: List[SNPSite], 
                       min_mapq: int, min_bq: int) -> List[Tuple[SNPSite, PileupCounts]]:
    """
    Process a chunk of SNP sites in a worker process.
    
    This function is designed to be called by worker processes in the pool.
    Each worker opens its own BAM file handle to avoid threading issues.
    
    Args:
        bam_path (Path): Path to BAM file
        sites_chunk (List[SNPSite]): Chunk of SNP sites to process
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        
    Returns:
        List[Tuple[SNPSite, PileupCounts]]: List of (site, counts) pairs for this chunk
        
    Raises:
        Exception: If BAM file cannot be opened or processing fails
    """
    results = []
    
    try:
        # Each worker opens its own BAM file handle
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for site in sites_chunk:
                counts = process_pileup_site(bam, site, min_mapq, min_bq)
                results.append((site, counts))
    except Exception as e:
        # Return partial results with error information
        console.print(f"[yellow]Warning: Error in worker process: {e}[/yellow]")
        raise
    
    return results


def generate_pileup_data(bam_file: Path, sites: List[SNPSite],
                        min_mapq: int, min_bq: int, ncpus: int,
                        progress: Progress, task_id: TaskID) -> List[Tuple[SNPSite, PileupCounts]]:
    """
    Generate pileup data for all SNP sites using parallel processing.
    
    Sites are divided into chunks and processed in parallel by multiple worker processes.
    Each worker opens its own BAM file handle to avoid threading issues. Chunk size is
    automatically calculated to balance memory usage and parallelization efficiency.
    
    Args:
        bam_file (Path): Path to BAM file
        sites (List[SNPSite]): List of SNP sites to process
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        ncpus (int): Number of parallel processes to use
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        List[Tuple[SNPSite, PileupCounts]]: List of (site, counts) pairs
        
    Raises:
        FileNotFoundError: If BAM file doesn't exist
        ValueError: If BAM file is invalid
    """
    if not bam_file.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_file}")
    
    if len(sites) == 0:
        console.print("[yellow]Warning: No sites to process[/yellow]")
        return []
    
    progress.update(task_id, description=f"Processing pileup data with {ncpus} workers...")
    
    try:
        # Calculate chunk size: aim for at least 2-4 chunks per CPU for load balancing
        # but not too small to avoid overhead
        min_chunk_size = 50  # Minimum sites per chunk
        target_chunks = ncpus * 3  # 3 chunks per CPU for good load balancing
        chunk_size = max(min_chunk_size, len(sites) // target_chunks)
        
        # Split sites into chunks
        site_chunks = [sites[i:i + chunk_size] for i in range(0, len(sites), chunk_size)]
        
        console.print(f"[blue]Processing {len(sites):,} sites in {len(site_chunks)} chunks "
                     f"(~{chunk_size} sites/chunk) using {ncpus} workers[/blue]")
        
        results = []
        completed_sites = 0
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=ncpus) as executor:
            # Submit all chunks to the pool
            future_to_chunk = {
                executor.submit(process_sites_chunk, bam_file, chunk, min_mapq, min_bq): chunk
                for chunk in site_chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    completed_sites += len(chunk)
                    
                    # Update progress based on completed sites
                    progress_pct = (completed_sites / len(sites)) * 100
                    progress.update(task_id, completed=progress_pct)
                    
                except Exception as e:
                    console.print(f"[red]Error processing chunk of {len(chunk)} sites: {e}[/red]")
                    raise
        
        progress.update(task_id, completed=100)
        console.print(f"[green]✓[/green] Processed pileup data for {len(results):,} sites")
        
        return results
        
    except Exception as e:
        console.print(f"[red]Error processing BAM file: {e}[/red]")
        raise


def save_pileup_output(results: List[Tuple[SNPSite, PileupCounts]], output_prefix: str,
                      progress: Progress, task_id: TaskID) -> Path:
    """
    Save pileup data to compressed TSV file.
    
    Args:
        results (List[Tuple[SNPSite, PileupCounts]]): Pileup results
        output_prefix (str): Output file prefix
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        Path: Path to saved output file
    """
    output_file = Path(f"{output_prefix}_pileup.tsv.gz")
    
    progress.update(task_id, description="Saving pileup data...")
    
    try:
        # Sort results by genomic coordinate for deterministic output
        def sort_key(item):
            site, _ = item
            # Natural chromosome sorting (chr1, chr2, ..., chr10, ..., chrX, chrY)
            chr_name = site.chr
            if chr_name.startswith('chr'):
                chr_part = chr_name[3:]
            else:
                chr_part = chr_name
            
            # Try to convert to int for numeric chromosomes
            try:
                chr_num = int(chr_part)
                return (0, chr_num, site.pos)  # Numeric chromosomes first
            except ValueError:
                return (1, chr_part, site.pos)  # Non-numeric chromosomes second
        
        sorted_results = sorted(results, key=sort_key)
        
        # Write header and data
        with gzip.open(output_file, 'wt') as f:
            # Write header
            header = ['chr', 'pos', 'ref', 'alt', 'af', 
                     'cfDNA_ref_reads', 'cfDNA_alt_reads', 'current_depth']
            f.write('\t'.join(header) + '\n')
            
            # Write data rows
            for site, counts in sorted_results:
                row = [
                    site.chr,
                    str(site.pos),
                    site.ref,
                    site.alt,
                    f"{site.af:.6f}",
                    f"{counts.cfDNA_ref_reads}",
                    f"{counts.cfDNA_alt_reads}",
                    f"{counts.current_depth}"
                ]
                f.write('\t'.join(row) + '\n')
        
        progress.update(task_id, advance=100)
        console.print(f"[green]✓[/green] Pileup data saved to: {output_file}")
        
        return output_file
        
    except Exception as e:
        console.print(f"[red]Error saving output file: {e}[/red]")
        raise


@click.command()
@click.option(
    '--input-bam',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to input BAM file from bisulfite sequencing'
)
@click.option(
    '--known-sites',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to known SNP sites TSV file (VCF-like format)'
)
@click.option(
    '--output',
    required=True,
    type=str,
    help='Output file prefix (will create {prefix}_pileup.tsv.gz)'
)
@click.option(
    '--min-mapq',
    default=20,
    type=int,
    help='Minimum mapping quality threshold (default: 20)'
)
@click.option(
    '--min-bq',
    default=13,
    type=int,
    help='Minimum base quality threshold (default: 13)'
)
@click.option(
    '--ncpus',
    default=8,
    type=int,
    help='Number of parallel processes to use (default: 8)'
)
def main(input_bam: Path, known_sites: Path,
         output: str, min_mapq: int, min_bq: int, ncpus: int) -> None:
    """
    Generate pileup data from bisulfite sequencing BAM files with strand-aware filtering.
    
    This tool processes BAM files from bisulfite methylation sequencing to compute
    raw reference and alternate allele counts at known SNP sites. It applies
    bisulfite-specific strand filtering to avoid counting chemistry-induced variations:
    
    - For C↔T SNPs: only counts reverse strand reads (flags 16/83/163)
    - For G↔A SNPs: only counts forward strand reads (flags 0/99/147)
    - For other SNPs: counts all strand reads
    
    This filtering is essential because bisulfite conversion introduces:
    - C→T changes on forward strand reads
    - G→A changes on reverse strand reads
    
    The tool also applies additional methylation chemistry base classification rules
    and quality filtering to produce accurate pileup statistics for bisulfite
    sequencing analysis workflows.
    """
    console.print("\n[bold blue]Hard Filter BAM to Pileup Converter (Bisulfite-Aware)[/bold blue]")
    console.print("="*70)
    
    # Display input parameters
    params_table = Table(title="Input Parameters", show_header=True, header_style="bold magenta")
    params_table.add_column("Parameter", style="cyan", no_wrap=True)
    params_table.add_column("Value", style="white")
    
    params_table.add_row("Input BAM", str(input_bam))
    params_table.add_row("Known Sites", str(known_sites))
    params_table.add_row("Output Prefix", output)
    params_table.add_row("Min MAPQ", str(min_mapq))
    params_table.add_row("Min Base Quality", str(min_bq))
    params_table.add_row("Parallel Workers", str(ncpus))
    params_table.add_row("Bisulfite Filtering", "Enabled (strand-aware)")
    
    console.print(params_table)
    console.print()
    
    # Display bisulfite filtering rules
    console.print("[bold yellow]Bisulfite Conversion Filtering Rules:[/bold yellow]")
    console.print("  • [cyan]C↔T SNPs:[/cyan] Only reverse strand reads (flags 16/83/163)")
    console.print("  • [cyan]G↔A SNPs:[/cyan] Only forward strand reads (flags 0/99/147)")
    console.print("  • [cyan]Other SNPs:[/cyan] All strand reads\n")
    
    try:
        with Progress(console=console) as progress:
            # Create progress tasks
            sites_task = progress.add_task("Parsing known sites...", total=100)
            pileup_task = progress.add_task("Processing pileup...", total=100)
            save_task = progress.add_task("Saving output...", total=100)
            
            # Parse known sites (use all sites, no BED filtering)
            all_sites = parse_known_sites(known_sites, progress, sites_task)
            
            # Generate pileup data with bisulfite-aware filtering and parallel processing
            pileup_results = generate_pileup_data(
                input_bam, all_sites, min_mapq, min_bq, ncpus,
                progress, pileup_task
            )
            
            # Save output
            output_file = save_pileup_output(pileup_results, output, progress, save_task)
        
        # Calculate summary statistics
        total_sites = len(pileup_results)
        if total_sites > 0:
            total_raw_depth = sum(counts.current_depth for _, counts in pileup_results)
            mean_raw_depth = total_raw_depth / total_sites
            
            sites_with_coverage = sum(1 for _, counts in pileup_results 
                                    if counts.current_depth > 0)
            
            # Calculate statistics by SNP type
            ct_snps = sum(1 for site, _ in pileup_results 
                         if (site.ref == 'C' and site.alt == 'T') or (site.ref == 'T' and site.alt == 'C'))
            ga_snps = sum(1 for site, _ in pileup_results 
                         if (site.ref == 'G' and site.alt == 'A') or (site.ref == 'A' and site.alt == 'G'))
            other_snps = total_sites - ct_snps - ga_snps
        else:
            mean_raw_depth = 0.0
            sites_with_coverage = 0
            ct_snps = 0
            ga_snps = 0
            other_snps = 0
        
        # Display summary statistics
        summary_table = Table(title="Processing Summary", show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Count", style="white", justify="right")
        
        summary_table.add_row("Total SNP sites processed", f"{total_sites:,}")
        summary_table.add_row("Sites with coverage", f"{sites_with_coverage:,}")
        summary_table.add_row("Mean depth (unweighted)", f"{mean_raw_depth:.2f}")
        summary_table.add_row("", "")  # Spacer
        summary_table.add_row("[bold]SNP Type Breakdown:[/bold]", "")
        summary_table.add_row("  C↔T SNPs (reverse strand only)", f"{ct_snps:,}")
        summary_table.add_row("  G↔A SNPs (forward strand only)", f"{ga_snps:,}")
        summary_table.add_row("  Other SNPs (all strands)", f"{other_snps:,}")
        
        console.print(summary_table)
        console.print(f"\n[bold green]✓ Processing completed successfully![/bold green]")
        console.print(f"Output file: [cyan]{output_file}[/cyan]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during processing:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

