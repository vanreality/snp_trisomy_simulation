#!/usr/bin/env python3
"""
Hard Filter BAM to Pileup Converter

This script processes BAM files of cfDNA reads to generate simple pileup data
at targeted SNP sites. It performs basic read counting without probability weighting,
reporting raw reference and alternate allele counts.

The script handles methylation chemistry artifacts and applies quality filtering to produce
accurate pileup statistics for cell-free DNA analysis workflows.
"""

import sys
import gzip
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

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


def filter_sites_by_bed(sites: List[SNPSite], bed_file: Optional[Path], 
                       progress: Progress, task_id: TaskID) -> List[SNPSite]:
    """
    Filter SNP sites by BED file regions if provided.
    
    Args:
        sites (List[SNPSite]): List of SNP sites to filter
        bed_file (Optional[Path]): Path to BED file, or None to skip filtering
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        List[SNPSite]: Filtered list of SNP sites
        
    Raises:
        FileNotFoundError: If BED file doesn't exist
    """
    if bed_file is None:
        progress.update(task_id, advance=50)
        console.print("[blue]No BED file provided, using all sites[/blue]")
        return sites
    
    if not bed_file.exists():
        raise FileNotFoundError(f"BED file not found: {bed_file}")
    
    progress.update(task_id, description="Filtering sites by BED file...")
    
    try:
        # Load BED regions into a set for efficient lookup
        bed_regions_data = pd.read_csv(bed_file, sep='\t', header=None, names=['chr', 'start', 'end'])
        
        chr_name = bed_regions_data['chr']
        pos = bed_regions_data['end']

        bed_regions = set(zip(chr_name, pos))
        
        # Filter sites that overlap with BED regions
        filtered_sites = []
        for site in sites:
            if (site.chr, site.pos) in bed_regions:
                filtered_sites.append(site)
        
        progress.update(task_id, advance=50)
        console.print(f"[green]✓[/green] Filtered to {len(filtered_sites):,} sites within BED regions")
        
        return filtered_sites
        
    except Exception as e:
        console.print(f"[red]Error filtering sites by BED file: {e}[/red]")
        raise


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
    Process a single SNP site to compute raw pileup counts.
    
    Args:
        bam_file (pysam.AlignmentFile): Opened BAM file
        site (SNPSite): SNP site to process
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        
    Returns:
        PileupCounts: Raw counts for this site
    """
    counts = PileupCounts()
    processed_templates = set()  # Track processed template names to avoid double-counting
    
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


def generate_pileup_data(bam_file: Path, sites: List[SNPSite],
                        min_mapq: int, min_bq: int,
                        progress: Progress, task_id: TaskID) -> List[Tuple[SNPSite, PileupCounts]]:
    """
    Generate pileup data for all SNP sites.
    
    Args:
        bam_file (Path): Path to BAM file
        sites (List[SNPSite]): List of SNP sites to process
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
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
    
    progress.update(task_id, description="Processing pileup data...")
    
    try:
        # Open BAM file
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            results = []
            
            # Process each site
            for i, site in enumerate(sites):
                if i % 100 == 0:  # Update progress every 100 sites
                    progress.update(task_id, advance=100 * 100 / len(sites) if len(sites) > 0 else 0)
                
                counts = process_pileup_site(bam, site, min_mapq, min_bq)
                results.append((site, counts))
            
            progress.update(task_id, advance=100)
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
    help='Path to input BAM file'
)
@click.option(
    '--known-sites',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to known SNP sites TSV file (VCF-like format)'
)
@click.option(
    '--bed',
    type=click.Path(exists=True, path_type=Path),
    help='Optional BED file to subset SNPs to specific regions'
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
def main(input_bam: Path, known_sites: Path,
         bed: Optional[Path], output: str, min_mapq: int, min_bq: int) -> None:
    """
    Generate simple pileup data from cfDNA BAM files using hard filtering.
    
    This tool processes BAM files of cell-free DNA reads to compute raw
    reference and alternate allele counts at known SNP sites. It handles
    methylation chemistry artifacts and produces unweighted pileup counts
    for cell-free DNA analysis workflows.
    
    Unlike the probability-weighted version, this tool does not require
    read-level probability predictions and reports only raw allele counts.
    """
    console.print("\n[bold blue]Hard Filter BAM to Pileup Converter[/bold blue]")
    console.print("="*60)
    
    # Display input parameters
    params_table = Table(title="Input Parameters", show_header=True, header_style="bold magenta")
    params_table.add_column("Parameter", style="cyan", no_wrap=True)
    params_table.add_column("Value", style="white")
    
    params_table.add_row("Input BAM", str(input_bam))
    params_table.add_row("Known Sites", str(known_sites))
    params_table.add_row("BED Filter", str(bed) if bed else "None")
    params_table.add_row("Output Prefix", output)
    params_table.add_row("Min MAPQ", str(min_mapq))
    params_table.add_row("Min Base Quality", str(min_bq))
    
    console.print(params_table)
    console.print()
    
    try:
        with Progress(console=console) as progress:
            # Create progress tasks
            sites_task = progress.add_task("Parsing known sites...", total=100)
            filter_task = progress.add_task("Filtering sites...", total=100)
            pileup_task = progress.add_task("Processing pileup...", total=100)
            save_task = progress.add_task("Saving output...", total=100)
            
            # Parse known sites
            all_sites = parse_known_sites(known_sites, progress, sites_task)
            
            # Filter sites by BED file if provided
            filtered_sites = filter_sites_by_bed(all_sites, bed, progress, filter_task)
            
            # Generate pileup data
            pileup_results = generate_pileup_data(
                input_bam, filtered_sites, min_mapq, min_bq,
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
        else:
            mean_raw_depth = 0.0
            sites_with_coverage = 0
        
        # Display summary statistics
        summary_table = Table(title="Processing Summary", show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Count", style="white", justify="right")
        
        summary_table.add_row("Total known sites", f"{len(all_sites):,}")
        summary_table.add_row("Sites after BED filtering", f"{len(filtered_sites):,}")
        summary_table.add_row("Final pileup entries", f"{total_sites:,}")
        summary_table.add_row("Sites with coverage", f"{sites_with_coverage:,}")
        summary_table.add_row("Mean depth (unweighted)", f"{mean_raw_depth:.2f}")
        
        console.print(summary_table)
        console.print(f"\n[bold green]✓ Processing completed successfully![/bold green]")
        console.print(f"Output file: [cyan]{output_file}[/cyan]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during processing:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

