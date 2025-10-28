#!/usr/bin/env python3
"""
Probability-Weighted BAM to Pileup Converter for Bisulfite Sequencing

This script processes BAM files from bisulfite methylation sequencing along with 
read-specific fetal probability predictions to generate probability-weighted pileup data. 
It computes both fetal-weighted and maternal-weighted reference/alternate allele counts 
for targeted SNP analysis.

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
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, List
from collections import defaultdict
from dataclasses import dataclass

import click
import pandas as pd
import pysam
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich import print as rprint

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
    Data class for storing both probability-weighted and raw pileup counts.
    
    Attributes:
        fetal_ref_reads_from_model (float): Fetal-weighted reference allele count
        fetal_alt_reads_from_model (float): Fetal-weighted alternate allele count
        maternal_ref_reads (float): Maternal-weighted reference allele count
        maternal_alt_reads (float): Maternal-weighted alternate allele count
        cfDNA_ref_reads (int): Raw unweighted reference allele count
        cfDNA_alt_reads (int): Raw unweighted alternate allele count
    """
    fetal_ref_reads_from_model: float = 0.0
    fetal_alt_reads_from_model: float = 0.0
    maternal_ref_reads: float = 0.0
    maternal_alt_reads: float = 0.0
    cfDNA_ref_reads: int = 0
    cfDNA_alt_reads: int = 0
    
    @property
    def fetal_current_depth_from_model(self) -> float:
        """Total fetal-weighted depth."""
        return self.fetal_ref_reads_from_model + self.fetal_alt_reads_from_model
    
    @property
    def maternal_current_depth(self) -> float:
        """Total maternal-weighted depth."""
        return self.maternal_ref_reads + self.maternal_alt_reads
    
    @property
    def current_depth(self) -> int:
        """Total raw unweighted depth."""
        return self.cfDNA_ref_reads + self.cfDNA_alt_reads


def load_probability_table(prob_file: Path, progress: Progress, task_id: TaskID) -> Dict[str, float]:
    """
    Load read probability table and create read name to fetal probability mapping.
    
    Args:
        prob_file (Path): Path to the probability TSV file
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        Dict[str, float]: Mapping from read name to fetal probability [0,1]
        
    Raises:
        FileNotFoundError: If probability file doesn't exist
        ValueError: If required columns are missing or invalid
    """
    if not prob_file.exists():
        raise FileNotFoundError(f"Probability file not found: {prob_file}")
    
    progress.update(task_id, description="Loading probability table...")
    
    try:
        # Read TSV with header, only loading required columns for memory efficiency
        prob_data = pd.read_csv(prob_file, sep='\t', usecols=['name', 'prob_class_1'])
        
        # Validate required columns were successfully loaded
        if 'name' not in prob_data.columns:
            raise ValueError("Required column 'name' not found in probability file")
        if 'prob_class_1' not in prob_data.columns:
            raise ValueError("Required column 'prob_class_1' not found in probability file")
        
        # Build probability mapping, using first occurrence for duplicates
        # Use drop_duplicates for efficiency with large files
        prob_data_dedup = prob_data.drop_duplicates(subset=['name'], keep='first').copy()
        
        # Clamp probabilities to [0, 1] and create mapping
        prob_data_dedup['prob_class_1'] = prob_data_dedup['prob_class_1'].clip(0.0, 1.0)
        prob_map = dict(zip(prob_data_dedup['name'].astype(str), prob_data_dedup['prob_class_1']))
        
        progress.update(task_id, advance=100)
        console.print(f"[green]✓[/green] Loaded probabilities for {len(prob_map):,} reads")
        
        return prob_map
        
    except Exception as e:
        console.print(f"[red]Error loading probability file: {e}[/red]")
        raise


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
                       prob_map: Dict[str, float], min_mapq: int, min_bq: int) -> PileupCounts:
    """
    Process a single SNP site to compute probability-weighted pileup counts with bisulfite-aware flag filtering.
    
    For bisulfite sequencing data, reads are filtered by SAM flags based on SNP type
    to avoid counting chemistry-induced base changes:
    - C↔T SNPs: only count reverse strand reads (flags 16/83/163)
    - G↔A SNPs: only count forward strand reads (flags 0/99/147)
    - Other SNPs: count all strand reads
    
    Args:
        bam_file (pysam.AlignmentFile): Opened BAM file
        site (SNPSite): SNP site to process
        prob_map (Dict[str, float]): Read name to fetal probability mapping
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        
    Returns:
        PileupCounts: Probability-weighted counts for this site
        
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
                
                # Determine effective flag for bisulfite filtering
                # We care about the strand orientation
                if read.is_reverse:
                    # Reverse strand alignment
                    if read.is_read1:
                        effective_flag = 83  # First in pair, reverse strand
                    elif read.is_read2:
                        effective_flag = 163  # Second in pair, reverse strand (but forward original)
                    else:
                        effective_flag = 16  # Single-end or unmapped pair, reverse strand
                else:
                    # Forward strand alignment
                    if read.is_read1:
                        effective_flag = 99  # First in pair, forward strand
                    elif read.is_read2:
                        effective_flag = 147  # Second in pair, forward strand (but reverse original)
                    else:
                        effective_flag = 0  # Single-end or unmapped pair, forward strand
                
                # Apply flag filtering if this SNP type requires it
                if allowed_flags is not None:
                    if effective_flag not in allowed_flags:
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
                
                # Get fetal probability for this read
                fetal_prob = prob_map.get(template_name, 0.0)
                maternal_prob = 1.0 - fetal_prob
                
                # Update counts based on classification
                if classification == 'REF':
                    counts.fetal_ref_reads_from_model += fetal_prob
                    counts.maternal_ref_reads += maternal_prob
                    counts.cfDNA_ref_reads += 1  # Raw unweighted count
                elif classification == 'ALT':
                    counts.fetal_alt_reads_from_model += fetal_prob
                    counts.maternal_alt_reads += maternal_prob
                    counts.cfDNA_alt_reads += 1  # Raw unweighted count
            
            break  # We only need the column at our target position
            
    except Exception as e:
        console.print(f"[yellow]Warning: Error processing site {site.chr}:{site.pos}: {e}[/yellow]")
    
    return counts


def generate_pileup_data(bam_file: Path, sites: List[SNPSite], prob_map: Dict[str, float],
                        min_mapq: int, min_bq: int,
                        progress: Progress, task_id: TaskID) -> List[Tuple[SNPSite, PileupCounts]]:
    """
    Generate probability-weighted pileup data for all SNP sites.
    
    Args:
        bam_file (Path): Path to BAM file
        sites (List[SNPSite]): List of SNP sites to process
        prob_map (Dict[str, float]): Read name to fetal probability mapping
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
                
                counts = process_pileup_site(bam, site, prob_map, min_mapq, min_bq)
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
    Save probability-weighted pileup data to compressed TSV file.
    
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
                     'fetal_ref_reads_from_model', 'fetal_alt_reads_from_model', 'fetal_current_depth_from_model',
                     'maternal_ref_reads', 'maternal_alt_reads', 'maternal_current_depth',
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
                    f"{counts.fetal_ref_reads_from_model:.6f}",
                    f"{counts.fetal_alt_reads_from_model:.6f}",
                    f"{counts.fetal_current_depth_from_model:.6f}",
                    f"{counts.maternal_ref_reads:.6f}",
                    f"{counts.maternal_alt_reads:.6f}",
                    f"{counts.maternal_current_depth:.6f}",
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
    help='Path to input BAM file from bisulfite sequencing (read names without /1 or /2 suffixes)'
)
@click.option(
    '--input-txt',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to read probability TSV file with "name" and "prob_class_1" columns'
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
def main(input_bam: Path, input_txt: Path, known_sites: Path,
         output: str, min_mapq: int, min_bq: int) -> None:
    """
    Generate probability-weighted pileup data from bisulfite sequencing BAM files with strand-aware filtering.
    
    This tool processes BAM files from bisulfite methylation sequencing along with read-specific
    fetal probability predictions to compute probability-weighted pileup counts. It applies
    bisulfite-specific strand filtering to avoid counting chemistry-induced variations:
    
    - For C↔T SNPs: only counts reverse strand reads (flags 16/83/163)
    - For G↔A SNPs: only counts forward strand reads (flags 0/99/147)
    - For other SNPs: counts all strand reads
    
    This filtering is essential because bisulfite conversion introduces:
    - C→T changes on forward strand reads
    - G→A changes on reverse strand reads
    
    The tool also applies additional methylation chemistry base classification rules
    and quality filtering. The output includes separate probability-weighted counts for 
    fetal and maternal contributions, enabling more accurate cell-free DNA analysis workflows.
    """
    console.print("\n[bold blue]Probability-Weighted BAM to Pileup Converter (Bisulfite-Aware)[/bold blue]")
    console.print("="*75)
    
    # Display input parameters
    params_table = Table(title="Input Parameters", show_header=True, header_style="bold magenta")
    params_table.add_column("Parameter", style="cyan", no_wrap=True)
    params_table.add_column("Value", style="white")
    
    params_table.add_row("Input BAM", str(input_bam))
    params_table.add_row("Probability Table", str(input_txt))
    params_table.add_row("Known Sites", str(known_sites))
    params_table.add_row("Output Prefix", output)
    params_table.add_row("Min MAPQ", str(min_mapq))
    params_table.add_row("Min Base Quality", str(min_bq))
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
            prob_task = progress.add_task("Loading probability table...", total=100)
            sites_task = progress.add_task("Parsing known sites...", total=100)
            pileup_task = progress.add_task("Processing pileup...", total=100)
            save_task = progress.add_task("Saving output...", total=100)
            
            # Load probability table
            prob_map = load_probability_table(input_txt, progress, prob_task)
            
            # Parse known sites (use all sites, no BED filtering)
            all_sites = parse_known_sites(known_sites, progress, sites_task)
            
            # Generate pileup data with bisulfite-aware filtering
            pileup_results = generate_pileup_data(
                input_bam, all_sites, prob_map, min_mapq, min_bq,
                progress, pileup_task
            )
            
            # Save output
            output_file = save_pileup_output(pileup_results, output, progress, save_task)
        
        # Calculate summary statistics
        total_sites = len(pileup_results)
        if total_sites > 0:
            total_fetal_depth = sum(counts.fetal_current_depth_from_model for _, counts in pileup_results)
            total_maternal_depth = sum(counts.maternal_current_depth for _, counts in pileup_results)
            total_raw_depth = sum(counts.current_depth for _, counts in pileup_results)
            mean_fetal_depth = total_fetal_depth / total_sites
            mean_maternal_depth = total_maternal_depth / total_sites
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
            mean_fetal_depth = mean_maternal_depth = mean_raw_depth = 0.0
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
        summary_table.add_row("Mean fetal depth (weighted)", f"{mean_fetal_depth:.2f}")
        summary_table.add_row("Mean maternal depth (weighted)", f"{mean_maternal_depth:.2f}")
        summary_table.add_row("Mean raw depth (unweighted)", f"{mean_raw_depth:.2f}")
        summary_table.add_row("Read probabilities loaded", f"{len(prob_map):,}")
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
