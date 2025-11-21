#!/usr/bin/env python3
"""
Extract Ground Truth for Fetal Read Classification

This script extracts ground truth labels for fetal reads from pre-split bisulfite sequencing 
BAM files (target and background). The BAM files have been split based on probability threshold:
- Target BAM: reads with prob_class_1 > 0.5 (predicted as fetal)
- Background BAM: reads with prob_class_1 <= 0.5 (predicted as maternal)

The script uses a filtered pileup file containing SNPs where maternal genotype is homozygous 
and fetal genotype is heterozygous, allowing identification of true fetal reads based on which
allele they support.

For SNPs with raw_vaf in (0, 0.2): ALT-supporting reads are from fetus
For SNPs with raw_vaf in (0.8, 1.0): REF-supporting reads are from fetus

Ground truth labels:
- Reads from target BAM supporting fetal allele: classified_label = 1 (correct classification)
- Reads from background BAM supporting fetal allele: classified_label = 0 (incorrect classification)

The script matches these fetal reads with their predicted probabilities (from input txt file)
to create a ground truth dataset for evaluating fetal read classification models.
"""

import sys
import gzip
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Data class representing a SNP site from filtered pileup.
    
    Attributes:
        chr (str): Chromosome name
        pos (int): 1-based genomic position
        ref (str): Reference allele (single nucleotide)
        alt (str): Alternate allele (single nucleotide)
        af (float): Allele frequency
        raw_vaf (float): Raw variant allele frequency
        target_vaf (float): Target (fetal) VAF
        background_vaf (float): Background (maternal) VAF
        fetal_allele (str): Which allele is fetal ('REF' or 'ALT')
    """
    chr: str
    pos: int
    ref: str
    alt: str
    af: float
    raw_vaf: float
    target_vaf: float
    background_vaf: float
    fetal_allele: str


@dataclass
class ReadRecord:
    """
    Data class for a single read record.
    
    Attributes:
        chr (str): Chromosome name
        pos (int): SNP position
        ref (str): Reference allele
        alt (str): Alternate allele
        af (float): Allele frequency
        raw_vaf (float): Raw variant allele frequency
        target_vaf (float): Target (fetal) VAF
        background_vaf (float): Background (maternal) VAF
        name (str): Read name
        prob_class_1 (float): Probability of being fetal
        support_base (str): Base supported by this read ('REF' or 'ALT')
        classified_label (int): 1 if correctly classified, 0 otherwise
    """
    chr: str
    pos: int
    ref: str
    alt: str
    af: float
    raw_vaf: float
    target_vaf: float
    background_vaf: float
    name: str
    prob_class_1: float
    support_base: str
    classified_label: int


def load_probability_table(prob_file: Path, progress: Progress, task_id: TaskID) -> Dict[str, float]:
    """
    Load read probability table and create read name to fetal probability mapping.
    Uses chunked reading for memory efficiency with large files.
    
    Args:
        prob_file (Path): Path to the probability TSV file
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        Dict[str, float]: Mapping from read name to fetal probability [0,1]
    """
    if not prob_file.exists():
        raise FileNotFoundError(f"Probability file not found: {prob_file}")
    
    progress.update(task_id, description="Loading probability table...")
    
    try:
        prob_map = {}
        chunk_size = 100000  # Process 100K rows at a time
        
        # Define dtype to use less memory (float32 instead of float64)
        dtype_dict = {'name': str, 'prob_class_1': 'float32'}
        
        # Read file in chunks to avoid memory issues
        chunk_iter = pd.read_csv(
            prob_file, 
            sep='\t', 
            usecols=['name', 'prob_class_1'],
            dtype=dtype_dict,
            chunksize=chunk_size,
            engine='c'  # Use faster C engine
        )
        
        total_rows_processed = 0
        for chunk in chunk_iter:
            # Validate required columns (only need to check first chunk)
            if total_rows_processed == 0:
                if 'name' not in chunk.columns:
                    raise ValueError("Required column 'name' not found in probability file")
                if 'prob_class_1' not in chunk.columns:
                    raise ValueError("Required column 'prob_class_1' not found in probability file")
            
            # Drop duplicates within chunk (keep first occurrence)
            chunk_dedup = chunk.drop_duplicates(subset=['name'], keep='first')
            
            # Clip probabilities to [0, 1]
            chunk_dedup['prob_class_1'] = chunk_dedup['prob_class_1'].clip(0.0, 1.0)
            
            # Update dictionary (only add if not already present - first occurrence wins)
            for name, prob in zip(chunk_dedup['name'], chunk_dedup['prob_class_1']):
                if name not in prob_map:
                    prob_map[name] = float(prob)
            
            total_rows_processed += len(chunk)
            
            # Update progress periodically
            if total_rows_processed % (chunk_size * 10) == 0:
                progress.update(task_id, advance=10)
        
        progress.update(task_id, completed=100)
        console.print(f"[green]✓[/green] Loaded probabilities for {len(prob_map):,} unique reads")
        console.print(f"[blue]Processed {total_rows_processed:,} total rows[/blue]")
        
        return prob_map
        
    except Exception as e:
        console.print(f"[red]Error loading probability file: {e}[/red]")
        raise


def parse_filtered_pileup(pileup_file: Path, progress: Progress, task_id: TaskID) -> List[SNPSite]:
    """
    Parse filtered pileup file to extract SNP sites with ground truth information.
    
    The filtered pileup contains SNPs where maternal is homozygous and fetal is heterozygous.
    Based on raw_vaf, we can determine which allele is fetal.
    
    Args:
        pileup_file (Path): Path to filtered pileup TSV file
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        List[SNPSite]: List of SNP sites with ground truth labels
    """
    if not pileup_file.exists():
        raise FileNotFoundError(f"Filtered pileup file not found: {pileup_file}")
    
    progress.update(task_id, description="Parsing filtered pileup...")
    
    try:
        # Read filtered pileup file
        pileup_data = pd.read_csv(pileup_file, sep='\t')
        
        # Validate required columns
        required_cols = ['chr', 'pos', 'ref', 'alt', 'af', 'raw_vaf', 'target_vaf', 'background_vaf']
        for col in required_cols:
            if col not in pileup_data.columns:
                raise ValueError(f"Required column '{col}' not found in filtered pileup file")
        
        progress.update(task_id, advance=30)
        
        # Convert bases to uppercase
        pileup_data['ref'] = pileup_data['ref'].str.upper()
        pileup_data['alt'] = pileup_data['alt'].str.upper()
        
        # Make a copy to avoid SettingWithCopyWarning
        pileup_data = pileup_data.copy()
        
        # Determine fetal allele based on raw_vaf
        # raw_vaf in (0, 0.2): ALT is fetal (maternal is REF/REF, fetal is REF/ALT)
        # raw_vaf in (0.8, 1.0): REF is fetal (maternal is ALT/ALT, fetal is ALT/REF)
        def determine_fetal_allele(row):
            if 0 < row['raw_vaf'] < 0.2:
                return 'ALT'
            elif 0.8 < row['raw_vaf'] < 1.0:
                return 'REF'
            else:
                return None  # Outside valid ranges
        
        pileup_data['fetal_allele'] = pileup_data.apply(determine_fetal_allele, axis=1)
        
        # Filter to only valid sites
        pileup_data = pileup_data[pileup_data['fetal_allele'].notna()].copy()
        
        progress.update(task_id, advance=40)
        
        # Convert to list of SNPSite objects
        sites = [
            SNPSite(
                chr=row['chr'],
                pos=int(row['pos']),
                ref=row['ref'],
                alt=row['alt'],
                af=float(row['af']),
                raw_vaf=float(row['raw_vaf']),
                target_vaf=float(row['target_vaf']),
                background_vaf=float(row['background_vaf']),
                fetal_allele=row['fetal_allele']
            )
            for _, row in pileup_data.iterrows()
        ]
        
        progress.update(task_id, advance=30)
        
        alt_fetal = sum(1 for s in sites if s.fetal_allele == 'ALT')
        ref_fetal = sum(1 for s in sites if s.fetal_allele == 'REF')
        console.print(f"[green]✓[/green] Parsed {len(sites):,} SNP sites")
        console.print(f"  • {alt_fetal:,} sites with ALT as fetal allele (low VAF)")
        console.print(f"  • {ref_fetal:,} sites with REF as fetal allele (high VAF)")
        
        return sites
        
    except Exception as e:
        console.print(f"[red]Error parsing filtered pileup file: {e}[/red]")
        raise


def get_allowed_flags_for_snp(ref: str, alt: str) -> Optional[set]:
    """
    Determine which SAM flags are allowed for a given SNP type in bisulfite sequencing.
    
    Same logic as in bam_to_pileup scripts.
    """
    ref = ref.upper()
    alt = alt.upper()
    
    forward_flags = {0, 99, 147}
    reverse_flags = {16, 83, 163}
    
    if (ref == 'C' and alt == 'T') or (ref == 'T' and alt == 'C'):
        return reverse_flags
    elif (ref == 'G' and alt == 'A') or (ref == 'A' and alt == 'G'):
        return forward_flags
    else:
        return None


def classify_base_with_methylation(base: str, ref: str, alt: str) -> Optional[str]:
    """
    Classify a sequenced base as REF or ALT considering methylation chemistry.
    
    Same logic as in bam_to_pileup scripts.
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


def extract_site_read_names(bam_file: pysam.AlignmentFile, site: SNPSite,
                           min_mapq: int, min_bq: int) -> set:
    """
    Extract fetal read names from a single SNP site (without probability lookup).
    
    Args:
        bam_file (pysam.AlignmentFile): Opened BAM file
        site (SNPSite): SNP site to process
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        
    Returns:
        set: Set of fetal read names from this site
    """
    read_names = set()
    processed_templates = set()
    
    # Determine which SAM flags are allowed for this SNP type
    allowed_flags = get_allowed_flags_for_snp(site.ref, site.alt)
    
    try:
        # Get pileup column at the specified position
        for pileup_column in bam_file.pileup(
            site.chr,
            site.pos - 1,  # Convert to 0-based for pysam
            site.pos,
            stepper='nofilter',
            ignore_overlaps=True,
            ignore_orphans=True,
            min_base_quality=min_bq,
            min_mapping_quality=min_mapq
        ):
            if pileup_column.pos != site.pos - 1:
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
                read_flag = read.flag
                
                if allowed_flags is not None:
                    if read_flag not in allowed_flags:
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
                
                # Avoid double-counting paired reads
                template_name = read.query_name
                template_key = (template_name, site.chr, site.pos)
                if template_key in processed_templates:
                    continue
                processed_templates.add(template_key)
                
                # Classify base according to methylation chemistry rules
                classification = classify_base_with_methylation(query_base, site.ref, site.alt)
                if classification is None:
                    continue
                
                # Only keep reads that support the fetal allele
                if classification == site.fetal_allele:
                    read_names.add(template_name)
            
            break  # We only need the column at our target position
            
    except Exception as e:
        pass  # Silently skip errors during name extraction
    
    return read_names


def extract_site_reads(bam_file: pysam.AlignmentFile, site: SNPSite,
                      prob_map: Dict[str, float], bam_source: str,
                      min_mapq: int, min_bq: int) -> List[ReadRecord]:
    """
    Extract fetal reads from a single SNP site.
    
    Based on the site's fetal_allele, only reads supporting that allele are fetal reads.
    
    Args:
        bam_file (pysam.AlignmentFile): Opened BAM file
        site (SNPSite): SNP site to process
        prob_map (Dict[str, float]): Read name to fetal probability mapping (from input txt)
        bam_source (str): 'target' or 'background' indicating which BAM this is
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        
    Returns:
        List[ReadRecord]: List of fetal read records from this site
    """
    records = []
    processed_templates = set()
    
    # Determine which SAM flags are allowed for this SNP type
    allowed_flags = get_allowed_flags_for_snp(site.ref, site.alt)
    
    try:
        # Get pileup column at the specified position
        for pileup_column in bam_file.pileup(
            site.chr,
            site.pos - 1,  # Convert to 0-based for pysam
            site.pos,
            stepper='nofilter',
            ignore_overlaps=True,
            ignore_orphans=True,
            min_base_quality=min_bq,
            min_mapping_quality=min_mapq
        ):
            if pileup_column.pos != site.pos - 1:
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
                read_flag = read.flag
                
                if allowed_flags is not None:
                    if read_flag not in allowed_flags:
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
                
                # Avoid double-counting paired reads
                template_name = read.query_name
                template_key = (template_name, site.chr, site.pos)
                if template_key in processed_templates:
                    continue
                processed_templates.add(template_key)
                
                # Classify base according to methylation chemistry rules
                classification = classify_base_with_methylation(query_base, site.ref, site.alt)
                if classification is None:
                    continue
                
                # Only keep reads that support the fetal allele
                if classification == site.fetal_allele:
                    # This is a fetal read
                    # Get probability from map
                    prob_class_1 = prob_map.get(template_name, None)
                    
                    # If not found, use default (should rarely happen if filtering worked)
                    if prob_class_1 is None:
                        prob_class_1 = 0.75 if bam_source == 'target' else 0.25
                    
                    # Classified label based on which BAM the read came from
                    # target BAM = correctly classified (1), background BAM = incorrectly classified (0)
                    classified_label = 1 if bam_source == 'target' else 0
                    
                    record = ReadRecord(
                        chr=site.chr,
                        pos=site.pos,
                        ref=site.ref,
                        alt=site.alt,
                        af=site.af,
                        raw_vaf=site.raw_vaf,
                        target_vaf=site.target_vaf,
                        background_vaf=site.background_vaf,
                        name=template_name,
                        prob_class_1=prob_class_1,
                        support_base=classification,
                        classified_label=classified_label
                    )
                    records.append(record)
            
            break  # We only need the column at our target position
            
    except Exception as e:
        console.print(f"[yellow]Warning: Error processing site {site.chr}:{site.pos}: {e}[/yellow]")
    
    return records


def process_sites_chunk(bam_path: Path, sites_chunk: List[SNPSite],
                       prob_map: Dict[str, float], bam_source: str,
                       min_mapq: int, min_bq: int, debug_first_chunk: bool = False) -> List[ReadRecord]:
    """
    Process a chunk of SNP sites in a worker process.
    
    Args:
        bam_path (Path): Path to BAM file
        sites_chunk (List[SNPSite]): Chunk of SNP sites to process
        prob_map (Dict[str, float]): Read name to fetal probability mapping
        bam_source (str): 'target' or 'background' indicating which BAM this is
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        debug_first_chunk (bool): Whether to print debug info
        
    Returns:
        List[ReadRecord]: List of read records from this chunk
    """
    records = []
    
    # Debug first chunk
    if debug_first_chunk:
        console.print(f"[blue]Worker received prob_map with {len(prob_map):,} entries[/blue]")
        if prob_map:
            sample_keys = list(prob_map.keys())[:3]
            console.print(f"[blue]Sample prob_map keys: {sample_keys}[/blue]")
    
    try:
        # Each worker opens its own BAM file handle
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for site in sites_chunk:
                site_records = extract_site_reads(bam, site, prob_map, bam_source, min_mapq, min_bq)
                records.extend(site_records)
                
                # Debug first site
                if debug_first_chunk and len(records) > 0 and len(records) <= 5:
                    console.print(f"[blue]Sample read from {bam_source}: {records[-1].name}, prob={records[-1].prob_class_1}[/blue]")
                    
    except Exception as e:
        console.print(f"[yellow]Warning: Error in worker process: {e}[/yellow]")
        raise
    
    return records


def collect_names_from_chunk(bam_path: Path, sites_chunk: List[SNPSite],
                             min_mapq: int, min_bq: int) -> set:
    """Worker function to collect read names from a chunk of sites."""
    all_names = set()
    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for site in sites_chunk:
                site_names = extract_site_read_names(bam, site, min_mapq, min_bq)
                all_names.update(site_names)
    except Exception as e:
        # Silently continue on errors during collection
        pass
    return all_names


def collect_fetal_read_names(target_bam: Path, background_bam: Path, sites: List[SNPSite],
                             min_mapq: int, min_bq: int, ncpus: int,
                             progress: Progress, task_id: TaskID) -> set:
    """
    Collect all fetal read names from SNP sites across both BAM files.
    This is done as a first pass to filter the probability table efficiently.
    
    Args:
        target_bam (Path): Path to target BAM file
        background_bam (Path): Path to background BAM file
        sites (List[SNPSite]): List of SNP sites to process
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        ncpus (int): Number of parallel processes to use
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        set: Set of all fetal read names
    """
    progress.update(task_id, description=f"Collecting read names with {ncpus} workers...")
    
    try:
        # Calculate chunk size
        min_chunk_size = 50
        target_chunks = ncpus * 3
        chunk_size = max(min_chunk_size, len(sites) // target_chunks)
        
        # Split sites into chunks
        site_chunks = [sites[i:i + chunk_size] for i in range(0, len(sites), chunk_size)]
        
        console.print(f"[blue]Collecting read names from {len(sites):,} sites in {len(site_chunks)} chunks...[/blue]")
        
        all_read_names = set()
        completed_sites = 0
        total_sites = len(sites) * 2  # Both BAMs
        
        # Collect names in parallel
        with ProcessPoolExecutor(max_workers=ncpus) as executor:
            # Submit chunks for both BAMs
            futures = []
            for chunk in site_chunks:
                futures.append(executor.submit(collect_names_from_chunk, target_bam, chunk, min_mapq, min_bq))
                futures.append(executor.submit(collect_names_from_chunk, background_bam, chunk, min_mapq, min_bq))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_names = future.result()
                    all_read_names.update(chunk_names)
                    completed_sites += len(sites) // len(site_chunks)
                    
                    progress_pct = min(95, (completed_sites / total_sites) * 100)
                    progress.update(task_id, completed=progress_pct)
                    
                except Exception as e:
                    console.print(f"[red]Error collecting names: {e}[/red]")
        
        progress.update(task_id, completed=100)
        console.print(f"[green]✓[/green] Collected {len(all_read_names):,} unique fetal read names")
        
        return all_read_names
        
    except Exception as e:
        console.print(f"[red]Error collecting read names: {e}[/red]")
        raise


def filter_probability_map(prob_file: Path, needed_reads: set,
                           progress: Progress, task_id: TaskID) -> Dict[str, float]:
    """
    Filter probability table to only include reads that are actually needed.
    Uses chunked reading for memory efficiency.
    
    Args:
        prob_file (Path): Path to the probability TSV file
        needed_reads (set): Set of read names that are actually needed
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        Dict[str, float]: Filtered mapping from read name to fetal probability
    """
    progress.update(task_id, description="Filtering probability table...")
    
    try:
        prob_map = {}
        chunk_size = 100000
        
        # Define dtype to use less memory
        dtype_dict = {'name': str, 'prob_class_1': 'float32'}
        
        # Read file in chunks
        chunk_iter = pd.read_csv(
            prob_file, 
            sep='\t', 
            usecols=['name', 'prob_class_1'],
            dtype=dtype_dict,
            chunksize=chunk_size,
            engine='c'
        )
        
        total_rows_processed = 0
        total_matches = 0
        
        # Debug: sample some needed reads
        sample_needed = list(needed_reads)[:5] if needed_reads else []
        console.print(f"[blue]Sample needed read names: {sample_needed}[/blue]")
        
        for chunk in chunk_iter:
            # Debug first chunk
            if total_rows_processed == 0:
                sample_chunk = chunk['name'].head(5).tolist()
                console.print(f"[blue]Sample read names from prob file: {sample_chunk}[/blue]")
            
            # Filter to only needed reads
            chunk_filtered = chunk[chunk['name'].isin(needed_reads)]
            matches_in_chunk = len(chunk_filtered)
            total_matches += matches_in_chunk
            
            # Clip probabilities to [0, 1]
            if len(chunk_filtered) > 0:
                chunk_filtered = chunk_filtered.copy()
                chunk_filtered['prob_class_1'] = chunk_filtered['prob_class_1'].clip(0.0, 1.0)
                
                # Update dictionary (first occurrence wins)
                for name, prob in zip(chunk_filtered['name'], chunk_filtered['prob_class_1']):
                    if name not in prob_map:
                        prob_map[name] = float(prob)
            
            total_rows_processed += len(chunk)
            
            # Update progress periodically
            if total_rows_processed % (chunk_size * 10) == 0:
                progress.update(task_id, advance=5)
        
        progress.update(task_id, completed=100)
        console.print(f"[green]✓[/green] Filtered to {len(prob_map):,} needed read probabilities (from {total_rows_processed:,} total rows)")
        console.print(f"[blue]Total matches found during filtering: {total_matches:,}[/blue]")
        
        # Warn if many needed reads are missing
        missing_reads = len(needed_reads) - len(prob_map)
        if missing_reads > 0:
            match_rate = (len(prob_map) / len(needed_reads) * 100) if needed_reads else 0
            console.print(f"[yellow]⚠[/yellow] {missing_reads:,} reads not found in probability file (match rate: {match_rate:.2f}%)")
        
        return prob_map
        
    except Exception as e:
        console.print(f"[red]Error filtering probability file: {e}[/red]")
        raise


def extract_ground_truth(target_bam: Path, background_bam: Path, sites: List[SNPSite], 
                         prob_map: Dict[str, float], min_mapq: int, min_bq: int, ncpus: int,
                         progress: Progress, task_id: TaskID) -> Tuple[List[ReadRecord], Dict[str, int]]:
    """
    Extract ground truth read records from all SNP sites using parallel processing.
    Processes both target and background BAM files.
    
    Args:
        target_bam (Path): Path to target BAM file (predicted fetal reads)
        background_bam (Path): Path to background BAM file (predicted maternal reads)
        sites (List[SNPSite]): List of SNP sites to process
        prob_map (Dict[str, float]): Read name to fetal probability mapping
        min_mapq (int): Minimum mapping quality threshold
        min_bq (int): Minimum base quality threshold
        ncpus (int): Number of parallel processes to use
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        Tuple[List[ReadRecord], Dict[str, int]]: List of all read records and statistics
    """
    if not target_bam.exists():
        raise FileNotFoundError(f"Target BAM file not found: {target_bam}")
    if not background_bam.exists():
        raise FileNotFoundError(f"Background BAM file not found: {background_bam}")
    
    if len(sites) == 0:
        console.print("[yellow]Warning: No sites to process[/yellow]")
        return [], {}
    
    progress.update(task_id, description=f"Extracting reads with {ncpus} workers...")
    
    console.print(f"[blue]Probability map contains {len(prob_map):,} entries[/blue]")
    
    try:
        # Calculate chunk size
        min_chunk_size = 50
        target_chunks = ncpus * 3
        chunk_size = max(min_chunk_size, len(sites) // target_chunks)
        
        # Split sites into chunks
        site_chunks = [sites[i:i + chunk_size] for i in range(0, len(sites), chunk_size)]
        
        console.print(f"[blue]Processing {len(sites):,} sites in {len(site_chunks)} chunks "
                     f"(~{chunk_size} sites/chunk) using {ncpus} workers[/blue]")
        console.print(f"[blue]Processing both target and background BAM files...[/blue]")
        
        all_records = []
        completed_sites = 0
        total_sites = len(sites) * 2  # Processing both BAMs
        
        # Process chunks in parallel for target BAM
        with ProcessPoolExecutor(max_workers=ncpus) as executor:
            # Submit all chunks for target BAM
            future_to_chunk = {}
            for i, chunk in enumerate(site_chunks):
                debug_flag = (i == 0)  # Debug first chunk only
                future = executor.submit(process_sites_chunk, target_bam, chunk, prob_map, 'target', min_mapq, min_bq, debug_flag)
                future_to_chunk[future] = ('target', chunk)
            
            # Submit all chunks for background BAM
            for i, chunk in enumerate(site_chunks):
                debug_flag = (i == 0)  # Debug first chunk only
                future = executor.submit(process_sites_chunk, background_bam, chunk, prob_map, 'background', min_mapq, min_bq, debug_flag)
                future_to_chunk[future] = ('background', chunk)
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                bam_source, chunk = future_to_chunk[future]
                try:
                    chunk_records = future.result()
                    all_records.extend(chunk_records)
                    completed_sites += len(chunk)
                    
                    # Update progress based on completed sites
                    progress_pct = (completed_sites / total_sites) * 100
                    progress.update(task_id, completed=progress_pct)
                    
                except Exception as e:
                    console.print(f"[red]Error processing {bam_source} chunk of {len(chunk)} sites: {e}[/red]")
                    raise
        
        progress.update(task_id, completed=100)
        console.print(f"[green]✓[/green] Extracted {len(all_records):,} fetal read records")
        
        # Calculate lookup statistics
        reads_with_prob = sum(1 for r in all_records if r.prob_class_1 not in [0.25, 0.75])
        reads_with_default = len(all_records) - reads_with_prob
        
        stats = {
            'total_records': len(all_records),
            'with_probability': reads_with_prob,
            'with_default': reads_with_default
        }
        
        console.print(f"[blue]Reads with actual probability: {reads_with_prob:,} ({reads_with_prob/len(all_records)*100:.2f}%)[/blue]")
        console.print(f"[blue]Reads with default probability: {reads_with_default:,} ({reads_with_default/len(all_records)*100:.2f}%)[/blue]")
        
        return all_records, stats
        
    except Exception as e:
        console.print(f"[red]Error processing BAM files: {e}[/red]")
        raise


def save_ground_truth(records: List[ReadRecord], output_prefix: str,
                     progress: Progress, task_id: TaskID) -> Path:
    """
    Save ground truth read records to compressed TSV file.
    
    Args:
        records (List[ReadRecord]): Read records
        output_prefix (str): Output file prefix
        progress (Progress): Rich progress bar instance
        task_id (TaskID): Task ID for progress tracking
        
    Returns:
        Path: Path to saved output file
    """
    output_file = Path(f"{output_prefix}_ground_truth.tsv.gz")
    
    progress.update(task_id, description="Saving ground truth data...")
    
    try:
        # Sort records by genomic coordinate and read name for deterministic output
        def sort_key(record):
            chr_name = record.chr
            if chr_name.startswith('chr'):
                chr_part = chr_name[3:]
            else:
                chr_part = chr_name
            
            try:
                chr_num = int(chr_part)
                return (0, chr_num, record.pos, record.name)
            except ValueError:
                return (1, chr_part, record.pos, record.name)
        
        sorted_records = sorted(records, key=sort_key)
        
        # Write header and data
        with gzip.open(output_file, 'wt') as f:
            # Write header
            header = ['chr', 'pos', 'ref', 'alt', 'af', 'raw_vaf', 'target_vaf', 'background_vaf',
                     'name', 'prob_class_1', 'support_base', 'classified_label']
            f.write('\t'.join(header) + '\n')
            
            # Write data rows
            for record in sorted_records:
                row = [
                    record.chr,
                    str(record.pos),
                    record.ref,
                    record.alt,
                    f"{record.af:.6f}",
                    f"{record.raw_vaf:.6f}",
                    f"{record.target_vaf:.6f}",
                    f"{record.background_vaf:.6f}",
                    record.name,
                    f"{record.prob_class_1:.6f}",
                    record.support_base,
                    str(record.classified_label)
                ]
                f.write('\t'.join(row) + '\n')
        
        progress.update(task_id, advance=100)
        console.print(f"[green]✓[/green] Ground truth data saved to: {output_file}")
        
        return output_file
        
    except Exception as e:
        console.print(f"[red]Error saving output file: {e}[/red]")
        raise


@click.command()
@click.option(
    '--input-target-bam',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to target BAM file (predicted fetal reads with prob_class_1 > 0.5)'
)
@click.option(
    '--input-background-bam',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to background BAM file (predicted maternal reads with prob_class_1 <= 0.5)'
)
@click.option(
    '--input-txt',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to read probability TSV file with "name" and "prob_class_1" columns'
)
@click.option(
    '--filtered-pileup',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Path to filtered pileup TSV file with ground truth SNP sites'
)
@click.option(
    '--output',
    required=True,
    type=str,
    help='Output file prefix (will create {prefix}_ground_truth.tsv.gz)'
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
def main(input_target_bam: Path, input_background_bam: Path, input_txt: Path, 
         filtered_pileup: Path, output: str, min_mapq: int, min_bq: int, ncpus: int) -> None:
    """
    Extract ground truth labels for fetal read classification from bisulfite sequencing data.
    
    This tool processes pre-split BAM files (target and background) along with filtered pileup 
    data to identify true fetal reads. The BAM files have been split based on probability threshold:
    - Target BAM: reads with prob_class_1 > 0.5 (predicted as fetal)
    - Background BAM: reads with prob_class_1 <= 0.5 (predicted as maternal)
    
    The filtered pileup contains SNPs where maternal genotype is homozygous and fetal is 
    heterozygous, allowing identification of fetal reads based on allele support:
    
    - For SNPs with raw_vaf in (0, 0.2): ALT-supporting reads are fetal
    - For SNPs with raw_vaf in (0.8, 1.0): REF-supporting reads are fetal
    
    Ground truth labels:
    - Reads from target BAM supporting fetal allele: classified_label = 1 (correct classification)
    - Reads from background BAM supporting fetal allele: classified_label = 0 (incorrect classification)
    
    The output contains read-level information including predicted probabilities and 
    ground truth labels for model evaluation.
    """
    console.print("\n[bold blue]Extract Ground Truth for Fetal Read Classification[/bold blue]")
    console.print("="*75)
    
    # Display input parameters
    params_table = Table(title="Input Parameters", show_header=True, header_style="bold magenta")
    params_table.add_column("Parameter", style="cyan", no_wrap=True)
    params_table.add_column("Value", style="white")
    
    params_table.add_row("Target BAM", str(input_target_bam))
    params_table.add_row("Background BAM", str(input_background_bam))
    params_table.add_row("Probability Table", str(input_txt))
    params_table.add_row("Filtered Pileup", str(filtered_pileup))
    params_table.add_row("Output Prefix", output)
    params_table.add_row("Min MAPQ", str(min_mapq))
    params_table.add_row("Min Base Quality", str(min_bq))
    params_table.add_row("Parallel Workers", str(ncpus))
    
    console.print(params_table)
    console.print()
    
    try:
        with Progress(console=console) as progress:
            # Create progress tasks
            pileup_task = progress.add_task("Parsing filtered pileup...", total=100)
            collect_task = progress.add_task("Collecting read names...", total=100)
            filter_task = progress.add_task("Filtering probability table...", total=100)
            extract_task = progress.add_task("Extracting reads...", total=100)
            save_task = progress.add_task("Saving output...", total=100)
            
            # Parse filtered pileup
            sites = parse_filtered_pileup(filtered_pileup, progress, pileup_task)
            
            # First pass: Collect all fetal read names from SNP sites
            needed_reads = collect_fetal_read_names(
                input_target_bam, input_background_bam, sites, min_mapq, min_bq, ncpus,
                progress, collect_task
            )
            
            # Filter probability table to only needed reads
            prob_map = filter_probability_map(input_txt, needed_reads, progress, filter_task)
            
            # Second pass: Extract ground truth reads with probabilities
            records, stats = extract_ground_truth(
                input_target_bam, input_background_bam, sites, prob_map, min_mapq, min_bq, ncpus,
                progress, extract_task
            )
            
            # Save output
            output_file = save_ground_truth(records, output, progress, save_task)
        
        # Calculate summary statistics
        if len(records) > 0:
            total_reads = len(records)
            unique_reads = len(set(r.name for r in records))
            correctly_classified = sum(1 for r in records if r.classified_label == 1)
            accuracy = (correctly_classified / total_reads) * 100 if total_reads > 0 else 0
            
            ref_supporting = sum(1 for r in records if r.support_base == 'REF')
            alt_supporting = sum(1 for r in records if r.support_base == 'ALT')
            
            avg_prob = sum(r.prob_class_1 for r in records) / total_reads
            
            reads_per_snp = total_reads / len(sites) if len(sites) > 0 else 0
        else:
            total_reads = unique_reads = correctly_classified = 0
            accuracy = avg_prob = reads_per_snp = 0.0
            ref_supporting = alt_supporting = 0
        
        # Display summary statistics
        summary_table = Table(title="Extraction Summary", show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="white", justify="right")
        
        summary_table.add_row("Total SNP sites processed", f"{len(sites):,}")
        summary_table.add_row("Total fetal read records", f"{total_reads:,}")
        summary_table.add_row("Unique fetal reads", f"{unique_reads:,}")
        summary_table.add_row("Reads per SNP (avg)", f"{reads_per_snp:.2f}")
        summary_table.add_row("", "")  # Spacer
        summary_table.add_row("[bold]Classification Results:[/bold]", "")
        summary_table.add_row("  Correctly classified (from target BAM)", f"{correctly_classified:,}")
        summary_table.add_row("  Accuracy", f"{accuracy:.2f}%")
        summary_table.add_row("  Average probability", f"{avg_prob:.4f}")
        summary_table.add_row("", "")  # Spacer
        summary_table.add_row("[bold]Allele Support:[/bold]", "")
        summary_table.add_row("  REF-supporting reads", f"{ref_supporting:,}")
        summary_table.add_row("  ALT-supporting reads", f"{alt_supporting:,}")
        
        console.print(summary_table)
        console.print(f"\n[bold green]✓ Extraction completed successfully![/bold green]")
        console.print(f"Output file: [cyan]{output_file}[/cyan]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during processing:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

