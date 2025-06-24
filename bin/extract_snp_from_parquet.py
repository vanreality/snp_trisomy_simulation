import pandas as pd
import numpy as np
import click
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import sys
import os
import pyarrow.parquet as pq
from pathlib import Path
from intervaltree import Interval, IntervalTree
from rich.console import Console
from rich.progress import Progress

console = Console()

def read_reference_genome(fasta_file: str) -> dict:
    """Reads the reference genome from a FASTA file.

    Args:
        fasta_file: The path to the FASTA file.

    Returns:
        A dictionary mapping chromosome IDs to their sequences.
    """
    console.log(f"Reading reference genome from: {fasta_file}")
    reference = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        reference[record.id] = str(record.seq).upper()
    return reference

def extract_region_from_reference(reference: dict, chrom: str, start: int, end: int) -> str:
    """Extracts a region from the reference genome.

    Args:
        reference: A dictionary of chromosome sequences.
        chrom: The chromosome name.
        start: The start coordinate.
        end: The end coordinate.

    Returns:
        The extracted sequence, or None if the region is invalid.
    """
    if chrom not in reference:
        alt_chrom = chrom.replace('chr', '') if chrom.startswith('chr') else f'chr{chrom}'
        if alt_chrom in reference:
            chrom = alt_chrom
        else:
            console.log(f"[bold red]ERROR:[/bold red] Chromosome {chrom} not found in reference genome")
            return None
    
    try:
        return reference[chrom][start:end]
    except IndexError:
        console.log(f"[bold red]ERROR:[/bold red] Invalid coordinates for chromosome {chrom}: {start}-{end}")
        return None

def local_realign(seq: str, ref_seq: str) -> PairwiseAligner:
    """Performs local re-alignment between a read and a reference sequence.

    Args:
        seq: The read sequence.
        ref_seq: The reference sequence.

    Returns:
        The best alignment object, or None if no alignment is found.
    """
    seq_for_alignment = seq.replace('M', 'C')
    aligner = PairwiseAligner(scoring='blastn')
    aligner.mode = 'local'
    alignments = aligner.align(ref_seq, seq_for_alignment)
    
    if not alignments:
        return None
    
    return alignments[0]

def extract_AF(row: pd.Series) -> float:
    """Parses the Allele Frequency (AF) from the INFO column of a VCF record.

    Args:
        row: A pandas Series representing a variant, requiring an 'info' column.

    Returns:
        The allele frequency as a float, or np.nan if parsing fails.
    """
    try:
        info = row['info']
        if 'AF=' not in info:
            return np.nan
        
        af_str = info.split('AF=')[1].split(';')[0]
        af = float(af_str)
        
        if not 0 <= af <= 1:
            return np.nan
            
        return af
    except (IndexError, ValueError):
        return np.nan

def load_snp_data(file_path: Path) -> pd.DataFrame:
    """Loads and validates SNP data from a VCF-like file.

    Args:
        file_path: The path to the SNP data file.

    Returns:
        A validated DataFrame with an 'af' column.

    Raises:
        ValueError: If the file is not found, empty, or cannot be processed.
    """
    try:
        snp_df = pd.read_csv(
            file_path,
            sep='\t',
            names=['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
            dtype={'chr': str, 'pos': int, 'id': str, 'ref': str, 'alt': str},
            usecols=[0, 1, 2, 3, 4, 5, 6, 7]
        )
        
        if snp_df.empty:
            raise ValueError("SNP data file is empty")
        
        console.log("Extracting allele frequencies (AF)...")
        with Progress() as progress:
            task = progress.add_task("[cyan]Extracting AF...", total=len(snp_df))
            af_values = []
            for _, row in snp_df.iterrows():
                af_values.append(extract_AF(row))
                progress.update(task, advance=1)
        
        snp_df['af'] = af_values
        
        invalid_count = snp_df['af'].isna().sum()
        snp_df.dropna(subset=['af'], inplace=True)

        if invalid_count > 0:
            console.log(f"[yellow]WARNING:[/yellow] Removed {invalid_count} variants due to missing or invalid AF.")

        if snp_df.empty:
            raise ValueError("No valid variants remaining after AF extraction")
        
        return snp_df
        
    except FileNotFoundError:
        raise ValueError(f"SNP data file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"SNP data file is empty: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to load SNP data: {e}") from e

def build_snp_trees(snp_list: Path) -> dict:
    """Builds interval trees for each chromosome from a list of SNPs.

    Args:
        snp_list: Path to the SNP data file.

    Returns:
        A dictionary mapping chromosome names to their IntervalTree.
    """
    console.log(f"Loading SNP data from {snp_list}...")
    snp_df = load_snp_data(snp_list)
    
    snp_trees = {}
    console.log("Building SNP interval trees...")
    
    chromosomes = [f'chr{i}' for i in range(1, 23)]
    snp_df['chr'] = snp_df['chr'].astype(str)
    snp_df['chr_norm'] = snp_df['chr'].apply(lambda x: f'chr{x}' if not x.startswith('chr') else x)
    
    snp_df = snp_df[snp_df['chr_norm'].isin(chromosomes)]
    
    grouped_chroms = snp_df.groupby('chr_norm')
    with Progress() as progress:
        task = progress.add_task("[cyan]Building trees...", total=len(grouped_chroms))
        for chrom, group in grouped_chroms:
            tree = IntervalTree()
            for _, snp in group.iterrows():
                pos = snp['pos']
                start = pos - 1
                end = start + len(snp['ref'])
                tree[start:end] = snp.to_dict()
            snp_trees[chrom] = tree
            progress.update(task, advance=1)
        
    console.log(f"Built interval trees for {len(snp_trees)} chromosomes.")
    return snp_trees

def extract_snp_info_from_alignment(
    alignment: PairwiseAligner, 
    ref_start_pos: int, 
    original_query: str, 
    snp_sites: set
) -> list:
    """Determines if a read supports REF or ALT alleles for given SNP sites.

    Based on local realignment, this function checks each overlapping SNP site
    to determine if the read contains the reference or alternative allele.

    Args:
        alignment: The alignment object from Biopython.
        ref_start_pos: The 0-based start position of the reference sequence
            in the chromosome.
        original_query: The original read sequence.
        snp_sites: A set of Interval objects for overlapping SNPs.

    Returns:
        A list of dictionaries, each containing info about a SNP found
        in the read. Status is 0 for REF, 1 for ALT.
    """
    results = []
    
    if not snp_sites:
        return results

    aligned_ref = str(alignment[0])
    aligned_query = str(alignment[1])
    aligned_ref_pos = alignment.coordinates[0][0]
    algined_query_pos = alignment.coordinates[1][0]

    # Recover the M bases in the query sequence
    # recovered_query = ""
    # for q_base in aligned_query:
    #     if q_base != '-':
    #         recovered_query += original_query[algined_query_pos]
    #         algined_query_pos += 1
    #     else:
    #         recovered_query += '-'
    # aligned_query = recovered_query

    for snp_interval in snp_sites:
        snp_data = snp_interval.data
        snp_pos = snp_data['pos']  # 1-based from VCF
        snp_ref = snp_data['ref']
        snp_alt = snp_data['alt']

        if len(snp_ref) != 1 or len(snp_alt) != 1:
            continue

        off_set = snp_pos - 1 - ref_start_pos - aligned_ref_pos
        idx = 0
        read_base = 'N'
        for i, base in enumerate(aligned_ref):
            if base != '-':
                if idx == off_set:
                    if i >= len(aligned_query):
                        read_base = 'N'
                        break
                    read_base = aligned_query[i]
                    break
                idx += 1
        
        status = -1
        if read_base == snp_alt:
            status = 1
        elif read_base == snp_ref:
            status = 0
        
        if status != -1:
            results.append({
                'pos': snp_pos,
                'status': status,
                'chr': snp_data['chr_norm'], # Use normalized chromosome
                'ref': snp_ref,
                'alt': snp_alt,
                'af': snp_data['af'],
            })
            
    return results

def process_batch(
    batch: pd.DataFrame, 
    reference_genome: dict, 
    snp_trees: dict, 
    n_bp_downstream: int, 
    n_bp_upstream: int
) -> list:
    """Processes a batch of sequences from the Parquet file.

    Args:
        batch: DataFrame with a batch of sequence records.
        reference_genome: Dictionary of chromosome sequences.
        snp_trees: Dictionary of SNP interval trees.
        n_bp_downstream: Bases downstream for reference extraction.
        n_bp_upstream: Bases upstream for reference extraction.

    Returns:
        A list of dictionaries with SNP site pileup data.
    """
    results = []
    
    for _, row in batch.iterrows():
        chr_name = row['chr']
        chr_name_norm = f'chr{chr_name}' if not chr_name.startswith('chr') else chr_name

        if chr_name_norm not in snp_trees:
            continue

        snp_tree = snp_trees[chr_name_norm]
        start_pos, end_pos = row['start'], row['end']

        overlapping_snps = snp_tree.overlap(start_pos, end_pos)
        if overlapping_snps:
            ref_extract_start = start_pos - n_bp_upstream
            ref_extract_end = end_pos + n_bp_downstream
            ref_region = extract_region_from_reference(
                reference_genome, chr_name_norm, ref_extract_start, ref_extract_end
            )
            
            if not ref_region:
                continue
            
            sequence = row['seq']
            alignment = local_realign(sequence, ref_region)
            
            if alignment:
                snp_results = extract_snp_info_from_alignment(
                    alignment, ref_extract_start, sequence, overlapping_snps
                )
                
                for snp in snp_results:
                    results.append({
                        **snp,
                        'prob_class_1': row['prob_class_1'],
                        'name': row['name'],
                        'insert_size': row['insert_size'],
                        'chr_dmr': row['chr_dmr'],
                        'start_dmr': row['start_dmr'],
                        'end_dmr': row['end_dmr']
                    })
    
    return results

def process_parquet_file(
    parquet_file: str, 
    reference_genome: dict, 
    snp_trees: dict, 
    batch_size: int = 1000, 
    num_workers: int = None, 
    n_bp_downstream: int = 20, 
    n_bp_upstream: int = 20
) -> pd.DataFrame:
    """Processes a Parquet file to determine SNP site pileup.

    Args:
        parquet_file: Path to the Parquet file.
        reference_genome: Dictionary of chromosome sequences.
        snp_trees: Dictionary of SNP interval trees.
        batch_size: Number of records per batch.
        num_workers: Number of worker processes.
        n_bp_downstream: Bases downstream for reference extraction.
        n_bp_upstream: Bases upstream for reference extraction.

    Returns:
        A DataFrame with SNP site pileup data.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    console.log(f"Processing parquet file: {parquet_file} with {num_workers} workers")
    
    pq_file = pq.ParquetFile(parquet_file)
    
    required_columns = ['chr', 'start', 'end', 'seq', 'prob_class_1', 'name', 'insert_size', 'chr_dmr', 'start_dmr', 'end_dmr']
    missing_columns = [col for col in required_columns if col not in pq_file.schema.names]
    if missing_columns:
        msg = f"Missing required columns in Parquet file: {', '.join(missing_columns)}"
        console.log(f"[bold red]ERROR:[/bold red] {msg}")
        raise ValueError(msg)
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        batch_processor = partial(
            process_batch, 
            reference_genome=reference_genome, 
            snp_trees=snp_trees, 
            n_bp_downstream=n_bp_downstream, 
            n_bp_upstream=n_bp_upstream
        )
        
        futures = {
            executor.submit(batch_processor, batch.to_pandas()) 
            for batch in pq_file.iter_batches(batch_size=batch_size)
        }
        
        console.log(f"Submitted {len(futures)} batches for processing.")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing batches...", total=len(futures))
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    if batch_results:
                        all_results.extend(batch_results)
                except Exception as e:
                    console.log(f"[bold red]ERROR:[/bold red] A batch failed to process: {e}", exc_info=True)
                progress.update(task, advance=1)

    if not all_results:
        console.log("[yellow]WARNING:[/yellow] No SNP information was extracted. The output file will be empty.")
        return pd.DataFrame()

    return pd.DataFrame(all_results)

def snp_pileup(results_df: pd.DataFrame, mode: str, threshold: float = None) -> pd.DataFrame:
    """Calculates pileup data for each SNP site.

    Args:
        results_df: DataFrame with per-read SNP information.
        mode: The calculation mode, either 'prob_weighted' or 'hard_filter'.
        threshold: Probability threshold required for 'hard_filter' mode.

    Returns:
        A DataFrame with aggregated pileup data per SNP.
        
    Raises:
        ValueError: If an invalid mode is provided.
    """
    console.log(f"Calculating SNP site pileup using '{mode}' mode.")
    
    if results_df.empty:
        console.log("[yellow]WARNING:[/yellow] Input for pileup calculation is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    
    df = results_df.copy()
    
    if mode == 'prob_weighted':
        df['prob_ref'] = df.apply(lambda row: row['prob_class_1'] if row['status'] == 0 else 0.0, axis=1)
        df['prob_alt'] = df.apply(lambda row: row['prob_class_1'] if row['status'] == 1 else 0.0, axis=1)

        pileup_df = (
            df
            .groupby(['chr','pos'], as_index=False)
            .agg(
                ref=('ref', 'first'),
                alt=('alt', 'first'),
                af=('af', 'first'),
                cfDNA_ref_reads=('status', lambda x: (x == 0).sum()),
                cfDNA_alt_reads=('status', lambda x: (x == 1).sum()),
                current_depth=('status', 'size'),
                total_prob=('prob_class_1', 'sum'),
                sum_prob_ref=('prob_ref', 'sum'),
                sum_prob_alt=('prob_alt', 'sum'),
            )
        )

        mask = pileup_df['total_prob'] > 0
        pileup_df['fetal_ref_reads_from_model'] = np.where(
            mask,
            pileup_df['sum_prob_ref'],
            0
        )
        pileup_df['fetal_alt_reads_from_model'] = np.where(
            mask,
            pileup_df['sum_prob_alt'],
            0
        )

        pileup_df.drop(columns=['total_prob','sum_prob_ref','sum_prob_alt'], inplace=True)

        pileup_df['fetal_ref_reads_from_model'] = pileup_df['fetal_ref_reads_from_model'].astype(int)
        pileup_df['fetal_alt_reads_from_model'] = pileup_df['fetal_alt_reads_from_model'].astype(int)

    elif mode == 'hard_filter':
        base_pileup = df.groupby(['chr', 'pos'], as_index=False).agg(
            ref=('ref', 'first'),
            alt=('alt', 'first'),
            af=('af', 'first'),
            cfDNA_ref_reads=('status', lambda x: (x == 0).sum()),
            cfDNA_alt_reads=('status', lambda x: (x == 1).sum()),
            current_depth=('status', 'size')
        )

        fetal_df = df[df['prob_class_1'] > threshold].copy()

        fetal_counts = fetal_df.groupby(['chr', 'pos']).agg(
            fetal_ref_reads_from_model=('status', lambda x: (x == 0).sum()),
            fetal_alt_reads_from_model=('status', lambda x: (x == 1).sum())
        ).reset_index()

        pileup_df = pd.merge(base_pileup, fetal_counts, on=['chr', 'pos'], how='left')
        pileup_df['fetal_ref_reads_from_model'] = pileup_df['fetal_ref_reads_from_model'].fillna(0).astype(int)
        pileup_df['fetal_alt_reads_from_model'] = pileup_df['fetal_alt_reads_from_model'].fillna(0).astype(int)

    else:
        raise ValueError(f"Invalid mode specified: {mode}")
        
    return pileup_df

@click.command()
@click.option('--parquet', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the input parquet file.')
@click.option('--fasta', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the reference genome FASTA file.')
@click.option('--snp-list', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the SNP list file (VCF-like format).')
@click.option('--output', required=True, type=str, help='Prefix for output files.')
@click.option('--mode', required=True, type=click.Choice(['hard_filter', 'prob_weighted']), help='Mode for calculating fetal reads.')
@click.option('--threshold', type=float, help='Probability threshold for hard_filter mode.')
@click.option('--batch-size', default=100000, show_default=True, type=int, help='Number of records to process in each batch.')
@click.option('--num-workers', default=None, type=int, help='Number of worker processes. Defaults to number of CPU cores.')
@click.option('--n-bp-downstream', default=20, show_default=True, type=int, help='Bases downstream to include in the reference extract.')
@click.option('--n-bp-upstream', default=20, show_default=True, type=int, help='Bases upstream to include in the reference extract.')
def main(parquet, fasta, snp_list, output, mode, threshold, batch_size, num_workers, n_bp_downstream, n_bp_upstream):
    """Processes sequencing data from a Parquet file to calculate SNP pileup."""
    console.log("Starting SNP pileup analysis pipeline.")
    
    if mode == 'hard_filter' and threshold is None:
        console.log("[bold red]ERROR:[/bold red] --threshold must be set when using --mode hard_filter.")
        sys.exit(1)
    
    reference_genome = read_reference_genome(fasta)
    snp_trees = build_snp_trees(Path(snp_list))
    
    results_df = process_parquet_file(
        parquet, 
        reference_genome, 
        snp_trees,
        batch_size=batch_size, 
        num_workers=num_workers,
        n_bp_downstream=n_bp_downstream,
        n_bp_upstream=n_bp_upstream
    )
    
    if not results_df.empty:
        raw_output_path = f"{output}_raw_snp_calls.tsv.gz"
        results_df.to_csv(raw_output_path, index=False, sep='\t', compression='gzip')
        console.log(f"Raw per-read results written to: {raw_output_path}")

    pileup_results_df = snp_pileup(results_df, mode=mode, threshold=threshold)

    if not pileup_results_df.empty:
        tsv_output = f"{output}_pileup.tsv.gz"
        pileup_results_df.to_csv(tsv_output, index=False, sep='\t', compression='gzip')
        console.log(f"Pileup results written to: {tsv_output}")
    else:
        console.log("[yellow]WARNING:[/yellow] Final pileup data is empty. No output file was generated.")

    console.log("Analysis complete.")

if __name__ == '__main__':
    main()
