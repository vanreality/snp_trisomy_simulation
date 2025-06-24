import pandas as pd
import numpy as np
import click
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import pyarrow.parquet as pq
from pathlib import Path
from intervaltree import Interval, IntervalTree
from rich.console import Console
from rich.progress import Progress
from collections import defaultdict
from Bio import SeqIO
from Bio.Align import PairwiseAligner

console = Console()

def read_reference_genome(fasta_file: str) -> dict:
    """Reads the reference genome from a FASTA file."""
    console.log(f"Reading reference genome from: {fasta_file}")
    reference = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        reference[record.id] = str(record.seq).upper()
    return reference

def extract_region_from_reference(reference: dict, chrom: str, start: int, end: int) -> str:
    """Extracts a region from the reference genome."""
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
    """Performs local re-alignment between a read and a reference sequence."""
    seq_for_alignment = seq.replace('M', 'C')
    aligner = PairwiseAligner(scoring='blastn')
    aligner.mode = 'local'
    alignments = aligner.align(ref_seq, seq_for_alignment)
    
    if not alignments:
        return None
    
    return alignments[0]

def get_base_from_alignment(
    alignment: PairwiseAligner, 
    ref_start_pos: int, 
    original_query: str, 
    snp_sites: set
) -> list:
    """Determines the base in a read at given SNP sites based on local realignment."""
    results = []
    
    if not snp_sites or not alignment:
        return results

    aligned_ref = str(alignment[0])
    aligned_query = str(alignment[1])
    aligned_ref_pos = alignment.coordinates[0][0]
    
    # This logic for recovering the query is based on extract_snp_from_parquet.py
    # to ensure consistency.
    original_query_idx = alignment.coordinates[1][0]
    recovered_aligned_query = ""
    for base in aligned_query:
        if base == '-':
            recovered_aligned_query += '-'
        else:
            if original_query_idx < len(original_query):
                recovered_aligned_query += original_query[original_query_idx]
                original_query_idx += 1
            else:
                # This case handles if alignment goes past original query length
                recovered_aligned_query += 'N'

    for snp_interval in snp_sites:
        snp_data = snp_interval.data
        snp_pos = snp_data['pos']

        # Offset of the SNP within the aligned reference part
        offset_in_aligned_ref = snp_pos - 1 - ref_start_pos - aligned_ref_pos

        read_base = 'N'
        # Find the base in the read corresponding to the SNP position
        if offset_in_aligned_ref >= 0:
            ref_bases_count = 0
            for i, base in enumerate(aligned_ref):
                if base != '-':
                    if ref_bases_count == offset_in_aligned_ref:
                        if i < len(recovered_aligned_query):
                            read_base = recovered_aligned_query[i]
                        break
                    ref_bases_count += 1
        
        results.append({
            'chr': snp_data['chr_norm'],
            'pos': snp_pos,
            'ref': snp_data['ref'],
            'alt': snp_data['alt'],
            'base': read_base if read_base in 'ACGT' else 'N'
        })
            
    return results

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

def process_batch(
    batch: pd.DataFrame, 
    reference_genome: dict,
    snp_trees: dict, 
    n_bp_downstream: int, 
    n_bp_upstream: int
) -> list:
    """Processes a batch of sequences to find SNP intersections and get base counts."""
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
                base_results = get_base_from_alignment(
                    alignment, ref_extract_start, sequence, overlapping_snps
                )
                results.extend(base_results)
            
    return results

def process_parquet_file(
    parquet_file: str, 
    reference_genome: dict,
    snp_trees: dict, 
    batch_size: int, 
    num_workers: int,
    n_bp_downstream: int,
    n_bp_upstream: int
) -> pd.DataFrame:
    """Processes a Parquet file to determine SNP site coverage and base counts."""
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    console.log(f"Processing parquet file: {parquet_file} with {num_workers} workers")
    
    pq_file = pq.ParquetFile(parquet_file)
    
    required_columns = ['chr', 'start', 'end', 'seq']
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
            for batch in pq_file.iter_batches(batch_size=batch_size, columns=required_columns)
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
        return pd.DataFrame()

    return pd.DataFrame(all_results)

@click.command()
@click.option('--parquet', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the input parquet file.')
@click.option('--fasta', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the reference genome FASTA file.')
@click.option('--snp-list', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the SNP list file (VCF-like format).')
@click.option('--output', required=True, type=str, help='Path for the output TSV file.')
@click.option('--batch-size', default=100000, show_default=True, type=int, help='Number of records to process in each batch.')
@click.option('--num-workers', default=None, type=int, help='Number of worker processes. Defaults to number of CPU cores.')
@click.option('--n-bp-downstream', default=20, show_default=True, type=int, help='Bases downstream to include in the reference extract.')
@click.option('--n-bp-upstream', default=20, show_default=True, type=int, help='Bases upstream to include in the reference extract.')
def main(parquet, fasta, snp_list, output, batch_size, num_workers, n_bp_downstream, n_bp_upstream):
    """Checks intersection between reads in a Parquet file and a SNP list with local realignment."""
    console.log("Starting SNP intersection analysis with local realignment.")
    
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
    
    if results_df.empty:
        console.log("[yellow]WARNING:[/yellow] No SNP intersections found. The output file will be empty.")
        with open(output, 'w') as f:
            f.write("chr\tpos\tref\talt\tcoverage\tA_count\tC_count\tG_count\tT_count\tN_count\n")
        return

    console.log("Aggregating base counts per SNP site...")
    
    base_counts = results_df.groupby(['chr', 'pos', 'base']).size().unstack(fill_value=0)
    
    # Ensure all base columns exist
    for base in ['A', 'C', 'G', 'T', 'N']:
        if base not in base_counts.columns:
            base_counts[base] = 0
            
    base_counts = base_counts[['A', 'C', 'G', 'T', 'N']]
    base_counts.rename(columns={b: f"{b}_count" for b in base_counts.columns}, inplace=True)
    
    coverage = results_df.groupby(['chr', 'pos']).size().to_frame('coverage')
    
    ref_alt_info = results_df[['chr', 'pos', 'ref', 'alt']].drop_duplicates().set_index(['chr', 'pos'])
    
    summary_df = ref_alt_info.join(coverage).join(base_counts).reset_index()
    summary_df.sort_values(['chr', 'pos'], inplace=True)

    summary_df.to_csv(output, index=False, sep='\t')
    console.log(f"Coverage results written to: {output}")

    console.log("Analysis complete.")

if __name__ == '__main__':
    main()
