import pandas as pd
import numpy as np
import click
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import sys
import os
import logging
import pyarrow.parquet as pq
from pathlib import Path
from intervaltree import Interval, IntervalTree
from rich.logging import RichHandler
from tqdm import tqdm

# Configure logging using rich for better output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt='[%X]',
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

def read_reference_genome(fasta_file: str) -> dict:
    """
    Read the reference genome from FASTA file.
    Returns a dictionary of chromosome sequences.
    """
    logger.info(f"Reading reference genome from: {fasta_file}")
    reference = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Convert sequence to uppercase
        reference[record.id] = str(record.seq).upper()
    return reference

def extract_region_from_reference(reference: dict, chrom: str, start: int, end: int) -> str:
    """
    Extract a region from the reference genome.
    """
    if chrom not in reference:
        # Try adding/removing 'chr' prefix
        alt_chrom = chrom.replace('chr', '') if chrom.startswith('chr') else f'chr{chrom}'
        if alt_chrom in reference:
            chrom = alt_chrom
        else:
            logger.error(f"Chromosome {chrom} not found in reference genome")
            return None
    
    try:
        return reference[chrom][start:end]
    except IndexError:
        logger.error(f"Invalid coordinates for chromosome {chrom}: {start}-{end}")
        return None

def local_realign(seq: str, ref_seq: str) -> PairwiseAligner:
    """
    Perform local re-alignment between a read sequence and reference sequence
    using the Bio.Align.PairwiseAligner.
    Returns the alignment.
    """
    # Replace 'M' with 'C' for alignment purposes
    seq_for_alignment = seq.replace('M', 'C')
    
    # Configure aligner for local alignment
    aligner = PairwiseAligner(scoring='blastn')
    aligner.mode = 'local'
    
    # Perform alignment
    alignments = aligner.align(ref_seq, seq_for_alignment)
    
    # Get the best alignment
    if not alignments:
        return None
    
    return alignments[0]

def extract_AF(row: pd.Series) -> float:
    """
    Parse the AF (allele frequency) field from the INFO column.
    
    Args:
        row: Pandas Series containing variant information with 'info' column
        
    Returns:
        Allele frequency as float
        
    Raises:
        ValueError: If AF field cannot be parsed
    """
    try:
        info = row['info']
        if 'AF=' not in info:
            raise ValueError(f"AF field not found in INFO column: {info}")
        
        af_str = info.split('AF=')[1].split(';')[0]
        af = float(af_str)
        
        if not 0 <= af <= 1:
            raise ValueError(f"Invalid allele frequency: {af}")
            
        return af
    except (IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse AF from INFO: {info}") from e

def load_snp_data(file_path: Path) -> pd.DataFrame:
    """
    Load and validate SNP data from file.
    
    Args:
        file_path: Path to the SNP data file
        
    Returns:
        Validated DataFrame with AF column added
        
    Raises:
        SNPSimulationError: If file cannot be loaded or is invalid
    """
    try:
        # Try to load the file
        potential_df = pd.read_csv(
            file_path,
            sep='\t',
            names=['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
            dtype={'chr': str, 'pos': int, 'id': str, 'ref': str, 'alt': str},
            usecols=[0, 1, 2, 3, 4, 5, 6, 7]
        )
        
        if potential_df.empty:
            raise ValueError("SNP data file is empty")
        
        # Apply AF extraction with error handling
        logger.info("Extracting allele frequencies (AF)...")
        af_values = []
        invalid_count = 0
        
        for idx, row in tqdm(potential_df.iterrows(), total=len(potential_df), desc="Extracting AF"):
            try:
                af = extract_AF(row)
                af_values.append(af)
            except ValueError:
                af_values.append(np.nan)
                invalid_count += 1
        
        potential_df['af'] = af_values
        
        # Remove rows with invalid AF values
        potential_df = potential_df.dropna(subset=['af'])

        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} variants due to missing or invalid AF.")

        if potential_df.empty:
            raise ValueError("No valid variants remaining after AF extraction")
        
        return potential_df
        
    except FileNotFoundError:
        raise ValueError(f"SNP data file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"SNP data file is empty: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to load SNP data: {e}") from e

def build_snp_trees(snp_list: Path) -> dict:
    """
    Build interval trees for each chromosome from a list of SNPs.
    """
    logger.info(f"Loading SNP data from {snp_list}...")
    snp_df = load_snp_data(snp_list)
    
    snp_trees = {}
    logger.info("Building SNP interval trees...")
    
    # Filter for standard chromosomes
    chromosomes = [f'chr{i}' for i in range(1, 23)]
    snp_df['chr'] = snp_df['chr'].astype(str)
    # Handle both 'chr1' and '1' formats
    snp_df['chr_norm'] = snp_df['chr'].apply(lambda x: f'chr{x}' if not x.startswith('chr') else x)
    
    # Use only standard chromosomes
    snp_df = snp_df[snp_df['chr_norm'].isin(chromosomes)]
    
    for chrom, group in tqdm(snp_df.groupby('chr_norm'), desc="Building trees"):
        tree = IntervalTree()
        for _, snp in group.iterrows():
            pos = snp['pos']
            # VCF is 1-based, IntervalTree is 0-based, end-exclusive
            start = pos - 1
            end = start + len(snp['ref'])
            tree[start:end] = snp.to_dict()
        snp_trees[chrom] = tree
        
    logger.info(f"Built interval trees for {len(snp_trees)} chromosomes.")
    return snp_trees

def extract_snp_info_from_alignment(
    reference_seq: str, 
    alignment: PairwiseAligner, 
    ref_start_pos: int, 
    original_query: str, 
    snp_sites: set
) -> list:
    """
    Determine if a read supports REF or ALT alleles for given SNP sites.

    Based on the local realignment, this function checks each overlapping SNP site
    to see if the read contains the reference or alternative allele.

    Args:
        reference_seq (str): The reference sequence slice used for alignment.
        alignment (PairwiseAligner): The alignment object from Biopython.
        ref_start_pos (int): The 0-based starting position of the reference_seq
            in the chromosome.
        original_query (str): The original read sequence.
        snp_sites (set): A set of Interval objects representing overlapping SNPs.

    Returns:
        list: A list of dictionaries, where each dictionary contains info about a
              SNP found in the read. Status is 0 for REF, 1 for ALT.
    """
    results = []
    
    if not snp_sites:
        return results

    # alignment.coordinates is a (2, N) array
    # row 0: indices in the target (reference)
    # row 1: indices in the query (read)
    # -1 indicates a gap
    ref_coords = alignment.coordinates[0]
    query_coords = alignment.coordinates[1]

    for snp_interval in snp_sites:
        snp_data = snp_interval.data
        snp_pos = snp_data['pos']  # 1-based from VCF
        snp_ref = snp_data['ref']
        snp_alt = snp_data['alt']

        # We only handle simple SNPs for now for simplicity and performance
        if len(snp_ref) != 1 or len(snp_alt) != 1:
            continue

        # Convert 1-based SNP position to 0-based index in reference_seq
        snp_index_in_ref_seq = snp_pos - 1 - ref_start_pos

        if not (0 <= snp_index_in_ref_seq < len(ref_coords)):
            continue

        # Find where in the alignment this SNP position is
        # np.where returns a tuple of arrays, one for each dimension
        alignment_indices = np.where(ref_coords == snp_index_in_ref_seq)[0]

        if alignment_indices.size == 0:
            # SNP position is not in the aligned part of the reference (e.g., in an intron for RNA-seq)
            continue

        alignment_idx = alignment_indices[0]
        query_idx = query_coords[alignment_idx]

        if query_idx == -1:
            # Deletion in read at SNP position, doesn't support REF or ALT for a simple SNP
            continue
        
        if query_idx >= len(original_query):
            # Should not happen with a valid alignment, but as a safeguard
            continue

        read_base = original_query[int(query_idx)]
        
        status = -1  # -1 for unknown/other
        if read_base == snp_alt:
            status = 1
        elif read_base == snp_ref:
            status = 0
        
        if status != -1:
            results.append({
                'pos': snp_pos,
                'status': status,
                'chr': snp_data['chr_norm'], # Use normalized chromosome
            })
            
    return results

def process_batch(batch: pd.DataFrame, reference_genome: dict, snp_trees: dict, n_bp_downstream: int, n_bp_upstream: int) -> list:
    """
    Process a batch of sequences from the parquet file.
    
    Args:
        batch: Pandas DataFrame containing a batch of sequence records
        reference_genome: Dictionary of chromosome sequences
        n_bp_downstream: Number of base pairs downstream to include in the reference extract
        n_bp_upstream: Number of base pairs upstream to include in the reference extract
        
    Returns:
        List of dictionaries with SNP site pileup data
    """
    results = []
    
    for _, row in batch.iterrows():
        chr_name = row['chr']
        # Normalize chromosome name to match snp_trees keys
        chr_name_norm = f'chr{chr_name}' if not chr_name.startswith('chr') else chr_name

        start_pos = row['start']
        end_pos = row['end']

        if chr_name_norm in snp_trees:
            snp_tree = snp_trees[chr_name_norm]
        else:
            # This can happen for non-standard chromosomes, so not an error
            continue

        # Check if the read overlaps with any SNPs
        overlapping_snps = snp_tree.overlap(start_pos, end_pos)
        if overlapping_snps:
            sequence = row['seq']
            prob = row['prob_class_1']
            name = row['name']
            insert_size = row['insert_size']
            chr_dmr = row['chr_dmr']
            start_dmr = row['start_dmr']
            end_dmr = row['end_dmr']
            
            # Extract reference region with extra bases upstream and downstream
            ref_extract_start = start_pos - n_bp_upstream
            ref_extract_end = end_pos + n_bp_downstream
            ref_region = extract_region_from_reference(
                reference_genome, chr_name_norm, ref_extract_start, ref_extract_end
            )
            
            if ref_region is None or len(ref_region) == 0:
                continue
            
            # Perform local re-alignment
            alignment = local_realign(sequence, ref_region)
            
            if alignment:
                # Identify SNP sites and pileup status
                snp_results = extract_snp_info_from_alignment(
                    ref_region, alignment, ref_extract_start, sequence, overlapping_snps
                )
                
                # Add data to results
                for snp in snp_results:
                    results.append({
                        'chr': snp['chr'],
                        'pos': snp['pos'],
                        'status': snp['status'],
                        'prob_class_1': prob,
                        'name': name,
                        'insert_size': insert_size,
                        'chr_dmr': chr_dmr,
                        'start_dmr': start_dmr,
                        'end_dmr': end_dmr
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
    """
    Process sequences from a parquet file to determine SNP site pileup.
    
    Args:
        parquet_file: Path to the parquet file
        reference_genome: Dictionary of chromosome sequences
        batch_size: Number of records to process in each batch
        num_workers: Number of worker processes for parallel processing
        n_bp_downstream: Number of base pairs downstream to include in the reference extract
        n_bp_upstream: Number of base pairs upstream to include in the reference extract
        
    Returns:
        Pandas DataFrame with SNP site pileup data
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    logger.info(f"Processing parquet file: {parquet_file} with {num_workers} workers")
    
    pq_file = pq.ParquetFile(parquet_file)
    
    required_columns = ['chr', 'start', 'end', 'seq', 'prob_class_1', 'name', 'insert_size', 'chr_dmr', 'start_dmr', 'end_dmr']
    schema_cols = pq_file.schema.names
    missing_columns = [col for col in required_columns if col not in schema_cols]
    if missing_columns:
        msg = f"Missing required columns in Parquet file: {', '.join(missing_columns)}"
        logger.error(msg)
        raise ValueError(msg)
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create a partial function with fixed arguments
        batch_processor = partial(
            process_batch, 
            reference_genome=reference_genome, 
            snp_trees=snp_trees, 
            n_bp_downstream=n_bp_downstream, 
            n_bp_upstream=n_bp_upstream
        )
        
        # Create futures for all batches
        futures = [
            executor.submit(batch_processor, batch.to_pandas()) 
            for batch in pq_file.iter_batches(batch_size=batch_size)
        ]
        
        logger.info(f"Submitted {len(futures)} batches for processing.")
        
        # Process results as they complete
        for future in tqdm(futures, desc="Processing batches"):
            try:
                batch_results = future.result()
                if batch_results:
                    all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"A batch failed to process: {e}", exc_info=True)

    if not all_results:
        logger.warning("No SNP information was extracted. The output file will be empty.")
        return pd.DataFrame(columns=required_columns + ['pos', 'status'])

    results_df = pd.DataFrame(all_results)
    
    return results_df

def snp_pileup(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the pileup for each SNP site using the formula:
    sum(status * prob_class_1) / sum(prob_class_1) * 100
    
    Args:
        results_df: DataFrame containing SNP site pileup data
        
    Returns:
        DataFrame with SNP site pileup data
    """
    logger.info("Calculating SNP site pileup")
    
    if results_df.empty:
        logger.warning("Input for pileup calculation is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Group by SNP site location
    grouped = results_df.groupby(['chr', 'pos'])
    
    # Calculate pileup
    pileup_df = grouped.apply(
        lambda g: pd.Series({
            'chr': g['chr'].iloc[0],
            'pos': g['pos'].iloc[0],
            'ref': g['ref'].iloc[0],
            'alt': g['alt'].iloc[0],
            'af': g['af'].iloc[0],
            'cfDNA_ref_reads': g[g['status'] == 0].shape[0],
            'cfDNA_alt_reads': g[g['status'] == 1].shape[0],
            'current_depth': g.shape[0],
            'fetal_ref_reads_from_model': int(g[g['status'] == 0]['prob_class_1'].sum() / g['prob_class_1'].sum()) if g['prob_class_1'].sum() > 0 else 0,
            'fetal_alt_reads_from_model': int(g[g['status'] == 1]['prob_class_1'].sum() / g['prob_class_1'].sum()) if g['prob_class_1'].sum() > 0 else 0,
        })
    ).reset_index()
    
    return pileup_df

@click.command()
@click.option('--parquet', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the input parquet file.')
@click.option('--fasta', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the reference genome FASTA file.')
@click.option('--snp-list', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), help='Path to the SNP list file (VCF-like format).')
@click.option('--output', required=True, type=str, help='Prefix for output files.')
@click.option('--batch-size', default=100000, show_default=True, type=int, help='Number of records to process in each batch.')
@click.option('--num-workers', default=None, type=int, help='Number of worker processes. Defaults to number of CPU cores.')
@click.option('--n-bp-downstream', default=20, show_default=True, type=int, help='Bases downstream to include in the reference extract.')
@click.option('--n-bp-upstream', default=20, show_default=True, type=int, help='Bases upstream to include in the reference extract.')
def main(parquet, fasta, snp_list, output, batch_size, num_workers, n_bp_downstream, n_bp_upstream):
    """
    Process sequencing data from a Parquet file to calculate SNP pileup at specified sites.
    """
    logger.info("Starting SNP pileup analysis pipeline.")
    
    # Read reference genome and build SNP trees
    reference_genome = read_reference_genome(fasta)
    snp_trees = build_snp_trees(Path(snp_list))
    
    # Process parquet file to get per-read SNP status
    results_df = process_parquet_file(
        parquet, 
        reference_genome, 
        snp_trees,
        batch_size=batch_size, 
        num_workers=num_workers,
        n_bp_downstream=n_bp_downstream,
        n_bp_upstream=n_bp_upstream
    )
    
    # Calculate final pileup from all read data
    pileup_results_df = snp_pileup(results_df)
    
    # Write detailed pileup results to a TSV file
    if not pileup_results_df.empty:
        tsv_output = f"{output}_pileup.tsv.gz"
        pileup_results_df.to_csv(tsv_output, index=False, sep='\t', compression='gzip')
        logger.info(f"Pileup results written to: {tsv_output}")
    else:
        logger.warning("Final pileup data is empty. No output file was generated.")

    # Optionally, write raw per-read results for debugging
    raw_output_path = f"{output}_raw_snp_calls.tsv.gz"
    results_df.to_csv(raw_output_path, index=False, sep='\t', compression='gzip')
    logger.info(f"Raw per-read results written to: {raw_output_path}")

    logger.info("Analysis complete.")

if __name__ == '__main__':
    main()
