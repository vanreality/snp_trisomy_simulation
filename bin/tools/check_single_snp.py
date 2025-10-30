import click
import pandas as pd
import pyarrow.parquet as pq
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from rich.console import Console

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
        alt_chrom = chrom.replace('chr', '') if not chrom.startswith('chr') else chrom.replace('chr', '', 1)
        if alt_chrom in reference:
            chrom = alt_chrom
        else:
            console.log(f"[bold yellow]Warning:[/bold yellow] Chromosome {chrom} not found in reference genome.")
            return None
    
    try:
        # Handle coordinates that might be out of bounds
        start = max(0, start)
        end = min(len(reference[chrom]), end)
        return reference[chrom][start:end]
    except IndexError:
        console.log(f"[bold red]ERROR:[/bold red] Invalid coordinates for chromosome {chrom}: {start}-{end}")
        return None

def local_realign(seq: str, ref_seq: str) -> PairwiseAligner:
    """Performs local re-alignment between a read and a reference sequence."""
    if not ref_seq:
        return None
    seq_for_alignment = seq.replace('M', 'C')
    aligner = PairwiseAligner(scoring='blastn')
    aligner.mode = 'local'
    alignments = aligner.align(ref_seq, seq_for_alignment)
    
    if not alignments:
        return None
    
    return alignments[0]

@click.command()
@click.option('--snp', required=True, type=str, help="The SNP to visualize in '{chr}_{pos}' format (e.g., 'chr1_12345').")
@click.option('--parquet', required=True, type=click.Path(exists=True, dir_okay=False), help='Path to the input Parquet file.')
@click.option('--fasta', required=True, type=click.Path(exists=True, dir_okay=False), help='Path to the reference genome FASTA file.')
@click.option('--output', required=True, type=click.Path(), help='Path for the output visualization TXT file.')
@click.option('--window-size', default=30, show_default=True, type=int, help='Number of bases to show on each side of the SNP.')
@click.option('--flank-size', default=20, show_default=True, type=int, help='Number of bases to include for local realignment context.')
def main(snp, parquet, fasta, output, window_size, flank_size):
    """
    Extracts reads covering a single SNP, performs local realignment, and
    creates a text-based visualization similar to IGV.
    """
    try:
        chrom, pos_str = snp.split('_')
        pos = int(pos_str)
    except ValueError:
        console.log("[bold red]ERROR:[/bold red] SNP format must be '{chr}_{pos}' (e.g., 'chr1_12345').")
        return

    reference_genome = read_reference_genome(fasta)
    
    # Normalize chromosome name
    chrom_norm = f'chr{chrom}' if not chrom.startswith('chr') else chrom
    if chrom_norm not in reference_genome:
        # Try the other format
        chrom_alt = chrom.replace('chr', '')
        if chrom_alt in reference_genome:
            chrom_norm = chrom_alt
        else:
            console.log(f"[bold red]ERROR:[/bold red] Chromosome '{chrom}' not found in FASTA file.")
            return

    pq_file = pq.ParquetFile(parquet)
    overlapping_reads_data = []

    console.log(f"Scanning {parquet} for reads covering {chrom_norm}:{pos}...")
    for batch in pq_file.iter_batches(columns=['chr', 'start', 'end', 'seq']):
        df = batch.to_pandas()
        
        df_chrom = df[df['chr'] == chrom_norm]
        if df_chrom.empty:
            continue

        # SNP position is 1-based, dataframe start is 0-based
        df_overlapping = df_chrom[(df_chrom['start'] < pos) & (df_chrom['end'] >= pos)]

        for _, row in df_overlapping.iterrows():
            ref_extract_start = row['start'] - flank_size
            ref_extract_end = row['end'] + flank_size
            
            ref_region = extract_region_from_reference(reference_genome, chrom_norm, ref_extract_start, ref_extract_end)
            if not ref_region:
                continue
            
            alignment = local_realign(row['seq'], ref_region)
            if alignment:
                overlapping_reads_data.append({
                    'alignment': alignment,
                    'ref_region_start': ref_extract_start,
                })

    console.log(f"Found and realigned {len(overlapping_reads_data)} reads covering the site.")
    if not overlapping_reads_data:
        console.log("[yellow]No overlapping reads found. No output file will be generated.[/yellow]")
        return

    # Generate visualization
    viz_start_0based = pos - 1 - window_size
    viz_end_0based = pos + window_size
    ref_window_seq = extract_region_from_reference(reference_genome, chrom_norm, viz_start_0based, viz_end_0based)

    if not ref_window_seq:
        console.log("[bold red]ERROR:[/bold red] Could not extract reference sequence for the visualization window.")
        return
        
    with open(output, 'w') as f:
        f.write(f"# SNP: {chrom_norm}:{pos}\n")
        f.write(f"# VISUALIZATION WINDOW: {chrom_norm}:{viz_start_0based + 1}-{viz_end_0based}\n\n")

        # Write reference and marker
        f.write("\t".join(list(ref_window_seq)) + "\n")
        
        marker_list = [' '] * len(ref_window_seq)
        snp_index_in_window = pos - 1 - viz_start_0based
        if 0 <= snp_index_in_window < len(marker_list):
            marker_list[snp_index_in_window] = "^"
        f.write("\t".join(marker_list) + "\n")

        for i, read_data in enumerate(overlapping_reads_data):
            alignment = read_data['alignment']
            ref_region_start = read_data['ref_region_start']
            
            aligned_ref_str = str(alignment[0])
            aligned_query_str = str(alignment[1])
            align_start_in_ref_region_0based = alignment.coordinates[0][0]
            
            align_start_in_genome_0based = ref_region_start + align_start_in_ref_region_0based
            
            viz_read_list = [' '] * len(ref_window_seq)
            
            ref_bases_traversed = 0
            for align_idx in range(len(aligned_ref_str)):
                ref_char = aligned_ref_str[align_idx]
                
                if ref_char != '-':
                    current_genome_pos_0based = align_start_in_genome_0based + ref_bases_traversed
                    
                    if viz_start_0based <= current_genome_pos_0based < viz_end_0based:
                        idx_in_viz = current_genome_pos_0based - viz_start_0based
                        query_char = aligned_query_str[align_idx]
                        
                        # Use '.' for match to make mismatches stand out
                        # if query_char == ref_window_seq[idx_in_viz]:
                        #      viz_read_list[idx_in_viz] = '.'
                        # else:
                        #      viz_read_list[idx_in_viz] = query_char
                        viz_read_list[idx_in_viz] = query_char

                    ref_bases_traversed += 1

            f.write("\t".join(viz_read_list) + "\n")
    
    console.log(f"Visualization written to {output}")


if __name__ == '__main__':
    main()
