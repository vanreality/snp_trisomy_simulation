import pandas as pd
import numpy as np
import click
from pathlib import Path
from Bio import SeqIO
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from collections import defaultdict
import sys

console = Console()

def load_reference_genome(fasta_file: str) -> dict:
    """Loads the reference genome from a FASTA file.
    
    Args:
        fasta_file: Path to the reference genome FASTA file.
        
    Returns:
        Dictionary mapping chromosome names to their sequences (uppercase).
        
    Raises:
        ValueError: If the FASTA file cannot be read or is empty.
    """
    console.log(f"Loading reference genome from: {fasta_file}")
    
    try:
        reference = {}
        record_count = 0
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            reference[record.id] = str(record.seq).upper()
            record_count += 1
            
        if record_count == 0:
            raise ValueError("No sequences found in FASTA file")
            
        console.log(f"Loaded {record_count} chromosomes from reference genome")
        return reference
        
    except Exception as e:
        raise ValueError(f"Failed to load reference genome: {e}") from e

def normalize_chromosome_name(chrom: str) -> str:
    """Normalizes chromosome names to ensure consistency.
    
    Args:
        chrom: Original chromosome name.
        
    Returns:
        Normalized chromosome name with 'chr' prefix.
    """
    chrom = str(chrom).strip()
    if not chrom.startswith('chr'):
        return f'chr{chrom}'
    return chrom

def load_snp_data(tsv_file: str) -> pd.DataFrame:
    """Loads and validates SNP data from a TSV file.
    
    Args:
        tsv_file: Path to the TSV file containing SNP data.
        
    Returns:
        DataFrame with validated SNP data including normalized chromosome names.
        
    Raises:
        ValueError: If the file cannot be read, is empty, or has invalid format.
    """
    console.log(f"Loading SNP data from: {tsv_file}")
    
    try:
        # Read TSV file without header, using VCF-like column names
        snp_df = pd.read_csv(
            tsv_file,
            sep='\t',
            header=None,
            names=['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'],
            dtype={'chr': str, 'pos': int, 'ref': str, 'alt': str}
        )
        
        if snp_df.empty:
            raise ValueError("SNP data file is empty")
            
        # Validate required columns
        required_cols = ['chr', 'pos', 'ref', 'alt']
        for col in required_cols:
            if snp_df[col].isnull().any():
                raise ValueError(f"Missing values found in column: {col}")
                
        # Normalize chromosome names
        snp_df['chr_norm'] = snp_df['chr'].apply(normalize_chromosome_name)
        
        # Filter for single nucleotide variants only
        initial_count = len(snp_df)
        snp_df = snp_df[
            (snp_df['ref'].str.len() == 1) & 
            (snp_df['alt'].str.len() == 1) &
            (snp_df['ref'].str.match(r'^[ACGT]$')) &
            (snp_df['alt'].str.match(r'^[ACGT]$'))
        ]
        
        filtered_count = initial_count - len(snp_df)
        if filtered_count > 0:
            console.log(f"[yellow]WARNING:[/yellow] Filtered out {filtered_count} non-SNP variants")
            
        if snp_df.empty:
            raise ValueError("No valid SNP variants found after filtering")
            
        console.log(f"Loaded {len(snp_df)} valid SNP variants")
        return snp_df
        
    except FileNotFoundError:
        raise ValueError(f"SNP data file not found: {tsv_file}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"SNP data file is empty: {tsv_file}")
    except Exception as e:
        raise ValueError(f"Failed to load SNP data: {e}") from e

def get_reference_base(reference: dict, chrom: str, pos: int) -> str:
    """Gets the reference base at a specific genomic position.
    
    Args:
        reference: Dictionary of chromosome sequences.
        chrom: Chromosome name.
        pos: 1-based genomic position.
        
    Returns:
        Reference base at the position, or 'N' if not available.
    """
    # Try original chromosome name first, then normalized version
    chrom_to_try = [chrom]
    if chrom.startswith('chr'):
        chrom_to_try.append(chrom[3:])
    else:
        chrom_to_try.append(f'chr{chrom}')
        
    for chr_name in chrom_to_try:
        if chr_name in reference:
            try:
                # Convert to 0-based indexing
                return reference[chr_name][pos - 1]
            except IndexError:
                continue
                
    return 'N'

def check_cpg_status(reference: dict, chrom: str, pos: int, ref_allele: str, alt_allele: str) -> dict:
    """Checks if a SNP affects CpG sites.
    
    Args:
        reference: Dictionary of chromosome sequences.
        chrom: Chromosome name.
        pos: 1-based genomic position.
        ref_allele: Reference allele.
        alt_allele: Alternate allele.
        
    Returns:
        Dictionary with CpG analysis results including:
        - cpg_gain: Boolean indicating if SNP creates new CpG
        - cpg_loss: Boolean indicating if SNP destroys existing CpG
        - context_before: Trinucleotide context before mutation
        - context_after: Trinucleotide context after mutation
        - analysis_details: String describing the specific change
    """
    result = {
        'cpg_gain': False,
        'cpg_loss': False,
        'context_before': 'N/A',
        'context_after': 'N/A',
        'analysis_details': 'No CpG effect'
    }
    
    # Get surrounding bases
    upstream_base = get_reference_base(reference, chrom, pos - 1)
    downstream_base = get_reference_base(reference, chrom, pos + 1)
    
    # Verify reference allele matches
    ref_base_from_genome = get_reference_base(reference, chrom, pos)
    if ref_base_from_genome != ref_allele:
        result['analysis_details'] = f'Reference mismatch: expected {ref_allele}, found {ref_base_from_genome}'
        return result
    
    # Create trinucleotide contexts
    context_before = f'{upstream_base}{ref_allele}{downstream_base}'
    context_after = f'{upstream_base}{alt_allele}{downstream_base}'
    
    result['context_before'] = context_before
    result['context_after'] = context_after
    
    # Check for CpG gain scenarios
    if ref_allele == 'A' and alt_allele == 'G':
        # A->G: Check if upstream is C (CA -> CG)
        if upstream_base == 'C':
            result['cpg_gain'] = True
            result['analysis_details'] = f'CpG gain: {context_before} -> {context_after} (CA->CG)'
            
    elif ref_allele == 'T' and alt_allele == 'C':
        # T->C: Check if downstream is G (TG -> CG)
        if downstream_base == 'G':
            result['cpg_gain'] = True
            result['analysis_details'] = f'CpG gain: {context_before} -> {context_after} (TG->CG)'
    
    # Check for CpG loss scenarios
    elif ref_allele == 'C' and alt_allele == 'T':
        # C->T: Check if downstream is G (CG -> TG)
        if downstream_base == 'G':
            result['cpg_loss'] = True
            result['analysis_details'] = f'CpG loss: {context_before} -> {context_after} (CG->TG)'
            
    elif ref_allele == 'G' and alt_allele == 'A':
        # G->A: Check if upstream is C (CG -> CA)
        if upstream_base == 'C':
            result['cpg_loss'] = True
            result['analysis_details'] = f'CpG loss: {context_before} -> {context_after} (CG->CA)'
    
    return result

def analyze_snp_cpg_effects(reference: dict, snp_df: pd.DataFrame) -> tuple:
    """Analyzes CpG effects for all SNPs in the dataset.
    
    Args:
        reference: Dictionary of chromosome sequences.
        snp_df: DataFrame containing SNP data.
        
    Returns:
        Tuple containing (results_list, summary_stats).
    """
    console.log("Analyzing CpG effects for all SNPs...")
    
    results = []
    cpg_gain_count = 0
    cpg_loss_count = 0
    no_effect_count = 0
    error_count = 0
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Analyzing SNPs...", total=len(snp_df))
        
        for _, snp in snp_df.iterrows():
            try:
                cpg_result = check_cpg_status(
                    reference, 
                    snp['chr_norm'], 
                    snp['pos'], 
                    snp['ref'], 
                    snp['alt']
                )
                
                # Add SNP information to result
                cpg_result.update({
                    'chr': snp['chr'],
                    'pos': snp['pos'],
                    'ref': snp['ref'],
                    'alt': snp['alt']
                })
                
                results.append(cpg_result)
                
                # Update counters
                if cpg_result['cpg_gain']:
                    cpg_gain_count += 1
                elif cpg_result['cpg_loss']:
                    cpg_loss_count += 1
                else:
                    no_effect_count += 1
                    
            except Exception as e:
                error_count += 1
                console.log(f"[red]ERROR:[/red] Failed to analyze SNP at {snp['chr']}:{snp['pos']}: {e}")
                
            progress.update(task, advance=1)
    
    summary_stats = {
        'total_snps': len(snp_df),
        'cpg_gain': cpg_gain_count,
        'cpg_loss': cpg_loss_count,
        'no_effect': no_effect_count,
        'errors': error_count,
        'cpg_gain_ratio': cpg_gain_count / len(snp_df) if len(snp_df) > 0 else 0,
        'cpg_loss_ratio': cpg_loss_count / len(snp_df) if len(snp_df) > 0 else 0,
        'cpg_effect_ratio': (cpg_gain_count + cpg_loss_count) / len(snp_df) if len(snp_df) > 0 else 0
    }
    
    return results, summary_stats

def generate_report(results: list, summary_stats: dict, output_file: str):
    """Generates a comprehensive report of CpG analysis results.
    
    Args:
        results: List of analysis results for each SNP.
        summary_stats: Summary statistics.
        output_file: Path to output report file.
    """
    console.log(f"Generating report: {output_file}")
    
    with open(output_file, 'w') as f:
        # Write header and summary
        f.write("# SNP CpG Site Influence Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(f"Total SNPs analyzed: {summary_stats['total_snps']:,}\n")
        f.write(f"CpG gain events: {summary_stats['cpg_gain']:,} ({summary_stats['cpg_gain_ratio']:.4f})\n")
        f.write(f"CpG loss events: {summary_stats['cpg_loss']:,} ({summary_stats['cpg_loss_ratio']:.4f})\n")
        f.write(f"No CpG effect: {summary_stats['no_effect']:,}\n")
        f.write(f"Analysis errors: {summary_stats['errors']:,}\n")
        f.write(f"Overall CpG effect ratio: {summary_stats['cpg_effect_ratio']:.4f}\n\n")
        
        # Write detailed results
        f.write("## Detailed Results\n")
        f.write("Chr\tPos\tRef\tAlt\tCpG_Gain\tCpG_Loss\tContext_Before\tContext_After\tAnalysis_Details\n")
        
        for result in results:
            f.write(f"{result['chr']}\t{result['pos']}\t{result['ref']}\t{result['alt']}\t")
            f.write(f"{result['cpg_gain']}\t{result['cpg_loss']}\t")
            f.write(f"{result['context_before']}\t{result['context_after']}\t")
            f.write(f"{result['analysis_details']}\n")
    
    # Display summary in console
    table = Table(title="CpG Analysis Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Ratio", style="green")
    
    table.add_row("Total SNPs", f"{summary_stats['total_snps']:,}", "1.0000")
    table.add_row("CpG Gain", f"{summary_stats['cpg_gain']:,}", f"{summary_stats['cpg_gain_ratio']:.4f}")
    table.add_row("CpG Loss", f"{summary_stats['cpg_loss']:,}", f"{summary_stats['cpg_loss_ratio']:.4f}")
    table.add_row("No Effect", f"{summary_stats['no_effect']:,}", f"{(summary_stats['no_effect']/summary_stats['total_snps']):.4f}")
    table.add_row("Errors", f"{summary_stats['errors']:,}", f"{(summary_stats['errors']/summary_stats['total_snps']):.4f}")
    
    console.print(table)

def extract_cpg_affected_snps(results: list, original_snp_df: pd.DataFrame) -> tuple:
    """Extracts SNPs that have CpG gain or loss effects.
    
    Args:
        results: List of CpG analysis results for each SNP.
        original_snp_df: Original DataFrame containing all SNP data.
        
    Returns:
        Tuple containing (cpg_gain_snps_df, cpg_loss_snps_df) DataFrames.
        
    Raises:
        ValueError: If results and SNP data don't match in length.
    """
    if len(results) != len(original_snp_df):
        raise ValueError(f"Mismatch between results length ({len(results)}) and SNP data length ({len(original_snp_df)})")
    
    console.log("Extracting CpG gain and loss SNPs...")
    
    # Create lists to store indices of CpG affected SNPs
    cpg_gain_indices = []
    cpg_loss_indices = []
    
    for i, result in enumerate(results):
        if result['cpg_gain']:
            cpg_gain_indices.append(i)
        elif result['cpg_loss']:
            cpg_loss_indices.append(i)
    
    # Extract the corresponding SNPs from original dataframe
    cpg_gain_snps_df = original_snp_df.iloc[cpg_gain_indices].copy() if cpg_gain_indices else pd.DataFrame()
    cpg_loss_snps_df = original_snp_df.iloc[cpg_loss_indices].copy() if cpg_loss_indices else pd.DataFrame()
    
    console.log(f"Extracted {len(cpg_gain_snps_df)} CpG gain SNPs and {len(cpg_loss_snps_df)} CpG loss SNPs")
    
    return cpg_gain_snps_df, cpg_loss_snps_df

def save_cpg_snps_to_files(cpg_gain_snps_df: pd.DataFrame, cpg_loss_snps_df: pd.DataFrame, 
                          cpg_gain_output: str, cpg_loss_output: str):
    """Saves CpG gain and loss SNPs to separate TSV files.
    
    Args:
        cpg_gain_snps_df: DataFrame containing SNPs that cause CpG gain.
        cpg_loss_snps_df: DataFrame containing SNPs that cause CpG loss.
        cpg_gain_output: Output file path for CpG gain SNPs.
        cpg_loss_output: Output file path for CpG loss SNPs.
        
    Raises:
        IOError: If files cannot be written.
    """
    console.log(f"Saving CpG affected SNPs to files...")
    
    try:
        # Save CpG gain SNPs
        if not cpg_gain_snps_df.empty:
            # Remove the normalized chromosome column if it exists
            output_df = cpg_gain_snps_df.drop(columns=['chr_norm'], errors='ignore')
            output_df.to_csv(cpg_gain_output, sep='\t', header=False, index=False)
            console.log(f"[green]✓[/green] Saved {len(cpg_gain_snps_df)} CpG gain SNPs to: {cpg_gain_output}")
        else:
            # Create empty file if no CpG gain SNPs found
            Path(cpg_gain_output).touch()
            console.log(f"[yellow]![/yellow] No CpG gain SNPs found. Created empty file: {cpg_gain_output}")
        
        # Save CpG loss SNPs
        if not cpg_loss_snps_df.empty:
            # Remove the normalized chromosome column if it exists
            output_df = cpg_loss_snps_df.drop(columns=['chr_norm'], errors='ignore')
            output_df.to_csv(cpg_loss_output, sep='\t', header=False, index=False)
            console.log(f"[green]✓[/green] Saved {len(cpg_loss_snps_df)} CpG loss SNPs to: {cpg_loss_output}")
        else:
            # Create empty file if no CpG loss SNPs found
            Path(cpg_loss_output).touch()
            console.log(f"[yellow]![/yellow] No CpG loss SNPs found. Created empty file: {cpg_loss_output}")
            
    except Exception as e:
        raise IOError(f"Failed to save CpG affected SNPs to files: {e}") from e

@click.command()
@click.option('--fasta', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), 
              help='Path to the reference genome FASTA file.')
@click.option('--snp-list', required=True, type=click.Path(exists=True, dir_okay=False, readable=True), 
              help='Path to the SNP list TSV file (VCF-like format without header).')
@click.option('--output', required=True, type=str, 
              help='Path for the output report file.')
@click.option('--cpg-gain-output', default='cpg_gain_snps.tsv', type=str,
              help='Path for the CpG gain SNPs output file (default: cpg_gain_snps.tsv).')
@click.option('--cpg-loss-output', default='cpg_loss_snps.tsv', type=str,
              help='Path for the CpG loss SNPs output file (default: cpg_loss_snps.tsv).')
def main(fasta, snp_list, output, cpg_gain_output, cpg_loss_output):
    """Analyzes SNP influence on CpG sites.
    
    This tool examines how single nucleotide polymorphisms (SNPs) affect CpG dinucleotides
    in the genome. It identifies SNPs that either create new CpG sites (CpG gain) or 
    destroy existing CpG sites (CpG loss).
    
    CpG gain scenarios:
    - A->G mutation where upstream base is C (CA -> CG)
    - T->C mutation where downstream base is G (TG -> CG)
    
    CpG loss scenarios:
    - C->T mutation where downstream base is G (CG -> TG)
    - G->A mutation where upstream base is C (CG -> CA)
    """
    console.log("[bold blue]Starting CpG site influence analysis...[/bold blue]")
    
    try:
        # Load reference genome
        reference_genome = load_reference_genome(fasta)
        
        # Load SNP data
        snp_df = load_snp_data(snp_list)
        
        # Analyze CpG effects
        results, summary_stats = analyze_snp_cpg_effects(reference_genome, snp_df)
        
        # Generate report
        generate_report(results, summary_stats, output)
        
        # Extract and save CpG affected SNPs
        cpg_gain_snps_df, cpg_loss_snps_df = extract_cpg_affected_snps(results, snp_df)
        save_cpg_snps_to_files(cpg_gain_snps_df, cpg_loss_snps_df, cpg_gain_output, cpg_loss_output)
        
        console.log(f"[green]✓[/green] Analysis complete! Report saved to: {output}")
        console.log(f"[green]✓[/green] CpG affected SNPs saved to: {cpg_gain_output} and {cpg_loss_output}")
        
    except ValueError as e:
        console.log(f"[bold red]ERROR:[/bold red] {e}")
        sys.exit(1)
    except IOError as e:
        console.log(f"[bold red]FILE ERROR:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.log(f"[bold red]UNEXPECTED ERROR:[/bold red] {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
