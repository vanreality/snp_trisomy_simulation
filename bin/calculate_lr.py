#!/usr/bin/env python3
"""
Fetal Fraction Estimation and Log Likelihood Ratio Calculator for Aneuploidy Detection

This script analyzes cell-free DNA (cfDNA) sequencing data to:
1. Estimate fetal fraction from maternal plasma samples
2. Calculate log likelihood ratios for trisomy vs. disomy detection

The analysis uses SNP read counts and population allele frequencies to perform
statistical inference on fetal chromosomal abnormalities.
"""

import numpy as np
import pandas as pd
from multiprocessing import cpu_count
import click
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import the extracted classes
from FFEstimator import FFEstimator
from LLRCalculator import LLRCalculator


# Initialize rich console for beautiful output
console = Console()


@click.command()
@click.option(
    '--input-path', '-i',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to input TSV.GZ file containing SNP data'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    required=True,
    help='Output directory for results'
)
@click.option(
    '--ff-min',
    type=click.FloatRange(0.0, 1.0),
    default=0.001,
    help='Minimum fetal fraction for estimation (default: 0.001)'
)
@click.option(
    '--ff-max',
    type=click.FloatRange(0.0, 1.0),
    default=0.3,
    help='Maximum fetal fraction for estimation (default: 0.3)'
)
@click.option(
    '--ff-step',
    type=click.FloatRange(0.0001, 0.1),
    default=0.001,
    help='Step size for fetal fraction grid search (default: 0.001)'
)
@click.option(
    '--chromosomes',
    default='1-22',
    help='Chromosomes to analyze (e.g., "1-22", "1,2,3", or "21"). Default: 1-22'
)
@click.option(
    '--mode',
    type=click.Choice(['cfDNA', 'cfDNA+WBC', 'cfDNA+model', 'cfDNA+model+mGT'], case_sensitive=False),
    default='cfDNA',
    help='Analysis mode: cfDNA (standard), cfDNA+WBC (with maternal WBC), cfDNA+model (with modeled maternal reads), cfDNA+model+mGT (with modeled reads and maternal genotyping)'
)
@click.option(
    '--cfdna-ref-col',
    default='cfDNA_ref_reads',
    help='Column name for cfDNA reference reads (default: cfDNA_ref_reads)'
)
@click.option(
    '--cfdna-alt-col',
    default='cfDNA_alt_reads',
    help='Column name for cfDNA alternative reads (default: cfDNA_alt_reads)'
)
@click.option(
    '--wbc-ref-col',
    default='maternal_ref_reads',
    help='Column name for maternal WBC reference reads (default: maternal_ref_reads)'
)
@click.option(
    '--wbc-alt-col',
    default='maternal_alt_reads',
    help='Column name for maternal WBC alternative reads (default: maternal_alt_reads)'
)
@click.option(
    '--model-ref-col',
    default='fetal_ref_reads_from_model',
    help='Column name for modeled fetal reference reads (default: fetal_ref_reads_from_model)'
)
@click.option(
    '--model-alt-col',
    default='fetal_alt_reads_from_model',
    help='Column name for modeled fetal alternative reads (default: fetal_alt_reads_from_model)'
)
@click.option(
    '--min-raw-depth',
    type=click.IntRange(0, None),
    default=0,
    help='Minimum raw depth filter for cfDNA reads (default: 0)'
)
@click.option(
    '--min-model-depth',
    type=click.IntRange(0, None),
    default=0,
    help='Minimum model depth filter for model filtered reads (default: 0)'
)
@click.option(
    '--ncpus',
    type=click.IntRange(1, cpu_count()),
    default=cpu_count(),
    help=f'Number of CPU cores to use for parallel processing (default: {cpu_count()})'
)
@click.option(
    '--fast',
    is_flag=True,
    default=False,
    help='Fast mode: estimate fetal fraction once using all chromosomes instead of per-chromosome estimation (default: False)'
)
@click.option(
    '--beta-binomial',
    is_flag=True,
    default=False,
    help='Use beta-binomial distribution for allelic distribution (default: False)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def main(
    input_path: Path,
    output_dir: Path,
    ff_min: float,
    ff_max: float,
    ff_step: float,
    chromosomes: str,
    mode: str,
    cfdna_ref_col: str,
    cfdna_alt_col: str,
    wbc_ref_col: str,
    wbc_alt_col: str,
    model_ref_col: str,
    model_alt_col: str,
    min_raw_depth: int,
    min_model_depth: int,
    ncpus: int,
    fast: bool,
    beta_binomial: bool,
    verbose: bool
) -> None:
    """
    Fetal Fraction Estimation and Log Likelihood Ratio Calculator.
    
    This tool analyzes cell-free DNA sequencing data to estimate fetal fraction
    and calculate log likelihood ratios for trisomy detection across chromosomes.
    
    The input file should be a TSV.GZ file with columns:
    chr, pos, af, and read count columns (names configurable via CLI options)
     
    Supports four analysis modes:
    - cfDNA: Standard cell-free DNA analysis
    - cfDNA+WBC: Analysis with maternal white blood cell data
    - cfDNA+model: Analysis with modeled fetal reads
    - cfDNA+model+mGT: Analysis with modeled fetal reads and maternal genotyping filtering
    
    Analysis speed options:
    - Standard mode: Estimates fetal fraction per chromosome using background chromosomes
    - Fast mode (--fast): Estimates fetal fraction once using all chromosomes for faster processing
    
    Depth filtering options:
    - min-raw-depth: Filter SNPs by minimum cfDNA read depth
    - min-model-depth: Filter SNPs by minimum model filtered read depth (cfDNA+model modes only)
    
    Results are saved as TSV files with fetal fraction estimates and log likelihood ratios.
    
    Multi-threading is used by default to speed up fetal fraction estimation.
    Use --ncpus to control the number of CPU cores used.
    """
    # Configure console output
    if verbose:
        console.print("[blue]Verbose mode enabled[/blue]")
    
    try:
        # Validate and parse chromosome specification
        target_chromosomes = parse_chromosome_list(chromosomes)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which columns to display based on mode
        mode_info = f"Mode: {mode}"
        if mode == 'cfDNA':
            column_info = f"cfDNA columns: {cfdna_ref_col}, {cfdna_alt_col}"
        elif mode == 'cfDNA+WBC':
            column_info = f"cfDNA columns: {cfdna_ref_col}, {cfdna_alt_col}\nWBC columns: {wbc_ref_col}, {wbc_alt_col}"
        elif mode == 'cfDNA+model':
            column_info = f"cfDNA columns: {cfdna_ref_col}, {cfdna_alt_col}\nModel columns: {model_ref_col}, {model_alt_col}"
        elif mode == 'cfDNA+model+mGT':
            column_info = f"cfDNA columns: {cfdna_ref_col}, {cfdna_alt_col}\nModel columns: {model_ref_col}, {model_alt_col}\nCalculated maternal columns: {wbc_ref_col}, {wbc_alt_col}"
        
        # Build depth filter info
        depth_filter_info = f"Raw depth filter: ≥{min_raw_depth}"
        if mode in ['cfDNA+model', 'cfDNA+model+mGT']:
            depth_filter_info += f"\nModel depth filter: ≥{min_model_depth}"
        
        # Build analysis mode info
        analysis_mode_info = f"Analysis mode: {'Fast' if fast else 'Standard'}"
        if fast:
            analysis_mode_info += " (single FF estimation using all chromosomes)"
        else:
            analysis_mode_info += " (per-chromosome FF estimation using background chromosomes)"
        
        # Display startup information
        console.print(Panel.fit(
            f"[bold green]Fetal Fraction & Log Likelihood Ratio Calculator[/bold green]\n"
            f"Input: {input_path}\n"
            f"Output: {output_dir}\n"
            f"{mode_info}\n"
            f"{column_info}\n"
            f"Chromosomes: {', '.join(map(str, target_chromosomes))}\n"
            f"FF Range: {ff_min:.3f} - {ff_max:.3f} (step: {ff_step:.3f})\n"
            f"{analysis_mode_info}\n"
            f"{depth_filter_info}\n"
            f"CPU Cores: {ncpus}",
            title="Configuration"
        ))
        
        # Generate output filename based on input
        input_filename = input_path.stem.replace('.tsv', '')
        output_path = output_dir / f'{input_filename}_lr.tsv'
        
        # Load and validate input data
        console.print("[cyan]Loading input data...[/cyan]")
        df = load_and_validate_data(
            input_path,
            mode=mode,
            cfdna_ref_col=cfdna_ref_col,
            cfdna_alt_col=cfdna_alt_col,
            wbc_ref_col=wbc_ref_col,
            wbc_alt_col=wbc_alt_col,
            model_ref_col=model_ref_col,
            model_alt_col=model_alt_col,
            min_raw_depth=min_raw_depth,
            min_model_depth=min_model_depth
        )
        
        console.print(f"[green]✓ Loaded {len(df)} SNPs from {df['chr'].nunique()} chromosomes[/green]")
        
        # Initialize classes based on mode
        if mode in ['cfDNA', 'cfDNA+model']:
            # Use cfDNA mode for both standard and model-based analysis
            ff_estimator = FFEstimator(mode='cfDNA')
            llr_calculator = LLRCalculator(mode='cfDNA', beta_binomial=beta_binomial)
        elif mode == 'cfDNA+WBC':
            ff_estimator = FFEstimator(mode='cfDNA+WBC')
            llr_calculator = LLRCalculator(mode='cfDNA+WBC', beta_binomial=beta_binomial)
        elif mode == 'cfDNA+model+mGT':
            # Use cfDNA+model+mGT mode for both FF estimation and LLR calculation
            ff_estimator = FFEstimator(mode='cfDNA+model+mGT')
            llr_calculator = LLRCalculator(mode='cfDNA+model+mGT', beta_binomial=beta_binomial)
        
        # Initialize results storage
        results_list = []
        
        # Fast mode: Estimate fetal fraction once using all chromosomes
        global_fetal_fraction = None
        if fast:
            console.print(f"\n[bold yellow]Fast Mode: Estimating fetal fraction using all {len(df)} SNPs...[/bold yellow]")
            
            try:
                if mode == 'cfDNA':
                    global_fetal_fraction, _ = ff_estimator.estimate(
                        df,  # Use all data instead of background only
                        f_min=ff_min,
                        f_max=ff_max,
                        f_step=ff_step,
                        ncpus=ncpus,
                        ref_col=cfdna_ref_col,
                        alt_col=cfdna_alt_col
                    )
                elif mode == 'cfDNA+WBC':
                    global_fetal_fraction, _ = ff_estimator.estimate(
                        df,  # Use all data instead of background only
                        f_min=ff_min,
                        f_max=ff_max,
                        f_step=ff_step,
                        ncpus=ncpus,
                        ref_col=cfdna_ref_col,
                        alt_col=cfdna_alt_col,
                        maternal_ref_col=wbc_ref_col,
                        maternal_alt_col=wbc_alt_col
                    )
                elif mode in ['cfDNA+model', 'cfDNA+model+mGT']:
                    global_fetal_fraction, _ = ff_estimator.estimate(
                        df,  # Use all data instead of background only
                        f_min=ff_min,
                        f_max=ff_max,
                        f_step=ff_step,
                        ncpus=ncpus,
                        ref_col=model_ref_col,
                        alt_col=model_alt_col
                    )
                
                console.print(f"[bold green]✓ Global fetal fraction estimated: {global_fetal_fraction:.3f}[/bold green]")
                console.print("[cyan]This fetal fraction will be used for all chromosome analyses[/cyan]")
                
            except Exception as e:
                console.print(f"[red]✗ Failed to estimate global fetal fraction: {str(e)}[/red]")
                if verbose:
                    console.print_exception()
                console.print("[yellow]Falling back to standard per-chromosome estimation[/yellow]")
                fast = False  # Disable fast mode if global FF estimation fails
                global_fetal_fraction = None
        
        # Process each target chromosome
        for target_chr in target_chromosomes:
            chr_name = f"chr{target_chr}"
            console.print(f"\n[bold cyan]Processing chromosome {target_chr}[/bold cyan]")
            
            # Filter data for target chromosome
            target_data = df[df['chr'] == chr_name]
            background_data = df[df['chr'] != chr_name]
            
            if len(target_data) == 0:
                console.print(f"[yellow]Warning: No SNPs found for {chr_name}, skipping[/yellow]")
                continue
            
            if len(background_data) == 0:
                console.print(f"[red]Error: No background SNPs available for {chr_name}[/red]")
                continue
            
            console.print(f"Target SNPs: {len(target_data)}, Background SNPs: {len(background_data)}")
            
            try:
                # Determine fetal fraction to use
                if fast and global_fetal_fraction is not None:
                    # Fast mode: Use pre-computed global fetal fraction
                    est_ff = global_fetal_fraction
                    console.print(f"[cyan]Using global fetal fraction ({est_ff:.3f}) for {chr_name}...[/cyan]")
                else:
                    # Standard mode: Estimate fetal fraction using background chromosomes
                    console.print(f"[cyan]Estimating fetal fraction for {chr_name}...[/cyan]")
                    if mode == 'cfDNA':
                        est_ff, _ = ff_estimator.estimate(
                            background_data,
                            f_min=ff_min,
                            f_max=ff_max,
                            f_step=ff_step,
                            ncpus=ncpus,
                            ref_col=cfdna_ref_col,
                            alt_col=cfdna_alt_col
                        )
                    elif mode == 'cfDNA+WBC':
                        est_ff, _ = ff_estimator.estimate(
                            background_data,
                            f_min=ff_min,
                            f_max=ff_max,
                            f_step=ff_step,
                            ncpus=ncpus,
                            ref_col=cfdna_ref_col,
                            alt_col=cfdna_alt_col,
                            maternal_ref_col=wbc_ref_col,
                            maternal_alt_col=wbc_alt_col
                        )
                    elif mode in ['cfDNA+model', 'cfDNA+model+mGT']:
                        est_ff, _ = ff_estimator.estimate(
                            background_data,
                            f_min=ff_min,
                            f_max=ff_max,
                            f_step=ff_step,
                            ncpus=ncpus,
                            ref_col=model_ref_col,
                            alt_col=model_alt_col
                        )
                
                # Calculate log likelihood ratio for target chromosome
                console.print(f"[cyan]Calculating log likelihood ratio for {chr_name}...[/cyan]")
                if mode == 'cfDNA':
                    llr = llr_calculator.calculate(target_data, 
                                           est_ff, 
                                           ref_col=cfdna_ref_col, 
                                           alt_col=cfdna_alt_col)
                elif mode == 'cfDNA+WBC':
                    llr = llr_calculator.calculate(
                        target_data, 
                        est_ff, 
                        ref_col=cfdna_ref_col, 
                        alt_col=cfdna_alt_col, 
                        maternal_ref_col=wbc_ref_col, 
                        maternal_alt_col=wbc_alt_col
                    )
                elif mode == 'cfDNA+model':
                    llr = llr_calculator.calculate(
                        target_data, 
                        est_ff, 
                        ref_col=model_ref_col, 
                        alt_col=model_alt_col
                    )
                elif mode == 'cfDNA+model+mGT':
                    llr = llr_calculator.calculate(
                        target_data, 
                        est_ff, 
                        ref_col=model_ref_col, 
                        alt_col=model_alt_col,
                        maternal_ref_col=wbc_ref_col,
                        maternal_alt_col=wbc_alt_col
                    )
                
                # Store results
                results_list.append({
                    'Chrom': chr_name,
                    'Log_LR': llr,
                    'Fetal Fraction': est_ff
                })
                
                console.print(f"[green]✓ {chr_name}: FF = {est_ff:.3f}, Log LR = {llr:.2f}[/green]")

            except Exception as e:
                console.print(f"[red]✗ Error processing {chr_name}: {str(e)}[/red]")
                if verbose:
                    console.print_exception()
                continue
        
        # Save results
        if results_list:
            console.print(f"\n[cyan]Saving results to {output_path}...[/cyan]")
            results_df = pd.DataFrame(results_list, columns=['Chrom', 'Log_LR', 'Fetal Fraction'])
            results_df.to_csv(output_path, sep='\t', index=False)
            
            # Display summary table
            display_results_summary(results_df)
            console.print(f"[bold green]✓ Analysis complete! Results saved to {output_path}[/bold green]")
        else:
            console.print("[red]✗ No results generated - check input data and parameters[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]✗ Fatal error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def parse_chromosome_list(chr_spec: str) -> list:
    """
    Parse chromosome specification string into list of chromosome numbers.
    
    Args:
        chr_spec: String like "1-22", "1,2,3", or "21"
    
    Returns:
        List of chromosome numbers (integers)
    
    Raises:
        ValueError: If chromosome specification is invalid
    """
    try:
        if '-' in chr_spec:
            # Range specification (e.g., "1-22")
            start, end = map(int, chr_spec.split('-'))
            return list(range(start, end + 1))
        elif ',' in chr_spec:
            # Comma-separated list (e.g., "1,2,3")
            return [int(x.strip()) for x in chr_spec.split(',')]
        else:
            # Single chromosome (e.g., "21")
            return [int(chr_spec)]
    except ValueError:
        raise ValueError(f"Invalid chromosome specification: {chr_spec}")


def load_and_validate_data(
    input_path: Path, 
    mode: str = 'cfDNA',
    cfdna_ref_col: str = 'cfDNA_ref_reads',
    cfdna_alt_col: str = 'cfDNA_alt_reads',
    wbc_ref_col: str = 'maternal_ref_reads',
    wbc_alt_col: str = 'maternal_alt_reads',
    model_ref_col: str = 'fetal_ref_reads_from_model',
    model_alt_col: str = 'fetal_alt_reads_from_model',
    min_raw_depth: int = 0,
    min_model_depth: int = 0
) -> pd.DataFrame:
    """
    Load and validate input SNP data from TSV.GZ file.
    
    This function validates the presence of required columns based on the analysis mode
    and performs data type validation and cleaning. It also applies depth filters.
    
    Args:
        input_path (Path): Path to input file
        mode (str): Analysis mode ('cfDNA', 'cfDNA+WBC', or 'cfDNA+model')
        cfdna_ref_col (str): Column name for cfDNA reference reads
        cfdna_alt_col (str): Column name for cfDNA alternative reads
        wbc_ref_col (str): Column name for maternal WBC reference reads
        wbc_alt_col (str): Column name for maternal WBC alternative reads
        model_ref_col (str): Column name for modeled fetal reference reads
        model_alt_col (str): Column name for modeled fetal alternative reads
        min_raw_depth (int): Minimum depth for cfDNA reads (default: 0)
        min_model_depth (int): Minimum depth for modeled reads (default: 0)
    
    Returns:
        pd.DataFrame: Validated pandas DataFrame with proper data types
    
    Raises:
        ValueError: If data validation fails or required columns are missing
        FileNotFoundError: If input file doesn't exist
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        # Load data with appropriate compression detection
        df = pd.read_csv(input_path, sep='\t', compression='gzip' if input_path.suffix == '.gz' else None)
    except Exception as e:
        raise ValueError(f"Failed to load input file: {str(e)}")
    
    # Determine required columns based on mode
    base_columns = {'chr', 'pos', 'af'}
    required_columns = base_columns.copy()
    read_columns = []
    
    if mode == 'cfDNA':
        required_columns.update({cfdna_ref_col, cfdna_alt_col})
        read_columns = [cfdna_ref_col, cfdna_alt_col]
        console.print(f"[cyan]Validating cfDNA mode with columns: {cfdna_ref_col}, {cfdna_alt_col}[/cyan]")
    elif mode == 'cfDNA+WBC':
        required_columns.update({cfdna_ref_col, cfdna_alt_col, wbc_ref_col, wbc_alt_col})
        read_columns = [cfdna_ref_col, cfdna_alt_col, wbc_ref_col, wbc_alt_col]
        console.print(f"[cyan]Validating cfDNA+WBC mode with columns: {', '.join(read_columns)}[/cyan]")
    elif mode == 'cfDNA+model':
        required_columns.update({cfdna_ref_col, cfdna_alt_col, model_ref_col, model_alt_col})
        read_columns = [cfdna_ref_col, cfdna_alt_col, model_ref_col, model_alt_col]
        console.print(f"[cyan]Validating cfDNA+model mode with columns: {', '.join(read_columns)}[/cyan]")
    elif mode == 'cfDNA+model+mGT':
        required_columns.update({cfdna_ref_col, cfdna_alt_col, model_ref_col, model_alt_col})
        read_columns = [cfdna_ref_col, cfdna_alt_col, model_ref_col, model_alt_col]
        console.print(f"[cyan]Validating cfDNA+model+mGT mode with columns: {', '.join(read_columns)}[/cyan]")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: cfDNA, cfDNA+WBC, cfDNA+model, cfDNA+model+mGT")
    
    # Check for required columns
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns for mode '{mode}': {missing}")
    
    # Basic data validation
    if len(df) == 0:
        raise ValueError("Input file is empty")
    
    # Ensure proper data types for base columns
    df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
    df['af'] = pd.to_numeric(df['af'], errors='coerce')
    
    # Ensure proper data types for read count columns
    for col in read_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
    
    # Build invalid mask for base columns
    invalid_mask = (
        df['af'].isna() | (df['af'] < 0) | (df['af'] > 1)
    )
    
    # Add read count validation to invalid mask
    for col in read_columns:
        invalid_mask |= (df[col].isna() | (df[col] < 0))
    
    if invalid_mask.any():
        console.print(f"[yellow]Warning: Removing {invalid_mask.sum()} rows with invalid data[/yellow]")
        df = df[~invalid_mask]
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after filtering")
    
    # Apply depth filters
    initial_count = len(df)
    
    # Filter by minimum raw depth (cfDNA reads)
    if min_raw_depth > 0:
        df['cfdna_total_depth'] = df[cfdna_ref_col] + df[cfdna_alt_col]
        df = df[df['cfdna_total_depth'] >= min_raw_depth]
        remaining_count = len(df)
        percentage = (remaining_count / initial_count) * 100 if initial_count > 0 else 0
        console.print(f"[cyan]Raw depth filter (≥{min_raw_depth}): {remaining_count:,} SNPs remaining ({percentage:.1f}%)[/cyan]")
        df = df.drop(columns=['cfdna_total_depth']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No SNPs remaining after raw depth filtering")
    
    # Filter by minimum model depth (only for cfDNA+model mode)
    if min_model_depth > 0 and mode == 'cfDNA+model':
        current_count = len(df)
        df['model_total_depth'] = df[model_ref_col] + df[model_alt_col]
        df = df[df['model_total_depth'] >= min_model_depth]
        remaining_count = len(df)
        percentage = (remaining_count / current_count) * 100 if current_count > 0 else 0
        console.print(f"[cyan]Model depth filter (≥{min_model_depth}): {remaining_count:,} SNPs remaining ({percentage:.1f}%)[/cyan]")
        df = df.drop(columns=['model_total_depth']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No SNPs remaining after model depth filtering")
    elif min_model_depth > 0 and mode not in ['cfDNA+model', 'cfDNA+model+mGT']:
        console.print(f"[yellow]Warning: Model depth filter ignored for mode '{mode}' (only applies to 'cfDNA+model' modes)[/yellow]")
    
    # Calculate maternal reads for cfDNA+model+mGT mode
    if mode == 'cfDNA+model+mGT':
        console.print("[cyan]Calculating maternal reads as cfDNA - model reads[/cyan]")
        
        # Calculate maternal reads by subtracting model reads from cfDNA reads
        df[wbc_ref_col] = df[cfdna_ref_col] - df[model_ref_col]
        df[wbc_alt_col] = df[cfdna_alt_col] - df[model_alt_col]
        
        # Handle edge cases where subtraction results in negative values
        negative_ref = df[wbc_ref_col] < 0
        negative_alt = df[wbc_alt_col] < 0
        
        if negative_ref.any() or negative_alt.any():
            negative_count = (negative_ref | negative_alt).sum()
            console.print(f"[yellow]Warning: {negative_count} SNPs have negative maternal reads (model > cfDNA), setting to 0[/yellow]")
            df.loc[negative_ref, wbc_ref_col] = 0
            df.loc[negative_alt, wbc_alt_col] = 0
        
        # Filter out SNPs where maternal coverage is zero after calculation
        df['maternal_total_reads'] = df[wbc_ref_col] + df[wbc_alt_col]
        zero_maternal_coverage = df['maternal_total_reads'] == 0
        
        if zero_maternal_coverage.any():
            console.print(f"[yellow]Warning: Removing {zero_maternal_coverage.sum()} SNPs with zero maternal coverage[/yellow]")
            df = df[~zero_maternal_coverage]
        
        # Clean up temporary column
        df = df.drop(columns=['maternal_total_reads']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No SNPs remaining after calculating maternal reads")
        
        console.print(f"[green]Successfully calculated maternal reads for {len(df)} SNPs[/green]")
    
    return df


def display_results_summary(results_df: pd.DataFrame) -> None:
    """
    Display a formatted summary table of results.
    
    Args:
        results_df: DataFrame containing Log_LR and FF results with columns 'Chrom', 'Log_LR', 'Fetal Fraction'.
    """
    table = Table(title="Analysis Results Summary")
    table.add_column("Chromosome", justify="center", style="cyan")
    table.add_column("Fetal Fraction", justify="right", style="green")
    table.add_column("Log Likelihood Ratio", justify="right", style="yellow")
    table.add_column("Interpretation", justify="center", style="magenta")
    
    for _, row in results_df.iterrows():
        chr_name = row['Chrom']
        ff_val = row['Fetal Fraction']
        log_lr_val = row['Log_LR']
        
        # Simple interpretation logic based on log likelihood ratios
        # log(10) ≈ 2.3, log(1) = 0
        if log_lr_val > np.log(10):  # Equivalent to LR > 10
            interpretation = "Strong Evidence"
        elif log_lr_val > 0:  # Equivalent to LR > 1
            interpretation = "Moderate Evidence"
        elif log_lr_val == 0:  # Equivalent to LR = 1
            interpretation = "No Evidence"
        else:  # log_lr_val < 0, equivalent to LR < 1
            interpretation = "Against Trisomy"
        
        table.add_row(
            str(chr_name.replace("chr", "")),
            f"{ff_val:.3f}",
            f"{log_lr_val:.2f}",
            interpretation
        )
    
    console.print(table)


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set
    
    main()