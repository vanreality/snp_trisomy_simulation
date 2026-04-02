#!/usr/bin/env python3
"""
Filter pileup files based on Variant Allele Frequency (VAF) thresholds.

This script processes a single pileup file, computes various VAF metrics,
and filters variants based on specified VAF conditions.
"""

import sys
from pathlib import Path

import click
import polars as pl
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

console = Console()


def validate_input_file(input_path: Path) -> None:
    """
    Validate that the input file exists and has the correct format.

    Args:
        input_path: Path to the input pileup file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input file has an invalid extension.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    valid_extensions = {".tsv.gz", ".tsv"}
    if not any(str(input_path).endswith(ext) for ext in valid_extensions):
        raise ValueError(
            f"Invalid file extension. Expected .tsv or .tsv.gz, got: {input_path.suffix}"
        )


def validate_required_columns(schema: dict[str, pl.DataType]) -> None:
    """
    Validate that the schema contains all required columns.

    Args:
        schema: Column name -> DataType mapping from a LazyFrame/DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = ["cfDNA_alt_reads", "current_depth"]
    missing_columns = [col for col in required_columns if col not in schema]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def build_vaf_expressions(columns: list[str]) -> list[pl.Expr]:
    """
    Build a list of Polars expressions for VAF computation.

    Returns expressions for raw_vaf (always), target_vaf and background_vaf
    (only when the source columns are present).

    Division by zero produces null which is filled to 0.
    """
    exprs: list[pl.Expr] = [
        (pl.col("cfDNA_alt_reads") / pl.col("current_depth"))
        .fill_null(0)
        .fill_nan(0)
        .alias("raw_vaf"),
    ]

    if (
        "fetal_alt_reads_from_model" in columns
        and "fetal_current_depth_from_model" in columns
    ):
        exprs.append(
            (
                pl.col("fetal_alt_reads_from_model")
                / pl.col("fetal_current_depth_from_model")
            )
            .fill_null(0)
            .fill_nan(0)
            .alias("target_vaf")
        )

    if "maternal_alt_reads" in columns and "maternal_current_depth" in columns:
        exprs.append(
            (pl.col("maternal_alt_reads") / pl.col("maternal_current_depth"))
            .fill_null(0)
            .fill_nan(0)
            .alias("background_vaf")
        )

    return exprs


def build_vaf_filter() -> pl.Expr:
    """VAF filter: keep rows where raw_vaf is in (0, 0.2) or (0.8, 1.0)."""
    vaf = pl.col("raw_vaf")
    return (vaf.gt(0) & vaf.lt(0.2)) | (vaf.gt(0.8) & vaf.lt(1.0))


def process_pileup_file(input_path: Path) -> pl.DataFrame:
    """
    Process a single pileup file: load, compute VAF, and filter.

    Uses Polars lazy evaluation so that reading, VAF computation, and
    filtering are fused into a single optimised scan — only the rows
    that survive the filter are materialised in memory.

    Args:
        input_path: Path to the input pileup file.

    Returns:
        Filtered DataFrame with VAF metrics.
    """
    validate_input_file(input_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_read = progress.add_task("[cyan]Scanning pileup file...", total=1)

        try:
            lf = pl.scan_csv(input_path, separator="\t")
            schema = lf.collect_schema()
        except pl.exceptions.NoDataError:
            progress.update(task_read, completed=1)
            console.print("[yellow]Warning: Input file contains no data[/yellow]")
            return pl.DataFrame()
        except Exception as e:
            raise Exception(f"Error reading pileup file: {e}") from e

        progress.update(task_read, completed=1)

        task_val = progress.add_task("[cyan]Validating columns...", total=1)
        validate_required_columns(schema)
        progress.update(task_val, completed=1)
        console.print("[green]✓[/green] Required columns present")

        columns = list(schema.names())
        has_fetal = (
            "fetal_alt_reads_from_model" in columns
            and "fetal_current_depth_from_model" in columns
        )
        has_maternal = (
            "maternal_alt_reads" in columns and "maternal_current_depth" in columns
        )

        if has_fetal:
            console.print("[green]✓[/green] Fetal columns detected")
        if has_maternal:
            console.print("[green]✓[/green] Maternal columns detected")

        task_compute = progress.add_task(
            "[cyan]Computing VAF & filtering...", total=1
        )

        vaf_exprs = build_vaf_expressions(columns)
        lf = lf.with_columns(vaf_exprs).filter(build_vaf_filter())

        filtered_df = lf.collect(streaming=True)

        progress.update(task_compute, completed=1)

        vaf_computed = ["raw_vaf"]
        if has_fetal:
            vaf_computed.append("target_vaf")
        if has_maternal:
            vaf_computed.append("background_vaf")
        console.print(
            f"[green]✓[/green] VAF metrics computed: {', '.join(vaf_computed)}"
        )
        console.print(
            f"[green]✓[/green] Filtered to {filtered_df.height:,} variants"
        )

    return filtered_df


def get_output_filename(input_path: Path) -> str:
    """
    Generate output filename based on input filename.

    Extracts the sample name from the input filename and creates
    an output filename with '_filtered_pileup.tsv' suffix.

    Args:
        input_path: Path to the input file.

    Returns:
        Output filename string.

    Examples:
        >>> get_output_filename(Path("sample123_pileup.tsv.gz"))
        'sample123_filtered_pileup.tsv'
        >>> get_output_filename(Path("test_pileup.tsv"))
        'test_filtered_pileup.tsv'
    """
    filename = input_path.name

    if filename.endswith(".tsv.gz"):
        base_name = filename[:-7]
    elif filename.endswith(".tsv"):
        base_name = filename[:-4]
    else:
        base_name = filename

    if base_name.endswith("_pileup"):
        base_name = base_name[:-7]

    return f"{base_name}_filtered_pileup.tsv"


@click.command()
@click.option(
    "--input-path",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the input pileup file (TSV or TSV.GZ format).",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory where the filtered output file will be saved.",
)
def main(input_path: Path, output_dir: Path) -> None:
    """
    Filter pileup files based on Variant Allele Frequency (VAF) thresholds.

    This script processes a pileup file, computes VAF metrics (raw_vaf is always
    computed; target_vaf and background_vaf are computed if fetal/maternal columns
    exist), and filters variants where raw_vaf is in the ranges (0, 0.2) or (0.8, 1.0).

    Required columns:
        - cfDNA_alt_reads
        - current_depth

    Optional columns (for additional VAF metrics):
        - fetal_alt_reads_from_model, fetal_current_depth_from_model
        - maternal_alt_reads, maternal_current_depth

    Examples:
        $ python filter_pileup.py \\
            --input-path sample_pileup.tsv.gz \\
            --output-dir ./output/
    """
    console.print("[bold blue]Pileup Filter[/bold blue]", style="bold")
    console.print("=" * 60)
    console.print(f"Input file: {input_path}")
    console.print(f"Output directory: {output_dir}")
    console.print("=" * 60)
    console.print()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        filtered_df = process_pileup_file(input_path)

        output_filename = get_output_filename(input_path)
        output_path = output_dir / output_filename

        if filtered_df.is_empty():
            console.print(
                "[yellow]No variants passed the filtering criteria[/yellow]"
            )
            filtered_df.write_csv(output_path, separator="\t")
            console.print(f"[yellow]Empty output file created: {output_path}[/yellow]")
            return

        console.print("\n[cyan]Saving filtered results...[/cyan]")
        filtered_df.write_csv(output_path, separator="\t")

        console.print()
        console.print("[bold green]✓ Success![/bold green]")
        console.print(f"Output file: {output_path}")
        console.print(f"Output size: {filtered_df.height:,} variants")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        console.print(f"[bold red]Validation Error:[/bold red] {e}", err=True)
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()