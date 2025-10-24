'''
Copyright (c) 2025-10-23 by LiMingyang, YiLab, Peking University.

Author: Li Mingyang (limingyang200101@gmail.com)

Institute: AAIS, Peking University

File Name: /lustre1/cqyi/myli/bert/analysis_nipt/analysis_targets/220k/20251015-XML_igtc/DMRed_bed_intersect/assembed_to_bam.py

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''
from rich import print, pretty
from rich.traceback import install
from rich.progress import Progress, BarColumn, TimeRemainingColumn, MofNCompleteColumn
pretty.install()
install(show_locals=True)
import re
import sys
import click
import polars as pl
import pysam
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import tempfile
import os
import multiprocessing

# 准备BAM头部信息
HEADER_DICT = OrderedDict([
    ("HD", {"VN": "1.6", "SO": "coordinate"}),
    ("SQ", [
        {"LN": 248956422, "SN": "chr1"},
        {"LN": 242193529, "SN": "chr2"},
        {"LN": 198295559, "SN": "chr3"},
        {"LN": 190214555, "SN": "chr4"},
        {"LN": 181538259, "SN": "chr5"},
        {"LN": 170805979, "SN": "chr6"},
        {"LN": 159345973, "SN": "chr7"},
        {"LN": 145138636, "SN": "chr8"},
        {"LN": 138394717, "SN": "chr9"},
        {"LN": 133797422, "SN": "chr10"},
        {"LN": 135086622, "SN": "chr11"},
        {"LN": 133275309, "SN": "chr12"},
        {"LN": 114364328, "SN": "chr13"},
        {"LN": 107043718, "SN": "chr14"},
        {"LN": 101991189, "SN": "chr15"},
        {"LN": 90338345, "SN": "chr16"},
        {"LN": 83257441, "SN": "chr17"},
        {"LN": 80373285, "SN": "chr18"},
        {"LN": 58617616, "SN": "chr19"},
        {"LN": 64444167, "SN": "chr20"},
        {"LN": 46709983, "SN": "chr21"},
        {"LN": 50818468, "SN": "chr22"},
        {"LN": 156040895, "SN": "chrX"},
        {"LN": 57227415, "SN": "chrY"},
        {"LN": 16569, "SN": "chrM"}
    ]),
    ("RG", [{"ID": "SAMPLE", "SM": "SAMPLE"}])
])

# 创建染色体名到索引的映射
CHROM_TO_TID = {sq["SN"]: tid for tid, sq in enumerate(HEADER_DICT["SQ"])}

def create_aligned_segment(row: Dict) -> pysam.AlignedSegment:
    """创建单个AlignedSegment对象"""
    seg = pysam.AlignedSegment()
    
    # 设置基本属性
    seg.query_name = row["read_name"]
    seg.query_sequence = row["seq"]
    seg.flag = row["flag"]
    
    # 设置染色体位置
    chrom = row["chrom"]
    if chrom not in CHROM_TO_TID:
        return None
        
    seg.reference_id = CHROM_TO_TID[chrom]
    seg.reference_start = row["start"]
    
    # 设置质量值和CIGAR
    seg.query_qualities = pysam.qualitystring_to_array(row["qual"])
    seg.cigar = [(0, len(row["seq"]))]  # 完全匹配
    
    # 设置其他必填字段
    seg.mapping_quality = 255  # 映射质量未知
    seg.next_reference_id = -1  # 无配对read
    seg.next_reference_start = -1
    seg.template_length = 0  # insert size
    
    # 设置read group
    seg.set_tag("RG", HEADER_DICT["RG"][0]["ID"])
    
    return seg


def process_chunk_to_bam(chunk_df: pl.DataFrame, output_path: str) -> tuple:
    """处理数据块并写入临时BAM文件"""
    skipped = 0
    written = 0
    
    try:
        with pysam.AlignmentFile(output_path, "wb", header=dict(HEADER_DICT)) as bam:
            for row in chunk_df.rows(named=True):
                seg = create_aligned_segment(row)
                if seg is None:
                    skipped += 1
                    continue
                bam.write(seg)
                written += 1
        return (output_path, written, skipped)
    except Exception as e:
        return (output_path, 0, 0)


def merge_bam_files(bam_files: List[str], output_bam: str):
    """合并多个BAM文件"""
    if len(bam_files) == 1:
        # 只有一个文件，直接重命名
        os.rename(bam_files[0], output_bam)
    else:
        # 使用pysam合并
        pysam.merge("-f", output_bam, *bam_files)
        # 清理临时文件
        for bam_file in bam_files:
            try:
                os.remove(bam_file)
            except:
                pass


def process_bed_to_bam(input_path, output_bam, n_threads=4, chunk_size=50000):
    """将BED格式表格转换为BAM文件（多线程版本）"""
    
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        
        # 读取数据文件
        read_task = progress.add_task("[cyan]读取输入文件...", total=1)
        try:
            df = pl.read_csv(
                input_path,
                separator="\t",
                has_header=False,
                new_columns=[
                    "chrom", "start", "end", "read_name", 
                    "flag_str", "dot", "seq", "seq_len"
                ]
            )
            progress.update(read_task, completed=1)
        except Exception as e:
            click.echo(f"读取输入文件失败: {str(e)}", err=True)
            sys.exit(1)
        
        # 处理数据
        process_task = progress.add_task("[yellow]处理数据...", total=1)
        df = df.with_columns(
            # 序列转换：C→T, M→C
            pl.col("seq").str.replace_all('C','T',literal=True).str.replace_all('M','C',literal=True),
            
            # 设置flag值
            pl.lit(99).alias("flag"),
            
            # 创建质量字符串：全部设为'I'（Phred40）
            pl.col("seq").map_elements(
                lambda s: "I" * len(s), return_dtype=pl.String
            ).alias("qual"),
        )
        progress.update(process_task, completed=1)
        
        total_rows = df.height
        click.echo(f"总共 {total_rows:,} 条记录，使用 {n_threads} 个线程处理")
        
        # 分块处理
        n_chunks = (total_rows + chunk_size - 1) // chunk_size
        chunks = [df.slice(i * chunk_size, chunk_size) for i in range(n_chunks)]
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="bam_temp_")
        temp_bam_files = []
        
        # 多线程处理
        write_task = progress.add_task("[green]写入BAM文件...", total=n_chunks)
        
        try:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # 提交所有任务
                futures = []
                for i, chunk in enumerate(chunks):
                    temp_bam = os.path.join(temp_dir, f"chunk_{i:04d}.bam")
                    future = executor.submit(process_chunk_to_bam, chunk, temp_bam)
                    futures.append(future)
                
                # 收集结果
                total_written = 0
                total_skipped = 0
                
                for future in as_completed(futures):
                    bam_path, written, skipped = future.result()
                    temp_bam_files.append(bam_path)
                    total_written += written
                    total_skipped += skipped
                    progress.update(write_task, advance=1)
            
            if total_skipped > 0:
                click.echo(f"警告: 跳过 {total_skipped} 条未知染色体的记录", err=True)
            
            # 合并BAM文件
            if len(temp_bam_files) > 0:
                merge_task = progress.add_task("[magenta]合并BAM文件...", total=1)
                # 按文件名排序以保持顺序
                temp_bam_files.sort()
                merge_bam_files(temp_bam_files, output_bam)
                progress.update(merge_task, completed=1)
                
                click.echo(f"[green]✓[/green] 成功转换 {total_written:,} 条记录到 {output_bam}")
            else:
                click.echo("错误: 没有生成任何BAM文件", err=True)
                sys.exit(1)
                
        except Exception as e:
            click.echo(f"处理失败: {str(e)}", err=True)
            sys.exit(1)
        finally:
            # 清理临时目录
            try:
                os.rmdir(temp_dir)
            except:
                pass

@click.command()
@click.argument("input_bed", type=click.Path(exists=True))
@click.argument("output_bam", type=click.Path())
@click.option("-t", "--threads", default=None, type=int, help="并行线程数 (默认: 自动检测所有可用CPU)")
@click.option("-c", "--chunk-size", default=50000, type=int, help="每个块的记录数 (默认: 50000)")
def cli(input_bed, output_bam, threads, chunk_size):
    """将BED格式的比对表转换为BAM文件（多线程版本）
    
    INPUT_BED: 输入表格路径（BED格式，8列）
    OUTPUT_BAM: 输出BAM文件路径
    
    示例:
        python assembled_to_bam.py input.bed output.bam -t 8 -c 100000
    """
    # 自动检测CPU数量
    if threads is None:
        threads = multiprocessing.cpu_count()
        click.echo(f"自动检测到 {threads} 个CPU核心")
    
    process_bed_to_bam(input_bed, output_bam, n_threads=threads, chunk_size=chunk_size)

if __name__ == "__main__":
    cli()