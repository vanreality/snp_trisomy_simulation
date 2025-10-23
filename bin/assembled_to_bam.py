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
pretty.install()
install(show_locals=True)
import re
import sys
import click
import polars as pl
import pysam
from collections import OrderedDict

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

def process_bed_to_bam(input_path, output_bam):
    """将BED格式表格转换为BAM文件"""
    try:
        # 读取数据文件
        df = pl.read_csv(
            input_path,
            separator="\t",
            has_header=False,
            new_columns=[
                "chrom", "start", "end", "read_name", 
                "flag_str", "dot", "seq", "seq_len"
            ]
        )
    except Exception as e:
        click.echo(f"读取输入文件失败: {str(e)}", err=True)
        sys.exit(1)
    
    # 处理数据
    df = df.with_columns(
        # 序列转换：C→T, M→C
        pl.col("seq").str.replace_all('C','T',literal=True).str.replace_all('M','C',literal=True),
        
        # 提取最小flag值
        # pl.col("flag_str").map_elements(
        #     lambda f: min(int(x) for x in f.split('|')), return_dtype=pl.Int64
        # ).alias("flag"),
        pl.lit(99).alias("flag"),
        
        # 创建质量字符串：全部设为'I'（Phred40）
        pl.col("seq").map_elements(
            lambda s: "I" * len(s), return_dtype=pl.String
        ).alias("qual"),
    )
    
    # 写入BAM文件
    try:
        with pysam.AlignmentFile(output_bam, "wb", header=dict(HEADER_DICT)) as bam:
            for row in df.rows(named=True):
                seg = pysam.AlignedSegment()
                
                # 设置基本属性
                seg.query_name = row["read_name"]
                seg.query_sequence = row["seq"]
                seg.flag = row["flag"]
                
                # 设置染色体位置
                chrom = row["chrom"]
                if chrom not in CHROM_TO_TID:
                    click.echo(f"警告: 跳过未知染色体 '{chrom}'", err=True)
                    continue
                    
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
                
                bam.write(seg)
                
    except Exception as e:
        click.echo(f"写入BAM文件失败: {str(e)}", err=True)
        sys.exit(1)
        
    click.echo(f"成功转换 {df.height} 条记录到 {output_bam}")

@click.command()
@click.argument("input_bed", type=click.Path(exists=True))
@click.argument("output_bam", type=click.Path())
def cli(input_bed, output_bam):
    """将BED格式的比对表转换为BAM文件
    
    INPUT_BED: 输入表格路径（BED格式，8列）
    OUTPUT_BAM: 输出BAM文件路径
    """
    process_bed_to_bam(input_bed, output_bam)

if __name__ == "__main__":
    cli()