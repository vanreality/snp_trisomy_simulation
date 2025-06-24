process EXTRACT_SNP_FROM_PARQUET {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(parquet_file)
    path(fasta)
    path(fai)
    path(snp_list_file)
    val(filter_mode)
    val(threshold)
    path(script)
    
    output:
    tuple val(meta), path("*_raw_snp_calls.tsv.gz"), emit: raw_snp_calls
    tuple val(meta), path("*_pileup.tsv.gz"), emit: pileup
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --parquet ${parquet_file} \\
        --fasta ${fasta} \\
        --snp-list ${snp_list_file} \\
        --output ${meta.id} \\
        --mode ${filter_mode} \\
        --threshold ${threshold} \\
        --num-workers ${task.cpus} \\
        ${args}
    """
}
