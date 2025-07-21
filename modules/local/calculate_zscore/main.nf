process CALCULATE_ZSCORE {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(snp_pileup_file), path(maternal_genotype_file), path(fetal_genotype_file)
    path(reference_file)
    path(script)
    
    output:
    path("*.tsv"), emit: zscore_results
    path("${meta.id}.log"), emit: log_file
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --input-pileup ${snp_pileup_file} \\
        --maternal-genotype ${maternal_genotype_file} \\
        --fetal-genotype ${fetal_genotype_file} \\
        --reference-file ${reference_file} \\
        --output ${meta.id} \\
        ${args} \\
        > ${meta.id}.log 2>&1
    """
}
