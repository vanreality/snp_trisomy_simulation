process BUILD_ZSCORE_REFERENCE {
    tag "Build Zscore Reference"
    
    input:
    path(pileup_list_file)
    path(maternal_genotype_file)
    path(fetal_genotype_file)
    path(script)
    
    output:
    path("zscore_reference.tsv"), emit: zscore_reference
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --input-pileup-list ${pileup_list_file} \\
        --maternal-genotype ${maternal_genotype_file} \\
        --fetal-genotype ${fetal_genotype_file} \\
        ${args}
    """
}
