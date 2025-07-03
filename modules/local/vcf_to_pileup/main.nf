process VCF_TO_PILEUP {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(vcf_file)
    path(potential_snps_file)
    path(script)
    
    output:
    tuple val(meta), path("*.tsv.gz"), emit: pileup
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --vcf-file ${vcf_file} \\
        --potential-snps-file ${potential_snps_file} \\
        --output-file ${meta.id}_pileup.tsv.gz \\
        --verbose \\
        ${args}
    """
}
