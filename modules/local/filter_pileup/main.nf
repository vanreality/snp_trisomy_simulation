process FILTER_PILEUP {
    errorStrategy 'ignore'
    maxErrors 10
    maxRetries 1
    tag "$meta.id"
    
    input:
    tuple val(meta), path(snp_pileup_file)
    path(script)
    
    output:
    path("*.tsv"), emit: filtered_pileup
    
    script:
    """
    python3 ${script} \\
        --input-path ${snp_pileup_file} \\
        --output-dir ./ \\
    """
}
