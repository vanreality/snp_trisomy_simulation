process CALCULATE_LR {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(snp_pileup_file)
    val(mode)
    path(script)
    
    output:
    tuple val(meta), path("*.tsv.gz"), emit: lr_results
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --input-path ${snp_pileup_file} \\
        --output-dir ./ \\
        --ncpus ${task.cpus} \\
        --mode ${mode} \\
        ${args}
    """
}
