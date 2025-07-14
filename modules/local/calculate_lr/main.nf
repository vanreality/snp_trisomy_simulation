process CALCULATE_LR {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(snp_pileup_file)
    val(mode)
    val(min_raw_depth)
    val(min_model_depth)
    path(script)
    
    output:
    path("*.tsv"), emit: lr_results
    path("${meta.id}.log"), emit: log_file
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --input-path ${snp_pileup_file} \\
        --output-dir ./ \\
        --ncpus ${task.cpus} \\
        --mode ${mode} \\
        --min-raw-depth ${min_raw_depth} \\
        --min-model-depth ${min_model_depth} \\
        ${args} \\
        > ${meta.id}.log 2>&1
    """
}
