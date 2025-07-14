process MERGE_LR_OUTPUT {
    input:
    path(input_files)
    path(script)
    
    output:
    path("merged_lr_output.tsv.gz"), emit: merged_lr_output
    path("merge_lr_output.log"), emit: log_file
    
    script:
    def args = task.ext.args ?: ''
    def input_files_str = input_files.join(' ')
    """
    python3 ${script} \\
        --input-files "${input_files_str}" \\
        --output merged_lr_output.tsv.gz \\
        ${args} \\
        > merge_lr_output.log 2>&1
    """
}
