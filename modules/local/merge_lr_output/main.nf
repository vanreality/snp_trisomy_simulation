process MERGE_LR_OUTPUT {
    input:
    path(input_files)
    path(script)
    
    output:
    path("merged_lr_output.tsv.gz"), emit: merged_lr_output
    path("merge_lr_output.log"), emit: log_file
    
    script:
    def args = task.ext.args ?: ''
    """
    # Create file list to avoid "Argument list too long" error
    echo "${input_files.join('\n')}" > input_files_list.txt
    
    python3 ${script} \\
        --input-files-list input_files_list.txt \\
        --output merged_lr_output.tsv.gz \\
        ${args} \\
        > merge_lr_output.log 2>&1
    """
}
