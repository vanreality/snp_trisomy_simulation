process MERGE_TXT {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(txt_files, stageAs: "input_?.txt")
    path(merge_script)
    val(ncpgs)
    
    output:
    tuple val(meta), path("merged_txt.txt"), emit: merged_txt
    
    script:
    """
    python ${merge_script} \\
        --inputs "\$(ls ${txt_files.join(' ')} | tr '\\n' ' ')" \\
        --output merged_txt.txt \\
        --ncpgs ${ncpgs}
    """
}
