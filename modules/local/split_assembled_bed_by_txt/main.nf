process SPLIT_ASSEMBLED_BED_BY_TXT {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(txt_file), path(bed_file)
    val(threshold)
    path(split_script)
    path(convert_script)

    output:
    tuple val(meta), path("*_target.bam"), emit: target
    tuple val(meta), path("*_background.bam"), emit: background
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    # Split assembled bed file by txt file, with probability threshold
    python ${split_script} \
        --input_txt ${txt_file} \
        --input_bed ${bed_file} \
        --threshold ${threshold} \
        --output_prefix ${prefix}

    # Convert target bed file to bam file
    python ${convert_script} \
        ${prefix}_target.bed \
        ${prefix}_target.bam

    # Convert background bed file to bam file
    python ${convert_script} \
        ${prefix}_background.bed \
        ${prefix}_background.bam

    # Remove temporary files
    rm *.bed
    """
}
