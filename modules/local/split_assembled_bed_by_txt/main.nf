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
    def ncpus = task.cpus
    """
    # Split assembled bed file by txt file, with probability threshold
    python ${split_script} \
        --input_txt ${txt_file} \
        --input_bed ${bed_file} \
        --threshold ${threshold} \
        --output_prefix ${prefix}

    # Convert target bed file to bam file
    python ${convert_script} \
        --input_bed ${prefix}_target.bed \
        --output_bam ${prefix}_target.unsorted.bam

    samtools sort ${prefix}_target.unsorted.bam -o ${prefix}_target.bam

    # Convert background bed file to bam file
    python ${convert_script} \
        --input_bed ${prefix}_background.bed \
        --output_bam ${prefix}_background.unsorted.bam

    samtools sort -@ ${ncpus} ${prefix}_background.unsorted.bam -o ${prefix}_background.bam

    # Remove temporary files
    rm *.bed *.unsorted.bam
    """
}
