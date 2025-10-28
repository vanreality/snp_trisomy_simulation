process SPLIT_BAM_BY_TXT {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(txt_file), path(bam_file)
    val(threshold)

    output:
    tuple val(meta), path("*_target.bam"), emit: target
    tuple val(meta), path("*_background.bam"), emit: background
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    # Extract read names based on probability threshold
    # Skip header (first line) and extract reads with prob > 0.5 for target
    tail -n +2 ${txt_file} | awk -F '\t' '\$12 > ${threshold} {print \$7}' > target_reads.txt
    
    # Use samtools view -N to extract target reads and -U to output unmatched reads to background
    samtools view -@ ${task.cpus} -b -N target_reads.txt -U ${prefix}_background.bam -o ${prefix}_target.bam ${bam_file}
    """
}
