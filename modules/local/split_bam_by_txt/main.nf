process SPLIT_BAM_BY_TXT {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(txt_file), path(bam_file)
    val(threshold)

    output:
    tuple val(meta), path("*_target.bam"), emit: target
    tuple val(meta), path("*_background.bam"), emit: background
    tuple val(meta), path("*_unclassified.bam"), emit: unclassified
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    set -euo pipefail
    export LC_ALL=C

    # Extract read names based on probability threshold
    # Skip header (first line) and extract reads with prob > 0.5 for target, prob <= 0.5 for background
    tail -n +2 ${txt_file} | awk -F '\\t' -v t="${threshold}" '
      {
        name = \$7;
        prob = \$12 + 0.0;
        if (prob > t) {
          if (!(name in seen_t)) { print name >> "target_reads.txt";     seen_t[name]=1 }
        } else {
          if (!(name in seen_b)) { print name >> "background_reads.txt"; seen_b[name]=1 }
        }
        if (!(name in seen_all)) { print name >> "classified_reads.txt"; seen_all[name]=1 }
      }
    '
    
    # Use samtools view -N to extract reads
    samtools view -@ ${task.cpus} -b -N target_reads.txt     -o ${prefix}_target.bam ${bam_file}
    samtools view -@ ${task.cpus} -b -N background_reads.txt -o ${prefix}_background.bam ${bam_file}
    samtools view -@ ${task.cpus} -b -N classified_reads.txt -U ${prefix}_unclassified.bam ${bam_file}
    """
}
