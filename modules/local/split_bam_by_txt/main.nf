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
    # Automatically detect column numbers for 'name' and 'prob_class_1' from header
    awk -F '\\t' -v t="${threshold}" '
      NR == 1 {
        # Process header to find column indices
        for (i = 1; i <= NF; i++) {
          if (\$i == "name") name_col = i;
          if (\$i == "prob_class_1") prob_col = i;
        }
        # Validate that required columns were found
        if (name_col == 0) {
          print "ERROR: Column '\''name'\'' not found in header" > "/dev/stderr";
          exit 1;
        }
        if (prob_col == 0) {
          print "ERROR: Column '\''prob_class_1'\'' not found in header" > "/dev/stderr";
          exit 1;
        }
        next;
      }
      {
        # Process data rows using detected column indices
        name = \$name_col;
        prob = \$prob_col + 0.0;
        if (prob > t) {
          if (!(name in seen_t)) { print name >> "target_reads.txt";     seen_t[name]=1 }
        } else {
          if (!(name in seen_b)) { print name >> "background_reads.txt"; seen_b[name]=1 }
        }
        if (!(name in seen_all)) { print name >> "classified_reads.txt"; seen_all[name]=1 }
      }
    ' ${txt_file}
    
    # Use samtools view -N to extract reads
    samtools view -@ ${task.cpus} -b -N target_reads.txt     -o ${prefix}_target.bam ${bam_file}
    samtools view -@ ${task.cpus} -b -N background_reads.txt -o ${prefix}_background.bam ${bam_file}
    samtools view -@ ${task.cpus} -b -N classified_reads.txt -U ${prefix}_unclassified.bam ${bam_file} > /dev/null
    """
}
