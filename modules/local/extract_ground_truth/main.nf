process EXTRACT_GROUND_TRUTH {
    errorStrategy 'retry'
    maxRetries 3
    tag "$meta.id"
    
    input:
    tuple val(meta), path(txtFiles, stageAs: "input_?.txt"), path(target_bamFile), path(background_bamFile), path(filtered_pileup)
    path(extract_ground_truth_script)
    
    output:
    tuple val(meta), path("*_ground_truth.tsv.gz"), emit: ground_truth
    tuple val(meta), path("*_ground_truth.log"), emit: log_file
    
    script:
    def args = task.ext.args ?: ''
    """
    # Merge txt files - extract 'name' and 'prob_class_1' columns
    first_file=true
    for txt in input*.txt; do
      if [ "\$first_file" = true ]; then
        # First file: extract header and data
        awk -F'\t' '
          NR==1 {
            for(i=1;i<=NF;i++) {
              if(\$i=="name") name_col=i
              if(\$i=="prob_class_1") prob_col=i
            }
            print \$name_col"\t"\$prob_col
          }
          NR>1 {
            print \$name_col"\t"\$prob_col
          }
        ' "\$txt" > merged_mqres.txt
        first_file=false
      else
        # Subsequent files: skip header, extract only data
        awk -F'\t' '
          NR==1 {
            for(i=1;i<=NF;i++) {
              if(\$i=="name") name_col=i
              if(\$i=="prob_class_1") prob_col=i
            }
          }
          NR>1 {
            print \$name_col"\t"\$prob_col
          }
        ' "\$txt" >> merged_mqres.txt
      fi
    done

    for bam in *.bam; do
      samtools index -@ ${task.cpus} \${bam}
    done

    # Extract ground truth
    python ${extract_ground_truth_script} \\
      --input-target-bam ${target_bamFile} \\
      --input-background-bam ${background_bamFile} \\
      --input-txt merged_mqres.txt \\
      --filtered-pileup ${filtered_pileup} \\
      --output ${meta.id}_ground_truth.tsv.gz \\
      --ncpus ${task.cpus} \\
      ${args} \\
      > ${meta.id}_ground_truth.log 2>&1
    
    # Remove intermediate files
    rm -f merged_mqres.txt
    """
}
