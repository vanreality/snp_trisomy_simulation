process BAM_TO_PILEUP_HARD_FILTER {
    errorStrategy 'ignore'
    maxErrors 10
    maxRetries 1
    tag "$meta.id"
    
    input:
    tuple val(meta), path(bamFiles, stageAs: "input_?.bam")
    path(known_sites_tsv)
    path(pileup_script)
    path(merge_script)
    
    output:
    tuple val(meta), path("*_pileup.tsv.gz"), emit: pileup
    
    script:
    """
    # Process each BAM file
    for bam in input*.bam; do
        # Get base name for output files
        base=\$(basename \$bam .bam)
        
        samtools index \${bam}
            
        python ${pileup_script} \\
          --input-bam \${bam} \\
          --known-sites ${known_sites_tsv} \\
          --output \${base} \\
          --ncpus ${task.cpus}
    done

    # Merge all intermediate TSV outputs into final file
    python ${merge_script} \\
      --inputs "\$(ls *_pileup.tsv.gz | tr '\\n' ' ')" \\
      --output ${meta.id}_${meta.label}_pileup.tsv.gz

    # Remove intermediate files
    rm -f input*_pileup.tsv.gz
    """
}
