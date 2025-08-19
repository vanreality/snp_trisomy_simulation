process BAM_TO_PILEUP {
    errorStrategy 'ignore'
    maxErrors 10
    maxRetries 1
    tag "$meta.id"
    
    input:
    tuple val(meta), path(bamFiles, stageAs: "input_?.bam")
    path(fasta)
    path(fasta_index)
    path(known_sites_tsv)
    path(known_sites_bed)
    path(pileup_script)
    path(merge_script)
    
    output:
    tuple val(meta), path("${meta.id}_pileup.tsv.gz"), emit: pileup
    
    script:
    """
    # Process each BAM file
    for bam in input*.bam; do
        # Get base name for output files
        base=\$(basename \$bam .bam)
        
        # Index the BAM file
        samtools index \$bam

        # Generate BCF via bcftools mpileup and call
        bcftools mpileup \\
          -f ${fasta} \\
          --regions-file ${known_sites_bed} \\
          --annotate AD,DP \\
          -Ou \$bam \\
        | bcftools view -Oz -o \${base}.vcf.gz

        # Process the BCF with a Python script to extract desired metrics
        python ${pileup_script} \\
          --input-vcf \${base}.vcf.gz \\
          --known-sites ${known_sites_tsv} \\
          --output \${base}
    done

    # Merge all intermediate TSV outputs into final file
    python ${merge_script} \\
      --inputs "\$(ls input*_pileup.tsv.gz | tr '\\n' ' ')" \\
      --output ${meta.id}_pileup.tsv.gz
    """
}
