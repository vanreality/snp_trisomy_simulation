process BAM_TO_PILEUP {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(bamFiles, stageAs: "input_bam_?/*"), val(bamNames)
    path(fasta)
    path(fasta_index)
    path(known_sites_tsv)
    path(known_sites_bed)
    path(pileup_script)
    path(merge_script)
    
    output:
    tuple val(meta), path("*.tsv.gz"), emit: pileup
    
    script:
    // Create symlink commands for each BAM file to avoid name collisions
    def bamList = bamFiles.toList()
    def linkCmds = bamList.withIndex().collect { bam, idx ->
        "ln -s ${bam} ${bamNames[idx]}"
    }.join('\n')

    // For each symlinked BAM, run bcftools mpileup and then a Python processing script
    def processCmds = bamNames.collect { name ->
        """
        # Generate BCF via bcftools mpileup and call
        bcftools mpileup \\
          -f ${fasta} \\
          --regions-file ${known_sites_bed} \\
          --annotate AD,DP \\
          -Ou $name \\
        | bcftools view -Oz -o ${name}.vcf.gz

        # Process the BCF with a Python script to extract desired metrics
        python ${pileup_script} \\
          --input-vcf ${name}.vcf.gz \\
          --known-sites ${known_sites_tsv} \\
          --output ${name}.tsv
        """
    }.join('\n')

    // Merge all intermediate TSVs and compress final output
    def mergeCmd = """
    # Merge individual TSV outputs into a single file
    python ${merge_script} \\
      --inputs ${ bamNames.collect{ it + '.tsv.gz' }.join(' ') } \\
      --output ${meta.id}_pileup.tsv.gz
    """

    """
    // Step 1: Create unique symlinks for BAM files
    ${linkCmds}

    // Step 2: Run pileup and Python processing on each BAM
    ${processCmds}

    // Step 3: Merge and compress results
    ${mergeCmd}
    """
}
