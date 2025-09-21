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
    path(pileup_script)
    path(merge_script)
    path(split_site_script)
    
    output:
    tuple val(meta), path("${meta.id}_pileup.tsv.gz"), emit: pileup
    
    script:
    """
    # Process each BAM file
    for bam in input*.bam; do
        # Get base name for output files
        base=\$(basename \$bam .bam)
        
        # Split the tsv file, output full_depth.bed, half_depth_ct.bed, half_depth_ga.bed
        python ${split_site_script} ${known_sites_tsv}

        # Generate bcftools mpileup files
        vcf_files=""
        
        # Process full_depth regions if bed file is not empty
        if [ -s full_depth.bed ]; then
            # Extract regions for full depth analysis
            samtools view -b -L full_depth.bed \$bam -o \${base}_full_depth.bam
            samtools index \${base}_full_depth.bam
            
            bcftools mpileup \\
              -f ${fasta} \\
              --regions-file full_depth.bed \\
              --annotate AD,DP \\
              -Ou \${base}_full_depth.bam \\
            | bcftools view -Oz -o \${base}_full_depth.vcf.gz
            vcf_files="\${base}_full_depth.vcf.gz"
        fi

        # Process half_depth_ct regions if bed file is not empty
        if [ -s half_depth_ct.bed ]; then
            # Extract regions for half depth CT analysis
            samtools view -b -L half_depth_ct.bed \$bam -o \${base}_half_depth_ct_regions.bam
            samtools index \${base}_half_depth_ct_regions.bam
            
            # Split by CT flags (flag==99 || flag==147)
            samtools view -b -e 'flag==99 || flag==147' \${base}_half_depth_ct_regions.bam -o \${base}_half_depth_ct.bam
            samtools index \${base}_half_depth_ct.bam
            
            bcftools mpileup \\
              -f ${fasta} \\
              --regions-file half_depth_ct.bed \\
              --annotate AD,DP \\
              -Ou \${base}_half_depth_ct.bam \\
            | bcftools view -Oz -o \${base}_half_depth_ct.vcf.gz
            vcf_files="\$vcf_files \${base}_half_depth_ct.vcf.gz"
        fi
        
        # Process half_depth_ga regions if bed file is not empty
        if [ -s half_depth_ga.bed ]; then
            # Extract regions for half depth GA analysis
            samtools view -b -L half_depth_ga.bed \$bam -o \${base}_half_depth_ga_regions.bam
            samtools index \${base}_half_depth_ga_regions.bam
            
            # Split by GA flags (flag==83 || flag==163)
            samtools view -b -e 'flag==83 || flag==163' \${base}_half_depth_ga_regions.bam -o \${base}_half_depth_ga.bam
            samtools index \${base}_half_depth_ga.bam
            
            bcftools mpileup \\
              -f ${fasta} \\
              --regions-file half_depth_ga.bed \\
              --annotate AD,DP \\
              -Ou \${base}_half_depth_ga.bam \\
            | bcftools view -Oz -o \${base}_half_depth_ga.vcf.gz
            vcf_files="\$vcf_files \${base}_half_depth_ga.vcf.gz"
        fi

        # Merge existing vcf files
        if [ -n "\$vcf_files" ]; then
            bcftools concat \$vcf_files -Oz -o \${base}.vcf.gz
        else
            echo "Error: No VCF files to concatenate for \$base"
            exit 1
        fi

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
