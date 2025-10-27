process BAM_TO_PILEUP_PROB_WEIGHTED {
    errorStrategy 'ignore'
    maxErrors 10
    maxRetries 1
    tag "$meta.id"
    
    input:
    tuple val(meta), path(txtFiles, stageAs: "input_?.txt"), path(bamFiles, stageAs: "input_?.bam")
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
        
        # Process full_depth regions if bed file is not empty
        if [ -s full_depth.bed ]; then
            # Extract regions for full depth analysis
            samtools view -b -L full_depth.bed \$bam -o \${base}_full_depth.bam
            samtools index \${base}_full_depth.bam

            python ${pileup_script} \\
              --input-bam \${base}_full_depth.bam \\
              --input-txt \${base}.txt \\
              --known-sites ${known_sites_tsv} \\
              --bed full_depth.bed \\
              --output \${base}_full_depth
        fi

        # Process half_depth_ct regions if bed file is not empty
        if [ -s half_depth_ct.bed ]; then
            # Extract regions for half depth CT analysis
            samtools view -b -L half_depth_ct.bed \$bam -o \${base}_half_depth_ct_regions.bam
            samtools index \${base}_half_depth_ct_regions.bam
            
            # Split by CT flags (flag==99 || flag==147)
            samtools view -b -e 'flag==99 || flag==147' \${base}_half_depth_ct_regions.bam -o \${base}_half_depth_ct.bam
            samtools index \${base}_half_depth_ct.bam
            
            python ${pileup_script} \\
              --input-bam \${base}_half_depth_ct.bam \\
              --input-txt \${base}.txt \\
              --known-sites ${known_sites_tsv} \\
              --bed half_depth_ct.bed \\
              --output \${base}_half_depth_ct
        fi
        
        # Process half_depth_ga regions if bed file is not empty
        if [ -s half_depth_ga.bed ]; then
            # Extract regions for half depth GA analysis
            samtools view -b -L half_depth_ga.bed \$bam -o \${base}_half_depth_ga_regions.bam
            samtools index \${base}_half_depth_ga_regions.bam
            
            # Split by GA flags (flag==83 || flag==163)
            samtools view -b -e 'flag==83 || flag==163' \${base}_half_depth_ga_regions.bam -o \${base}_half_depth_ga.bam
            samtools index \${base}_half_depth_ga.bam
            
            python ${pileup_script} \\
              --input-bam \${base}_half_depth_ga.bam \\
              --input-txt \${base}.txt \\
              --known-sites ${known_sites_tsv} \\
              --bed half_depth_ga.bed \\
              --output \${base}_half_depth_ga
        fi
    done

    # Merge all intermediate TSV outputs into final file
    python ${merge_script} \\
      --inputs "\$(ls *_pileup.tsv.gz | tr '\\n' ' ')" \\
      --output ${meta.id}_pileup.tsv.gz

    # Remove intermediate files
    rm -f full_depth.bed half_depth_ct.bed half_depth_ga.bed
    rm -f input*_full_depth.bam input*_half_depth_ct*.bam input*_half_depth_ga*.bam
    rm -f *.bai
    rm -f *.csi
    rm -f input*.vcf.gz input*.tsv.gz
    """
}
