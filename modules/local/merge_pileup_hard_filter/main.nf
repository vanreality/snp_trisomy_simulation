process MERGE_PILEUP_HARD_FILTER {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(target_pileup), path(background_pileup)
    path(merge_script)
    
    output:
    tuple val(meta), path("${meta.id}_pileup.tsv.gz"), emit: pileup

    script:
    """
    python ${merge_script} \\
      --target_pileup ${target_pileup} \\
      --background_pileup ${background_pileup} \\
      --output ${meta.id}_pileup
    """
}
