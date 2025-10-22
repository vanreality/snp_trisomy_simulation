// Local modules
include { SPLIT_PARQUET_BY_SAMPLE } from './modules/local/split_parquet_by_sample/main.nf'
include { EXTRACT_SNP_FROM_PARQUET } from './modules/local/extract_snp_from_parquet/main.nf'
include { CALCULATE_LR } from './modules/local/calculate_lr/main.nf'
include { VCF_TO_PILEUP } from './modules/local/vcf_to_pileup/main.nf'
include { MERGE_LR_OUTPUT } from './modules/local/merge_lr_output/main.nf'
include { SPLIT_BAM_BY_TXT } from './modules/local/split_bam_by_txt/main.nf'
include { BAM_TO_PILEUP_HARD_FILTER } from './modules/local/bam_to_pileup_hard_filter/main.nf'
include { BAM_TO_PILEUP_HARD_FILTER as BAM_TO_PILEUP_HARD_FILTER_TARGET } from './modules/local/bam_to_pileup_hard_filter/main.nf'
include { BAM_TO_PILEUP_HARD_FILTER as BAM_TO_PILEUP_HARD_FILTER_BACKGROUND } from './modules/local/bam_to_pileup_hard_filter/main.nf'
include { BAM_TO_PILEUP_PROB_WEIGHTED } from './modules/local/bam_to_pileup_prob_weighted/main.nf'
include { MERGE_PILEUP_HARD_FILTER } from './modules/local/merge_pileup_hard_filter/main.nf'
include { FILTER_PILEUP } from './modules/local/filter_pileup/main.nf'
include { SAMTOOLS_MERGE as SAMTOOLS_MERGE_TARGET } from './modules/nf-core/samtools/merge/main.nf'
include { SAMTOOLS_MERGE as SAMTOOLS_MERGE_BACKGROUND } from './modules/nf-core/samtools/merge/main.nf'

workflow {
    // 1. Input samplesheet(txt, bam, parquet) processing
    // ====================================
    
    // Determine input source (samplesheet or parquet)
    if (params.input_parquet_samplesheet) {
        // Skip SPLIT_PARQUET_BY_SAMPLE process and directly parse the provided CSV
        Channel
            .fromPath(params.input_parquet_samplesheet)
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample]
                // Check if parquet file exists
                def parquetFile = file(row.parquet)
                if (!parquetFile.exists()) {
                    error "Parquet file not found: ${row.parquet}"
                }
                return [meta, parquetFile]
            }
            .set { ch_parquet_samplesheet }
    } else if (params.input_txt_samplesheet) {
        // Skip SPLIT_PARQUET_BY_SAMPLE process and directly parse the provided CSV
        Channel
            .fromPath(params.input_txt_samplesheet)
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample]
                // Check if txt file exists
                def txtFile = file(row.txt)
                if (!txtFile.exists()) {
                    error "txt file not found: ${row.txt}"
                }
                def bamFile = file(row.bam)
                if (!bamFile.exists()) {
                    error "bam file not found: ${row.bam}"
                }
                return [meta, txtFile, bamFile]
            }
            .set { ch_txt_samplesheet }
    } else if (params.input_parquet) {
        // Run split_parquet_by_sample process to split input parquet file by sample
        SPLIT_PARQUET_BY_SAMPLE(
            [[id: "sample"], file(params.input_parquet)],
            file("${workflow.projectDir}/bin/split_parquet_by_sample.py")
        )

        // Parse the generated samplesheet CSV to create channel
        SPLIT_PARQUET_BY_SAMPLE.out.samplesheet
            .map { _meta, samplesheet -> samplesheet }
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample]
                // Check if parquet file exists
                def parquetFile = file(row.parquet)
                if (!parquetFile.exists()) {
                    error "Parquet file not found: ${row.parquet}"
                }
                return [meta, parquetFile]
            }
            .set { ch_parquet_samplesheet }
    }

    // 2. SNP pileup processing
    // ====================================

    if (params.input_pileup_samplesheet) {
        // Skip EXTRACT_SNP_FROM_PARQUET process and directly parse the provided CSV
        Channel
            .fromPath(params.input_pileup_samplesheet)
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample]
                return [meta, file(row.pileup)]
            }
            .set { ch_pileup_samplesheet }
    } else if (params.input_vcf_samplesheet) {
        // Run VCF_TO_PILEUP process to convert VCF to pileup
        Channel
            .fromPath(params.input_vcf_samplesheet)
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample]
                return [meta, file(row.vcf)]
            }
            .set { ch_vcf_samplesheet }

        VCF_TO_PILEUP(
            ch_vcf_samplesheet,
            file(params.input_potential_snps),
            file("${workflow.projectDir}/bin/vcf_to_pileup.py")
        )
        VCF_TO_PILEUP.out.pileup
            .set { ch_pileup_samplesheet }
    } else if (params.input_bam_samplesheet) {
        Channel
            .fromPath(params.input_bam_samplesheet)
            .splitCsv(header: true)
            .map { row -> 
                tuple(row.sample, file(row.bam))
            }
            .groupTuple(by: 0)
            .map { sample, bamFiles ->
                def bamList = bamFiles.toList()
                def meta = [id: sample]
                return tuple(meta, bamList)
            }
            .set { ch_bam_samplesheet }

        BAM_TO_PILEUP_HARD_FILTER(
            ch_bam_samplesheet,
            file(params.fasta),
            file(params.fasta_index),
            file(params.known_sites_tsv),
            file("${workflow.projectDir}/bin/bam_to_pileup.py"),
            file("${workflow.projectDir}/bin/merge_pileups.py"),
            file("${workflow.projectDir}/bin/split_site_tsv.py")
        )
        BAM_TO_PILEUP_HARD_FILTER.out.pileup
            .set { ch_pileup_samplesheet }
    } else if (params.input_txt_samplesheet && params.filter_mode == "hard_filter") {
        SPLIT_BAM_BY_TXT(
            ch_txt_samplesheet,
            params.threshold
        )

        SPLIT_BAM_BY_TXT.out.target
            .groupTuple(by: 0)
            .map { meta, target ->
                def new_meta = [id: meta.id, label: "target"]
                def bamList = target.toList()
                return tuple(new_meta, bamList)
            }
            .set { ch_target_samplesheet }

        SPLIT_BAM_BY_TXT.out.background
            .groupTuple(by: 0)
            .map { meta, background ->
                def new_meta = [id: meta.id, label: "background"]
                def bamList = background.toList()
                return tuple(new_meta, bamList)
            }
            .set { ch_background_samplesheet }

        SAMTOOLS_MERGE_TARGET(
            ch_target_samplesheet,
            [[:], file(params.fasta)],
            [[:], file(params.fasta_index)],
            [[:], [:]]
        )
        
        SAMTOOLS_MERGE_BACKGROUND(
            ch_background_samplesheet,
            [[:], file(params.fasta)],
            [[:], file(params.fasta_index)],
            [[:], [:]]
        )

        BAM_TO_PILEUP_HARD_FILTER_TARGET(
            ch_target_samplesheet,
            file(params.fasta),
            file(params.fasta_index),
            file(params.known_sites_tsv),
            file("${workflow.projectDir}/bin/bam_to_pileup.py"),
            file("${workflow.projectDir}/bin/merge_pileups.py"),
            file("${workflow.projectDir}/bin/split_site_tsv.py")
        )
        BAM_TO_PILEUP_HARD_FILTER_TARGET.out.pileup
            .set { ch_target_pileup_samplesheet }

        BAM_TO_PILEUP_HARD_FILTER_BACKGROUND(
            ch_background_samplesheet,
            file(params.fasta),
            file(params.fasta_index),
            file(params.known_sites_tsv),
            file("${workflow.projectDir}/bin/bam_to_pileup.py"),
            file("${workflow.projectDir}/bin/merge_pileups.py"),
            file("${workflow.projectDir}/bin/split_site_tsv.py")
        )
        BAM_TO_PILEUP_HARD_FILTER_BACKGROUND.out.pileup
            .set { ch_background_pileup_samplesheet }

        ch_merged = ch_target_pileup_samplesheet.join(
            ch_background_pileup_samplesheet,
            by: [{it[0].id}, {it[0].id}]
        ).map {meta_target, pileup_target, meta_background, pileup_background -> 
            def new_meta = [id: meta_target.id]
            return tuple(new_meta, pileup_target, pileup_background)
        }

        MERGE_PILEUP_HARD_FILTER(
            ch_merged,
            file("${workflow.projectDir}/bin/merge_pileup_hard_filter.py")
        )
        MERGE_PILEUP_HARD_FILTER.out.pileup
            .set { ch_pileup_samplesheet }

        FILTER_PILEUP(
            ch_pileup_samplesheet,
            file("${workflow.projectDir}/bin/filter_pileup.py")
        )
    } else if (params.input_txt_samplesheet && params.filter_mode == "prob_weighted") {
        ch_txt_samplesheet.groupTuple(by: 0)
            .map { meta, txtFile, bamFile ->
                def txtList = txtFile.toList()
                def bamList = bamFile.toList()
                return tuple(meta, txtList, bamList)
            }
            .set { ch_txt_samplesheet_grouped }

        BAM_TO_PILEUP_PROB_WEIGHTED(
            ch_txt_samplesheet_grouped,
            file(params.known_sites_tsv),
            file("${workflow.projectDir}/bin/bam_to_pileup_prob_weighted.py"),
            file("${workflow.projectDir}/bin/merge_pileups.py"),
            file("${workflow.projectDir}/bin/split_site_tsv.py")
        )
        BAM_TO_PILEUP_PROB_WEIGHTED.out.pileup
            .set { ch_pileup_samplesheet }

        FILTER_PILEUP(
            ch_pileup_samplesheet, 
            file("${workflow.projectDir}/bin/filter_pileup.py")
        )
    } else {
        EXTRACT_SNP_FROM_PARQUET(
            ch_parquet_samplesheet,
            file(params.fasta),
            file(params.fasta_index),
            file(params.input_potential_snps),
            params.filter_mode,
            params.threshold,
            file("${workflow.projectDir}/bin/extract_snp_from_parquet.py")
        )
        EXTRACT_SNP_FROM_PARQUET.out.pileup
            .set { ch_pileup_samplesheet }
    }


    if (params.run_lr_calculator) {
        // 3. LR calculation
        // ====================================
        CALCULATE_LR(
            ch_pileup_samplesheet,
            params.lr_mode,
            params.min_raw_depth,
            params.min_model_depth,
            file("${workflow.projectDir}/bin/calculate_lr.py")
        )

        // 4. Merge LR output
        // ====================================
        MERGE_LR_OUTPUT(
            CALCULATE_LR.out.lr_results.collect(),
            file("${workflow.projectDir}/bin/merge_lr_output.py")
        )
    }


}


