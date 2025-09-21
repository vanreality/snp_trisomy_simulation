// Local modules
include { SPLIT_PARQUET_BY_SAMPLE } from './modules/local/split_parquet_by_sample/main.nf'
include { EXTRACT_SNP_FROM_PARQUET } from './modules/local/extract_snp_from_parquet/main.nf'
include { CALCULATE_LR } from './modules/local/calculate_lr/main.nf'
include { VCF_TO_PILEUP } from './modules/local/vcf_to_pileup/main.nf'
include { MERGE_LR_OUTPUT } from './modules/local/merge_lr_output/main.nf'
include { BAM_TO_PILEUP } from './modules/local/bam_to_pileup/main.nf'

workflow {
    // 1. Input parquet processing
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
                return [meta, txtFile]
            }
            .set { ch_parquet_samplesheet }
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

        BAM_TO_PILEUP(
            ch_bam_samplesheet,
            file(params.fasta),
            file(params.fasta_index),
            file(params.known_sites_tsv),
            file("${workflow.projectDir}/bin/bam_to_pileup.py"),
            file("${workflow.projectDir}/bin/merge_pileups.py"),
            file("${workflow.projectDir}/bin/split_site_tsv.py")
        )
        BAM_TO_PILEUP.out.pileup
            .set { ch_pileup_samplesheet }
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
            file("${workflow.projectDir}/bin/lr_calculator.py")
        )

        // 4. Merge LR output
        // ====================================
        MERGE_LR_OUTPUT(
            CALCULATE_LR.out.lr_results.collect(),
            file("${workflow.projectDir}/bin/merge_lr_output.py")
        )
    }


}


