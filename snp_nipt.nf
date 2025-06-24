// Local modules
include { SPLIT_PARQUET_BY_SAMPLE } from './modules/local/split_parquet_by_sample/main.nf'
include { EXTRACT_SNP_FROM_PARQUET } from './modules/local/extract_snp_from_parquet/main.nf'
include { CALCULATE_LR } from './modules/local/calculate_lr/main.nf'

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
    } else {
        EXTRACT_SNP_FROM_PARQUET(
            ch_parquet_samplesheet,
            file(params.fasta),
            file(params.fasta_index),
            file(params.snp_list),
            params.filter_mode,
            params.threshold,
            file("${workflow.projectDir}/bin/extract_snp_from_parquet.py")
        )
        EXTRACT_SNP_FROM_PARQUET.out.pileup
            .set { ch_pileup_samplesheet }
    }
    
    // 3. LR calculation
    // ====================================

    CALCULATE_LR(
        ch_pileup_samplesheet,
        params.lr_mode,
        file("${workflow.projectDir}/bin/calculate_lr.py")
    )
}


