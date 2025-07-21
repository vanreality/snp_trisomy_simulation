include { BUILD_ZSCORE_REFERENCE } from './modules/local/build_zscore_reference/main.nf'
include { CALCULATE_ZSCORE } from './modules/local/calculate_zscore/main.nf'

workflow {
    // 1. Build zscore reference
    // ====================================
    BUILD_ZSCORE_REFERENCE(
        params.input_reference_pileup_samplesheet,
        params.reference_maternal_genotype,
        params.reference_fetal_genotype,
        "${baseDir}/bin/build_zscore_reference.py"
    )

    // 2. Calculate zscore
    // ====================================
    Channel
        .fromPath(params.input_zscore_pileup_samplesheet)
        .splitCsv(header: true)
        .map { row ->
            def meta = [id: row.sample]
            return [meta, file(row.pileup), file(row.maternal_genotype), file(row.fetal_genotype)]
        }
        .set { ch_zscore_pileup_samplesheet }

    CALCULATE_ZSCORE(
        ch_zscore_pileup_samplesheet,
        BUILD_ZSCORE_REFERENCE.output.zscore_reference,
        "${baseDir}/bin/zscore_calculator.py"
    )
}