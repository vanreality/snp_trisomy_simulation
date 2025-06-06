import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from numpy.random import default_rng
import math

# Initialize a single RNG
rng = default_rng()

def extract_AF(x):
    """Parse the AF (allele frequency) field out of the INFO column."""
    af = x['info'].split('AF=')[1].split(';')[0]
    return float(af)

def generate_genotype(af):
    """Generates a genotype based on allele frequency (Hardy-Weinberg).

    Args:
        af (float): Allele frequency of the alternative allele.

    Returns:
        int: Genotype code (0 = hom-ref, 1 = het, 2 = hom-alt).
    """
    p_hom_ref = (1 - af) ** 2
    p_het = 2 * af * (1 - af)
    p_hom_alt = af ** 2

    return rng.choice([0, 1, 2], p=[p_hom_ref, p_het, p_hom_alt])

def get_disomy_fetal_genotype(maternal_gt, paternal_gt):
    """Determines the fetal genotype given parental genotypes, using RNG.

    Args:
        maternal_gt (int): Maternal genotype (0, 1, or 2).
        paternal_gt (int): Paternal genotype (0, 1, or 2).

    Returns:
        int: Fetal genotype (0, 1, or 2).
    """
    if maternal_gt == 0:
        if paternal_gt == 0:
            return 0
        elif paternal_gt == 1:
            return rng.choice([0, 1])      # R,R or R,A
        else:  # paternal_gt == 2
            return 1                       # R,A
    elif maternal_gt == 1:
        if paternal_gt == 0:
            return rng.choice([0, 1])     # R,R or R,A
        elif paternal_gt == 1:
            return rng.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
        else:  # paternal_gt == 2
            return rng.choice([1, 2])     # R,A or A,A
    else:  # maternal_gt == 2
        if paternal_gt == 0:
            return 1                       # R,A
        elif paternal_gt == 1:
            return rng.choice([1, 2])     # R,A or A,A
        else:
            return 2                       # A,A

def get_trisomy_fetal_genotype(maternal_gt, paternal_gt):
    """Determines the trisomy fetal genotype given parental genotypes.

    The extra (third) chromosome is randomly inherited from the mother or father.
    The genotype is encoded as the count of alternate alleles (0-3).

    Args:
        maternal_gt (int): Maternal genotype (0, 1, or 2).
        paternal_gt (int): Paternal genotype (0, 1, or 2).

    Returns:
        int: Trisomy fetal genotype (0 = 0 alternate alleles, 1, 2, or 3).
    """
    # First get a disomy genotype (0, 1, or 2 alt alleles)
    disomy_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt)
    # Convert disomy genotype code to count of alternate alleles
    # (0→0, 1→1, 2→2)
    disomy_alt_count = disomy_gt

    # Randomly choose whether the extra chromosome comes from mother or father
    origin = rng.choice(['maternal', 'paternal'])
    if origin == 'maternal':
        # Sample one allele from the mother's two alleles
        if maternal_gt == 0:
            extra_alt = 0
        elif maternal_gt == 1:
            # One ref, one alt → 50:50 chance
            extra_alt = rng.choice([0, 1])
        else:  # maternal_gt == 2
            extra_alt = 1
    else:
        # Sample one allele from the father's two alleles
        if paternal_gt == 0:
            extra_alt = 0
        elif paternal_gt == 1:
            extra_alt = rng.choice([0, 1])
        else:  # paternal_gt == 2
            extra_alt = 1

    # Trisomy genotype is total count of alternate alleles (0-3)
    trisomy_alt_count = disomy_alt_count + extra_alt
    return trisomy_alt_count

def simulate_disomy_sequencing(depth, fetal_fraction, maternal_gt, fetal_gt):
    """Simulates disomy sequencing reads (binomial).

    Args:
        depth (int): Total sequencing depth for this SNP.
        fetal_fraction (float): Proportion of fetal cfDNA.
        maternal_gt (int): Maternal genotype (0,1,2).
        fetal_gt (int): Fetal genotype (0,1,2).

    Returns:
        tuple[int,int,int,int,float]: (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads, cfDNA_af).
    """
    # Simulate maternal and fetal reads counts
    fetal_reads = rng.binomial(depth, fetal_fraction)
    maternal_reads = depth - fetal_reads

    # Simulate maternal allele counts
    maternal_alt_prob = maternal_gt / 2
    maternal_alt_reads = rng.binomial(maternal_reads, maternal_alt_prob)
    maternal_ref_reads = maternal_reads - maternal_alt_reads

    # Simulate fetal allele counts
    fetal_alt_prob = fetal_gt / 2
    fetal_alt_reads = rng.binomial(fetal_reads, fetal_alt_prob)
    fetal_ref_reads = fetal_reads - fetal_alt_reads

    # Calculate cfDNA allele frequency
    cfDNA_alt_reads = maternal_alt_reads + fetal_alt_reads
    cfDNA_ref_reads = maternal_ref_reads + fetal_ref_reads
    if cfDNA_alt_reads + cfDNA_ref_reads > 0:
        cfDNA_af = cfDNA_alt_reads / (cfDNA_alt_reads + cfDNA_ref_reads)
    else:
        cfDNA_af = 0

    return maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads, cfDNA_af


def simulate_trisomy_sequencing(depth, fetal_fraction, maternal_gt, fetal_gt):
    """Simulates trisomy sequencing reads (binomial).

    Args:
        depth (int): Total sequencing depth for this SNP.
        fetal_fraction (float): Proportion of fetal cfDNA.
        maternal_gt (int): Maternal genotype (0,1,2).
        fetal_gt (int): Fetal genotype (0,1,2).

    Returns:
        tuple[int,int,int,int,float]: (maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads, cfDNA_af).
    """
    # Simulate maternal and fetal reads counts
    fetal_reads = rng.binomial(depth, fetal_fraction)
    maternal_reads = depth - fetal_reads

    # Simulate maternal allele counts
    maternal_alt_prob = maternal_gt / 2
    maternal_alt_reads = rng.binomial(maternal_reads, maternal_alt_prob)
    maternal_ref_reads = maternal_reads - maternal_alt_reads

    # Simulate fetal allele counts
    fetal_alt_prob = fetal_gt / 3
    fetal_alt_reads = rng.binomial(fetal_reads, fetal_alt_prob)
    fetal_ref_reads = fetal_reads - fetal_alt_reads

    # Calculate cfDNA allele frequency
    cfDNA_alt_reads = maternal_alt_reads + fetal_alt_reads
    cfDNA_ref_reads = maternal_ref_reads + fetal_ref_reads
    if cfDNA_alt_reads + cfDNA_ref_reads > 0:
        cfDNA_af = cfDNA_alt_reads / (cfDNA_alt_reads + cfDNA_ref_reads)
    else:
        cfDNA_af = 0

    return maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads, cfDNA_af


def classifier_filtering(model_accuracy, maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads):
    """Use a ML or LLM model to filter out maternal reads
    Args:
        model_accuracy (float): Accuracy of the model.
        maternal_ref_reads (int): Number of maternal reference reads.
        maternal_alt_reads (int): Number of maternal alternate reads.
        fetal_ref_reads (int): Number of fetal reference reads.
        fetal_alt_reads (int): Number of fetal alternate reads.

    Returns:
        tuple[int,int,int,int,float]: (correct_fetal_ref, correct_fetal_alt, misclassified_maternal_ref, misclassified_maternal_alt, filtered_cfDNA_af).
    """
    # True positive
    correct_fetal_ref = rng.binomial(fetal_ref_reads, model_accuracy)
    correct_fetal_alt = rng.binomial(fetal_alt_reads, model_accuracy)

    # False positive
    misclassified_maternal_ref = rng.binomial(maternal_ref_reads, 1 - model_accuracy)
    misclassified_maternal_alt = rng.binomial(maternal_alt_reads, 1 - model_accuracy)

    # cfDNA allele frequency after filtering
    filtered_cfDNA_alt_reads = correct_fetal_alt + misclassified_maternal_alt
    filtered_cfDNA_ref_reads = correct_fetal_ref + misclassified_maternal_ref
    if filtered_cfDNA_alt_reads + filtered_cfDNA_ref_reads > 0:
        filtered_cfDNA_af = filtered_cfDNA_alt_reads / (filtered_cfDNA_alt_reads + filtered_cfDNA_ref_reads)
    else:
        filtered_cfDNA_af = 0

    return correct_fetal_ref, correct_fetal_alt, misclassified_maternal_ref, misclassified_maternal_alt, filtered_cfDNA_af


def run_single_disomy_simulation_set(variants_df, depth_lambda, fetal_fraction_value, model_accuracy):
    """Runs multiple disomy simulation replicates for a given (depth, fetal_fraction).

    Args:
        variants_df (pd.DataFrame): Must have column 'af'.
        depth_lambda (int): Poisson λ for per-SNP depth.
        fetal_fraction_value (float): Fraction of fetal cfDNA.
        model_accuracy (float): Accuracy of the model.

    Returns:
        pd.DataFrame: DataFrame containing simulation results for each variant.
    """
    records = []
    for _, row in variants_df.iterrows():
        af = row['af']

        maternal_gt = generate_genotype(af)
        paternal_gt = generate_genotype(af)
        fetal_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt)

        # Draw a Poisson depth for this SNP
        current_depth = rng.poisson(depth_lambda)
        if current_depth == 0:
            continue

        (maternal_ref_reads, 
            maternal_alt_reads, 
            fetal_ref_reads, 
            fetal_alt_reads, 
            cfDNA_af) = simulate_disomy_sequencing(current_depth, fetal_fraction_value, maternal_gt, fetal_gt)
        (filtered_fetal_ref, 
            filtered_fetal_alt, 
            misclassified_maternal_ref, 
            misclassified_maternal_alt, 
            filtered_cfDNA_af) = classifier_filtering(model_accuracy, maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads)
        
        records.append({
            'chr': row['chr'],
            'pos': row['pos'],
            'ref': row['ref'],
            'alt': row['alt'],
            'af': af,
            'maternal_gt': maternal_gt,
            'paternal_gt': paternal_gt,
            'fetal_gt': fetal_gt,
            'current_depth': current_depth,
            'maternal_ref_reads': maternal_ref_reads,
            'maternal_alt_reads': maternal_alt_reads,
            'fetal_ref_reads': fetal_ref_reads,
            'fetal_alt_reads': fetal_alt_reads,
            'cfDNA_af': cfDNA_af,
            'filtered_fetal_ref': filtered_fetal_ref,
            'filtered_fetal_alt': filtered_fetal_alt,
            'misclassified_maternal_ref': misclassified_maternal_ref,
            'misclassified_maternal_alt': misclassified_maternal_alt,
            'filtered_cfDNA_af': filtered_cfDNA_af
        })

    return pd.DataFrame(records)


def run_single_trisomy_simulation_set(variants_df, depth_lambda, fetal_fraction_value, model_accuracy, trisomy_chr):
    """Runs multiple disomy simulation replicates for a given (depth, fetal_fraction).

    Args:
        variants_df (pd.DataFrame): Must have column 'af'.
        depth_lambda (int): Poisson λ for per-SNP depth.
        fetal_fraction_value (float): Fraction of fetal cfDNA.
        model_accuracy (float): Accuracy of the model.

    Returns:
        pd.DataFrame: DataFrame containing simulation results for each variant.
    """
    records = []
    for _, row in variants_df.iterrows():
        af = row['af']

        maternal_gt = generate_genotype(af)
        paternal_gt = generate_genotype(af)

        if row['chr'] == trisomy_chr:
            fetal_gt = get_trisomy_fetal_genotype(maternal_gt, paternal_gt)
        else:
            fetal_gt = get_disomy_fetal_genotype(maternal_gt, paternal_gt)

        # Draw a Poisson depth for this SNP
        current_depth = rng.poisson(depth_lambda)
        if current_depth == 0:
            continue

        # Simulate sequencing reads
        if row['chr'] == trisomy_chr:
            (maternal_ref_reads, 
                maternal_alt_reads, 
                fetal_ref_reads, 
                fetal_alt_reads, 
                cfDNA_af) = simulate_trisomy_sequencing(current_depth, fetal_fraction_value, maternal_gt, fetal_gt)
        else:
            (maternal_ref_reads, 
                maternal_alt_reads, 
                fetal_ref_reads, 
                fetal_alt_reads, 
                cfDNA_af) = simulate_disomy_sequencing(current_depth, fetal_fraction_value, maternal_gt, fetal_gt)
            
        # Filtering using the model
        (filtered_fetal_ref, 
            filtered_fetal_alt, 
            misclassified_maternal_ref, 
            misclassified_maternal_alt, 
            filtered_cfDNA_af) = classifier_filtering(model_accuracy, maternal_ref_reads, maternal_alt_reads, fetal_ref_reads, fetal_alt_reads)
        
        records.append({
            'chr': row['chr'],
            'pos': row['pos'],
            'ref': row['ref'],
            'alt': row['alt'],
            'af': af,
            'maternal_gt': maternal_gt,
            'paternal_gt': paternal_gt,
            'fetal_gt': fetal_gt,
            'current_depth': current_depth,
            'maternal_ref_reads': maternal_ref_reads,
            'maternal_alt_reads': maternal_alt_reads,
            'fetal_ref_reads': fetal_ref_reads,
            'fetal_alt_reads': fetal_alt_reads,
            'cfDNA_af': cfDNA_af,
            'filtered_fetal_ref': filtered_fetal_ref,
            'filtered_fetal_alt': filtered_fetal_alt,
            'misclassified_maternal_ref': misclassified_maternal_ref,
            'misclassified_maternal_alt': misclassified_maternal_alt,
            'filtered_cfDNA_af': filtered_cfDNA_af
        })

    return pd.DataFrame(records)


def estimate_fetal_fraction(background_chr_df, f_min=0.001, f_max=0.5, f_step=0.001):
    """
    Estimate fetal fraction (FF) from maternal cfDNA SNP read counts using a simple MLE approach.
    
    We assume:
      - For each SNP i, population allele frequency = p_i (“af” column).
      - Mother's genotype G_i ∈ {0/0, 0/1, 1/1} with Hardy-Weinberg priors:
          P(G=0/0) = (1 - p_i)^2
          P(G=0/1) = 2 * p_i * (1 - p_i)
          P(G=1/1) = (p_i)^2
      - Fetal genotype depends on mother's genotype and a random paternal allele drawn from population AF p_i.
        We compute expected fetal alt-allele fraction E[fetal_alt_i | G_mother]:
          • If mother = 0/0:  fetal_alt_frac = 0.5 * p_i
          • If mother = 0/1:  fetal_alt_frac = 0.25 + 0.5 * p_i
          • If mother = 1/1:  fetal_alt_frac = 0.5 + 0.5 * p_i
      - Maternal alt-allele fraction is:
          • 0/0 → 0.0
          • 0/1 → 0.5
          • 1/1 → 1.0
      - Mixture model: observed alt reads at SNP i ~ Binomial(n_i, μ_i), where
            μ_i = (1 - FF) * maternal_alt_frac(G) + FF * fetal_alt_frac(G, p_i)
      - We marginalize over G_mother to get P(obs_alt_i | FF), then maximize the product of likelihoods across SNPs.
    
    Parameters:
    -----------
    background_chr_df : pandas.DataFrame
        DataFrame containing SNP info with columns:
          'chr'       : chromosome (unused in computation, but kept for reference)
          'pos'       : genomic position (unused in computation, but kept for reference)
          'af'        : population allele frequency p_i (float in [0,1])
          'ref_reads' : number of reads supporting reference allele at this SNP
          'alt_reads' : number of reads supporting alternative allele at this SNP
    f_min : float
        Minimum fetal fraction to consider (default 0.001)
    f_max : float
        Maximum fetal fraction to consider (default 0.5)
    f_step : float
        Step size for grid search over FF in [f_min, f_max] (default 0.001)
    
    Returns:
    --------
    best_ff : float
        Estimated fetal fraction that maximizes the log-likelihood.
    log_likelihoods : dict
        A dictionary mapping FF values (keys) to their total log-likelihood (values).
    """
    # Precompute genotype‐specific constants per SNP:
    #   prior_G = [P(G=0/0), P(G=0/1), P(G=1/1)]
    #   maternal_alt_frac = [0.0, 0.5, 1.0]
    #   fetal_alt_frac = [0.5*p, 0.25 + 0.5*p, 0.5 + 0.5*p]  (all depend on p_i)
    #
    # Then for each FF candidate, we compute for each SNP:
    #   L_i(FF) = sum_{g=0,1,2} [ prior_G[g] * BinomialPMF(alt_reads_i; n_i, μ_i(g, FF)) ]
    # where μ_i(g,FF) = (1−FF)*maternal_alt_frac[g] + FF*fetal_alt_frac[g].
    #
    # We work in log‐space using log-sum-exp for numerical stability:
    #   log L_i(FF) = log( sum_g [ exp(log prior_G[g] + log BinomialPMF_i(g; FF)) ] ).
    # Then total log-likelihood = sum_i log L_i(FF). We pick FF maximizing that.
    
    # Helper functions:
    def _log_binomial_coeff(n, k):
        """Compute log(comb(n,k)) via gammaln for numerical stability."""
        return (
            math.lgamma(n + 1) 
            - math.lgamma(k + 1) 
            - math.lgamma(n - k + 1)
        )
    
    def _log_binomial_pmf(k, n, p):
        """Compute log of Binomial PMF: C(n,k) * p^k * (1-p)^(n-k)."""
        if p <= 0.0:
            # If expected prob p=0, then PMF = 1 if k=0, else 0 (log PMF = -inf)
            return 0.0 if k == 0 else float("-inf")
        if p >= 1.0:
            # If p=1, PMF = 1 if k=n, else 0
            return 0.0 if k == n else float("-inf")
        # Otherwise standard log PMF
        log_coeff = _log_binomial_coeff(n, k)
        return log_coeff + k * math.log(p) + (n - k) * math.log(1 - p)
    
    def _log_sum_exp(log_values):
        """Stable log-sum-exp over a list of log-values."""
        m = max(log_values)
        if m == float("-inf"):
            return float("-inf")
        total = sum(math.exp(x - m) for x in log_values)
        return m + math.log(total)
    
    # Extract columns as numpy arrays for speed
    p_arr = background_chr_df["af"].to_numpy(dtype=float)             # population allele freq
    ref_arr = background_chr_df["ref_reads"].to_numpy(dtype=int)
    alt_arr = background_chr_df["alt_reads"].to_numpy(dtype=int)
    n_arr = ref_arr + alt_arr                                         # total coverage per SNP
    
    # Filter out any SNPs with zero total coverage (no information)
    valid_mask = n_arr > 0
    p_arr = p_arr[valid_mask]
    ref_arr = ref_arr[valid_mask]
    alt_arr = alt_arr[valid_mask]
    n_arr = n_arr[valid_mask]
    
    # Precompute genotype priors and maternal/fetal alt fractions per SNP
    # For each SNP index i and genotype g in {0,1,2}:
    #   prior_G[i, g], maternal_alt_frac[g], fetal_alt_frac[i, g]
    num_snps = len(p_arr)
    
    # maternal_alt_frac is genotype-specific but not SNP-specific: [0.0, 0.5, 1.0]
    maternal_alt_frac = np.array([0.0, 0.5, 1.0], dtype=float)
    
    # Prepare arrays of shape (num_snps, 3) for genotype-specific priors and fetal fractions
    prior_G = np.zeros((num_snps, 3), dtype=float)
    fetal_alt_frac = np.zeros((num_snps, 3), dtype=float)
    
    for i in range(num_snps):
        p_i = p_arr[i]
        # Hardy-Weinberg priors for maternal genotype
        prior_G[i, 0] = (1 - p_i) ** 2        # P(mother = 0/0)
        prior_G[i, 1] = 2 * p_i * (1 - p_i)    # P(mother = 0/1)
        prior_G[i, 2] = p_i ** 2              # P(mother = 1/1)
        
        # Expected fetal alt-allele fraction conditioned on maternal genotype g:
        #   g=0 (0/0): fetal f_alt = 0.5 * p_i
        #   g=1 (0/1): fetal f_alt = 0.25 + 0.5 * p_i
        #   g=2 (1/1): fetal f_alt = 0.5 + 0.5 * p_i
        fetal_alt_frac[i, 0] = 0.5 * p_i
        fetal_alt_frac[i, 1] = 0.25 + 0.5 * p_i
        fetal_alt_frac[i, 2] = 0.5 + 0.5 * p_i
    
    # Precompute log(prior_G) for numeric stability
    # If prior_G is zero (very rare unless p_i=0 or 1), log-prior = -inf
    with np.errstate(divide="ignore"):
        log_prior_G = np.log(prior_G)
    
    # Pre-allocate a dictionary to store total log-likelihood for each FF candidate
    log_likelihoods = {}
    
    # Grid of FF values to search over [f_min, f_max] inclusive
    f_values = np.arange(f_min, f_max + f_step, f_step)
    
    # Main loop: evaluate log-likelihood for each FF candidate
    for FF in f_values:
        total_log_lik = 0.0
        
        # For each SNP, compute log L_i(FF) and add to total
        for i in range(num_snps):
            n_i = int(n_arr[i])
            k_i = int(alt_arr[i])
            
            # Prepare a list of log-probabilities for G = 0,1,2
            log_probs = []
            for g in (0, 1, 2):
                # Maternal alt fraction m_g
                m_g = maternal_alt_frac[g]
                # Fetal alt fraction f_g for this SNP/genotype
                f_g = fetal_alt_frac[i, g]
                # Mixture allele fraction at FF: μ = (1−FF)*m_g + FF*f_g
                mu = (1 - FF) * m_g + FF * f_g
                
                # Compute log BinomialPMF(k_i; n_i, mu)
                log_pmf = _log_binomial_pmf(k_i, n_i, mu)
                
                # log prior + log binomial PMF
                log_probs.append(log_prior_G[i, g] + log_pmf)
            
            # Use log-sum-exp to get log L_i(FF)
            log_L_i = _log_sum_exp(log_probs)
            total_log_lik += log_L_i
        
        log_likelihoods[FF] = total_log_lik
    
    # Select the FF value that maximizes the total log-likelihood
    best_ff = max(log_likelihoods, key=log_likelihoods.get)
    return best_ff, log_likelihoods


def estimate_fetal_fraction_with_model(background_chr_df, model_accuracy):
    """
    Estimate fetal fraction (FF) from maternal cfDNA SNP read counts using a model.
    """
    pass


def LR_calculator(input_df: pd.DataFrame, fetal_fraction: float) -> float:
    """
    Calculate the likelihood ratio (LR) of trisomy vs. disomy on a target chromosome
    using SNP read counts (ref_reads, alt_reads) in maternal plasma cfDNA.
    This implementation sums over all possible maternal/paternal/fetal genotype combinations
    under both disomy and trisomy hypotheses.

    Args:
        input_df (pd.DataFrame): DataFrame containing cfDNA sequencing data with columns:
            - 'chr': chromosome of SNP (string or int)
            - 'pos': genomic coordinate of SNP (not used in computation, but retained for reference)
            - 'af': population allele frequency of the ALT allele (float between 0 and 1)
            - 'ref_reads': number of reads supporting the REF allele (non-negative int)
            - 'alt_reads': number of reads supporting the ALT allele (non-negative int)
        fetal_fraction (float): Estimated fetal fraction in maternal plasma (0 < fetal_fraction < 1).

    Returns:
        float: Likelihood ratio LR = L_trisomy / L_disomy. If L_disomy is zero, returns np.inf (infinite LR).
    """

    # ----------------------------
    # Helper functions
    # ----------------------------

    def genotype_priors(af: float) -> dict:
        """
        Compute genotype prior probabilities under Hardy-Weinberg equilibrium
        for a biallelic SNP with population ALT allele frequency 'af'.

        Returns a dict mapping genotype string '0/0', '0/1', '1/1' to their probabilities.
        """
        p_hom_ref = (1.0 - af) ** 2
        p_het     = 2.0 * af * (1.0 - af)
        p_hom_alt = af ** 2
        return {
            '0/0': p_hom_ref,
            '0/1': p_het,
            '1/1': p_hom_alt
        }

    def get_fetal_prob_disomy(gm: str, gp: str) -> dict:
        """
        Given maternal genotype gm and paternal genotype gp (each '0/0', '0/1', or '1/1'),
        return a dict mapping fetal genotype ('0/0', '0/1', '1/1') to its probability under disomy (diploid).
        """
        # Count maternal allele frequencies in her genotype
        # e.g., gm = '0/1' -> maternal_allele_probs = {0: 0.5, 1: 0.5}
        def allele_probs_from_genotype(gt: str) -> dict:
            a1, a2 = gt.split('/')
            a1, a2 = int(a1), int(a2)
            # Count how many alt-allele copies in the two alleles
            counts = {0: 0, 1: 0}
            counts[a1] += 1
            counts[a2] += 1
            total = 2.0
            return {0: counts[0] / total, 1: counts[1] / total}

        pm = allele_probs_from_genotype(gm)
        pp = allele_probs_from_genotype(gp)

        # Initialize fetal genotype probabilities
        fetal_probs = {'0/0': 0.0, '0/1': 0.0, '1/1': 0.0}

        # Iterate over maternal allele i and paternal allele j
        for mat_allele, p_mat in pm.items():
            for pat_allele, p_pat in pp.items():
                prob_combo = p_mat * p_pat
                if mat_allele == 0 and pat_allele == 0:
                    fetal_probs['0/0'] += prob_combo
                elif (mat_allele == 0 and pat_allele == 1) or (mat_allele == 1 and pat_allele == 0):
                    fetal_probs['0/1'] += prob_combo
                else:  # mat_allele == 1 and pat_allele == 1
                    fetal_probs['1/1'] += prob_combo

        return fetal_probs

    def get_fetal_prob_trisomy(gm: str, gp: str) -> dict:
        """
        Given maternal genotype gm and paternal genotype gp, return a dict mapping
        fetal ALT-allele dosage (0, 1, 2, 3) to its probability under trisomy (ploidy = 3).
        We approximate maternal nondisjunction by drawing two maternal alleles with replacement,
        and paternal by drawing one paternal allele. 
        The fetal genotype dosage d = (# ALT alleles among the three copies).
        """
        # First, compute single-allele probabilities from mother & father
        def allele_probs(gt: str) -> dict:
            a1, a2 = gt.split('/')
            a1, a2 = int(a1), int(a2)
            counts = {0: 0, 1: 0}
            counts[a1] += 1
            counts[a2] += 1
            total = 2.0
            return {0: counts[0] / total, 1: counts[1] / total}

        pm = allele_probs(gm)  # maternal single-allele distribution
        pp = allele_probs(gp)  # paternal single-allele distribution

        # Compute maternal gamete distribution of k ALT alleles out of 2 draws (with replacement)
        # P_maternal_gamete[k] = P(k ALT alleles in two independent draws from pm)
        p_maternal_alt = pm[1]
        p_maternal_pair = {
            0: (1 - p_maternal_alt) ** 2,
            1: 2 * p_maternal_alt * (1 - p_maternal_alt),
            2: p_maternal_alt ** 2
        }

        # Paternal single-allele distribution: P_paternal_allele[c] where c in {0,1}
        p_paternal_allele = pp  # already {0: prob, 1: prob}

        # Now compute fetal dosage distribution: d = k_maternal + c_paternal,
        # where k_maternal in {0,1,2}, c_paternal in {0,1}.
        fetal_dosage_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for k_m in [0, 1, 2]:
            for c_p in [0, 1]:
                d = k_m + c_p
                prob = p_maternal_pair[k_m] * p_paternal_allele[c_p]
                fetal_dosage_probs[d] += prob

        return fetal_dosage_probs

    def log_binomial_pmf(k: int, n: int, p: float) -> float:
        """
        Compute log of Binomial(n, p) probability mass at k: log( C(n,k) * p^k * (1-p)^(n-k) ).
        Handles edge cases p == 0 or p == 1 explicitly.
        """
        if p < 0.0 or p > 1.0:
            return -math.inf
        if n < 0 or k < 0 or k > n:
            return -math.inf

        # Handle exact p == 0
        if p == 0.0:
            return 0.0 if k == 0 else -math.inf
        # Handle exact p == 1
        if p == 1.0:
            return 0.0 if k == n else -math.inf

        # General case
        # log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        log_comb = (math.lgamma(n + 1)
                    - math.lgamma(k + 1)
                    - math.lgamma(n - k + 1))
        return log_comb + k * math.log(p) + (n - k) * math.log(1.0 - p)

    def logsumexp(log_values: np.ndarray) -> float:
        """
        Stable log-sum-exp: returns log(sum(exp(log_values_i))) for possibly very negative values.
        """
        if len(log_values) == 0:
            return -math.inf
        m = np.max(log_values)
        if m == -math.inf:
            return -math.inf
        sum_exp = np.sum(np.exp(log_values - m))
        return m + math.log(sum_exp)

    # ----------------------------
    # Input validation
    # ----------------------------
    required_cols = {'chr', 'pos', 'af', 'ref_reads', 'alt_reads'}
    if not required_cols.issubset(input_df.columns):
        missing = required_cols - set(input_df.columns)
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    # Validate fetal_fraction
    if not (0.0 < fetal_fraction < 1.0):
        raise ValueError("fetal_fraction must be between 0 and 1 (exclusive).")

    # Validate SNP numbers
    if input_df.shape[0] == 0:
        raise ValueError(f"No SNPs found on chromosome '{trisomy_chr}' in input_df.")

    # ----------------------------
    # Main likelihood computation
    # ----------------------------
    logL_disomy = 0.0  # sum of log-likelihoods under disomy
    logL_trisomy = 0.0  # sum of log-likelihoods under trisomy

    # Pre-define possible maternal/paternal genotype labels
    genotype_labels = ['0/0', '0/1', '1/1']

    # Iterate over each SNP on the target chromosome
    for idx, snp in input_df.iterrows():
        # Extract observed counts
        try:
            ref_count = int(snp['ref_reads'])
            alt_count = int(snp['alt_reads'])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid read counts at index {idx}: ref_reads={snp['ref_reads']}, alt_reads={snp['alt_reads']}")

        depth = ref_count + alt_count
        # If no coverage at this SNP, it contributes no information; skip it.
        if depth == 0:
            continue

        # Population allele frequency
        af = float(snp['af'])
        if not (0.0 <= af <= 1.0):
            raise ValueError(f"Invalid population allele frequency (af={af}) at index {idx}.")

        # Compute maternal and paternal genotype priors
        geno_priors = genotype_priors(af)

        # For disomy: collect log-likelihood contributions for each combination (gm, gp, gf)
        log_terms_disomy = []

        # Loop over maternal genotype (gm) and paternal genotype (gp)
        for gm in genotype_labels:
            p_gm = geno_priors[gm]
            if p_gm <= 0.0:
                continue
            log_p_gm = math.log(p_gm)

            for gp in genotype_labels:
                p_gp = geno_priors[gp]
                if p_gp <= 0.0:
                    continue
                log_p_gp = math.log(p_gp)

                # Get fetal genotype distribution under disomy
                fetal_dist_dis = get_fetal_prob_disomy(gm, gp)

                # Count ALT alleles in maternal genotype gm
                d_maternal = sum(int(allele) for allele in gm.split('/'))  # 0, 1, or 2 ALT alleles

                # Enumerate fetal genotype gf and compute P(obs | gm, gf, fetal_fraction)
                for gf, p_gf in fetal_dist_dis.items():
                    if p_gf <= 0.0:
                        continue
                    log_p_gf = math.log(p_gf)

                    # Count ALT alleles in fetal genotype gf (diploid)
                    d_fetal = sum(int(allele) for allele in gf.split('/'))  # 0, 1, or 2 ALT alleles

                    # Expected ALT allele fraction in mixed cfDNA:
                    # maternal contribution: (1 - ff) * (d_maternal / 2)
                    # fetal contribution: ff * (d_fetal / 2)
                    p_alt_exp = (1.0 - fetal_fraction) * (d_maternal / 2.0) + fetal_fraction * (d_fetal / 2.0)

                    # Compute log-probability of observing (alt_count) ALT reads out of (depth) total
                    log_p_obs = log_binomial_pmf(alt_count, depth, p_alt_exp)
                    # If log_p_obs is -inf, this genotype combination contributes zero likelihood
                    if log_p_obs == -math.inf:
                        continue

                    # Sum log-terms: log(P(gm)) + log(P(gp)) + log(P(gf)) + log(P(obs))
                    log_terms_disomy.append(log_p_gm + log_p_gp + log_p_gf + log_p_obs)

        # Compute SNP log-likelihood under disomy via log-sum-exp
        logL_snp_disomy = logsumexp(np.array(log_terms_disomy, dtype=float))
        # If the SNP yields zero likelihood under disomy (i.e., logL = -inf), 
        # it means no genotype combination could explain the data => overall likelihood will be zero.
        # We propagate this by setting total logL to -inf immediately.
        if logL_snp_disomy == -math.inf:
            return 0.0  # L_disomy = 0 => LR = 0

        logL_disomy += logL_snp_disomy

        # For trisomy: collect log-likelihood contributions for each combination (gm, gp, fetal dosage)
        log_terms_trisomy = []

        for gm in genotype_labels:
            p_gm = geno_priors[gm]
            if p_gm <= 0.0:
                continue
            log_p_gm = math.log(p_gm)

            for gp in genotype_labels:
                p_gp = geno_priors[gp]
                if p_gp <= 0.0:
                    continue
                log_p_gp = math.log(p_gp)

                # Get fetal dosage distribution under trisomy (ploidy=3)
                fetal_dist_tri = get_fetal_prob_trisomy(gm, gp)

                # Count ALT alleles in maternal genotype gm
                d_maternal = sum(int(allele) for allele in gm.split('/'))

                # For each possible fetal ALT allele dosage d_f (0..3)
                for d_fetal, p_fd in fetal_dist_tri.items():
                    if p_fd <= 0.0:
                        continue
                    log_p_fd = math.log(p_fd)

                    # Expected ALT allele fraction in mixed cfDNA under trisomy:
                    # maternal: (1 - ff) * (d_maternal/2)
                    # fetal: ff * (d_fetal / 3)
                    p_alt_exp_tri = (1.0 - fetal_fraction) * (d_maternal / 2.0) + fetal_fraction * (d_fetal / 3.0)

                    log_p_obs_tri = log_binomial_pmf(alt_count, depth, p_alt_exp_tri)
                    if log_p_obs_tri == -math.inf:
                        continue

                    log_terms_trisomy.append(log_p_gm + log_p_gp + log_p_fd + log_p_obs_tri)

        # Compute SNP log-likelihood under trisomy via log-sum-exp
        logL_snp_trisomy = logsumexp(np.array(log_terms_trisomy, dtype=float))
        if logL_snp_trisomy == -math.inf:
            # If no genotype combination under trisomy can explain the data, 
            # then L_trisomy for the whole dataset = 0 => LR = 0
            return 0.0

        logL_trisomy += logL_snp_trisomy

    # Final LR = exp(logL_trisomy - logL_disomy)
    delta_logL = logL_trisomy - logL_disomy

    # If disomy log-likelihood is extremely low (logL_disomy very negative), delta_logL could be large positive.
    # Compute LR carefully to avoid overflow:
    if delta_logL > 700:  # threshold to avoid math.exp overflow
        return float('inf')
    elif delta_logL < -700:
        return 0.0
    else:
        return math.exp(delta_logL)


def LR_calculator_with_model(input_df, fetal_fraction):
    """Calculates the likelihood ratio for a given SNP sequencing simulation dataframe.
    """
    pass


def worker(task):
    """Wrapper for multiprocessing"""
    i, j, depth_lambda, ff_value, trisomy_chr, n_repeats = task

    for repeat_idx in range(n_repeats):
        # Disomy simulation
        disomy_result = run_single_disomy_simulation_set(potential_df, depth_lambda, ff_value, model_accuracy)
        disomy_result['ref_reads'] = disomy_result['maternal_ref_reads'] + disomy_result['fetal_ref_reads']
        disomy_result['alt_reads'] = disomy_result['maternal_alt_reads'] + disomy_result['fetal_alt_reads']

        # Trisomy simulation
        trisomy_result = run_single_trisomy_simulation_set(potential_df, depth_lambda, ff_value, model_accuracy, trisomy_chr)
        trisomy_result['ref_reads'] = trisomy_result['maternal_ref_reads'] + trisomy_result['fetal_ref_reads']
        trisomy_result['alt_reads'] = trisomy_result['maternal_alt_reads'] + trisomy_result['fetal_alt_reads']

        disomy_LR_list = []
        trisomy_LR_list = []
        # Calculate the likelihood ratio for each autosome
        for target_chr in [f'chr{i}' for i in range(1, 23)]:
            disomy_background_df = disomy_result[disomy_result['chr'] != target_chr]
            disomy_target_df = disomy_result[disomy_result['chr'] == target_chr]
            disomy_ff, _ = estimate_fetal_fraction(disomy_background_df, f_max=0.2)
            disomy_LR = LR_calculator(disomy_target_df, disomy_ff)

            trisomy_background_df = trisomy_result[trisomy_result['chr'] != trisomy_chr]
            trisomy_target_df = trisomy_result[trisomy_result['chr'] == trisomy_chr]
            trisomy_ff, _ = estimate_fetal_fraction(trisomy_background_df, f_max=0.2)
            trisomy_LR = LR_calculator(trisomy_target_df, trisomy_ff)

            disomy_LR_list.append(disomy_LR)
            trisomy_LR_list.append(trisomy_LR)

        

    return disomy_LR_list, trisomy_LR_list


if __name__ == '__main__':
    # 1. Load & annotate the variants DataFrame:
    potential_df = pd.read_csv(
        'merged_probes_ChinaMAP_filtered.tsv',
        sep='\t',
        names=['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info']
    )
    potential_df['af'] = potential_df.apply(extract_AF, axis=1)

    # 2. Define parameter grids:
    min_depth = 50
    max_depth = 150
    depth_steps = 11
    min_ff = 0.005
    max_ff = 0.05
    ff_steps = 10
    model_accuracy = 0.81
    trisomy_chr = 'chr16'
    n_repeats = 10

    depth_param_range = np.linspace(min_depth, max_depth, depth_steps, dtype=int)
    ff_param_range = np.linspace(min_ff, max_ff, ff_steps)

    # 3. Prepare a zero matrix to store results:
    disomy_LR_matrix = np.zeros((len(depth_param_range), len(ff_param_range)))
    trisomy_LR_matrix = np.zeros((len(depth_param_range), len(ff_param_range)))

    # 4. Build a list of (i, j, depth_lambda, ff_value) tasks
    tasks = []
    for i, depth_lambda in enumerate(depth_param_range):
        for j, ff_value in enumerate(ff_param_range):
            tasks.append((i, j, depth_lambda, ff_value, trisomy_chr, n_repeats))

    # 5. Use multiprocessing to compute each cell in parallel:
    with Pool(cpu_count()) as pool:
        for i, j, disomy_LR_list, trisomy_LR_list in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
            disomy_LR_matrix[i, j] = disomy_LR_list
            trisomy_LR_matrix[i, j] = trisomy_LR_list

    # 6. Save the results_matrix to a TSV (no index column):
    pd.DataFrame(disomy_LR_matrix).to_csv('simulate_disomy_LR_result.tsv', sep='\t', index=False)
    pd.DataFrame(trisomy_LR_matrix).to_csv('simulate_trisomy_LR_result.tsv', sep='\t', index=False)
