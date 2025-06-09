import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from numpy.random import default_rng
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from functools import partial
from itertools import chain
import click


def estimate_fetal_fraction(background_chr_df, f_min=0.001, f_max=0.5, f_step=0.001):
    """
    Estimate fetal fraction (FF) from maternal cfDNA SNP read counts using a simple MLE approach.
    
    We assume:
      - For each SNP i, population allele frequency = p_i (“af” column).
      - Mother’s genotype G_i ∈ {0/0, 0/1, 1/1} with Hardy-Weinberg priors:
          P(G=0/0) = (1 - p_i)^2
          P(G=0/1) = 2 * p_i * (1 - p_i)
          P(G=1/1) = (p_i)^2
      - Fetal genotype depends on mother’s genotype and a random paternal allele drawn from population AF p_i.
        We compute expected fetal alt‐allele fraction E[fetal_alt_i | G_mother]:
          • If mother = 0/0:  fetal_alt_frac = 0.5 * p_i
          • If mother = 0/1:  fetal_alt_frac = 0.25 + 0.5 * p_i
          • If mother = 1/1:  fetal_alt_frac = 0.5 + 0.5 * p_i
      - Maternal alt‐allele fraction is:
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
        return a dict mapping fetal genotype ('0/0', '0/1', '1/1') to its probability under disomy (disomy).
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
    required_cols = {'chr', 'pos', 'af', 'cfDNA_ref_reads', 'cfDNA_alt_reads'}
    if not required_cols.issubset(input_df.columns):
        missing = required_cols - set(input_df.columns)
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    # Validate fetal_fraction
    if not (0.0 < fetal_fraction < 1.0):
        raise ValueError("fetal_fraction must be between 0 and 1 (exclusive).")

    # Validate SNP numbers
    if input_df.shape[0] == 0:
        raise ValueError(f"No SNPs found on target chromosome in input_df.")

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

                    # Count ALT alleles in fetal genotype gf (disomy)
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


def plot_lr_heatmap(input_df, output_file, fmt="{:.3f}"):
    """Creates & saves a heatmap from input_df DataFrame. 
    
    Args:
        input_df (pd.DataFrame): DataFrame with depth as index and ff as columns
        output_file (str): Path to save the heatmap PNG
        fmt (str): Python format string for cell annotations (e.g. "{:.3f}" or "d")
    """
    plt.figure(figsize=(12, 8))
    
    # Convert DataFrame values to numpy array for heatmap
    data = input_df.values
    
    # Create annotation matrix with formatted strings
    annot_matrix = np.empty_like(data, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            annot_matrix[i, j] = fmt.format(data[i, j]) if not np.isnan(data[i, j]) else ""

    ax = sns.heatmap(
        data,
        annot=annot_matrix,
        fmt="",                 # Already formatted in annot_matrix
        cmap="viridis",
        xticklabels=[f"{ff:.3f}" for ff in input_df.columns],
        yticklabels=input_df.index.astype(int)
    )
    
    # Improve label formatting
    ax.set_xlabel("Fetal Fraction", fontsize=12)
    ax.set_ylabel("Sequencing Depth (Poisson λ)", fontsize=12)
    ax.set_title("Estimated Fetal Fraction Heatmap", fontsize=14, pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lr_scatter(input_df, output_file, threshold=10, jitter_amount=0.2):
    """
    Creates and saves a scatter plot from input_df DataFrame, with increased horizontal jitter 
    and adjusted annotation positions to prevent overlap.

    Args:
        input_df (pd.DataFrame): DataFrame with columns ['label', 'depth', 'ff', 'repeat_idx', 'chr1', ..., 'chr22']
        output_file (str): Path to save the scatter plot PNG.
        threshold (int, optional): Threshold value for highlighting points. Defaults to 10.
        jitter_amount (float, optional): Base jitter to apply along the x-axis (in category units). 
                                         Effective jitter is increased by 50%. Defaults to 0.2.
    """
    # Define chromosome columns from 'chr1' to 'chr22'
    chr_cols = [f'chr{i}' for i in range(1, 23)]
    
    # Melt the DataFrame to long format for plotting
    df_melted = input_df.melt(
        id_vars=['label', 'depth', 'ff', 'repeat_idx'],
        value_vars=chr_cols,
        var_name='chromosome',
        value_name='lr'
    )
    
    # Convert 'chromosome' into an ordered categorical type so we can map to numeric codes
    df_melted['chromosome'] = pd.Categorical(
        df_melted['chromosome'],
        categories=chr_cols,
        ordered=True
    )
    
    # Map each chromosome to a numeric code (0 for 'chr1', 1 for 'chr2', ..., 21 for 'chr22')
    df_melted['chrom_code'] = df_melted['chromosome'].cat.codes
    
    # Increase jitter by 50%
    effective_jitter = jitter_amount * 1.5
    
    # Apply random jitter along the x-axis by adding a small uniform noise to the chromosome code
    # Each point will be shifted by up to ±effective_jitter category units
    df_melted['x_jitter'] = df_melted['chrom_code'] + np.random.uniform(
        low=-effective_jitter,
        high=effective_jitter,
        size=len(df_melted)
    )
    
    # Begin plotting
    plt.figure(figsize=(16, 8))
    
    # Scatterplot using numeric x positions (with jitter) and labeling by 'label'
    sns.scatterplot(
        x=df_melted['x_jitter'],
        y=df_melted['lr'],
        hue=df_melted['label'],
        palette='viridis',
        edgecolor=None,
        alpha=0.8
    )
    
    # Draw a horizontal line at the threshold value
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
    
    # Annotate points that exceed the threshold
    above_threshold = df_melted[df_melted['lr'] > threshold].copy()
    annotations = []  # Will store existing annotation positions for collision checking
    offset_y = 0.3    # Vertical offset for resolving overlap
    offset_x = jitter_amount  # Horizontal offset for resolving overlap
    
    for _, row in above_threshold.iterrows():
        annotation_text = f"{row['ff']:.3f}_{row['depth']}"
        x_pos = row['x_jitter']
        y_pos = row['lr']
        
        # Check for collisions with existing annotations; adjust both x and y as needed
        collision = True
        while collision:
            collision = False
            for (ax, ay) in annotations:
                # If new label is too close to an existing one in both x and y
                if abs(x_pos - ax) < offset_x and abs(y_pos - ay) < offset_y:
                    # Shift vertically first
                    y_pos += offset_y
                    # Then shift horizontally if still colliding
                    x_pos += offset_x
                    collision = True
                    # Re-check against all annotations from scratch
                    break
        
        # Record the final position of this annotation
        annotations.append((x_pos, y_pos))
        
        plt.annotate(
            annotation_text,
            xy=(row['x_jitter'], row['lr']),
            xytext=(x_pos+0.3, y_pos+1.2),
            ha='center',
            fontsize=8,
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5)
        )
    
    # Configure x-axis ticks: place them at integer positions with chromosome labels
    plt.xticks(
        ticks=np.arange(len(chr_cols)),
        labels=chr_cols,
        rotation=45
    )
    
    # Set plot titles and labels
    plt.title('Scatter Plot of Chromosome LR Values')
    plt.xlabel('')
    plt.ylabel('LR')
    
    # Move the legend outside the main plot area
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


# TODO:
# Click option input
def main():
    pass


if __name__ == "__main__":
    main()