import itertools
import allel
import numpy as np
import csv
import time
import sys

# ----------------------------
# Helper Functions (unchanged)
# ----------------------------

# Function to calculate Extended Haplotype Homozygosity decay (EHH)
def process_ehh(haplotypes):
    ehh_decay = allel.ehh_decay(haplotypes)
    return np.nanmean(ehh_decay)

# Function to calculate Fay and Wu's H statistic (From Ulas Isildak GitHub)
def calc_faywu_h(haplotypes):
    n_sam = haplotypes.shape[0]
    counts = haplotypes.sum(axis=0)
    S_i = [np.sum(counts == i) for i in range(1, n_sam)]
    i = np.arange(1, n_sam)
    n_i = n_sam - i
    thetaP = np.sum((n_i * i * np.array(S_i) * 2) / (n_sam * (n_sam - 1.0)))
    thetaH = np.sum((2 * np.array(S_i) * np.power(i, 2)) / (n_sam * (n_sam - 1.0)))
    return thetaP - thetaH

# Function to calculate Fu and Li's D* statistic (From Ulas Isildak GitHub)
def calc_fuli_d_star(haplotypes):
    n_sam = haplotypes.shape[0]
    n_pos = haplotypes.shape[1]
    # Efficient computation using numpy functions
    r = np.arange(1, n_sam)
    an = np.sum(1.0 / r)
    bn = np.sum(1.0 / (r ** 2))
    an1 = an + 1.0 / n_sam

    # Calculate cn and dn using formulae
    cn = (2 * (((n_sam * an) - 2 * (n_sam - 1))) / ((n_sam - 1) * (n_sam - 2)))
    dn = (cn + np.true_divide((n_sam - 2), ((n_sam - 1) ** 2)) + np.true_divide(2, (n_sam - 1)) * (
            3.0 / 2 - (2 * an1 - 3) / (n_sam - 2) - 1.0 / n_sam))

    # Calculate vds and uds
    vds = (((n_sam / (n_sam - 1.0)) ** 2) * bn + (an ** 2) * dn - 2 * (n_sam * an * (an + 1)) / (
            (n_sam - 1.0) ** 2)) / (an ** 2 + bn)
    uds = ((n_sam / (n_sam - 1.0)) * (an - n_sam / (n_sam - 1.0))) - vds

    # Count the number of singleton sites
    ss = np.sum(np.sum(haplotypes, axis=0) == 1)

    # Calculate Dstar
    Dstar = ((n_sam / (n_sam - 1.0)) * n_pos - (an * ss)) / (uds * n_pos + vds * (n_pos ^ 2)) ** 0.5
    return Dstar

# Calculates average pairwise differences - necessary to calculate Fu and Li's F*
def calc_pi(haplotypes):
    """
    Compute the average pairwise differences among haplotypes using
    allele frequencies. Assumes haplotypes is a binary (0/1) 2D array
    with shape (n_haplotypes, n_variants).
    
    Returns a scalar equal to the average number of differences per pair.
    """
    n, m = haplotypes.shape  # n: number of haplotypes, m: number of variants
    # Compute the allele frequency at each site
    p = haplotypes.sum(axis=0) / n
    # Expected difference at each site is 2 * p * (1-p)
    # (because the probability that two haplotypes differ is 2 * p * (1-p))
    # Average across sites, then multiply by m to get the total differences per pair.
    return np.mean(2 * p * (1 - p)) * m

# Function to calculate Fu and Li's F* statistic (From Ulas Isildak GitHub)
def calc_fuli_f_star(haplotypes):
    n_sam = haplotypes.shape[0]
    n_pos = haplotypes.shape[1]

    # Precompute harmonic sums
    r = np.arange(1, n_sam)
    an = np.sum(1.0 / r)
    bn = np.sum(1.0 / (r ** 2))
    an1 = an + 1.0 / n_sam

    # Calculate vfs and ufs using precomputed values
    vfs = (((2 * (n_sam ** 3) + 110 * (n_sam ** 2) - 255 * n_sam + 153) /
            (9 * (n_sam ** 2) * (n_sam - 1))) + ((2 * (n_sam - 1) * an) / (n_sam ** 2)) -
           ((8 * bn) / n_sam)) / (an ** 2 + bn)
    ufs = ((n_sam / (n_sam + 1) + (n_sam + 1) / (3 * (n_sam - 1)) - 4 / (n_sam * (n_sam - 1)) +
            ((2 * (n_sam + 1)) / ((n_sam - 1) ** 2)) * (an1 - ((2 * n_sam) / (n_sam + 1)))) / an) - vfs
    
    # Calculate pi and ss
    pi_est = calc_pi(haplotypes)
    ss = np.sum(np.sum(haplotypes, axis=0) == 1)

    # Returns Fstar
    return (pi_est - ((n_sam - 1) / n_sam) * ss) / np.sqrt(ufs * n_pos + vfs * (n_pos ** 2))

# Function to calculate Garud's H statistics (H1, H12, H123, H2/H1)
def process_garuds_h(haplotypes):
    h_stats = allel.garud_h(haplotypes)
    return h_stats[0], h_stats[1], h_stats[2], h_stats[3]

# Function to calculate observed and expected Heterozygosity
def process_heterozygosity(genotypes):
    obs_het = allel.heterozygosity_observed(genotypes)
    allele_counts = genotypes.count_alleles()
    allele_freqs = allele_counts.to_frequencies()
    exp_het = allel.heterozygosity_expected(allele_freqs, ploidy=2)
    return np.nanmean(obs_het), np.nanmean(exp_het)

# Function to calculate integrated haplotype score (iHS)
def process_ihs(haplotypes, positions):
    ihs = allel.ihs(haplotypes, positions, include_edges=True)
    return np.nanmean(ihs)

# Function to calculate and process normalised site-specific log-ratio of EHH (nSL)
def process_nsl(haplotypes):
    nsl = allel.nsl(haplotypes)
    return np.nanmean(nsl)

# Function to calculate non-central deviation statistic (NCD1) (From Ulas Isildak GitHub)
def process_ncd1(genotypes):
    allele_counts = genotypes.count_alleles()
    allele_frequencies = allele_counts.to_frequencies()

    # Use only polymorphic sites
    polymorphic_sites = (allele_counts.max_allele() > 0)
    allele_frequencies = allele_frequencies[polymorphic_sites]
    return np.mean(np.abs(allele_frequencies[:, 1] - 0.5))

# Function to calculate raggedness index (from Ulas Isildak GitHub)
def calc_raggedness(haplotypes):
    n_sam, n_var = haplotypes.shape
    hist = np.zeros(n_var + 1, dtype=np.int64)
    total_pairs = 0
    for i in range(n_sam - 1):
        dists = np.sum(haplotypes[i+1:] != haplotypes[i], axis=1)
        hist_update = np.bincount(dists, minlength=n_var + 1)
        hist[:len(hist_update)] += hist_update
        total_pairs += dists.shape[0]
    freqs = hist / total_pairs
    max_mist = np.max(np.nonzero(hist)[0])
    # Sum differences over bins 0..max_mist-1
    rgd = np.sum((freqs[1:max_mist+1] - freqs[:max_mist]) ** 2)
    # Add the boundary term: difference between bin max_mist and 0
    rgd += (0 - freqs[max_mist]) ** 2
    return rgd

# Function to calculate Tajima's D
def process_tajima_d(allele_counts):
    allele_counts = allele_counts[:, :2]
    return allel.tajima_d(allele_counts)

# Function to calculate Zeng's E statistic (from Ulas Isildak GitHub)
def calc_zeng_e(haplotypes):
    n_sam = haplotypes.shape[0]
    n_pos = haplotypes.shape[1]
    an = np.sum(np.divide(1.0, range(1, n_sam)))
    bn = np.sum(np.divide(1.0, np.power(range(1, n_sam), 2)))
    counts = haplotypes.sum(axis=0)
    S_i = [np.sum(counts == i) for i in range(1, n_sam)]
    thetaW = n_pos / an
    thetaL = np.sum(np.multiply(S_i, range(1, n_sam))) / (n_sam - 1.0)
    theta2 = (n_pos * (n_pos - 1.0)) / (an ** 2 + bn)
    var1 = (n_sam / (2.0 * (n_sam - 1.0)) - 1.0 / an) * thetaW
    var2 = theta2 * (bn / (an ** 2.0)) + 2 * bn * (n_sam / (n_sam - 1.0)) ** 2.0 - (2.0 * (n_sam * bn - n_sam + 1.0)) / ((n_sam - 1.0) * an) - (3.0 * n_sam + 1.0) / (n_sam - 1.0)
    varlw = var1 + var2
    return (thetaL - thetaW) / np.sqrt(varlw)

# Instead of separate functions for Kelly's ZnS and LD, compute LD once and reuse it:
def compute_ld_metrics(genotypes, positions):
    # Convert to alternate allele count array (make contiguous)
    n_alt = np.ascontiguousarray(genotypes.to_n_alt(fill=-1))
    ld_r = allel.rogers_huff_r(n_alt)
    mean_r2 = np.nanmean(ld_r ** 2)
    n_pos = len(positions)
    kelly_zns = (np.nansum(ld_r ** 2) * 2.0) / (n_pos * (n_pos - 1.0))
    return mean_r2, kelly_zns

# Function to calculate mean pairwise difference
def process_mpd(genotypes):
    pairwise_diff = allel.mean_pairwise_difference(genotypes.to_n_alt())
    mean_pairwise_diff = np.nanmean(pairwise_diff)

    return mean_pairwise_diff

# Function to calculate and process Hudson Fst
def process_hudson_fst(vcf_paths):
    callsets = [allel.read_vcf(path, fields='*') for path in vcf_paths]
    # Extract genotype data and positions
    genotypes = [allel.GenotypeArray(callset['calldata/GT']) for callset in callsets]
    positions = [allel.SortedIndex(callset['variants/POS']) for callset in callsets]
    # Generate all pairs of positions
    population_combinations = list(itertools.combinations(range(len(positions)), 2))

    # Intitialise list to store results
    mean_fst_results = []

    # Function to process each pair
    for i, j in population_combinations:
        # Find common positions between two populations
        intersect_pos = positions[i].intersect(positions[j])
        # Count number of intersected positions
        count_intersections = len(intersect_pos)
        
        if count_intersections > 0:
            # Locate indices of intersected positions in each dataset
            indices_i = positions[i].locate_keys(intersect_pos)
            indices_j = positions[j].locate_keys(intersect_pos)
            # Subselect genotypes at intersected positions
            genotypes_i_filt = genotypes[i].subset(indices_i)
            genotypes_j_filt = genotypes[j].subset(indices_j)
            # Calculate allele counts
            allele_counts_i = genotypes_i_filt.count_alleles()
            allele_counts_j = genotypes_j_filt.count_alleles()
            # Calculate Fst statistic
            num, den = allel.hudson_fst(allele_counts_i, allele_counts_j)
            mean_fst = np.sum(num)/np.sum(den)
            mean_fst_results.append((f'Pop.{i+1} vs Pop.{j+1}', mean_fst))
        else:
            mean_fst_results.append((f'Pop.{i+1} vs Pop.{j+1}', None))
    return mean_fst_results

# ----------------------------
# Compute Summary Stats per VCF
# ----------------------------
def compute_summary_stats(callset):
    # Create arrays once and ensure contiguity
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    haplotypes = np.ascontiguousarray(genotypes.to_haplotypes())
    positions = allel.SortedIndex(callset['variants/POS'])
    positions_unsorted = callset['variants/POS']
    allele_counts = genotypes.count_alleles()

    # Compute shared LD metrics only once
    mean_ld_r2, kelly_zns = compute_ld_metrics(genotypes, positions)

    stats = {}
    stats['mean_ehh_decay']     = process_ehh(haplotypes)
    stats['faywu_h']            = calc_faywu_h(haplotypes)
    stats['fuli_d']             = calc_fuli_d_star(haplotypes)
    stats['fuli_f']             = calc_fuli_f_star(haplotypes)
    H1, H12, H123, H2_H1         = process_garuds_h(haplotypes)
    stats['garud_H1']           = H1
    stats['garud_H12']          = H12
    stats['garud_H123']         = H123
    stats['garud_H2_H1']        = H2_H1
    stats['haplotype_diversity']= allel.haplotype_diversity(haplotypes)
    stats['mean_obs_het'], stats['mean_exp_het'] = process_heterozygosity(genotypes)
    stats['mean_ihs']           = process_ihs(haplotypes, positions_unsorted)
    stats['kelly_zns']          = kelly_zns
    stats['mean_ld_r2']         = mean_ld_r2
    stats['mean_pairwise_diff'] = process_mpd(genotypes)
    stats['ncd1']               = process_ncd1(genotypes)
    stats['mean_nsl']           = process_nsl(haplotypes)
    stats['nucleotide_diversity']= allel.sequence_diversity(positions_unsorted, allele_counts)
    stats['raggedness']         = calc_raggedness(haplotypes)
    stats['tajima_d']           = process_tajima_d(allele_counts)
    stats['watterson_theta']    = allel.watterson_theta(positions_unsorted, allele_counts)
    stats['zeng_e']             = calc_zeng_e(haplotypes)
    return stats

# ----------------------------
# Main Function (Refactored)
# ----------------------------
def main(base_path, start_sim, end_sim):
    start = time.perf_counter()
    
    # Prepare CSV files for writing
    output_stats = f'{base_path}/BS_statistics_results_{start_sim}_{end_sim}.csv'
    output_fst   = f'{base_path}/BS_fst_results_{start_sim}_{end_sim}.csv'
    
    with open(output_stats, 'w', newline='') as stats_file, open(output_fst, 'w', newline='') as fst_file:
        stats_writer = csv.writer(stats_file)
        fst_writer = csv.writer(fst_file)
        
        # Write header for main stats (modify as desired)
        stats_header = [
            'BS/Pop', 'mean_ehh_decay', 'faywu_h', 'fuli_d', 'fuli_f', 
            'garud_H1', 'garud_H12', 'garud_H123', 'garud_H2_H1', 
            'haplotype_diversity', 'mean_obs_het', 'mean_exp_het', 'mean_ihs', 
            'kelly_zns', 'mean_ld_r2', 'mean_pairwise_diff', 'ncd1', 'mean_nsl', 
            'nucleotide_diversity', 'raggedness', 'tajima_d', 'watterson_theta', 'zeng_e'
        ]
        stats_writer.writerow(stats_header)
        fst_writer.writerow(['BS/Pop', 'Mean Fst'])
        
        # Loop over simulations and populations
        for sim_num in range(start_sim, end_sim + 1):
            BS_all_stats = []
            BS_all_fst_results = []
            
            # Process each population VCF for this simulation
            for pop_num in range(1, 5):
                vcf_path = f'{base_path}/BS_{sim_num}_p{pop_num}.vcf'
                callset = allel.read_vcf(vcf_path, fields='*')
                stats = compute_summary_stats(callset)
                
                row = [f'BS.{sim_num}/Pop.{pop_num}'] + [stats[h] for h in stats_header[1:]]
                BS_all_stats.append(row)
            
            # Process Fst across populations for this simulation
            vcf_paths = [f'{base_path}/BS_{sim_num}_p{i}.vcf' for i in range(1, 5)]
            fst_results = process_hudson_fst(vcf_paths)
            for result in fst_results:
                BS_all_fst_results.append([f'BS.{sim_num}/{result[0]}', result[1]])
            
            stats_writer.writerows(BS_all_stats)
            fst_writer.writerows(BS_all_fst_results)
    
    end = time.perf_counter()
    print('Results saved to CSV files.')
    print(f'Time taken: {end - start:.6f} seconds')

# ----------------------------
# Execute Main
# ----------------------------
if __name__ == "__main__":
    base_path = sys.argv[1]
    start_sim = int(sys.argv[2])
    end_sim = int(sys.argv[3])
    main(base_path, start_sim, end_sim)

