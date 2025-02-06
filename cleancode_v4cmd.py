import itertools
import allel
import numpy as np
import csv
import time
import sys


# Function to calculate Extended Haplotype Homozygosity decay
def process_ehh(haplotypes):
    # EHH decay
    ehh_decay = allel.ehh_decay(haplotypes)
    mean_ehh_decay = np.nanmean(ehh_decay)

    return mean_ehh_decay


# Calculates Fay and Wu's H statistic
def calc_faywu_h(haplotypes):
    n_sam = haplotypes.shape[0]
    counts = haplotypes.sum(axis=0)
    S_i = [np.sum(counts == i) for i in range(1, n_sam)]
    i = np.arange(1, n_sam)
    n_i = n_sam - i
    thetaP = np.sum((n_i * i * np.array(S_i) * 2) / (n_sam * (n_sam - 1.0)))
    thetaH = np.sum((2 * np.array(S_i) * np.power(i, 2)) / (n_sam * (n_sam - 1.0)))
    Hstat = thetaP - thetaH
    return Hstat


# Calculates Fu and Li's D* statistic
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
    '''diffs = haplotypes[:, None, :] != haplotypes[None, :, :]
    num_diffs = diffs.sum(axis=2)
    pi_est = np.triu(num_diffs, k=1).sum() / ((haplotypes.shape[0] * (haplotypes.shape[0] - 1)) / 2)
    return pi_est'''
    return allel.mean_pairwise_difference(haplotypes)


# Calculates Fu and Li's F* statistic
def calc_fuli_f_star(haplotypes):
    n_sam = haplotypes.shape[0]
    n_pos = haplotypes.shape[1]

    # Precompute the harmonic sums
    r = np.arange(1, n_sam)
    an = np.sum(1.0 / r)
    bn = np.sum(1.0 / (r ** 2))
    an1 = an + 1.0 / n_sam

    # Calculate vfs and ufs using precomputed values
    vfs = (((2 * (n_sam ** 3) + 110 * (n_sam ** 2) - 255 * n_sam + 153) /
            (9 * (n_sam ** 2) * (n_sam - 1))) + ((2 * (n_sam - 1) * an) / (n_sam ** 2)) -
           ((8 * bn) / n_sam)) / (an ** 2 + bn)

    ufs = ((n_sam / (n_sam + 1) + (n_sam + 1) / (3 * (n_sam - 1)) - 4 /
            (n_sam * (n_sam - 1)) + ((2 * (n_sam + 1)) / ((n_sam - 1) ** 2)) *
            (an1 - ((2 * n_sam) / (n_sam + 1)))) / an) - vfs

    # Calculate pi and ss
    pi_est = calc_pi(haplotypes)
    ss = np.sum(np.sum(haplotypes, axis=0) == 1)

    # Calculate Fstar
    Fstar = (pi_est - ((n_sam - 1) / n_sam) * ss) / np.sqrt(ufs * n_pos + vfs * (n_pos ** 2))
    return Fstar


# Function to process and calculate Garud's H statistics
def process_garuds_h(haplotypes):
    # Garud's H statistics
    h_stats = allel.garud_h(haplotypes)
    H1, H12, H123, H2_H1 = h_stats[0], h_stats[1], h_stats[2], h_stats[3]
    return H1, H12, H123, H2_H1


# Function to calculate Observed and Expected Heterozygosity
def process_heterozygosity(genotypes):
    # Observed heterozygosity
    obs_het = allel.heterozygosity_observed(genotypes)
    mean_obs_het = np.nanmean(obs_het)

    # Expected heterozygosity
    allele_counts = genotypes.count_alleles()
    allele_freqs = allele_counts.to_frequencies()
    exp_het = allel.heterozygosity_expected(allele_freqs, ploidy=2)
    mean_exp_het = np.nanmean(exp_het)

    return mean_obs_het, mean_exp_het


# Function to calculate iHS
def process_ihs(haplotypes, positions):
    ihs = allel.ihs(haplotypes, positions, include_edges=True)
    mean_ihs = np.nanmean(ihs)

    return mean_ihs


# Function to calculate Kelly's Zns
def calc_kelly_zns(g, n_pos):
    gn = g.to_n_alt(fill=-1)
    LDr = allel.rogers_huff_r(gn)
    LDr2 = LDr ** 2
    kellyzn = (np.nansum(LDr2) * 2.0) / (n_pos * (n_pos - 1.0))
    return kellyzn


def process_kelly_zns(genotypes, positions):
    n_positions = len(positions)

    kzns = calc_kelly_zns(genotypes, n_positions)

    return kzns


# Function to calculate LD (Rogers-Huff r)
def process_ld(genotypes):
    ld_r = allel.rogers_huff_r(genotypes.to_n_alt())
    mean_r2 = np.nanmean(ld_r ** 2)

    return mean_r2


# Function to calculate mean pairwise difference
def process_mpd(genotypes):
    pairwise_diff = allel.mean_pairwise_difference(genotypes.to_n_alt())
    mean_pairwise_diff = np.nanmean(pairwise_diff)

    return mean_pairwise_diff


# Function to calculate NCD1
def calculate_ncd1(allele_frequencies, target_frequency=0.5):
    """Calculate NCD1 statistic"""
    deviations = np.abs(allele_frequencies - target_frequency)
    ncd1 = np.mean(deviations)
    return ncd1


# Function to process NCD1 Statistics
def process_ncd1(genotypes):
    # Calculate allele frequencies
    allele_counts = genotypes.count_alleles()
    allele_frequencies = allele_counts.to_frequencies()
    # Use only polymorphic sites (exclude monomorphic)
    polymorphic_sites = (allele_counts.max_allele() > 0)
    allele_frequencies = allele_frequencies[polymorphic_sites]

    # Calculate NCD1
    ncd1 = calculate_ncd1(allele_frequencies[:, 1])  # Frequency of the derived allele
    return ncd1


# Function to calculate and process nSL (Normalised Site-specific Log-Ratio of EHH)
def process_nsl(haplotypes):
    # Calculate nSL
    nsl = allel.nsl(haplotypes)
    mean_nsl = np.nanmean(nsl)
    return mean_nsl


# Calculate Raggedness index
def calc_raggedness(haplotypes):
    # Calculate pairwise mismatches in a vectorized manner
    diffs = np.sum(haplotypes[:, None, :] != haplotypes[None, :, :], axis=2)
    mist = diffs[np.triu_indices_from(diffs, k=1)]

    # Compute the total number of comparisons
    lnt = len(mist)

    # Compute the frequencies of each mismatch count
    max_mist = np.max(mist)
    fclass = np.zeros(max_mist + 2)  # +2 to include the range 1 to max_mist+1 safely
    for i in range(1, max_mist + 2):
        count_i = np.sum(mist == i)
        count_i_minus_1 = np.sum(mist == (i - 1))
        fclass[i] = ((count_i / lnt) - (count_i_minus_1 / lnt)) ** 2

    # Sum the squared differences
    rgd = np.sum(fclass)
    return rgd

# Function to calculate and process Tajima's D
def process_tajima_d(allele_counts):
    # Ensure allele_counts is a 2-dimensional array
    allele_counts = allele_counts[:, :2]

    tajima_d = allel.tajima_d(allele_counts)
    return tajima_d

# Calculates Zeng et al's E statistic.
def calc_zeng_e(haplotypes):
    n_sam = haplotypes.shape[0]
    n_pos = np.size(haplotypes, 1)
    an = np.sum(np.divide(1.0, range(1, n_sam)))
    bn = np.sum(np.divide(1.0, np.power(range(1, n_sam), 2)))
    counts = haplotypes.sum(axis=0)
    S_i = []
    for i in range(1, n_sam):
        S_i.append(sum(counts == i))
    thetaW = n_pos / an
    thetaL = np.sum(np.multiply(S_i, range(1, n_sam))) / (n_sam - 1.0)
    theta2 = (n_pos * (n_pos - 1.0)) / (an ** 2 + bn)

    var1 = (n_sam / (2.0 * (n_sam - 1.0)) - 1.0 / an) * thetaW
    var2 = theta2 * (bn / (an ** 2.0)) + 2 * bn * (n_sam / (n_sam - 1.0)) ** 2.0 - (
            2.0 * (n_sam * bn - n_sam + 1.0)) / ((n_sam - 1.0) * an) - (3.0 * n_sam + 1.0) / (
               (n_sam - 1.0))
    varlw = var1 + var2

    ZengE = (thetaL - thetaW) / (varlw) ** 0.5
    return ZengE

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
'''
# Function to calculate and process Patterson's F2
def process_f2(vcf_paths):
    try:
        callsets = [allel.read_vcf(path, fields='*') for path in vcf_paths]
        # Extract genotype data and positions
        genotypes = [allel.GenotypeArray(callset['calldata/GT']) for callset in callsets]
        positions = [allel.SortedIndex(callset['variants/POS']) for callset in callsets]
        # Generate all pairs of positions
        population_combinations = list(itertools.combinations(range(len(positions)), 2))

        # Initialize list to store F2 results
        f2_results = []

        # Function to process each pair
        for i, j in population_combinations:
            # Find common positions between the two populations
            intersect_pos = positions[i].intersect(positions[j])

            if len(intersect_pos) > 0:
                # Locate indices of intersected positions in each dataset
                indices_i = positions[i].locate_keys(intersect_pos)
                indices_j = positions[j].locate_keys(intersect_pos)

                # Subselect genotypes at intersected positions
                genotypes_i_filt = genotypes[i].subset(indices_i)
                genotypes_j_filt = genotypes[j].subset(indices_j)

                # Calculate allele counts
                allele_counts_i = genotypes_i_filt.count_alleles()
                allele_counts_j = genotypes_j_filt.count_alleles()

                # Filter out non-biallelic sites
                biallelic_i = allele_counts_i.is_biallelic()
                biallelic_j = allele_counts_j.is_biallelic()
                biallelic_sites = biallelic_i & biallelic_j

                allele_counts_i_biallelic = allele_counts_i.compress(biallelic_sites, axis=0)
                allele_counts_j_biallelic = allele_counts_j.compress(biallelic_sites, axis=0)

                # Calculate F2 statistic
                f2_statistic = allel.patterson_f2(allele_counts_i_biallelic, allele_counts_j_biallelic)
                mean_f2 = np.mean(f2_statistic)
                f2_results.append((f'Pop.{i+1} vs Pop.{j+1}', mean_f2))
            else:
                f2_results.append((f'Pop.{i+1} vs Pop.{j+1}', None))
    except Exception as e:
        print(f'Error processing F2: {e}')
        return []

    return f2_results

def process_f3(vcf_paths):
    try:
        callsets = [allel.read_vcf(path, fields='*') for path in vcf_paths]
        # Extract genotype data and positions
        genotypes = [allel.GenotypeArray(callset['calldata/GT']) for callset in callsets]
        positions = [allel.SortedIndex(callset['variants/POS']) for callset in callsets]
        # Generate all triplets of populations
        population_combinations = list(itertools.combinations(range(len(positions)), 3))

        # Initialize lists to store results
        f3_results = []

        # Function to process each triplet
        for i, j, k in population_combinations:
            # Find common positions between the three populations
            intersect_pos_ij = positions[i].intersect(positions[j])
            intersect_pos = intersect_pos_ij.intersect(positions[k])

            # Count the number of intersected positions
            count_intersections = len(intersect_pos)
            
            if count_intersections > 0:
                # Locate indices of intersected positions in each dataset
                indices_i = positions[i].locate_keys(intersect_pos)
                indices_j = positions[j].locate_keys(intersect_pos)
                indices_k = positions[k].locate_keys(intersect_pos)

                # Subselect genotypes at intersected positions
                genotypes_i_filt = genotypes[i].subset(indices_i)
                genotypes_j_filt = genotypes[j].subset(indices_j)
                genotypes_k_filt = genotypes[k].subset(indices_k)

                # Calculate allele counts
                allele_counts_i = genotypes_i_filt.count_alleles()
                allele_counts_j = genotypes_j_filt.count_alleles()
                allele_counts_k = genotypes_k_filt.count_alleles()

                # Filter out non-biallelic sites
                biallelic_i = allele_counts_i.is_biallelic()
                biallelic_j = allele_counts_j.is_biallelic()
                biallelic_k = allele_counts_k.is_biallelic()
                biallelic_sites = biallelic_i & biallelic_j & biallelic_k

                allele_counts_i_biallelic = allele_counts_i.compress(biallelic_sites, axis=0)
                allele_counts_j_biallelic = allele_counts_j.compress(biallelic_sites, axis=0)
                allele_counts_k_biallelic = allele_counts_k.compress(biallelic_sites, axis=0)

                # Calculate F3 statistic
                f3, T = allel.patterson_f3(allele_counts_k_biallelic, allele_counts_i_biallelic, allele_counts_j_biallelic)
                mean_f3 = np.mean(f3)
                f3_results.append((f'Pop.{k+1}; Pop.{i+1}, Pop.{j+1}', mean_f3))
            else:
                f3_results.append((f'Pop.{k+1}; Pop.{i+1}, Pop.{j+1}', None))
    except Exception as e:
        print(f'Error processing simulation: {e}')
        return []

    return f3_results

def process_f4(vcf_paths):
    try:
        callsets = [allel.read_vcf(path, fields='*') for path in vcf_paths]
        # Extract genotype data and positions
        genotypes = [allel.GenotypeArray(callset['calldata/GT']) for callset in callsets]
        positions = [allel.SortedIndex(callset['variants/POS']) for callset in callsets]
        
        # Find common positions across all four populations
        intersect_pos_ij = positions[0].intersect(positions[1])
        intersect_pos_ijk = intersect_pos_ij.intersect(positions[2])
        intersect_pos = intersect_pos_ijk.intersect(positions[3])

        # Count the number of intersected positions
        count_intersections = len(intersect_pos)

        f4_results = []

        if count_intersections > 0:
            # Locate indices of intersected positions in each dataset
            indices_i = positions[0].locate_keys(intersect_pos)
            indices_j = positions[1].locate_keys(intersect_pos)
            indices_k = positions[2].locate_keys(intersect_pos)
            indices_l = positions[3].locate_keys(intersect_pos)

            # Subselect genotypes at intersected positions
            genotypes_i_filt = genotypes[0].subset(indices_i)
            genotypes_j_filt = genotypes[1].subset(indices_j)
            genotypes_k_filt = genotypes[2].subset(indices_k)
            genotypes_l_filt = genotypes[3].subset(indices_l)

            # Calculate allele counts
            allele_counts_i = genotypes_i_filt.count_alleles()
            allele_counts_j = genotypes_j_filt.count_alleles()
            allele_counts_k = genotypes_k_filt.count_alleles()
            allele_counts_l = genotypes_l_filt.count_alleles()

            # Filter out non-biallelic sites
            biallelic_i = allele_counts_i.is_biallelic()
            biallelic_j = allele_counts_j.is_biallelic()
            biallelic_k = allele_counts_k.is_biallelic()
            biallelic_l = allele_counts_l.is_biallelic()
            biallelic_sites = biallelic_i & biallelic_j & biallelic_k & biallelic_l

            allele_counts_i_biallelic = allele_counts_i.compress(biallelic_sites, axis=0)
            allele_counts_j_biallelic = allele_counts_j.compress(biallelic_sites, axis=0)
            allele_counts_k_biallelic = allele_counts_k.compress(biallelic_sites, axis=0)
            allele_counts_l_biallelic = allele_counts_l.compress(biallelic_sites, axis=0)

            # Calculate F4 statistic
            f4, den = allel.patterson_d(allele_counts_i_biallelic, allele_counts_j_biallelic, allele_counts_k_biallelic, allele_counts_l_biallelic)
            mean_f4 = np.mean(f4)
            f4_results.append((f'P1 vs P2 vs P3 vs P4', mean_f4))
        else:
            f4_results.append((f'P1 vs P2 vs P3 vs P4', 'No intersected positions found'))
    except Exception as e:
        print(f'Error processing simulation: {e}')
        return []

    return f4_results

def process_patterson_fst(vcf_paths):
    try:
        callsets = [allel.read_vcf(path, fields='*') for path in vcf_paths]
        # Extract genotype data and positions
        genotypes = [allel.GenotypeArray(callset['calldata/GT']) for callset in callsets]
        positions = [allel.SortedIndex(callset['variants/POS']) for callset in callsets]
        # Generate all pairs of positions
        population_combinations = list(itertools.combinations(range(len(positions)), 2))

        # Initialize lists to store results
        mean_patterson_fst = []

        # Function to process each pair
        for i, j in population_combinations:
            # Find common positions between the two populations
            intersect_pos = positions[i].intersect(positions[j])

            # Count the number of intersected positions
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

                # Filter out non-biallelic sites
                biallelic_i = allele_counts_i.is_biallelic()
                biallelic_j = allele_counts_j.is_biallelic()
                biallelic_sites = biallelic_i & biallelic_j

                allele_counts_i_biallelic = allele_counts_i.compress(biallelic_sites, axis=0)
                allele_counts_j_biallelic = allele_counts_j.compress(biallelic_sites, axis=0)

                # Calculate Fst statistic
                num, den = allel.patterson_fst(allele_counts_i_biallelic, allele_counts_j_biallelic)
                mean_fst = np.sum(num) / np.sum(den)
                mean_patterson_fst.append((f'Pop.{i+1} vs Pop.{j+1}', mean_fst))
            else:
                mean_patterson_fst.append((f'Pop.{i+1} vs Pop.{j+1}', None))
    except Exception as e:
        print(f'Error processing simulation: {e}')
        return []

    return mean_patterson_fst
    '''
# Main function to process multiple simulations
def main(base_path, start_sim, end_sim):
    start = time.perf_counter()
    
    # Open the CSV files once before the loop and write the headers
    output_stats = f'statistics_results_{start_sim}_{end_sim}.csv'
    output_fst = f'fst_results_{start_sim}_{end_sim}.csv'
    
    with open(output_stats, 'w', newline='') as stats_file, open(output_fst, 'w', newline='') as fst_file:
        stats_writer = csv.writer(stats_file)
        fst_writer = csv.writer(fst_file)
        
        # Write the header for the main stats
        stats_writer.writerow([
            'Sim/Pop', 
            'Mean EHH Decay', 
            'Fay & Wu\'s H', 
            'Fu & Li\'s D*', 
            'Fu & Li\'s F*', 
            'H1', 'H12', 'H123', 'H2/H1',
            'Hap. Diversity',
            'Mean Obs. Heterozygosity', 
            'Mean Exp. Heterozygosity', 
            'Mean iHS', 
            'Kelly\'s Zns', 
            'Mean LD (r^2)', 
            'Mean Pairwise Difference',
            'NCD1',
            'nSL',
            'Nuc. Diversity',
            'Raggedness Index',
            'Tajima\'s D',
            'Watterson\'s Theta',
            'Zeng\'s E'])
        
        # Write the header for the Fst results
        fst_writer.writerow(['Sim/Pop', 'Mean Fst'])
        
        for sim_num in range(start_sim, end_sim + 1):
            all_stats = []
            all_fst_results = []
            
            for pop_num in range(1, 5):
                vcf_path = f'{base_path}/sim_{sim_num}_p{pop_num}.vcf'
                
                # Read and process the VCF file
                callset = allel.read_vcf(vcf_path, fields='*')
                genotypes = allel.GenotypeArray(callset['calldata/GT'])
                haplotypes = genotypes.to_haplotypes()
                positions = allel.SortedIndex(callset['variants/POS'])
                positions_unsorted = callset['variants/POS']
                allele_counts = genotypes.count_alleles()
                
                # Calculate statistics
                mean_ehh_decay = process_ehh(haplotypes)
                faywu_h = calc_faywu_h(haplotypes)
                fuli_d = calc_fuli_d_star(haplotypes)
                fuli_f = calc_fuli_f_star(haplotypes)
                H1, H12, H123, H2_H1 = process_garuds_h(haplotypes)
                hap_diversity = allel.haplotype_diversity(haplotypes)
                mean_obs_het, mean_exp_het = process_heterozygosity(genotypes)
                mean_ihs = process_ihs(haplotypes, positions_unsorted)
                kzns = process_kelly_zns(genotypes, positions)
                mean_r2 = process_ld(genotypes)
                mean_pairwise_diff = process_mpd(genotypes)
                ncd1 = process_ncd1(genotypes)
                mean_nsl = process_nsl(haplotypes)
                nuc_diversity = allel.sequence_diversity(positions_unsorted, allele_counts)
                raggedness = calc_raggedness(haplotypes)
                tajima_d = process_tajima_d(allele_counts)
                w_theta = allel.watterson_theta(positions_unsorted, allele_counts)
                zeng_e = calc_zeng_e(haplotypes)
                
                all_stats.append((
                    f'Sim.{sim_num}/Pop.{pop_num}',
                    mean_ehh_decay,
                    faywu_h,
                    fuli_d,
                    fuli_f,
                    H1, H12, H123, H2_H1,
                    hap_diversity,
                    mean_obs_het,
                    mean_exp_het,
                    mean_ihs,
                    kzns,
                    mean_r2,
                    mean_pairwise_diff,
                    ncd1,
                    mean_nsl,
                    nuc_diversity,
                    raggedness,
                    tajima_d,
                    w_theta,
                    zeng_e
                ))
            
            # Process and write Fst results
            vcf_paths = [
                f'{base_path}/sim_{sim_num}_p1.vcf',
                f'{base_path}/sim_{sim_num}_p2.vcf',
                f'{base_path}/sim_{sim_num}_p3.vcf',
                f'{base_path}/sim_{sim_num}_p4.vcf'
            ]
            mean_fst_results = process_hudson_fst(vcf_paths)
            for result in mean_fst_results:
                all_fst_results.append([f'Sim.{sim_num}/{result[0]}', result[1]])
            
            # Write the main stats for this simulation to the CSV file
            stats_writer.writerows(all_stats)
            
            # Write the Fst results for this simulation to the CSV file
            fst_writer.writerows(all_fst_results)
    
    end = time.perf_counter()
    elapsed = end - start
    print('Results saved to CSV files.')
    print(f'Time taken: {elapsed:.6f} seconds')


'''# Path to base directory containing VCF files
base_path = '/Users/Antoniopacheco/Documents/SimDeepBLUEs/sim_test1'

# Number of simulations to process
num_simulations = 25

# Execute main function
main(base_path, num_simulations)'''

if __name__ == "__main__":
    base_path = sys.argv[1]
    start_sim = int(sys.argv[2])
    end_sim = int(sys.argv[3])
    main(base_path, start_sim, end_sim)