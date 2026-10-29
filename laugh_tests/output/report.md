# Laugh test report

88 PASS, 1 KNOWN DEVIATION, 0 FAIL.

A PASS row matches an analytical result. A KNOWN DEVIATION row is a
textbook result the current code does not reproduce. The test is marked
strict, so the suite fails if that code starts matching the textbook and
the deviation note is not retired.

| Status | Test | Check | Expected | Computed | Reference |
| --- | --- | --- | --- | --- | --- |
| PASS | test_particle_filter::test_weights_match_the_gaussian_likelihood | normalised weights | [5.739865e-01, 3.481404e-01, 7.768062e-02, 1.925510e-04] | [5.739865e-01, 3.481404e-01, 7.768062e-02, 1.925510e-04] | Arulampalam et al. (2002): L = exp(-0.5 * (error / sigma)^2), then L/sum(L). sigma = 0.10 * \|SWE\| when that exceeds the floor. Gordon et al. (1993) normalise the likelihood |
| PASS | test_particle_filter::test_weights_match_the_gaussian_likelihood | weights are finite | True | True | A positive sigma keeps the Gaussian likelihood finite |
| PASS | test_particle_filter::test_weights_match_the_gaussian_likelihood | weights sum to 1 | 1 | 1 | Gordon, Salmond, and Smith (1993) |
| PASS | test_particle_filter::test_weights_match_the_gaussian_likelihood | exact match has the largest weight | True | True | A particle on the observation is the most likely |
| PASS | test_particle_filter::test_two_hrus_multiply_independent_gaussians | weights from each HRU's own sigma | [0.576117, 0.211942, 0.211942] | [0.576117, 0.211942, 0.211942] | Arulampalam et al. (2002): independent observations multiply, so the exponent sums ((x - z) / sigma)^2. sigma is 20 and 10 kg m-2 here, both above the floor of 5 |
| PASS | test_particle_filter::test_two_hrus_multiply_independent_gaussians | weights are finite and sum to 1 | True | True | Gordon, Salmond, and Smith (1993) |
| PASS | test_particle_filter::test_two_hrus_multiply_independent_gaussians | closest particle has the largest weight | True | True | A particle on both observations is the most likely |
| PASS | test_particle_filter::test_zero_swe_uses_the_absolute_floor | weights at observed SWE of 0 | [0.574097, 0.348207, 0.077696] | [0.574097, 0.348207, 0.077696] | sigma = max(0.10 * \|SWE\|, swe_sigma_floor). At SWE = 0 the floor of 5 kg m-2 is the standard deviation |
| PASS | test_particle_filter::test_zero_swe_uses_the_absolute_floor | weights are finite | True | True | The floor keeps sigma positive when observed SWE is zero |
| PASS | test_particle_filter::test_zero_swe_uses_the_absolute_floor | weights sum to 1 | 1 | 1 | Gordon, Salmond, and Smith (1993) |
| PASS | test_particle_filter::test_zero_swe_uses_the_absolute_floor | particle on the observation has the largest weight | True | True | A particle on the observation is the most likely |
| PASS | test_particle_filter::test_floor_binds_only_below_the_relative_term | weights when 0.10 * \|SWE\| is below the floor | [0.574097, 0.348207, 0.077696] | [0.574097, 0.348207, 0.077696] | sigma = max(0.10 * 20, 5) = 5 kg m-2 |
| PASS | test_particle_filter::test_floor_binds_only_below_the_relative_term | those weights differ from sigma = 0.10 * \|SWE\| | True | True | The floor replaces the relative term when the relative term is smaller |
| PASS | test_particle_filter::test_floor_binds_only_below_the_relative_term | weights when 0.10 * \|SWE\| is above the floor | [0.574097, 0.348207, 0.077696] | [0.574097, 0.348207, 0.077696] | sigma = max(0.10 * 100, 5) = 10 kg m-2 |
| PASS | test_particle_filter::test_weights_follow_hru_id_not_row_order | weights aligned by hruId | [9.996646e-01, 3.353501e-04] | [9.996646e-01, 3.353501e-04] | Observations and predictions are paired on hruId, not row position |
| PASS | test_particle_filter::test_weights_follow_hru_id_not_row_order | matching particle has the larger weight | True | True | Member 001 matches both HRUs; member 002 misses HRU 2 by 20 kg m-2 |
| PASS | test_particle_filter::test_identical_particles_share_weight_equally | equal weights | [0.333333, 0.333333, 0.333333] | [0.333333, 0.333333, 0.333333] | Equal errors produce equal likelihoods |
| PASS | test_particle_filter::test_state_estimate_matches_weighted_moments | weighted mean | 1.75 | 1.75 | Hand calculation of sum(w * x) |
| PASS | test_particle_filter::test_state_estimate_matches_weighted_moments | weighted population variance | 0.6875 | 0.6875 | Hand calculation of sum(w * (x - mean)^2), divisor 1 |
| PASS | test_particle_filter::test_state_estimate_matches_weighted_moments | equal-weight mean | 2 | 2 | Arithmetic mean of 1, 2, 3 |
| PASS | test_particle_filter::test_state_estimate_matches_weighted_moments | equal-weight population variance | 0.666667 | 0.666667 | Population variance of 1, 2, 3 is 2/3 |
| PASS | test_particle_filter::test_effective_sample_size | equal weights | 4 | 4 | Arulampalam et al. (2002): N_eff = 1 / sum(w_i^2) |
| PASS | test_particle_filter::test_effective_sample_size | one particle certain | 1 | 1 | Arulampalam et al. (2002): N_eff = 1 / sum(w_i^2) |
| PASS | test_particle_filter::test_effective_sample_size | two particles share the weight | 2 | 2 | Arulampalam et al. (2002): N_eff = 1 / sum(w_i^2) |
| PASS | test_particle_filter::test_systematic_resample_counts | copy counts for weights 0.5, 0.25, 0.25 | [2, 1, 1] | [2, 1, 1] | Kitagawa (1996): systematic resampling copies particle i floor(N*w_i) or ceil(N*w_i) times |
| PASS | test_particle_filter::test_systematic_resample_counts | seeds that missed those counts | 0 | 0 | Kitagawa (1996): systematic resampling copies particle i floor(N*w_i) or ceil(N*w_i) times |
| PASS | test_particle_filter::test_certain_particle_is_copied_everywhere | indices when the first weight is 1 | [0, 0, 0, 0] | [0, 0, 0, 0] | A weight of 1 puts every copy on that particle |
| PASS | test_particle_filter::test_relative_uncertainty_flattens_the_weights | weights at 10% of 200 kg m-2 | [0.574097, 0.348207, 0.077696] | [0.574097, 0.348207, 0.077696] | Arulampalam et al. (2002): Gaussian likelihood exp(-0.5 * (error / sigma)^2). SWE sigma is max(relative_uncertainty * \|observation\|, sigma_floor) |
| PASS | test_particle_filter::test_relative_uncertainty_flattens_the_weights | weights at 50% of 200 kg m-2 | [0.344434, 0.337614, 0.317953] | [0.344434, 0.337614, 0.317953] | Arulampalam et al. (2002): Gaussian likelihood exp(-0.5 * (error / sigma)^2). SWE sigma is max(relative_uncertainty * \|observation\|, sigma_floor) |
| PASS | test_particle_filter::test_relative_uncertainty_flattens_the_weights | the exact particle keeps more weight when the uncertainty is smaller | True | True | Arulampalam et al. (2002): Gaussian likelihood exp(-0.5 * (error / sigma)^2). SWE sigma is max(relative_uncertainty * \|observation\|, sigma_floor) |
| PASS | test_particle_filter::test_resample_when_effective_sample_size_is_low | indices when N_eff = 1 and the threshold is 2 | [0, 0, 0, 0] | [0, 0, 0, 0] | Arulampalam et al. (2002): resample when N_eff is at or below N_thr |
| PASS | test_particle_filter::test_resample_when_effective_sample_size_is_low | indices when N_eff = 4 and the threshold is 2 | None | None | A uniform ensemble is above the threshold, so it is kept |
| PASS | test_particle_filter::test_figure_resampling | effective sample sizes | [4., 2., 1.] | [4., 2., 1.] | Arulampalam et al. (2002): N_eff = 1 / sum(w_i^2) |
| PASS | test_particle_filter::test_figure_resampling | copy counts for weights 0.5, 0.25, 0.25 | [2, 1, 1] | [2, 1, 1] | Kitagawa (1996): systematic resampling copies particle i floor(N*w_i) or ceil(N*w_i) times |
| PASS | test_particle_filter::test_figure_weights_versus_distance | largest gap between the code and exp(-0.5 (error / sigma)^2) | 0 | 1.11022e-16 | Arulampalam et al. (2002): Gaussian likelihood. sigma = max(relative_uncertainty * \|observed SWE\|, 5 kg m-2) |
| PASS | test_quantile_mapping::test_hazen_plotting_positions | positions for n = 4 | [0.125, 0.375, 0.625, 0.875] | [0.125, 0.375, 0.625, 0.875] | Hazen plotting position p_i = (i - 0.5) / n |
| PASS | test_quantile_mapping::test_median_of_an_odd_sample_is_the_50th_percentile | F(3) for five sorted samples | 0.5 | 0.5 | With n = 5 the middle rank has p = (3 - 0.5) / 5 = 0.5 |
| PASS | test_quantile_mapping::test_identity_mapping | mapped values | [1. , 1.5, 2. , 3. , 4. , 5.5, 7. ] | [1. , 1.5, 2. , 3. , 4. , 5.5, 7. ] | F^{-1}(F(y)) = y when both distributions use the same knots |
| PASS | test_quantile_mapping::test_affine_mapping_is_exact | mapped values | [ 2.1 , 3.35, 7.1 , 14.6 , 19.6 ] | [ 2.1 , 3.35, 7.1 , 14.6 , 19.6 ] | A positive affine QQ relationship is reproduced exactly |
| PASS | test_quantile_mapping::test_exponential_quantiles_map_by_the_scale_ratio | mapped values along y* = 2y | [0.048195, 0.79091 , 1.533624, 2.276338, 3.019053, 3.761767, 4.504482, 5.247196, 5.98991 , 6.732625, 7.475339] | [0.048195, 0.79091 , 1.533624, 2.276338, 3.019053, 3.761767, 4.504482, 5.247196, 5.98991 , 6.732625, 7.475339] | Matched exponential quantiles lie on a straight QQ line |
| PASS | test_quantile_mapping::test_random_sample_error_shrinks_with_n | mean absolute error decreases (0.763 at n=10, 0.229 at n=100, 0.078 at n=1000) | True | True | Gudmundsson et al. (2012) QUANT: empirical CDFs, linear interpolation, no extrapolation; population map of Exp(1) onto Exp(2) is y* = 2y |
| PASS | test_quantile_mapping::test_mapping_is_monotone | largest decrease along an increasing grid | 0 | 0 | An empirical quantile map is non-decreasing |
| PASS | test_quantile_mapping::test_outside_the_sample_is_clipped | value below the obs sample | 10 | 10 | Gudmundsson et al. (2012) QUANT: empirical CDFs, linear interpolation, no extrapolation |
| PASS | test_quantile_mapping::test_outside_the_sample_is_clipped | value above the obs sample | 40 | 40 | Gudmundsson et al. (2012) QUANT: empirical CDFs, linear interpolation, no extrapolation |
| PASS | test_quantile_mapping::test_outside_the_sample_is_clipped | clipped low probability | 0.125 | 0.125 | Hazen plotting position p_i = (i - 0.5) / n |
| PASS | test_quantile_mapping::test_outside_the_sample_is_clipped | clipped high probability | 0.875 | 0.875 | Hazen plotting position p_i = (i - 0.5) / n |
| PASS | test_quantile_mapping::test_figure_error_shrinks_with_sample_size | mean absolute error decreases with n | True | True | Gudmundsson et al. (2012) QUANT: empirical CDFs, linear interpolation, no extrapolation; population map of Exp(1) onto Exp(2) is y* = 2y |
| PASS | test_quantile_mapping::test_figure_transfer_function | largest gap from y* = 2y at n = 20 and n = 200 | 0 | 1.06581e-14 | Matched quantiles of Exp(1) and Exp(2) lie on y* = 2y |
| PASS | test_schaake_shuffle::test_clark_2004_worked_example | rank index B, 1-based as in the paper | [ 3, 7, 2, 8, 1, 4, 9, 5, 10, 6] | [ 3, 7, 2, 8, 1, 4, 9, 5, 10, 6] | Clark et al. (2004), section 3b, equations 5-12 |
| PASS | test_schaake_shuffle::test_clark_2004_worked_example | shuffled forecast X_ss | [10.1, 8.8, 7.5, 10.3, 11.9, 15.3, 8.3, 9.7, 11.2, 12.5] | [10.1, 8.8, 7.5, 10.3, 11.9, 15.3, 8.3, 9.7, 11.2, 12.5] | Clark et al. (2004), section 3b, equations 5-12 |
| PASS | test_schaake_shuffle::test_marginal_distributions_are_unchanged | values at each site, sorted | [[-2.325031, -0.665195, -1.245911], [-1.265421, -0.623274, -0.703735], [-0.732267, -0.544259, -0.3163 ], ..., [ 0.411631, -0.132105, 0.35151 ], [ 1.304 , 0.947081, 0.361595], [ 1.366463, 1.042513, 0.640423]] | [[-2.325031, -0.665195, -1.245911], [-1.265421, -0.623274, -0.703735], [-0.732267, -0.544259, -0.3163 ], ..., [ 0.411631, -0.132105, 0.35151 ], [ 1.304 , 0.947081, 0.361595], [ 1.366463, 1.042513, 0.640423]] | Clark et al. (2004): the shuffle reorders members and does not change values |
| PASS | test_schaake_shuffle::test_ranks_match_the_template | largest difference between shuffled ranks and template ranks | 0 | 0 | Clark et al. (2004), equation 10: member q receives the value of rank r = B[q] |
| PASS | test_schaake_shuffle::test_spearman_correlation_matches_the_template | Spearman correlation matrix | [[ 1. , 0.014286, -0.232143], [ 0.014286, 1. , -0.160714], [-0.232143, -0.160714, 1. ]] | [[ 1. , 0.014286, -0.232143], [ 0.014286, 1. , -0.160714], [-0.232143, -0.160714, 1. ]] | Clark et al. (2004): the shuffle preserves the Spearman rank correlation |
| PASS | test_schaake_shuffle::test_comonotone_template_gives_perfect_positive_correlation | pairwise Spearman correlation | [1., 1., 1.] | [1., 1., 1.] | Identical ranks at every site give correlation +1 |
| PASS | test_schaake_shuffle::test_antimonotone_template_gives_perfect_negative_correlation | Spearman correlation | -1 | -1 | Reversed ranks give correlation -1 |
| PASS | test_schaake_shuffle::test_shuffling_a_field_by_itself_is_identity | schaake_shuffle(X, X) | [[1., 4.], [3., 1.], [2., 3.], [4., 2.]] | [[1., 4.], [3., 1.], [2., 3.], [4., 2.]] | Ranking a field by its own values puts every member back |
| PASS | test_schaake_shuffle::test_ties_break_in_member_order | permutation at a tied site | [0, 1, 2, 3] | [0, 1, 2, 3] | A deterministic jitter increasing with member index breaks ties |
| PASS | test_schaake_shuffle::test_ties_break_in_member_order | same order at the second site | [0, 1, 2, 3] | [0, 1, 2, 3] | The same tie break applies at every site |
| PASS | test_schaake_shuffle::test_ties_break_in_member_order | second call | [[0, 0], [1, 1], [2, 2], [3, 3]] | [[0, 0], [1, 1], [2, 2], [3, 3]] | Tie breaking does not use a random number |
| PASS | test_schaake_shuffle::test_figure_clark_worked_example | largest gap from the printed X_ss | 0 | 0 | Clark et al. (2004), section 3b, equations 5-12 |
| PASS | test_schaake_shuffle::test_figure_clark_worked_example | largest gap from the printed rank index B | 0 | 0 | Clark et al. (2004), section 3b, equations 5-12 |
| PASS | test_schaake_shuffle::test_figure_spatial_dependence | Spearman correlation of reversed sites | -1 | -1 | Clark et al. (2004), Figure 2: ranks follow the template |
| PASS | test_schaake_shuffle::test_figure_spatial_dependence | Spearman matrix equals the template | [[ 1. , 0.983677, -1. ], [ 0.983677, 1. , -0.983677], [-1. , -0.983677, 1. ]] | [[ 1. , 0.983677, -1. ], [ 0.983677, 1. , -0.983677], [-1. , -0.983677, 1. ]] | Clark et al. (2004), section 3b, equations 5-12 |
| PASS | test_snow5::test_precipitation_is_snow_only_below_the_threshold | snowfall | [5., 0., 0., 2.] | [5., 0., 0., 2.] | Slater et al. (2013): snowfall = precip * guc when T < src, else 0 |
| PASS | test_snow5::test_no_melt_at_or_below_the_melt_threshold | potential melt | [0., 0., 8.] | [0., 0., 8.] | Potential melt = melt factor * max(T - t_mlt, 0) |
| PASS | test_snow5::test_melt_cannot_exceed_swe | SWE after melt | [0., 3., 0.] | [0., 3., 0.] | Melt = min(SWE, potential), so storage cannot go negative |
| PASS | test_snow5::test_melt_cannot_exceed_swe | melt | [1., 2., 0.] | [1., 2., 0.] | Melt = min(SWE, potential), so storage cannot go negative |
| PASS | test_snow5::test_cold_day_stores_all_snowfall | snowfall | 5 | 5 | Slater et al. (2013): snowfall = precip * guc when T < src, else 0 |
| PASS | test_snow5::test_cold_day_stores_all_snowfall | melt | 0 | 0 | T < t_mlt, so potential melt is 0 |
| PASS | test_snow5::test_cold_day_stores_all_snowfall | SWE | 25 | 25 | SWE_after = SWE + snowfall - melt = 20 + 5 - 0 |
| PASS | test_snow5::test_timestep_conserves_swe | mass-balance residual | [0., 0.] | [0., 0.] | SWE_after = SWE + snowfall - melt at each HRU |
| PASS | test_snow5::test_timestep_conserves_swe | SWE stays non-negative | True | True | Melt is capped by the snow on the ground |
| KNOWN DEVIATION | test_snow5::test_equal_melt_bounds_give_a_constant_factor | minimum melt factor over the year when f_max = f_min = 2 | 2 | 0 | SNOW-17: Mf = (f_max + f_min)/2 + (f_max - f_min)/2 * sin(phase), which equals f when f_max = f_min |
| PASS | test_ssnowd::test_parameters_have_the_right_mean | exp(lambda + zeta^2/2) | 2.5 | 2.5 | Liston (2004): lognormal SWE, zeta = sqrt(ln(1+CV^2)), lambda = ln(mu) - zeta^2/2 |
| PASS | test_ssnowd::test_parameters_have_the_right_mean | scipy lognormal mean | 2.5 | 2.5 | Liston (2004): lognormal SWE, zeta = sqrt(ln(1+CV^2)), lambda = ln(mu) - zeta^2/2 |
| PASS | test_ssnowd::test_depletion_matches_the_lognormal | fSCA against the lognormal survival function | 0.955766 | 0.955766 | fSCA = P(SWE > dm) for the lognormal distribution |
| PASS | test_ssnowd::test_depletion_matches_the_lognormal | mean SWE against the numerical partial expectation | 0.602913 | 0.602913 | E[(SWE - dm)+] = integral_{dm}^{inf} (x - dm) f(x) dx |
| PASS | test_ssnowd::test_no_melt_leaves_full_cover | fSCA | 1 | 1 | With dm = 0 the whole grid is snow covered |
| PASS | test_ssnowd::test_no_melt_leaves_full_cover | mean SWE | 1.7 | 1.7 | With dm = 0 the mean is the distribution mean |
| PASS | test_ssnowd::test_erfc_complement_identity | largest departure of fSCA(z) + fSCA(-z) from 1 | 0 | 0 | erfc(x) + erfc(-x) = 2, so the two half-erfc values sum to 1 |
| PASS | test_ssnowd::test_tiny_cv_is_a_step_function | fSCA when dm < mu | 1 | 1 | A degenerate distribution at mu is fully covered until dm reaches mu |
| PASS | test_ssnowd::test_tiny_cv_is_a_step_function | mean SWE when dm < mu | 0.5 | 0.5 | Below the spike, mean SWE = mu - dm |
| PASS | test_ssnowd::test_tiny_cv_is_a_step_function | fSCA when dm > mu | 0 | 0 | Once dm passes mu, a degenerate distribution is snow free |
| PASS | test_ssnowd::test_tiny_cv_is_a_step_function | mean SWE when dm > mu | 0 | 0 | Snow-free grid mean is 0 |
| PASS | test_ssnowd::test_melt_depth_round_trip | recovered melt depth | 0.4 | 0.4 | Brent inversion of the partial expectation |
| PASS | test_ssnowd::test_melt_depth_round_trip | residual of the inverted mean | 0 | 0 | The solver residual should be within Brent's tolerance |
| PASS | test_ssnowd::test_figure_step_and_inversion | fSCA on either side of mu when CV = 0.001 | [1., 0.] | [1., 0.] | A degenerate distribution at mu is fully covered until dm reaches mu |
| PASS | test_ssnowd::test_figure_step_and_inversion | largest melt-depth recovery error | 0 | 3.48499e-13 | Brent inversion of the partial expectation |
| PASS | test_ssnowd::test_figure_depletion_curve | largest gap from the lognormal partial expectation | 0 | 2.22045e-16 | Liston (2004): lognormal SWE, zeta = sqrt(ln(1+CV^2)), lambda = ln(mu) - zeta^2/2 |

## Known deviations

- **test_snow5::test_equal_melt_bounds_give_a_constant_factor** — minimum melt factor over the year when f_max = f_min = 2
  - expected: 2
  - computed: 0
  - note: The SNOW-17 melt factor is (f_max+f_min)/2 + (f_max-f_min)/2*sin(phase), so equal bounds give that constant all year. compute_melt_factor multiplies mean * sin * diff, as the legacy port does, and equal bounds give zero melt.

## Figures

- [particle_resample.png](particle_resample.png)
- [particle_weights.png](particle_weights.png)
- [quantile_mapping_convergence.png](quantile_mapping_convergence.png)
- [quantile_mapping_transfer.png](quantile_mapping_transfer.png)
- [schaake_clark_example.png](schaake_clark_example.png)
- [schaake_rank_dependence.png](schaake_rank_dependence.png)
- [ssnowd_step_and_inversion.png](ssnowd_step_and_inversion.png)
- [ssnowd_depletion.png](ssnowd_depletion.png)
