import numpy as np
import params
from scipy import ndimage
from copy import deepcopy
from helpers.decorators import timeit
from gmm import MyGmm
from helpers.diarization_helpers import extract_features_for_diarization, likelihood_propagation_matrix, \
    single_gmm_update
from debug.debug_outputs import plot_6_7k


def return_diarization_index(likelihoods, active_segments):
    """Find higher likelihood, transform, remove nonactive segments"""
    diarization = np.argmax(likelihoods, axis=1)
    diarization += 1
    diarization *= active_segments
    likelihoods_diff = likelihoods[:, 0] - likelihoods[:, 1]
    return diarization, likelihoods_diff


def energy_based_diarization_no_interruptions(energy, vad):
    """Simple energy based diarization, if voice activity was detected, choose higher energy as speaker"""
    """"""
    higher_energy = np.argmax(energy, axis=1)
    vad_summed = np.sum(vad, axis=1)
    diarized = np.where(vad_summed > 0, higher_energy + 1, 0)
    diarized = ndimage.median_filter(diarized, params.mean_filter_diar)
    return diarized


@timeit
def gmm_mfcc_diarization_no_interruptions_2channels_single_iteration(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments, active_segments_index, features, energy_difference = extract_features_for_diarization(mfcc, vad,
                                                                                                           energy)

    # Train UBM GMM - speaker independent
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=32, verbose=1, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    # Test plot
    plot_6_7k(likelihoods_smoothed, active_segments)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    return diarization


@timeit
def gmm_mfcc_diarization_no_interruptions_2channels_single_iteration_delta(mfcc_dd, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments, active_segments_index, features, energy_difference = extract_features_for_diarization(mfcc_dd,
                                                                                                           vad,
                                                                                                           energy)

    # Train UBM GMM - speaker independent
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=32, verbose=1, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    # Test plot
    plot_6_7k(likelihoods_smoothed, active_segments)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    """Second iteration on learned likelihoods if needed"""
    # likelihoods = single_gmm_update(gmm1, gmm2, features, likelihoods_difference, active_segments_index)
    # likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)
    #
    # # Test plot
    # plot_6_7k(likelihoods_smoothed, active_segments)
    #
    # diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)
    # outputs.diarization_to_files(*diarization_with_timing(diarization))
    # calculate_success_rate.main()

    return diarization


@timeit
def gmm_mfcc_diarization_no_interruptions_1channel(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""

    active_segments, active_segments_index, features, energy_difference = extract_features_for_diarization(mfcc, vad,
                                                                                                           energy)

    # Train UBM GMM - speaker independent
    features = features[:, :, 0]
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=32, verbose=1, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    # Test plot
    plot_6_7k(likelihoods_smoothed, active_segments)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    return diarization
