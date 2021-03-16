import numpy as np
import params
from scipy import ndimage
from copy import deepcopy
from helpers.decorators import timeit
from gmm import MyGmm
import outputs
from statistics import diarization_with_timing
from tests import calculate_success_rate
from helpers.helpers import extract_features_for_diarization, likelihood_propagation_matrix, single_gmm_update
from debug.debug_outputs import plot_6_7k


def return_diarization_index(likelihoods, active_segments):
    """Find higher likelihood, transform, remove nonactive segments"""
    diarization = np.argmax(likelihoods, axis=1)
    diarization += 1
    diarization *= active_segments
    return diarization


def energy_based_diarization_no_interruptions(energy, vad):
    """Simple energy based diarization, if voice activity was detected, choose higher energy as speaker"""
    """"""
    higher_energy = np.argmax(energy, axis=1)
    vad_summed = np.sum(vad, axis=1)
    diarized = np.where(vad_summed > 0, higher_energy + 1, 0)
    diarized = ndimage.median_filter(diarized, params.filter_diar)
    return diarized


@timeit
def gmm_mfcc_diarization_no_interruptions_2channels_single_iteration(mfcc_dd, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments, active_segments_index, features, energy_difference = extract_features_for_diarization(mfcc_dd, vad,
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
    plot_6_7k(likelihoods, active_segments)
    plot_6_7k(likelihoods_smoothed, active_segments)

    diarization = return_diarization_index(likelihoods_smoothed, active_segments)

    outputs.diarization_to_files(*diarization_with_timing(diarization))
    calculate_success_rate.main()

    # likelihoods = single_gmm_update(gmm1, gmm2, features, llhs, active_segments_index)
    # likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)
    # diarization, llhs = choose_speaker(likelihoods_smoothed, active_segments, None, None, None)
    # diarization += 1
    # diarization *= active_segments
    # outputs.diarization_to_files(*diarization_with_timing(diarization, llhs))
    # calculate_success_rate.main()

    return diarization


@timeit
def gmm_mfcc_diarization_no_interruptions_1channel(mfcc_dd, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""

    active_segments, active_segments_index, features, energy_difference = extract_features_for_diarization(mfcc_dd, vad,
                                                                                                           energy)
    #
    # # Train gmm
    # gmm1 = MyGmm(n_components=32, verbose=1, covariance_type='diag').fit(channel1_mfcc)
    # gmm2 = deepcopy(gmm1)
    # gmm1.update_centers(channel1_mfcc[energy_difference > 0.5], params.means_shift)
    # gmm2.update_centers(channel1_mfcc[energy_difference < -0.5], params.means_shift)
    #
    # speaker1 = gmm1.score_samples(mfcc_dd[:, :, 0])
    # speaker2 = gmm2.score_samples(mfcc_dd[:, :, 0])
    # speaker1 = speaker1.reshape(-1, 1)
    # speaker2 = speaker2.reshape(-1, 1)
    # likelihoods = np.append(speaker1, speaker2, axis=1)
