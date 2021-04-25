import numpy as np
import params
from scipy import ndimage
from copy import deepcopy
from helpers.decorators import timeit
from helpers.gmm import MyGmm
from helpers.diarization_helpers import extract_features_for_diarization, likelihood_propagation_matrix, \
    single_gmm_update
from debug.debug_outputs import plot_6_7k
from sklearn.cluster import KMeans


def return_diarization_index(likelihoods, active_segments):
    """Find higher likelihood, transform, remove nonactive segments"""
    diarization = np.argmax(likelihoods, axis=1)
    diarization += 1
    diarization *= active_segments
    likelihoods_diff = likelihoods[:, 0] - likelihoods[:, 1]
    return diarization, likelihoods_diff

def energy_based_diarization_no_interruptions(energy, vad):
    """Simple energy based diarization, if voice activity was detected, choose higher energy as speaker"""
    higher_energy = np.argmax(energy, axis=1)
    vad_summed = np.sum(vad, axis=1)
    plot_6_7k(energy, vad_summed)
    diarized = np.where(vad_summed > 0, higher_energy + 1, 0)
    diarized = ndimage.median_filter(diarized, params.mean_filter_diar)
    return diarized


@timeit
def gmm_mfcc_diarization_no_interruptions_1channel(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""

    active_segments, active_segments_index, features, energy_difference = extract_features_for_diarization(mfcc, vad,
                                                                                                           energy)

    # Train UBM GMM - speaker independent
    features = features[:, 0: int(features.shape[1] / 2)]
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, verbose=1, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    # Test plot
    # plot_6_7k(likelihoods_smoothed, active_segments)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    return


@timeit
def gmm_mfcc_diarization_no_interruptions_1channel_k_means(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments = np.logical_or(vad[:, 0], vad[:, 1]).astype("i1")
    active_segments_index = np.argwhere(active_segments).reshape(-1)

    mfcc_active = mfcc[active_segments_index]
    k_means = KMeans(n_clusters=2, verbose=1, max_iter=params.gmm_max_iterations).fit(mfcc_active)

    clusters = k_means.predict(mfcc)
    likelihood = np.sum(mfcc - k_means.cluster_centers_[0], axis=1) ** 2

    # Train UBM GMM - speaker independent
    features = np.append(mfcc, energy[:, np.newaxis], axis=1)
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, verbose=1, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, likelihood, active_segments_index)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    # Test plot
    # plot_6_7k(likelihoods_smoothed, active_segments)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    return diarization


@timeit
def gmm_mfcc_diarization_no_interruptions_2channels_single_iteration(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments, active_segments_index, features1, features2, energy_difference = extract_features_for_diarization(
        mfcc, vad,
        energy)

    # Train UBM GMM - speaker independent
    gmm = MyGmm(n_components=params.gmm_components, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features1[active_segments_index])

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods1 = single_gmm_update(gmm1, gmm2, features1, energy_difference, active_segments_index)
    likelihoods1 = likelihood_propagation_matrix(likelihoods1)

    # Train UBM GMM - speaker independent
    gmm = MyGmm(n_components=params.gmm_components, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features2[active_segments_index])

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods2 = single_gmm_update(gmm1, gmm2, features2, energy_difference, active_segments_index)
    likelihoods2 = likelihood_propagation_matrix(likelihoods2)

    likelihoods_diff1 = likelihoods1[:, 0] - likelihoods1[:, 1]
    likelihoods_diff2 = likelihoods2[:, 0] - likelihoods2[:, 1]

    likelihood_both = np.append(likelihoods_diff1[:, np.newaxis], likelihoods_diff2[:, np.newaxis], axis=1)
    likelihood_both_abs = np.abs(likelihood_both)
    likelihood_max_ind = np.argmax(likelihood_both_abs, axis=1)
    likelihood_max = likelihood_both[np.arange(0, likelihood_max_ind.shape[0]), likelihood_max_ind]
    diarization = (likelihood_max < 0).astype('i1')
    diarization += 1
    diarization *= active_segments
    return diarization



@timeit
def gmm_mfcc_diarization_no_interruptions_2channels_2iterations(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments, active_segments_index, features, energy_difference = extract_features_for_diarization(mfcc, vad,
                                                                                                           energy)

    # Train UBM GMM - speaker independent
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, verbose=1, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    # Test plot
    # plot_6_7k(likelihoods_smoothed, active_segments)
    likelihoods_diff = likelihoods_smoothed[:, 0] - likelihoods_smoothed[:, 1]
    # Second update on learned likelihoods
    likelihoods = single_gmm_update(gmm1, gmm2, features, likelihoods_diff, active_segments_index)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    return diarization
