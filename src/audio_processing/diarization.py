import numpy as np
import params
from scipy import ndimage
from copy import deepcopy
from helpers.gmm import MyGmm
from helpers.diarization_helpers import extract_features_for_diarization, likelihood_propagation_matrix, \
    single_gmm_update
from sklearn.cluster import KMeans
from helpers.propagation import mean_filter
import matplotlib.pyplot as plt


def return_diarization_index(likelihoods, active_segments):
    """Find higher likelihood, transform, remove nonactive segments"""
    diarization = np.argmax(likelihoods, axis=1)
    diarization += 1
    diarization *= active_segments
    likelihoods_diff = likelihoods[:, 0] - likelihoods[:, 1]
    return diarization, likelihoods_diff


def energy_based_diarization(energy, vad):
    """Simple energy based diarization, if voice activity was detected, choose higher energy as speaker"""
    higher_energy = np.argmax(energy, axis=1)
    vad_summed = np.sum(vad, axis=1)
    diarized = np.where(vad_summed > 0, higher_energy + 1, 0)
    diarized_active = diarized[diarized > 0]
    diarized_active = ndimage.median_filter(diarized_active,
                                            int(params.diarization_output_mean_filter / params.window_stride))
    diarized[diarized > 0] = diarized_active
    return diarized


def gmm_mfcc_diarization_1channel_k_means(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments = np.logical_or(vad[:, 0], vad[:, 1]).astype("i1")
    active_segments_index = np.argwhere(active_segments).reshape(-1)

    mfcc_active = mfcc[active_segments_index]
    k_means = KMeans(n_clusters=2, max_iter=params.gmm_max_iterations).fit(mfcc_active)

    clusters = k_means.predict(mfcc)
    likelihood = np.sum(mfcc - k_means.cluster_centers_[0], axis=1) ** 2

    # Train UBM GMM - speaker independent
    features = np.append(mfcc, energy[:, np.newaxis], axis=1)
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, verbose=params.gmm_verbose, covariance_type='diag',
                max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, likelihood, active_segments_index)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    return diarization


def gmm_mfcc_diarization_1channel(mfcc, vad, energy, ref):
    """Train Gaussian mixture models with mfcc features over speech segments"""

    active_segments, active_segments_index, features_ch1, _, energy_difference = extract_features_for_diarization(mfcc,
                                                                                                                  vad,
                                                                                                                  energy)
    energy_difference = mean_filter(energy_difference,
                                    int(params.diarization_init_array_filter / params.window_stride))

    # Train UBM GMM - speaker independent
    features_active = features_ch1[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, verbose=params.gmm_verbose, covariance_type='diag',
                max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features_ch1, energy_difference, active_segments_index, ref)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    energy_mean = np.mean(energy_difference)
    zeros = np.zeros_like(likelihoods_difference)
    ref[ref == 2] = -1
    fig, axs = plt.subplots(nrows=5)
    axs[0].plot(ref, label='reference')
    axs[0].legend()
    axs[1].plot(energy_difference, label='diff')
    axs[1].plot(zeros, label='threshold1')
    axs[2].plot(likelihoods_difference, label='1channel_gmm')
    axs[2].plot(zeros, label='threshold')
    axs[2].legend()

    high_bound = np.percentile(energy_difference, 100 - params.likelihood_percentile)
    low_bound = np.percentile(energy_difference, params.likelihood_percentile)

    # Extract features
    features1 = (energy_difference > high_bound) * 1
    features2 = (energy_difference < low_bound) * -1

    axs[1].plot(features1, label='speaker1')
    axs[1].plot(features2, label='speaker2')
    axs[1].legend()

    hyp = np.sign(energy_difference)
    hyp *= active_segments

    confusion = np.sum((ref != hyp)) / ref.shape[0] * 100
    print(f'\nThresholding 0: {confusion:.2f}%')

    hyp = np.sign(likelihoods_difference)
    hyp *= active_segments

    confusion = np.sum((ref != hyp)) / ref.shape[0] * 100
    print(f'1 channel GMM: {confusion:.2f}%')

    # Train UBM GMM - speaker independent
    mfcc = np.append(mfcc[:, :, 0], mfcc[:, :, 1], axis=1)
    features_active = mfcc[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, verbose=params.gmm_verbose, covariance_type='diag',
                max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, mfcc, energy_difference, active_segments_index, ref)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)

    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    axs[3].plot(likelihoods_difference, label='2channels_gmm')
    axs[3].plot(zeros, label='threshold')
    axs[3].legend()

    hyp = np.sign(likelihoods_difference)
    hyp *= active_segments

    confusion = np.sum((ref != hyp)) / ref.shape[0] * 100
    print(f'2 channel GMM: {confusion:.2f}%')

    likelihoods_diff = likelihoods_smoothed[:, 0] - likelihoods_smoothed[:, 1]

    likelihoods = single_gmm_update(gmm1, gmm2, mfcc, likelihoods_diff, active_segments_index, ref)
    likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)

    diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    axs[4].plot(likelihoods_difference, label='2channels2iterations_gmm')
    axs[4].plot(zeros, label='threshold')
    axs[4].legend()

    hyp = np.sign(likelihoods_difference)
    hyp *= active_segments

    confusion = np.sum((ref != hyp)) / ref.shape[0] * 100
    print(f'2 channel 2 iterations GMM: {confusion:.2f}%')

    plt.show()


    return diarization


def gmm_mfcc_diarization_2channels(mfcc, vad, energy):
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


def gmm_mfcc_diarization_2channels_2iterations(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments, active_segments_index, features, energy_difference = extract_features_for_diarization(mfcc, vad,
                                                                                                           energy)

    # Train UBM GMM - speaker independent
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, verbose=params.gmm_verbose, covariance_type='diag',
                max_iter=params.gmm_max_iterations,
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
