import numpy as np
import params
from scipy import ndimage
from copy import deepcopy
from helpers.gmm import MyGmm
from helpers.diarization_helpers import extract_features_for_diarization, likelihood_propagation_matrix, \
    single_gmm_update
from helpers.propagation import mean_filter
from scipy.special import logsumexp


def return_diarization_index(likelihoods, active_segments):
    """Find higher likelihood, transform, remove nonactive segments"""
    diarization = np.argmax(likelihoods, axis=1)
    diarization += 1
    diarization *= active_segments
    likelihoods_diff = likelihoods[:, 0] - likelihoods[:, 1]
    return diarization, likelihoods_diff


def diar_thresholding_step(diarization, likelihoods_diff):
    """Process diar post-processing - waiting for edge bigger than threshold to change state"""
    diar_post = np.copy(diarization)
    new_val = -10
    thresholds = [np.abs(np.min(likelihoods_diff)) / params.prob_threshold_normalization, 0,
                  np.max(likelihoods_diff) / params.prob_threshold_normalization]
    for index, segment in enumerate(likelihoods_diff):
        # if speech was not present, nothing changes
        if diarization[index] != 0:
            sign = np.sign(segment).astype(int)
            # in case of finding edge
            if np.abs(segment) > thresholds[sign + 1]:
                new_val = sign
                # replace segments before that was not classified yet
                diar_post[diar_post == -10] = sign
            diar_post[index] = new_val
        else:
            # end of activity null speaker
            new_val = -10

    # Have not found any edge on the end so complete by last speaker
    diar_post[diar_post == -10] = diar_post[np.where(np.abs(diar_post) == 1)[0][-1]]
    return diar_post


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


def gmm_mfcc_diarization_1channel(mfcc, vad, energy):
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
    likelihoods = single_gmm_update(gmm1, gmm2, features_ch1, energy_difference, active_segments_index)

    # Normalize likelihoods and process forward backward propagation
    log_evidence = logsumexp(likelihoods, axis=1)
    likelihoods = np.exp(likelihoods - log_evidence[:, np.newaxis])
    likelihoods = likelihood_propagation_matrix(likelihoods)

    # Calculate difference of each model likelihoods
    likelihoods_difference = likelihoods[:, 0] - likelihoods[:, 1]

    # Smooth difference
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    # Get output diarization
    diarization = np.sign(likelihoods_difference) * active_segments

    # Process thresholding
    diarization = diar_thresholding_step(diarization, likelihoods_difference)

    # Change output to correspond with rttm notation
    diarization[diarization == -1] = 2
    diarization = diarization.astype(int)

    return diarization


def gmm_mfcc_diarization_2channels(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments, active_segments_index, _, _, energy_difference = extract_features_for_diarization(
        mfcc, vad,
        energy)

    # Train UBM GMM - speaker independent
    features = np.append(mfcc[:, :, 0], mfcc[:, :, 1], axis=1)
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)

    # Normalize likelihoods and process forward backward propagation
    log_evidence = logsumexp(likelihoods, axis=1)
    likelihoods = np.exp(likelihoods - log_evidence[:, np.newaxis])
    likelihoods = likelihood_propagation_matrix(likelihoods)

    # Calculate difference of each model likelihoods
    likelihoods_difference = likelihoods[:, 0] - likelihoods[:, 1]

    # Smooth difference
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    # Get output diarization
    diarization = np.sign(likelihoods_difference) * active_segments

    # Process thresholding
    diarization = diar_thresholding_step(diarization, likelihoods_difference)

    # Change output to correspond with rttm notation
    diarization[diarization == -1] = 2
    diarization = diarization.astype(int)
    return diarization


def gmm_mfcc_diarization_2channels_2iterations(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments in 2 iterations of adaptation"""
    active_segments, active_segments_index, _, _, energy_difference = extract_features_for_diarization(
        mfcc, vad,
        energy)

    # Train UBM GMM - speaker independent
    features = np.append(mfcc[:, :, 0], mfcc[:, :, 1], axis=1)
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)

    # Normalize likelihoods and process forward backward propagation
    log_evidence = logsumexp(likelihoods, axis=1)
    likelihoods = np.exp(likelihoods - log_evidence[:, np.newaxis])
    likelihoods = likelihood_propagation_matrix(likelihoods)

    # Calculate difference of each model likelihoods
    likelihoods_difference = likelihoods[:, 0] - likelihoods[:, 1]

    # Smooth difference
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    """Second iteration"""
    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, likelihoods_difference, active_segments_index)

    # Normalize likelihoods and process forward backward propagation
    log_evidence = logsumexp(likelihoods, axis=1)
    likelihoods = np.exp(likelihoods - log_evidence[:, np.newaxis])
    likelihoods = likelihood_propagation_matrix(likelihoods)

    # Calculate difference of each model likelihoods
    likelihoods_difference = likelihoods[:, 0] - likelihoods[:, 1]

    # Smooth difference
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    # Get output diarization
    diarization = np.sign(likelihoods_difference) * active_segments

    # Process thresholding
    diarization = diar_thresholding_step(diarization, likelihoods_difference)

    # Change output to correspond with rttm notation
    diarization[diarization == -1] = 2
    diarization = diarization.astype(int)
    return diarization
