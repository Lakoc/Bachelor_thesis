import numpy as np
from scipy import ndimage
import params


def extract_features_for_diarization(mfcc_dd, vad, energy):
    """Extract features from sound for diarization"""
    # Active segments over channels
    active_segments = np.logical_or(vad[:, 0], vad[:, 1]).astype("i1")
    active_segments_index = np.argwhere(active_segments).reshape(-1)

    # Extract speech mfcc features
    channel1_mfcc = mfcc_dd[:, :, 0]
    channel2_mfcc = mfcc_dd[:, :, 1]

    # Energy over channels
    channel1_energy = energy[:, 0].reshape(-1, 1)
    channel2_energy = energy[:, 1].reshape(-1, 1)

    # Difference
    energy_difference = (np.log(channel1_energy) - np.log(channel2_energy)).reshape(-1)

    # Append all features together
    features_ch1 = np.append(channel1_energy, channel1_mfcc, axis=1)
    features_ch2 = np.append(channel2_energy, channel2_mfcc, axis=1)
    features = np.append(features_ch1, features_ch2, axis=1)

    return active_segments, active_segments_index, features, energy_difference


def extract_features_for_diarization_with_delta(mfcc, delta, delta_delta, vad, energy):
    """Extract features from sound for diarization"""
    # Active segments over channels
    active_segments = np.logical_or(vad[:, 0], vad[:, 1]).astype("i1")
    active_segments_index = np.argwhere(active_segments).reshape(-1)

    # Extract speech mfcc features
    channel1_mfcc = mfcc[:, :, 0]
    channel1_delta = delta[:, :, 0]
    channel1_ddelta = delta_delta[:, :, 0]
    channel2_mfcc = mfcc[:, :, 1]
    channel2_delta = delta[:, :, 1]
    channel2_ddelta = delta_delta[:, :, 1]

    # Energy over channels
    channel1_energy = energy[:, 0].reshape(-1, 1)
    channel2_energy = energy[:, 1].reshape(-1, 1)

    # Difference
    energy_difference = (np.log(channel1_energy) - np.log(channel2_energy)).reshape(-1)

    # Append all features together
    features_ch1 = np.append(channel1_energy, channel1_mfcc, channel1_delta, channel1_ddelta, axis=1)
    features_ch2 = np.append(channel2_energy, channel2_mfcc, channel2_delta, channel2_ddelta, axis=1)
    features = np.append(features_ch1, features_ch2, axis=1)

    return active_segments, active_segments_index, features, energy_difference


def convolve2d_with_zeros(data, filter_to_convolve):
    """Append zeros, convolve, extract features"""
    index_start = int(round(filter_to_convolve.shape[0] / 2))
    padded = np.pad(data, ((filter_to_convolve.shape[0], filter_to_convolve.shape[0]), (0, 0)), 'constant',
                    constant_values=0)
    convolved = ndimage.convolve(padded[:, 0], filter_to_convolve).reshape(-1, 1), ndimage.convolve(padded[:, 1],
                                                                                                    filter_to_convolve).reshape(
        -1, 1)
    convolved = np.append(convolved[0], convolved[1], axis=1)
    return convolved, index_start


def mean_filter2d(data, filter_size):
    """Simple mean filter of N neighbours"""
    mean_filter = np.ones(filter_size)
    filter_sum = np.sum(mean_filter)
    convolved, index_start = convolve2d_with_zeros(data, mean_filter)
    convolved = convolved[index_start: data.shape[0] + index_start]
    return convolved / filter_sum


def likelihood_propagation_matrix(likelihood):
    """Front and back propagation"""
    powers = np.arange(0, min(likelihood.shape[0], params.propagation_size))
    power_filter = params.prop_coef ** powers
    convolved, index_start = convolve2d_with_zeros(likelihood, power_filter)
    front_propagated = convolved[index_start: index_start + likelihood.shape[0], :]

    flipped = np.flip(likelihood, axis=0)
    convolved, index_start = convolve2d_with_zeros(flipped, power_filter)
    back_propagated = np.flip(convolved[index_start: index_start + likelihood.shape[0], :], axis=0)

    """Sum propagation and add mean filter for even smoother effect"""
    likelihood_smoothed = front_propagated + back_propagated
    likelihood_smoothed = mean_filter2d(likelihood_smoothed, params.mean_filter_diar)

    return likelihood_smoothed


def single_gmm_update(gmm1, gmm2, features, data_full, active_segments_index):
    """Update means of gmm by X% in direction of new means"""

    features_active = features[active_segments_index]
    data_to_train = data_full[active_segments_index]

    # Find data on which should model update
    high_bound = np.percentile(data_to_train, 100 - params.likelihood_percentile)
    low_bound = np.percentile(data_to_train, params.likelihood_percentile)

    # Extract features
    features1 = features_active[data_to_train > high_bound]
    features2 = features_active[data_to_train < low_bound]

    # Update models
    gmm1.update_centers(features1, params.model_means_shift)
    gmm2.update_centers(features2, params.model_means_shift)

    # Score samples
    speaker1 = gmm1.score_samples(features)
    speaker2 = gmm2.score_samples(features)
    likelihoods = np.append(speaker1.reshape(-1, 1), speaker2.reshape(-1, 1), axis=1)

    return likelihoods


def smoother_diarization(diarized):
    """Remove small segments from diarization"""
    diarized = ndimage.median_filter(diarized, params.median_filter_diar)
    return diarized