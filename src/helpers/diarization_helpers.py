import numpy as np
from scipy import ndimage
import params
from helpers.propagation import forward_backward


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
    features_ch1 =  channel1_mfcc
    features_ch2 =  channel2_mfcc

    return active_segments, active_segments_index, features_ch1, features_ch2, energy_difference


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
    diar_min_speech_dur = params.diar_min_speech_dur
    n_tokens = likelihood.shape[1]
    sp = np.ones(n_tokens) / n_tokens
    tr = np.eye(diar_min_speech_dur * n_tokens, k=1)
    ip = np.zeros(diar_min_speech_dur * n_tokens)
    tr[diar_min_speech_dur - 1::diar_min_speech_dur, 0::diar_min_speech_dur] = (1 - params.diar_loop_prob) * sp
    tr[(np.arange(1, n_tokens + 1) * diar_min_speech_dur - 1,) * 2] += params.diar_loop_prob
    ip[::diar_min_speech_dur] = sp

    # hmm backward, forward propagation, and threshold silence component
    likelihood_smoothed, _, _, _ = forward_backward(likelihood.repeat(diar_min_speech_dur, axis=1), tr, ip)

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
