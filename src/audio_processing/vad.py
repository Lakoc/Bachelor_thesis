import numpy as np
import params
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from helpers.propagation import forward_backward, segments_filter
from copy import deepcopy


def calculate_initial_threshold(energy_segments):
    """ Calculate threshold over first and last 100ms of signal """
    threshold_segments = int(round(params.energy_threshold_interval / params.window_stride))
    start_thresholds = np.mean(energy_segments[0:threshold_segments, :], axis=0)
    end_thresholds = np.mean(energy_segments[energy_segments.shape[0] - threshold_segments:, :], axis=0)
    return np.mean([start_thresholds, end_thresholds])


def energy_vad_threshold(energy_segments, threshold):
    """Process voice activity detection with simple energy thresholding"""
    # Threshold speech segments
    vad = (energy_segments > threshold).astype("i1")
    return vad


def energy_vad_threshold_dynamic(energy_segments):
    """Process voice activity detection with simple energy thresholding"""
    threshold = calculate_initial_threshold(energy_segments)

    # threshold speech segments
    vad = (energy_segments > threshold).astype("i1")

    return vad


def energy_vad_threshold_with_adaptive_threshold(energy_segments):
    """Process vad with adaptive energy thresholding
    http://www.iaeng.org/IJCS/issues_v36/issue_4/IJCS_36_4_16.pdf
    """

    # Initial energies
    e_min = np.array([np.finfo(float).max, np.finfo(float).max])
    e_max = np.array([np.finfo(float).min, np.finfo(float).min])

    # Scaling factors
    j = np.array([1, 1], dtype=float)

    # Create output array
    vad = np.zeros(energy_segments.shape, dtype="i1")

    # Lambda init
    lam = np.array([params.lam, params.lam])
    for index, segment in enumerate(energy_segments):
        # Update value of e_min and j
        e_min = e_min * j
        j = np.where(segment < e_min, 1, j)
        j *= params.growth_coefficient

        e_min = np.minimum(e_min, segment)
        e_max = np.maximum(e_max, segment)

        # Update lambda and calculate threshold
        threshold = (1 - lam) * e_max + lam * e_min
        vad[index] = segment > threshold

    return vad


def energy_gmm_based(energy, propagation=False):
    """Estimate vad using gaussian mixture model"""
    # Initialization of model
    gmm1 = GaussianMixture(n_components=3, max_iter=params.vad_max_iter, tol=params.vad_error_rate, weights_init=np.array([1 / 3, 1 / 3, 1 / 3]),
                           means_init=np.array([np.min(energy[:, 0]), (np.max(energy[:, 0]) + np.min(energy[:, 0])) / 2,
                                                np.max(energy[:, 0])])[:, np.newaxis])
    gmm2 = deepcopy(gmm1)
    gmm2.means_init = np.array(
        [np.min(energy[:, 1]), (np.max(energy[:, 1]) + np.min(energy[:, 1])) / 2, np.max(energy[:, 1])])[:, np.newaxis]

    # Fit and predict posterior probabilities for each channel
    gmm1.fit(energy[:, 0][:, np.newaxis])
    likelihoods1 = gmm1.predict_proba(energy[:, 0][:, np.newaxis])

    gmm2.fit(energy[:, 1][:, np.newaxis])
    likelihoods2 = gmm2.predict_proba(energy[:, 1][:, np.newaxis])

    # Init propagation matrix for sil likelihood smoothing
    vad_min_speech_dur = params.vad_min_speech_dur
    n_tokens = likelihoods1.shape[1]
    sp = np.ones(n_tokens) / n_tokens
    ip = np.zeros(vad_min_speech_dur * n_tokens)
    ip[::vad_min_speech_dur] = sp
    p_change = (1 - params.p_loop) / 2
    vad_transition_matrix = [[params.p_loop, p_change, p_change], [p_change, params.p_loop, p_change],
                             [p_change, p_change, params.p_loop]]
    tr = np.array(vad_transition_matrix)

    # HMM backward, forward propagation, and threshold silence component
    if propagation:
        likelihoods1, _, _, _ = forward_backward(likelihoods1, tr, ip)
        likelihoods2, _, _, _ = forward_backward(likelihoods2, tr, ip)

    vad1 = likelihoods1[:, params.vad_component_to_threshold] > params.vad_min_likelihood
    vad2 = likelihoods2[:, params.vad_component_to_threshold] > params.vad_min_likelihood

    vad = np.append(vad1[:, np.newaxis], vad2[:, np.newaxis], axis=1)
    if params.vad_component_to_threshold == 0:
        vad = 1 - vad

    return vad


def apply_median_filter(vad):
    """Apply median filter to smooth output"""
    vad1 = ndimage.median_filter(vad[:, 0], int(round(params.vad_med_filter / params.window_stride)))
    vad2 = ndimage.median_filter(vad[:, 1], int(round(params.vad_med_filter / params.window_stride)))

    return np.append(vad1[:, np.newaxis], vad2[:, np.newaxis], axis=1)


def apply_silence_speech_removal(vad):
    """Smooth VAD by removing segments nested in group of other class segments"""
    vad1 = segments_filter(segments_filter(vad[:, 0], params.vad_filter_active, 1), params.vad_filter_non_active, 0)
    vad2 = segments_filter(segments_filter(vad[:, 1], params.vad_filter_active, 1), params.vad_filter_non_active, 0)
    return np.append(vad1[:, np.newaxis], vad2[:, np.newaxis], axis=1)
