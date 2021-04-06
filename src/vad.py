import numpy as np
import params
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from scipy.special import logsumexp
from helpers.decorators import timeit
from helpers.propagation import forward_backward


def calculate_initial_threshold(energy_segments):
    """ Calculate threshold over first and last 100ms of signal """
    threshold_segments = int(round(params.energy_threshold_interval / params.window_stride))
    start_thresholds = np.mean(energy_segments[0:threshold_segments, :], axis=0)
    end_thresholds = np.mean(energy_segments[energy_segments.shape[0] - threshold_segments:, :], axis=0)
    return np.mean([start_thresholds, end_thresholds])


@timeit
def energy_vad_threshold(energy_segments):
    """Process voice activity detection with simple energy thresholding"""
    threshold = calculate_initial_threshold(energy_segments)

    # threshold speech segments
    vad = (energy_segments > threshold).astype("i1")

    # apply median filter to remove unwanted
    vad = ndimage.median_filter(vad, int(round(params.med_filter / params.window_stride)))
    return vad


@timeit
def energy_vad_threshold_with_adaptive_threshold(energy_segments):
    """Process vad with adaptive energy thresholding
    http://www.iaeng.org/IJCS/issues_v36/issue_4/IJCS_36_4_16.pdf
    """

    # Initial energies
    e_min = np.array([np.finfo(float).max, np.finfo(float).max])
    e_max = np.array([np.finfo(float).min, np.finfo(float).min])
    e_max_ref = e_max

    # Scaling factors
    j = np.array([1, 1], dtype=float)
    k = np.array([1, 1], dtype=float)

    vad = np.zeros(energy_segments.shape, dtype="i1")

    # Lambda min in case of too fast scaling
    lam_min = np.array([0.95, 0.95])
    for index, segment in enumerate(energy_segments):
        # Actualize e_min and e_max
        e_min = e_min * j
        j = np.where(segment < e_min, 1, j)
        j *= 1.0001
        e_min = np.minimum(e_min, segment)

        e_max_ref = e_max_ref * k
        k = np.where(segment > e_max_ref, 1, k)
        k *= 0.999
        e_max = np.maximum(e_max, segment)
        e_max_ref = np.maximum(e_max_ref, segment)

        # Actualize lambda and calculate threshold
        lam = np.maximum(lam_min, (e_max_ref - e_min) / e_max_ref)
        # lam = np.maximum(lam_min, (e_max - e_min) / e_max)
        threshold = (1 - lam) * e_max + lam * e_min

        # 1 - speech, 0 - non speech
        vad[index] = segment > threshold

    # apply median filter to remove unwanted
    vad = ndimage.median_filter(vad, int(round(params.med_filter / params.window_stride)))

    return vad


@timeit
def energy_gmm_based_vad(normalized_energy):
    """Estimate vad using gaussian mixture model"""
    # Initialization of model
    means = np.array(
        [np.min(normalized_energy), (np.max(normalized_energy) + np.min(normalized_energy)) / 2,
         np.max(normalized_energy)])[
            :, np.newaxis]
    weights = np.array([1 / 3, 1 / 3, 1 / 3])

    gmm = GaussianMixture(n_components=3, weights_init=weights, means_init=means)

    # Reshape input
    normalized_energy = normalized_energy.reshape(-1, 1)

    # Fit and predict posterior probabilities
    gmm.fit(normalized_energy)

    # Calculate likelihoods and transform to (0,1) interval
    likelihoods = gmm.predict_proba(normalized_energy).reshape(-1, 2, 3)
    shift = logsumexp(likelihoods, axis=2)
    likelihoods = (likelihoods.transpose(2, 0, 1) - shift).transpose(1, 2, 0)
    likelihoods = np.exp(likelihoods)

    # If silence was detected with likelihood smaller than threshold, we mark segment as active
    vad = likelihoods[:, :, 0] < params.min_silence_likelihood

    # Apply median filter
    vad = ndimage.median_filter(vad, int(round(params.med_filter / params.window_stride)))

    return vad


def energy_gmm_based_vad_propagation(normalized_energy):
    """Estimate vad using gaussian mixture model"""
    # Initialization of model
    means = np.array(
        [np.min(normalized_energy), (np.max(normalized_energy) + np.min(normalized_energy)) / 2,
         np.max(normalized_energy)])[
            :, np.newaxis]
    weights = np.array([1 / 3, 1 / 3, 1 / 3])

    gmm = GaussianMixture(n_components=3, weights_init=weights, covariance_type='spherical', means_init=means)

    # Fit and predict posterior probabilities for each channel
    gmm.fit(normalized_energy[:, 0][:, np.newaxis])
    likelihoods1 = gmm.predict_proba(normalized_energy[:, 0][:, np.newaxis])
    shift = logsumexp(likelihoods1, axis=1)
    likelihoods1 = np.exp(likelihoods1 - shift[:, np.newaxis])

    gmm.fit(normalized_energy[:, 1][:, np.newaxis])
    likelihoods2 = gmm.predict_proba(normalized_energy[:, 1][:, np.newaxis])
    shift = logsumexp(likelihoods2, axis=1)
    likelihoods2 = np.exp(likelihoods2 - shift[:, np.newaxis])

    # Init propagation matrix
    vad_min_speech_dur = params.vad_min_speech_dur
    n_tokens = likelihoods1.shape[1]
    sp = np.ones(n_tokens) / n_tokens
    tr = np.eye(vad_min_speech_dur * n_tokens, k=1)
    ip = np.zeros(vad_min_speech_dur * n_tokens)
    tr[vad_min_speech_dur - 1::vad_min_speech_dur, 0::vad_min_speech_dur] = (1 - params.vad_loop_probability) * sp
    tr[(np.arange(1, n_tokens + 1) * vad_min_speech_dur - 1,) * 2] += params.vad_loop_probability
    ip[::vad_min_speech_dur] = sp

    # HMM backward, forward propagation, and threshold silence component
    likelihood_propagated, _, _, _ = forward_backward(likelihoods1.repeat(vad_min_speech_dur, axis=1), tr, ip)
    vad1 = likelihood_propagated[:, 0] < params.min_silence_likelihood
    likelihood_propagated, _, _, _ = forward_backward(likelihoods2.repeat(vad_min_speech_dur, axis=1), tr, ip)
    vad2 = likelihood_propagated[:, 0] < params.min_silence_likelihood
    vad = np.append(vad1[:, np.newaxis], vad2[:, np.newaxis], axis=1)

    return vad
