import matplotlib.pyplot as plt
import numpy as np
import params
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from scipy.special import logsumexp
from helpers.decorators import timeit
from helpers.propagation import forward_backward, segments_filter
from copy import deepcopy


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


def energy_gmm_based_vad_propagation(energy):
    """Estimate vad using gaussian mixture model"""
    # Initialization of model
    weights = np.array([1/3, 1/3, 1/3])

    gmm1 = GaussianMixture(n_components=3, max_iter=300, tol=1e-5, weights_init=weights, covariance_type='spherical',
                          means_init=np.array([np.min(energy[:,0]), (np.max(energy[:,0]) + np.min(energy[:,0])) / 2, np.max(energy[:,0])])[:, np.newaxis])
    gmm2 = deepcopy(gmm1)
    gmm2.means_init = np.array([np.min(energy[:,1]), (np.max(energy[:,1]) + np.min(energy[:,1])) / 2, np.max(energy[:,1])])[:, np.newaxis]


    # Fit and predict posterior probabilities for each channel
    while True:
        gmm1.fit(energy[:, 0][:, np.newaxis])

        if np.argmin(gmm1.means_) == 0:
            break
        else:
            print(gmm1.means_)
            # gmm1 = deepcopy(gmm)

    likelihoods1 = gmm1.predict_proba(energy[:, 0][:, np.newaxis])

    while True:
        gmm2.fit(energy[:, 1][:, np.newaxis])

        if np.argmin(gmm2.means_) == 0:
            break
        else:
            print(gmm2.means_)
            # gmm2 = deepcopy(gmm)

    likelihoods2 = gmm2.predict_proba(energy[:, 1][:, np.newaxis])

    # Init propagation matrix for sil likelihood smoothing
    vad_min_speech_dur = params.vad_min_speech_dur
    n_tokens = likelihoods1.shape[1]
    sp = np.ones(n_tokens) / n_tokens
    ip = np.zeros(vad_min_speech_dur * n_tokens)
    ip[::vad_min_speech_dur] = sp
    tr = np.array(params.transition_matrix)

    # HMM backward, forward propagation, and threshold silence component
    likelihood_propagated1, _, _, _ = forward_backward(likelihoods1, tr, ip)
    vad1 = likelihood_propagated1[:, 0] < params.min_silence_likelihood

    likelihood_propagated2, _, _, _ = forward_backward(likelihoods2, tr, ip)
    vad2 = likelihood_propagated2[:, 0] < params.min_silence_likelihood
    vad2 = ndimage.median_filter(vad2, int(round(params.med_filter / params.window_stride)))

    # Smooth activity segments and create arr containing both speakers
    vad1 = segments_filter(segments_filter(vad1, params.filter_active, 1), params.filter_non_active, 0)
    vad2 = segments_filter(segments_filter(vad2, params.filter_active, 1), params.filter_non_active, 0)
    vad = np.append(vad1[:, np.newaxis], vad2[:, np.newaxis], axis=1)

    #
    # fig, axs = plt.subplots(nrows=3,  figsize=(10, 6))
    # height = np.max(signal[:, 0])
    # threshold = np.full(likelihood_propagated1.shape[0], params.min_silence_likelihood)
    # time = np.linspace(0, signal.shape[0], vad.shape[0])
    #
    # axs[0].plot( likelihoods2[:, 0], label='Ticho')
    # axs[0].plot(likelihoods2[:, 1], label='Šum / tichá řeč')
    # axs[0].plot(likelihoods2[:, 2], label='Řeč')
    # axs[0].spines['right'].set_visible(False)
    # axs[0].spines['top'].set_visible(False)
    # axs[0].legend()
    # axs[0].set_ylabel('Pravděpodobnost')
    #
    # axs[1].plot(likelihood_propagated2[:, 2], label='Pravděpodobnost ticha')
    # axs[1].plot( threshold, label='Práh')
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['top'].set_visible(False)
    # axs[1].set_ylabel('Velikost')
    # axs[1].legend()
    #
    # axs[2].plot(time, vad2, label='Řečová aktivita', color='black', linewidth=3)
    # axs[2].plot( signal[:, 1] / height, label='Normalizovaný signál')
    # axs[2].spines['right'].set_visible(False)
    # axs[2].spines['top'].set_visible(False)
    # axs[2].set_ylabel('Velikost')
    # axs[2].legend()
    # axs[2].set_xlabel('Čas [s]')
    #
    # plt.show()
    #
    # fig, axs = plt.subplots(nrows=3, figsize=(10, 6))
    # height = np.max(signal[:, 0])
    # threshold = np.full(likelihood_propagated1.shape[0], params.min_silence_likelihood)
    # time = np.linspace(0, signal.shape[0], vad.shape[0])
    #
    # axs[0].plot(likelihoods1[:, 0], label='Ticho')
    # axs[0].plot(likelihoods1[:, 1], label='Šum / tichá řeč')
    # axs[0].plot(likelihoods1[:, 2], label='Řeč')
    # axs[0].spines['right'].set_visible(False)
    # axs[0].spines['top'].set_visible(False)
    # axs[0].legend()
    # axs[0].set_ylabel('Pravděpodobnost')
    #
    # axs[1].plot(likelihood_propagated1[:, 2], label='Pravděpodobnost ticha')
    # axs[1].plot(threshold, label='Práh')
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['top'].set_visible(False)
    # axs[1].set_ylabel('Velikost')
    # axs[1].legend()
    #
    # axs[2].plot(time, vad1, label='Řečová aktivita', color='black', linewidth=3)
    # axs[2].plot(signal[:, 0] / height, label='Normalizovaný signál')
    # axs[2].spines['right'].set_visible(False)
    # axs[2].spines['top'].set_visible(False)
    # axs[2].set_ylabel('Velikost')
    # axs[2].legend()
    # axs[2].set_xlabel('Čas [s]')
    #
    # plt.show()

    return vad
