import matplotlib.pyplot as plt
import numpy
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
    e_min_arr = np.empty(energy_segments.shape)
    e_min = np.array([np.finfo(float).max, np.finfo(float).max])
    e_max = np.array([np.finfo(float).min, np.finfo(float).min])
    e_max_arr = np.empty(energy_segments.shape)
    e_max_ref = e_max
    e_max_ref_arr = np.empty(energy_segments.shape)
    threshold_arr = np.empty(energy_segments.shape)

    # Scaling factors
    j = np.array([1, 1], dtype=float)
    k = np.array([1, 1], dtype=float)

    vad = np.zeros(energy_segments.shape, dtype="i1")

    # Lambda min in case of too fast scaling
    lam_min = np.array([0.95, 0.95])
    for index, segment in enumerate(energy_segments):
        # Update value of e_min and e_max
        e_min = e_min * j
        j = np.where(segment < e_min, 1, j)
        j *= params.growth_coefficient
        e_min = np.minimum(e_min, segment)
        e_min_arr[index] = e_min
        e_max_ref = e_max_ref * k
        k = np.where(segment > e_max_ref, 1, k)
        k *= params.descent_coefficient
        e_max = np.maximum(e_max, segment)
        e_max_arr[index] = e_max
        e_max_ref = np.maximum(e_max_ref, segment)
        e_max_ref_arr[index] = e_max_ref

        # Update lambda and calculate threshold
        lam = np.maximum(lam_min, (e_max_ref - e_min) / e_max_ref)
        threshold = (1 - lam) * e_max + lam * e_min
        threshold_arr[index] = threshold
        # 1 - speech, 0 - non speech
        vad[index] = segment > threshold

        # apply median filter to remove unwanted

    fig, axs = plt.subplots(nrows=2, figsize=(7, 4), sharex=True)
    axs[0].plot( e_min_arr[:, 0] / np.max(e_max_arr[:, 0]), label='Minimální energie')
    # axs[0].plot( e_max_arr[:, 0] / np.max(e_max_arr[:, 0]), label='Maximální energie')
    axs[0].plot( e_max_ref_arr[:, 0] / np.max(e_max_ref_arr[:, 0]), label='Maximální energie ref')
    axs[0].legend()


    axs[1].plot( energy_segments[:, 0] / np.max(energy_segments[:, 0]), label='Energie')
    axs[1].plot( threshold_arr[:, 0] / np.max(energy_segments[:, 0]), label='Práh')
    # axs[1].plot( vad[:, 0], label='Řečová aktivita', color='black', linewidth=1.0)
    axs[1].legend()
    fig.tight_layout()
    # plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/vad_adaptive.pdf')
    plt.show()
    exit(0)
    return vad


def energy_gmm_based(energy, propagation=False):
    """Estimate vad using gaussian mixture model"""
    # Initialization of model
    gmm1 = GaussianMixture(n_components=3, max_iter=300, tol=1e-5, weights_init=np.array([1 / 3, 1 / 3, 1 / 3]),
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
    tr = np.array(params.vad_transition_matrix)

    # HMM backward, forward propagation, and threshold silence component
    if propagation:
        likelihoods1, _, _, _ = forward_backward(likelihoods1, tr, ip)
        likelihoods2, _, _, _ = forward_backward(likelihoods2, tr, ip)

    vad1 = likelihoods1[:, params.vad_component_to_threshold] > params.vad_min_likelihood
    vad2 = likelihoods2[:, params.vad_component_to_threshold] > params.vad_min_likelihood

    vad = np.append(vad1[:, np.newaxis], vad2[:, np.newaxis], axis=0)

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
    # # axs[0].legend()
    # axs[0].set_ylabel('Pravděpodobnost')
    #
    # axs[1].plot(likelihood_propagated2[:, 2], label='Pravděpodobnost ticha')
    # axs[1].plot( threshold, label='Práh')
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['top'].set_visible(False)
    # axs[1].set_ylabel('Velikost')
    # # axs[1].legend()
    #
    # axs[2].plot(time, vad2, label='Řečová aktivita', color='black', linewidth=3)
    # axs[2].plot( signal[:, 1] / height, label='Normalizovaný signál')
    # axs[2].spines['right'].set_visible(False)
    # axs[2].spines['top'].set_visible(False)
    # axs[2].set_ylabel('Velikost')
    # # axs[2].legend()
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
    # # axs[0].legend()
    # axs[0].set_ylabel('Pravděpodobnost')
    #
    # axs[1].plot(likelihood_propagated1[:, 2], label='Pravděpodobnost ticha')
    # axs[1].plot(threshold, label='Práh')
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['top'].set_visible(False)
    # axs[1].set_ylabel('Velikost')
    # # axs[1].legend()
    #
    # axs[2].plot(time, vad1, label='Řečová aktivita', color='black', linewidth=3)
    # axs[2].plot(signal[:, 0] / height, label='Normalizovaný signál')
    # axs[2].spines['right'].set_visible(False)
    # axs[2].spines['top'].set_visible(False)
    # axs[2].set_ylabel('Velikost')
    # # axs[2].legend()
    # axs[2].set_xlabel('Čas [s]')
    #
    # plt.show()

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
