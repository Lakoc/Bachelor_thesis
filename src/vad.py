import numpy as np
import params
from decorators import deprecated
from external import energy_vad
from scipy import ndimage
import matplotlib.pyplot as plt


def calculate_initial_threshold(energy_segments):
    """ Calculate threshold over first and last 100ms of signal """
    threshold_segments = int(round(params.energy_threshold_interval / params.window_stride))
    start_thresholds = np.mean(energy_segments[0:threshold_segments, :], axis=0)
    end_thresholds = np.mean(energy_segments[energy_segments.shape[0] - threshold_segments:, :], axis=0)
    return np.mean([start_thresholds, end_thresholds])


def energy_vad_threshold(energy_segments):
    """Process voice activity detection with simple energy thresholding"""
    threshold = calculate_initial_threshold(energy_segments)

    # threshold speech segments
    vad = (energy_segments > threshold).astype("i1")

    # apply median filter to remove unwanted
    vad = ndimage.median_filter(vad, int(round(params.med_filter / params.window_stride)))
    return vad


def energy_vad_threshold_with_adaptive_threshold(energy_segments):
    """Process vad with adaptive energy thresholding
    http://www.iaeng.org/IJCS/issues_v36/issue_4/IJCS_36_4_16.pdf
    """
    e_min = np.array([np.finfo(float).max, np.finfo(float).max])
    e_max = np.array([np.finfo(float).min, np.finfo(float).min])
    e_max_ref = e_max

    j = np.array([1, 1], dtype=float)
    k = np.array([1, 1], dtype=float)

    vad = np.zeros(energy_segments.shape, dtype="i1")
    # e_min_a = np.zeros(energy_segments.shape)
    # e_max_a = np.zeros(energy_segments.shape)
    # threshold_a = np.zeros(energy_segments.shape)
    # lam_a = np.zeros(energy_segments.shape)

    lam_min = np.array([0.95, 0.95])

    for index, segment in enumerate(energy_segments):
        e_min = e_min * j
        j = np.where(segment < e_min, 1, j)
        j *= 1.0001
        e_min = np.minimum(e_min, segment)

        e_max_ref = e_max_ref * k
        k = np.where(segment > e_max_ref, 1, k)
        k *= 0.999
        e_max = np.maximum(e_max, segment)
        e_max_ref = np.maximum(e_max_ref, segment)

        lam = np.maximum(lam_min, (e_max_ref - e_min) / e_max_ref)
        # lam_a[index] = lam

        threshold = (1 - lam) * e_max + lam * e_min

        vad[index] = segment > threshold
        # threshold_a[index] = threshold
        # e_min_a[index] = e_min
        # e_max_a[index] = e_max

    # TODO: append zeros at start
    # apply median filter to remove unwanted
    vad = ndimage.median_filter(vad, int(round(params.med_filter / params.window_stride)))

    # fig, ax = plt.subplots()
    # ax.set_title('Energy')
    # ax.plot(energy_segments[:, 0])
    # # ax.plot(e_max_a[:, 0])
    # # ax.plot(e_min_a[:, 0])
    # # ax.plot(lam_a[:, 0])
    # # ax.plot(threshold_a[:,0])
    # ax.plot(vad[:,0] * 10000)
    # # ax.plot(lam_a[:,0])
    # # axs[1].set_title('Energy')
    # # axs[1].plot(energy)
    # # axs[2].set_title('Vad')
    # # axs[2].plot(vad)
    # fig.tight_layout()
    # fig.show()
    return vad

# @deprecated
# def process_voice_activity_detection_via_gmm(wav_file, remove_pitches_size, win_length):
#     """Process voice activity detection with gmm"""
#     vad0, _ = energy_vad.compute_vad(wav_file[:, 0], win_length=win_length, win_overlap=win_length // 2)
#     vad1, _ = energy_vad.compute_vad(wav_file[:, 1], win_length=win_length, win_overlap=win_length // 2)
#     vad = [vad0, vad1]
#     return post_process_vad_energy(vad, remove_pitches_size)
