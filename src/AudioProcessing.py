import numpy as np
from scipy.io import wavfile
from external import energy_vad
from outputs import plot_energy_with_wav

from decorators import deprecated

# We work with stereo file, where track 1 is therapist and track 2 client
NUMBER_OF_TRACKS = 2


def post_process_vad_energy(vad_coefficients, peak_width):
    """Post process vad segments to remove pitches"""
    for track_index, track in enumerate(vad_coefficients):
        # find indexes on each value changed
        changes = np.where(track[:-1] != track[1:])[0]
        should_skip = False
        for index, change in enumerate(changes):
            if index == len(changes) - 1 and index % 2 == 0 and len(track) - change < peak_width:
                track[changes[index - 1]: len(track)] = 0
                continue
            if index == 0 or should_skip:
                should_skip = False
                continue
            if change - changes[index - 1] < peak_width:
                should_skip = True
                track[changes[index - 1]: change + 1] = 1 - track[change]

        vad_coefficients[track_index] = track
    return vad_coefficients


# def post_process_vad_energy_v2(vad_coefficients, peak_width):
#     for track_index, track in enumerate(vad_coefficients):
#         track = track[np.newaxis, :]
#         N = peak_width
#         padded = np.pad(track, ((0, 0), (N, N)), 'constant', constant_values=0)
#         conv = np.convolve(padded[0], np.ones(N), 'valid')
#
#         left = conv[:-N - 1]
#         right = conv[N + 1:]
#
#         neighbours = np.add(left, right)
#
#         mask = (track[0] == 1) & (neighbours < N * 0.8)  # flip 1 to 0
#         mask |= (track[0] == 0) & (neighbours > N * 0.8)  # flip 0 to 1
#         vad_coefficients[track_index] = np.logical_xor(track, mask).astype(np.int)
#     return vad_coefficients


def process_voice_activity_detection_via_energy(energy_segments, zero_crossings, threshold, remove_pitches_size):
    """Process voice activity detection with simple energy thresholding"""

    def is_speaking(val):
        (energy, crossings) = val
        return energy > threshold

    vad = np.zeros(energy_segments.shape, dtype=np.int8)

    for index in range(len(energy_segments)):
        vad[index] = np.array(
            [is_speaking(val) for val in zip(energy_segments[index], zero_crossings[index])])

    return post_process_vad_energy(vad, remove_pitches_size)


def process_voice_activity_detection_via_gmm(wav_file, remove_pitches_size, win_length):
    """Process voice activity detection with gmm"""
    vad0, _ = energy_vad.compute_vad(wav_file[:, 0], win_length=win_length, win_overlap=win_length // 2)
    vad1, _ = energy_vad.compute_vad(wav_file[:, 1], win_length=win_length, win_overlap=win_length // 2)
    vad = [vad0, vad1]
    return post_process_vad_energy(vad, remove_pitches_size)


def read_wav_file(path):
    """Read wav file and returns numpy array and sampling rate"""
    sampling_rate, wav_file = wavfile.read(path)
    return wav_file.astype(int), sampling_rate


def process_pre_emphasis(signal, coefficient):
    """Amplify high frequencies, and balance frequency spectrum"""
    return np.append([signal[0]], signal[1:] - coefficient * signal[:-1], axis=0)


def process_hamming(signal, emphasis_coefficient, sampling_rate, window_size, window_overlap):
    """Process pre emphasis over input signal, segment it and apply window function"""

    if window_size < window_overlap:
        exit("Window size smaller than overlap!")

    amplified_signal = process_pre_emphasis(signal, emphasis_coefficient)

    window_size = int(round(sampling_rate * window_size))
    window_overlap = int(round(sampling_rate * window_overlap))
    window_stride = window_size - window_overlap

    signal_len = len(amplified_signal)

    # size_of_signal = ((number_of_frames - 1) * (window_size - window_overlap)) + window_size
    number_of_frames = int(np.ceil((signal_len - window_size) / window_stride + 1))
    signal_len_new = (number_of_frames - 1) * window_stride + window_size

    def pad_segment_hamming(sig):
        amplified_signal_padded = np.append(sig, np.zeros((
                signal_len_new - signal_len)))

        # np.arange ends on stop so add 1 more step
        indices = np.tile(np.arange(0, window_size), (number_of_frames, 1)) + np.tile(
            np.arange(0, signal_len_new - window_size + window_stride, window_stride),
            (window_size, 1)).T

        windows = amplified_signal_padded[indices.astype(np.int32, copy=False)]
        windows *= np.hamming(window_size)
        return windows

    # apply segmentation over each axis
    return np.apply_along_axis(pad_segment_hamming, 0, amplified_signal)


def calculate_energy_over_segments(signal):
    """Iterate over all segments and count energy ( E = sum(x^2))"""
    signal_power2 = signal ** 2
    return np.sum(signal_power2, axis=1)


# def calculate_energy_over_segments(segmented_tracks, display_energy):
#     """Iterate over all segments and count energy ( E = sum(x^2))"""
#     energies_per_segment = np.zeros((NUMBER_OF_TRACKS, len(segmented_tracks[0])))
#     for track_index, track in enumerate(segmented_tracks):
#         track = track * (2 ** 15 - 1)  # short size
#         for segment_index, segment in enumerate(track):
#             energies_per_segment[track_index][segment_index] = 10 * np.log10(max(np.sum(segment ** 2), 1000))
#         # to show detected energy plot uncomment next line
#         if display_energy:
#             plot_energy_with_wav(track, energies_per_segment[track_index])
#     return energies_per_segment


@deprecated
def calculate_sign_changes(segmented_tracks):
    """Iterate over all segments and count sign change"""
    sign_changes = np.zeros((NUMBER_OF_TRACKS, len(segmented_tracks[0])))
    for track_index, track in enumerate(segmented_tracks):
        for segment_index, segment in enumerate(track):
            # check for sign by comparing sign bit of two neighbor numbers
            sign_changes[track_index][segment_index] = np.sum(np.diff(np.signbit(segment)))
    return sign_changes
