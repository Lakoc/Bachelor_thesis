import numpy as np
from soundfile import read
import matplotlib.pyplot as plt
from external import energy_vad

# We work with stereo file, where track 1 is therapist and track 2 client
NUMBER_OF_TRACKS = 2


def post_process_vad_energy(vad_coefficients, peak_width):
    """Post process vad segments to remove pitches"""
    for track_index, track in enumerate(vad_coefficients):
        # we find indexes on each value changed
        changes = np.where(track[:-1] != track[1:])[0]
        should_skip = False
        for index, change in enumerate(changes):
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
    return read(path)


def process_pre_emphasis(signal, coefficient):
    """Amplify high frequencies, and balance frequency spectrum"""
    return np.append([signal[0]], signal[1:] - coefficient * signal[:-1], axis=0)


def process_hamming(signal_to_process, emphasis_coefficient, sampling_rate, window_size, overlap_size):
    """Process hamming windowing and pre emphasis over input signal, return segmented signal"""
    amplified_signal = process_pre_emphasis(signal_to_process, emphasis_coefficient)

    # hamming window size 25 ms (1/40 s)
    window_size = int(sampling_rate * window_size)

    # overlap 10 ms (1/100s)
    shift = int(sampling_rate * overlap_size)

    # size of window part which doesn't overlap with next window
    overlapped = window_size - shift

    # length of each track
    length_of_track = len(amplified_signal[:, 0])

    # values will be ignored if we don't add zeros at the end
    overlay = ((length_of_track - overlapped) % window_size)

    # windows per track
    windows_count = ((length_of_track - window_size) // overlapped) + 1

    # create return numpy array
    segmented_tracks = np.zeros((NUMBER_OF_TRACKS, windows_count, window_size))

    for track_index in range(NUMBER_OF_TRACKS):
        # get current working track
        track = signal_to_process[:, track_index]

        # we add zeros at the end here
        if overlay:
            missing_part = window_size - overlay
            zeros = np.zeros(missing_part)
            track = np.append(track, zeros)

        # iterate over track and write values multiplied by hamming window to returned array
        for window in range(windows_count):
            # get starting array
            current_stop_index = (overlapped * window) + window_size
            # change value in return array sliced from input wav and multiply it by hamming window
            segmented_tracks[track_index][window] = track[
                                                    current_stop_index - window_size: current_stop_index] * np.hamming(
                window_size)
    return segmented_tracks


def plot_energy_with_wav(track, energies_per_segment):
    """Plots wav and energy of signal"""
    fig, axs = plt.subplots(2)
    axs[0].set_title('Vaw')
    track_shape = track.shape
    axs[0].plot(np.reshape(track, track_shape[0] * track_shape[1]))
    axs[1].set_title('Energy')
    axs[1].plot(energies_per_segment)
    fig.show()


def calculate_energy_over_segments(segmented_tracks, display_energy):
    """Iterate over all segments and count energy ( E = sum(x^2))"""
    energies_per_segment = np.zeros((NUMBER_OF_TRACKS, len(segmented_tracks[0])))
    for track_index, track in enumerate(segmented_tracks):
        for segment_index, segment in enumerate(track):
            # energies_per_segment[track_index][segment_index] = np.sum(segment ** 2)
            energies_per_segment[track_index][segment_index] = np.log(np.sum(segment ** 2))
        energies_per_segment[track_index] += np.abs(np.min(energies_per_segment[track_index]))
        # to show detected energy plot uncomment next line
        if display_energy:
            plot_energy_with_wav(track, energies_per_segment[track_index])
    return energies_per_segment


def calculate_sign_changes(segmented_tracks):
    """Iterate over all segments and count sign change"""
    sign_changes = np.zeros((NUMBER_OF_TRACKS, len(segmented_tracks[0])))
    for track_index, track in enumerate(segmented_tracks):
        for segment_index, segment in enumerate(track):
            # check for sign by comparing sign bit of two neighbor numbers
            sign_changes[track_index][segment_index] = np.sum(np.diff(np.signbit(segment)))
    return sign_changes
