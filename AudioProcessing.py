import numpy as np
from sys import maxsize
from soundfile import read
import matplotlib.pyplot as plt

NUMBER_OF_TRACKS = 2


def detect_cross_talks(vad_segments):
    cross_talks = np.zeros(len(vad_segments[0]))
    for index in range(len(vad_segments[0])):
        cross_talks[index] = vad_segments[0][index] and vad_segments[1][index]
    return cross_talks


def post_process_vad_energy(vad_coefficients, peak_width):
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


def process_voice_activity_detection_via_energy(energy_segments, zero_crossings, threshold, remove_pitches_size):
    def is_speaking(val):
        (energy, crossings) = val
        return energy > threshold

    vad = np.zeros(energy_segments.shape, dtype=np.int8)

    for index in range(len(energy_segments)):
        vad[index] = np.array(
            [is_speaking(val) for val in zip(energy_segments[index], zero_crossings[index])])

    return post_process_vad_energy(vad, remove_pitches_size)


def detect_spaces(vad_segments):
    spaces_ret = [{'min': maxsize,
                   'max': 0, 'mean': 0}, {'min': maxsize,
                                          'max': 0, 'mean': 0}]
    for index in range(len(vad_segments)):
        values = vad_segments[index]
        length = len(values)
        spaces = np.where(values[:-1] != values[1:])[0]
        if len(spaces) % 2:
            np.append(spaces, length)
        for x in range(len(spaces) // 2):
            interval = (spaces[(2 * x) + 1] - spaces[(2 * x)]) / 50
            if spaces_ret[index]['max'] < interval < 2:
                spaces_ret[index]['max'] = interval
                spaces_ret[index]['mean'] += interval
            if spaces_ret[index]['min'] > interval > 0.1:
                spaces_ret[index]['min'] = interval
                spaces_ret[index]['mean'] += interval
        spaces_ret[index]['mean'] /= (len(spaces) // 2)
    return spaces_ret


def read_wav_file(path):
    return read(path)


def process_pre_emphasis(signal, coefficient):
    # amplifying high frequencies, and balance frequency spectrum
    return np.append([signal[0]], signal[1:] - coefficient * signal[:-1], axis=0)


def process_hamming(signal_to_process, emphasis_coefficient, sampling_rate):
    amplified_signal = process_pre_emphasis(signal_to_process, emphasis_coefficient)

    # hamming window size 25 ms (1/40 s)
    window_size = sampling_rate // 40

    # overlap 10 ms (1/100s)
    shift = sampling_rate // 100

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
    fig, axs = plt.subplots(2)
    axs[0].set_title('Vaw')
    track_shape = track.shape
    axs[0].plot(np.reshape(track, track_shape[0] * track_shape[1]))
    axs[1].set_title('Energy')
    axs[1].plot(energies_per_segment)
    fig.show()


def calculate_energy_over_segments(segmented_tracks):
    # iterate over all segments and count energy ( E = sum(x^2))
    energies_per_segment = np.zeros((NUMBER_OF_TRACKS, len(segmented_tracks[0])))
    for track_index, track in enumerate(segmented_tracks):
        for segment_index, segment in enumerate(track):
            # energies_per_segment[track_index][segment_index] = np.sum(segment ** 2)
            energies_per_segment[track_index][segment_index] = np.log(np.sum(segment ** 2))
        energies_per_segment[track_index] += np.abs(np.min(energies_per_segment[track_index]))
        # to show detected energy plot uncomment next line
        # plot_energy_with_wav(track, energies_per_segment[track_index])
    return energies_per_segment


def calculate_sign_changes(segmented_tracks):
    # iterate over all segments and count sign change
    sign_changes = np.zeros((NUMBER_OF_TRACKS, len(segmented_tracks[0])))
    for track_index, track in enumerate(segmented_tracks):
        for segment_index, segment in enumerate(track):
            # check for sign by comparing sign bit of two neighbor numbers
            sign_changes[track_index][segment_index] = np.sum(np.diff(np.signbit(segment)))
    return sign_changes

    # def get_normalized_energies(self):
    #     # subtract mean value for better VAD
    #     energies_per_segment = self.calculate_energy_over_segments()
    #     for index, channel in enumerate(energies_per_segment):
    #         channel_mean_energy = np.mean(channel)
    #         energies_per_segment[index] = channel - channel_mean_energy
    #     return energies_per_segment
