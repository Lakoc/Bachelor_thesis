import numpy as np
from sys import maxsize
from soundfile import read


def detect_cross_talks(vad_segments):
    cross_talks = np.zeros(len(vad_segments[0]))

    def detect_it(index_to_check):
        if vad_segments[0][index_to_check]:
            return np.sometrue(vad_segments[1][index_to_check - 1: index_to_check + 1])
        elif vad_segments[1][index_to_check]:
            return np.sometrue(vad_segments[0][index_to_check - 1: index_to_check + 1])
        return 0

    for index in range(len(vad_segments[0])):
        cross_talks[index] = detect_it(index)

    return cross_talks


def process_voice_activity_detection(energy_segments, zero_crossings):
    def is_speaking(val):
        (energy, crossings) = val
        return energy > 0.0001 and crossings < 500

    vad = np.zeros(energy_segments.shape)
    for index in range(len(energy_segments)):
        vad[index] = np.array(
            [is_speaking(val) for val in zip(energy_segments[index], zero_crossings[index])])
    return vad


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


class AudioProcessing:
    def __init__(self, path):
        # should use value somewhere between 0.9 - 1
        self.pre_emphasis = 0.97
        # read wav file and each sampling rate
        self.audio, self.sampling_rate = read(path)

        # attribute for segmented_tracks multiplied by hamming window
        self.segmented_tracks = None
        self.tracks = len(self.audio[0])
        self.windows_count = 0

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_audio(self):
        return self.audio

    def process_pre_emphasis(self):
        # amplifying high frequencies, and balance frequency spectrum
        self.audio = np.append([self.audio[0]], self.audio[1:] - self.pre_emphasis * self.audio[:-1], axis=0)

    def process_hamming(self):
        self.process_pre_emphasis()
        # iterate for each track in fetched audio
        for track_index in range(self.tracks):
            # get current working track
            track = self.audio[:, track_index]

            # hamming window size 25 ms (1/40 s)
            window_size = self.sampling_rate // 40

            # overlap 5 ms (1/200s)
            overlap_size = self.sampling_rate // 200

            # size of each window overlapped
            shift = window_size - overlap_size

            # add zeros to end of track to process all values
            overlay = ((len(track)) % shift)
            if overlay:
                missing_part = window_size - overlay
                zeros = np.zeros(missing_part)
                track = np.append(track, zeros)
            # windows per track
            self.windows_count = (len(track) + overlap_size) // shift

            # result array
            if self.segmented_tracks is None:
                self.segmented_tracks = np.zeros((self.tracks, self.windows_count, window_size))

            # iterate over track and write values multiplied by hamming window to returned array
            for window in range(self.windows_count):
                # get starting array
                current_starting_index = window * shift
                # change value in return array sliced from input wav and multiply it by hamming window
                self.segmented_tracks[track_index][window] = track[
                                                             current_starting_index: current_starting_index + window_size] * np.hamming(
                    window_size)

    def calculate_energy_over_segments(self):
        # iterate over all segments and count energy ( E = sum(x^2))
        self.process_hamming()
        energies_per_segment = np.zeros((self.tracks, self.windows_count))
        for track_index, track in enumerate(self.segmented_tracks):
            for segment_index, segment in enumerate(track):
                energies_per_segment[track_index][segment_index] = np.sum(segment ** 2)
        return energies_per_segment

    def calculate_sign_changes(self):
        # iterate over all segments and count sign change
        sign_changes = np.zeros((self.tracks, self.windows_count))
        for track_index, track in enumerate(self.segmented_tracks):
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
