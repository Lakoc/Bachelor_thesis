import numpy as np
from decorators import deprecated
from external import energy_vad


@deprecated
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

@deprecated
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


@deprecated
def process_voice_activity_detection_via_gmm(wav_file, remove_pitches_size, win_length):
    """Process voice activity detection with gmm"""
    vad0, _ = energy_vad.compute_vad(wav_file[:, 0], win_length=win_length, win_overlap=win_length // 2)
    vad1, _ = energy_vad.compute_vad(wav_file[:, 1], win_length=win_length, win_overlap=win_length // 2)
    vad = [vad0, vad1]
    return post_process_vad_energy(vad, remove_pitches_size)
