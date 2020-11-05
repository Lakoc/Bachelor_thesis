import numpy as np
from soundfile import read
from external import energy_vad
from outputs import plot_energy_with_wav

# We work with stereo file, where track 1 is therapist and track 2 client
NUMBER_OF_TRACKS = 2


def post_process_vad_energy(vad_coefficients, peak_width):
    """Post process vad segments to remove pitches"""
    for track_index, track in enumerate(vad_coefficients):
        # find indexes on each value changed
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

    # values will be ignored if zeros won't be added at the end
    overlay = ((length_of_track - overlapped) % window_size)

    # windows per track
    windows_count = ((length_of_track - window_size) // overlapped) + 1

    # create return numpy array
    segmented_tracks = np.zeros((NUMBER_OF_TRACKS, windows_count, window_size))

    for track_index in range(NUMBER_OF_TRACKS):
        # get current working track
        track = signal_to_process[:, track_index]

        # add zeros at the end here
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


def calculate_lpc(signal, number_of_coefficients):
    """Return linear predictive coefficients for segment of speech using Levinson-Durbin prediction algorithm"""
    # Compute autocorrelation coefficients R_0 which is energy of signal in segment
    r_vector = [signal @ signal]

    # if energy of signal is 0 return array like [1,0 x m,-1]
    # TODO: Ask if because of hamming window
    if r_vector[0] == 0:
        return np.array(([1] + [0] * (number_of_coefficients - 2) + [-1]))
    else:
        # shift signal to calculate rest of autocorrelation coefficients
        for shift in range(1, number_of_coefficients + 1):
            # multiply vectors of signal[current index : end] @ signal[0:current shift] and append to r vector
            r = signal[shift:] @ signal[:-shift]
            r_vector.append(r)
        r_vector = np.array(r_vector)

        # set initial value of a[0,m] to 1 and calculate next coefficient
        a_vector = np.array([1, -r_vector[1] / r_vector[0]])

        # calculate initial error
        error = r_vector[0] + r_vector[1] * a_vector[1]

        # for each step add new coefficient and update coefficients by alpha
        for i in range(1, number_of_coefficients):
            # set error to the small float to prevent division by zero
            if error == 0:
                error = 10e-20

            # calculate new alpha by multiplying existing coefficient with R[1:i+1] flipped, same as a[m,m]
            alpha = - (a_vector[:i + 1] @ r_vector[i + 1:0:-1]) / error
            # add 0 as new coefficient
            a_vector = np.hstack([a_vector, 0])
            # add flipped coefficients multiplied by alpha to coefficients
            a_vector = a_vector + alpha * a_vector[::-1]
            # update error by alpha
            error *= (1 - alpha ** 2)
        return a_vector
