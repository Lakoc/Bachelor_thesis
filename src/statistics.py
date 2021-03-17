import numpy as np
import params
from helpers.decorators import timeit

# 15 ms not overlapping frame -> corresponding to 1000 // 15
NORMALIZATION_COEFFICIENT = 67


def find_non_zero_intervals(vad):
    """Find indexes where speech started and ended for next processing"""
    n = vad.shape[1]

    # find indexes where value changes
    loc_run_start = np.empty(vad.shape, dtype=bool)
    loc_run_start[:, 0] = True
    np.not_equal(vad[:, :-1], vad[:, 1:], out=loc_run_start[:, 1:])
    run_indexes = np.nonzero(loc_run_start)
    run_starts = np.append(run_indexes[1][np.where(run_indexes[0] == 0)], n), np.append(
        run_indexes[1][np.where(run_indexes[0])], n)
    # find values on starting index
    run_values = vad[0, loc_run_start[0]], vad[1, loc_run_start[1]]

    # find lengths of intervals
    run_lengths = np.diff(run_starts[0]), np.diff(run_starts[1])

    # null intervals with silence for next
    non_zero_runs_indexes = np.where(run_values[0]), np.where(run_values[1])
    return np.column_stack((run_starts[0][non_zero_runs_indexes[0]],
                            run_starts[0][non_zero_runs_indexes[0]] + run_lengths[0][
                                non_zero_runs_indexes[0]])), np.column_stack((run_starts[1][non_zero_runs_indexes[1]],
                                                                              run_starts[1][non_zero_runs_indexes[1]] +
                                                                              run_lengths[1][
                                                                                  non_zero_runs_indexes[1]]))


def detect_cross_talks(vad_segments):
    """Detects cross talks by vad coefficients"""
    cross_talks = np.logical_and(vad_segments[0], vad_segments[1])
    cross_talks_bounds = np.where(np.diff(cross_talks))[0].reshape(-1, 2) - [0, 1]
    interruptions = np.all(vad_segments[:, cross_talks_bounds], axis=2)
    interruptions = interruptions.shape[1] - np.count_nonzero(interruptions, axis=1)
    return cross_talks, interruptions


def get_next_segment_in_bounds(segments, start_bound, end_bound, current_index, length):
    """Find next segment in opposite speech after silence"""
    while current_index < length:
        # get items and bounds
        item = segments[current_index]
        item_left_bound = item[0]
        item_right_bound = item[1]
        # if response start in our speech and ends after our speech space is not detected
        if item_left_bound > start_bound and item_right_bound < end_bound:
            return None, current_index
        # item has been found
        if item_left_bound > start_bound:
            return item, current_index
        current_index += 1
    # no item founded
    return None, current_index


def get_last_segment_within_bounds(segments, end_bound, current_index, length):
    """Find last segment, that has silence and is before opposite speech"""
    item = segments[current_index]
    item_left_bound = item[0]
    while item_left_bound < end_bound:
        current_index += 1
        # last item was searched item
        if current_index == length:
            break
        item = segments[current_index]
        item_left_bound = item[0]

    return segments[current_index - 1], current_index - 1


def detect_spaces(segments1, segments2):
    """Detects silence after when speaker ends speech and waits for response"""
    # indexes to detect which segments are already processed
    current_done_index1 = 0
    current_done_index2 = 0
    len_seg1 = len(segments1)
    len_seg2 = len(segments2)
    spaces = []
    while current_done_index1 < len_seg1:
        bound_left = segments1[current_done_index1][0]
        bound_right = segments1[current_done_index1][1]
        # first segment after silence
        opposite_item, current_done_index2 = get_next_segment_in_bounds(segments2, bound_left, bound_right,
                                                                        current_done_index2,
                                                                        len_seg2)
        if opposite_item is None:
            current_done_index1 += 1
            continue
        # check if any later segment of our speech could precede pause, because of some small pauses
        current_item, current_done_index1 = get_last_segment_within_bounds(segments1, opposite_item[0],
                                                                           current_done_index1,
                                                                           len_seg1)
        # check if space was found or cross talk
        response = opposite_item[0] - current_item[1]
        if response > 0:
            spaces.append(opposite_item[0] - current_item[1])
        current_done_index1 += 1
    return np.array(spaces)


def generate_response_and_speech_time_statistics(vad):
    """Process vad to get histograms with lengths of response time and speech time"""
    track_speech_segments = find_non_zero_intervals(vad)
    sentences_lengths = track_speech_segments[0][:, 1] - track_speech_segments[0][:, 0], \
                        track_speech_segments[1][:, 1] - track_speech_segments[1][:, 0]
    responses = detect_spaces(track_speech_segments[0], track_speech_segments[1]), detect_spaces(
        track_speech_segments[1], track_speech_segments[0])
    return responses, sentences_lengths


def count_mean_energy_when_speaking(energy_segments, vad):
    """Count mean energy when speaking"""
    mean_energy0 = np.mean(energy_segments[0][np.where(vad[0])])
    mean_energy1 = np.mean(energy_segments[1][np.where(vad[1])])
    return mean_energy0, mean_energy1


def get_speech_time(vad):
    """Count speech time of each speaker"""
    return np.sum(vad[:], axis=1) / NORMALIZATION_COEFFICIENT


# def count_monolog_hesitations(vad, hesitations_max_size):
#     """Detects hesitations in monolog"""
#     # get indexes of speech
#     hesitations = np.where(np.diff(vad[0]))[0].reshape(-1, 2) + [1, 0], np.where(np.diff(vad[1]))[0].reshape(-1, 2) + [
#         1, 0]
#     # count differences between start and stop of following speech segments
#     hesitations = np.diff(
#         np.append(hesitations[0][0, 1], hesitations[0][1:].reshape(-1, 1))[:-1].reshape(-1, 2)), np.diff(
#         np.append(hesitations[1][0, 1], hesitations[1][1:].reshape(-1, 1))[:-1].reshape(-1, 2))
#     # filter differences greater than our threshold
#     return hesitations[0][hesitations[0] < hesitations_max_size], hesitations[1][hesitations[1] < hesitations_max_size]


def print_min_max_mean_sum(data):
    """Print min, max, mean and sum of selected values"""
    for value, label, normalize in data:
        print(f'Therapist {label} min:{np.min(value[0]) / normalize:.2f}, max:{np.max(value[0]) / normalize:.2f}, '
              f'mean:{np.mean(value[0]) / normalize:.2f}, sum:{np.sum(value[0]) / normalize:.2f}')
        print(f'Client {label} min:{np.min(value[1]) / normalize:.2f}, max:{np.max(value[1]) / normalize:.2f},'
              f' mean:{np.mean(value[1]) / normalize:.2f}, sum:{np.sum(value[1]) / normalize:.2f}\n')


def generate_text_statistics(mean_energies, speech_time, hesitations, responses, sentences_lengths,
                             interruptions):
    """Generate text statistics about speech"""
    is_channel1_louder = mean_energies[0] > mean_energies[1]
    print(f'Therapist speech mean energy: {mean_energies[0]:.2f}')
    print(f'Client speech mean energy: {mean_energies[1]:.2f}')
    print(
        f'Overall {"Therapist" if is_channel1_louder else "Client"} was '
        f'{mean_energies[0] / mean_energies[1] if is_channel1_louder else mean_energies[1] / mean_energies[0]:.2f} '
        f'louder')
    print(f'Therapist speech time: {speech_time[0]:.2f}s')
    print(f'Client speech time: {speech_time[1]:.2f}s\n')
    print(f'Therapist interruptions count: {interruptions[0]}')
    print(f'Client interruptions count: {interruptions[1]}\n')
    print_min_max_mean_sum([(hesitations, 'hesitation time', NORMALIZATION_COEFFICIENT),
                            (responses, 'response time', NORMALIZATION_COEFFICIENT),
                            (sentences_lengths, 'sentences length', NORMALIZATION_COEFFICIENT)])


def join_vad_and_count_hesitations(vad, hesitations_max_size):
    """Join sentences with hesitations to one block"""
    # get indexes of speech
    hesitations_ind = np.where(np.diff(vad[0]))[0].reshape(-1, 2) + [1, 0], np.where(np.diff(vad[1]))[0].reshape(-1,
                                                                                                                 2) + [
                          1, 0]
    # count differences between start and stop of following speech segments
    hesitations = np.diff(
        np.append(hesitations_ind[0][0, 1], hesitations_ind[0][1:].reshape(-1, 1))[:-1].reshape(-1, 2)), np.diff(
        np.append(hesitations_ind[1][0, 1], hesitations_ind[1][1:].reshape(-1, 1))[:-1].reshape(-1, 2))

    # filter differences greater than our threshold
    hesitations_ind_filtered = hesitations[0] < hesitations_max_size, hesitations[1] < hesitations_max_size

    # append false to indexes because of removing start and end index
    hesitations_ind_filtered_with_false = np.append(hesitations_ind_filtered[0], False), np.append(
        hesitations_ind_filtered[1], False)

    # find indexes where values in vad should be changed to 1
    spaces_to_fill = np.stack((hesitations_ind[0][hesitations_ind_filtered_with_false[0]][:, 1],
                               hesitations_ind[0][np.roll(hesitations_ind_filtered_with_false[0], 1)][:, 0]),
                              axis=1), np.stack((hesitations_ind[1][hesitations_ind_filtered_with_false[1]][:, 1],
                                                 hesitations_ind[1][np.roll(hesitations_ind_filtered_with_false[1], 1)][
                                                 :,
                                                 0]), axis=1)
    joined_vad = vad.copy()
    for index, channel in enumerate(spaces_to_fill):
        for space in channel:
            joined_vad[index][space[0]: space[1]] = 1
    # filter values that are bigger than threshold
    hesitations_filtered = hesitations[0][hesitations_ind_filtered[0]], hesitations[1][hesitations_ind_filtered[1]]
    return joined_vad, hesitations_filtered


def diarization_with_timing(diarization):
    """Process diarization output and extract speaker with time bounds"""
    speech_start = np.empty(diarization.shape, dtype=bool)
    speech_start[0] = True
    speech_start[1:] = np.not_equal(diarization[:-1], diarization[1:])
    segment_indexes = np.argwhere(speech_start).reshape(-1)
    speaker = diarization[segment_indexes]

    # Append end of audio
    segment_indexes = np.append(segment_indexes, diarization.shape[0])

    segment_time = segment_indexes * params.window_stride
    return speaker, segment_time
