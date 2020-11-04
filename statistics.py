import numpy as np

from ouputs import create_histogram

# 15 ms not overlapping frame we get that by dividing by this number
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


def get_next_segment_in_bounds(segments, start_bound, end_bound, current_index, length):
    """Find next segment in opposite speech after silence"""
    while current_index < length:
        # get items and bounds
        item = segments[current_index]
        item_left_bound = item[0]
        item_right_bound = item[1]
        # if response start in our speech and ends after our speech we don't detect space
        if item_left_bound > start_bound and item_right_bound < end_bound:
            return None, current_index
        # we have found our item
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
        # last item was item we searched
        if current_index == length:
            break
        item = segments[current_index]
        item_left_bound = item[0]

    return segments[current_index - 1], current_index - 1


def detect_spaces(segments1, segments2):
    """Detects silence after when speaker ends speech and waits for response"""
    # indexes to detect which segments we already processed segments
    current_done_index1 = 0
    current_done_index2 = 0
    len_seg1 = len(segments1)
    len_seg2 = len(segments2)
    spaces = []
    while current_done_index1 < len_seg1:
        bound_left = segments1[current_done_index1][0]
        bound_right = segments1[current_done_index1][1]
        # we found first segment after silence
        opposite_item, current_done_index2 = get_next_segment_in_bounds(segments2, bound_left, bound_right,
                                                                        current_done_index2,
                                                                        len_seg2)
        if opposite_item is None:
            current_done_index1 += 1
            continue
        # we check if we don't find any later segment of our speech, because of some small pauses
        current_item, current_done_index1 = get_last_segment_within_bounds(segments1, opposite_item[0],
                                                                           current_done_index1,
                                                                           len_seg1)
        # we check if we found space or we found cross talk
        response = opposite_item[0] - current_item[1]
        if response > 0:
            spaces.append(opposite_item[0] - current_item[1])
        current_done_index1 += 1
    return spaces


def generate_response_and_speech_time_statistics(vad, save_to):
    """Process vad to get histograms with lengths of response time and speech time"""
    track_speech_segments = find_non_zero_intervals(vad)
    sentences_lengths = track_speech_segments[0][:, 1] - track_speech_segments[0][:, 0], \
                        track_speech_segments[1][:, 1] - track_speech_segments[1][:, 0]
    spaces = np.array([detect_spaces(track_speech_segments[0], track_speech_segments[1]),
                       detect_spaces(track_speech_segments[1], track_speech_segments[0])])
    return spaces, sentences_lengths


def count_mean_energy_when_speaking(energy_segments, vad):
    """Count mean energy when speaking"""
    mean_energy0 = np.mean(energy_segments[0][np.where(vad[0])])
    mean_energy1 = np.mean(energy_segments[1][np.where(vad[1])])
    return mean_energy0, mean_energy1


def get_speech_time(vad):
    """Count speech time of each speaker"""
    return np.sum(vad[:], axis=1) / NORMALIZATION_COEFFICIENT


def generate_text_statistics(mean_energies, speech_time):
    """Generate text statistics about speech"""
    is_channel1_louder = mean_energies[0] > mean_energies[1]
    print(f'Channel 1 speech mean energy: {mean_energies[0]:.2f}')
    print(f'Channel 2 speech mean energy: {mean_energies[1]:.2f}')
    print(
        f'Overall channel {1 if is_channel1_louder else 2} was '
        f'{mean_energies[0] / mean_energies[1] if is_channel1_louder else mean_energies[1] / mean_energies[0]:.2f} '
        f'louder.')
    print(f'Channel 1 speech time: {speech_time[0]:.2f}s')
    print(f'Channel 2 speech time: {speech_time[1]:.2f}s')
