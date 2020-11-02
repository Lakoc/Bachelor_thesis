import matplotlib.pyplot as plt
import numpy as np

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


def remove_zeros(segments):
    """Remove zeros from intervals"""
    return segments[np.where(segments.any(axis=1))]


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


def create_histogram(data, titles):
    """Plots histogram"""
    graphs = len(titles)
    fig, axes = plt.subplots(nrows=graphs, ncols=1, figsize=(8, 6), sharex=True, sharey=True)
    for index in range(graphs):
        labels, counts = np.unique(data[index] / NORMALIZATION_COEFFICIENT, return_counts=True)
        counts_probability = counts / counts.shape[0]
        axes[index].bar(labels, counts_probability, width=0.1)
        axes[index].grid(axis='y', color='black', linewidth=.5, alpha=.5)
        axes[index].spines["top"].set_visible(False)
        axes[index].spines["right"].set_visible(False)
        axes[index].tick_params(left=False, bottom=False)
        axes[index].set_ylabel("Probability")
        axes[index].set_xlabel("Length (s)")
        axes[index].set_title(titles[index], fontsize=20)
    fig.show()


def detect_response_speech_times(vad):
    """Process vad to get histograms with lengths of response time and speech time"""
    track_speech_segments = find_non_zero_intervals(vad)
    sentences_lengths = track_speech_segments[0][:, 1] - track_speech_segments[0][:, 0], \
                        track_speech_segments[1][:, 1] - track_speech_segments[1][:, 0]
    spaces = np.array([detect_spaces(track_speech_segments[0], track_speech_segments[1]),
                       detect_spaces(track_speech_segments[1], track_speech_segments[0])])
    create_histogram(spaces, ['Client response time', 'Therapist response time'])
    create_histogram(sentences_lengths, ['Therapist sentences length', 'Client sentences length'])


def plot_wav_with_detection(sampling_rate, wav, vad_segments, cross_talks):
    """Plot signals with vad and cross talks"""
    fig, axs = plt.subplots(5, sharex=True)
    wav_len = len(wav[:, 0])
    length_in_sec = wav_len / sampling_rate
    time_audio = np.linspace(0, length_in_sec, num=wav_len)
    time_vad = np.linspace(0, length_in_sec, num=len(vad_segments[0]))
    axs[0].set_title('Left channel')
    axs[0].plot(time_audio, wav[:, 0])
    axs[1].set_title('Left channel vad')
    axs[1].plot(time_vad, vad_segments[0])
    axs[2].set_title('Right channel')
    axs[2].plot(time_audio, wav[:, 1])
    axs[3].set_title('Right channel vad')
    axs[3].plot(time_vad, vad_segments[1])
    axs[4].set_title('Cross talks')
    axs[4].set_xlabel("Time (s)")
    axs[4].plot(time_vad, cross_talks)
    # uncomment if we want to save figures to file
    # plt.savefig('output/plots.png', dpi=1200)
    fig.show()


def plot_speech_time_comparison(energies, speech_time):
    fig, axs = plt.subplots(2)
    axs[0].set_title('Speech energy')
    axs[0].pie(energies, labels=['Therapist', 'Client'], autopct='%1.1f%%',
               startangle=90)
    axs[1].set_title('Speech time')
    axs[1].pie(speech_time, labels=['Therapist', 'Client'], autopct='%1.1f%%',
               startangle=90)
    # axs[1].set_title('Speech time')
    # axs[1].plot(time_vad, vad_segments[0])
    # uncomment if we want to save figures to file
    # plt.savefig('output/plots.png', dpi=1200)
    fig.show()


def count_energy_when_speaking(energy_segments, vad):
    """Count mean energy when speaking"""
    counter = 0
    energy = 0
    for index, value in enumerate(vad):
        if value:
            energy += energy_segments[index]
            counter += 1
    if counter == 0:
        return 0
    return energy / counter


def generate_statistics(energy_over_segments, vad):
    """Generate text statistics about speech"""
    channel1_energy = count_energy_when_speaking(energy_over_segments[0], vad[0])
    channel2_energy = count_energy_when_speaking(energy_over_segments[1], vad[1])

    detect_response_speech_times(vad)
    is_channel1_louder = channel1_energy > channel2_energy
    print(f'Channel 1 speech mean energy: {channel1_energy:.2f}')
    print(f'Channel 2 speech mean energy: {channel2_energy:.2f}')
    print(
        f'Overall channel {1 if is_channel1_louder else 2} was '
        f'{channel1_energy / channel2_energy if is_channel1_louder else channel2_energy / channel1_energy:.2f} '
        f'louder.')

    speech_time = np.sum(vad[0]) / NORMALIZATION_COEFFICIENT, np.sum(vad[1]) / NORMALIZATION_COEFFICIENT
    print(f'Channel 1 speech time: {speech_time[0]:.2f}s')
    print(f'Channel 2 speech time: {speech_time[1]:.2f}s')

    plot_speech_time_comparison([channel1_energy, channel2_energy], speech_time)
