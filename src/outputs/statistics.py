import numpy as np
import params
import re
import text2emotion as te


def closest_interval(dict_keys, value_to_fit):
    """Find closest interval to provided value in transcription"""
    return dict_keys[min(range(len(dict_keys)),
                         key=lambda i: abs(dict_keys[i][1] - value_to_fit) + abs(dict_keys[i][0] - value_to_fit))]


def detect_cross_talks(vad, min_len, transcription):
    """Detects cross talks in vad segments"""
    min_len_segments = int(min_len / params.window_stride)
    cross_talks = np.logical_and(vad[:, 0], vad[:, 1])
    cross_talks = np.append(np.append([0], cross_talks), [0])
    cross_talks_bounds = np.where(np.diff(cross_talks))[0].reshape(-1, 2)
    cross_talks_bounds = cross_talks_bounds[cross_talks_bounds[:, 1] - cross_talks_bounds[:, 0] > min_len_segments]
    pre_cross_talk = cross_talks_bounds[:, 0] - 1
    cross_talk_origin = (vad[pre_cross_talk, 1] == 0).astype('i1')[:, np.newaxis]

    cross_talks = np.append(cross_talk_origin, cross_talks_bounds, axis=1)

    dictionary_keys = [list(transcription[0].keys()), list(transcription[1].keys())]
    cross_talks_texts = ([], [])

    """Find transcriptions for cross talks"""
    for cross_talk in cross_talks:
        origin = cross_talk[0]
        start = cross_talk[1]
        end = cross_talk[2]
        mean_time = (end + start) / 2
        interruption_origin = transcription[origin][closest_interval(dictionary_keys[origin], mean_time)]['text']
        interrupted = transcription[1 - origin][closest_interval(dictionary_keys[1 - origin], mean_time)]['text']
        cross_talks_texts[origin].append([interruption_origin, interrupted])

    cross_talks_as_tuple = cross_talks[cross_talks[:, 0] == 0][:, 1:], cross_talks[cross_talks[:, 0] == 1][:, 1:]

    return cross_talks_as_tuple, cross_talks_texts


# def detect_interruptions(vad):
#     """Detects interruptions (speaker 1 is speaking from 0-10, speaker 2 from 3-6) in vad segments"""
#     cross_talks = np.logical_and(vad[:, 0], vad[:, 1])
#     cross_talks = np.append(np.append([0], cross_talks), [0])
#     cross_talks_bounds = np.where(np.diff(cross_talks))[0].reshape(-1, 2)
#     vad_appended = np.append(vad, [[0, 0]], axis=0)
#     pre_cross_talk = vad_appended[cross_talks_bounds[:, 0] - 1]
#     post_cross_talk = vad_appended[cross_talks_bounds[:, 1]]
#
#     interruptions = np.all(post_cross_talk == pre_cross_talk, axis=1)
#     interruptions_arg = np.argwhere(interruptions)
#     interruption_origin = pre_cross_talk[interruptions_arg, 0]
#
#     interruptions = np.append(interruption_origin, cross_talks_bounds[np.squeeze(interruptions_arg)], axis=1)
#
#     return interruptions[interruptions[:, 0] == 0][:, 1:], interruptions[interruptions[:, 0] == 1][:, 1:]


def detect_loudness(energy, vad, percentile=90, min_len=20):
    """Detect loud parts of speech"""
    energy_active_segments1 = np.where(vad[:, 0], energy[:, 0], 0)
    energy_active_segments2 = np.where(vad[:, 1], energy[:, 1], 0)

    loudness1 = np.squeeze(np.argwhere(energy_active_segments1 > np.percentile(energy_active_segments1, percentile)))
    loudness2 = np.squeeze(np.argwhere(energy_active_segments2 > np.percentile(energy_active_segments2, percentile)))

    sequences1 = np.split(loudness1, np.array(np.where(np.diff(loudness1) > 1)[0]) + 1)
    sequences2 = np.split(loudness2, np.array(np.where(np.diff(loudness2) > 1)[0]) + 1)

    # Filter sequences by their lengths and create numpy arr
    loudness_bounds_filtered = np.array([[s[0], s[-1]] for s in sequences1 if len(s) >= min_len]), np.array(
        [[s[0], s[-1]] for s in sequences2 if len(s) >= min_len])

    return loudness_bounds_filtered


def process_volume_changes(energy, vad):
    """Get energy of voice active segments"""
    energy_active_segments1 = np.where(vad[:, 0], energy[:, 0], 0)
    energy_active_segments2 = np.where(vad[:, 1], energy[:, 1], 0)

    return energy_active_segments1, energy_active_segments2


def calculate_speech_ratio(vad):
    """Calculate percentage of time when each speaker is active"""
    speaker1 = np.sum(vad[:, 0]) / vad.shape[0]
    speaker2 = np.sum(vad[:, 1]) / vad.shape[0]

    speakers = speaker1 + speaker2
    return np.array([[speaker1 / speakers, speaker2 / speakers], [speaker1, speaker2]])


def calculate_speech_speed(transcriptions):
    """Calculate on transcribed segments speed of speech"""
    for speaker in range(len(transcriptions)):
        for key in transcriptions[speaker]:
            duration = key[1] - key[0]
            text = transcriptions[speaker][key]['text']
            if text == '':
                transcriptions[speaker][key]['words_per_segment'] = 0
            else:
                text = re.sub(r'%[a-zA-Z]*|{[a-zA-Z ]*}|\[[a-zA-Z ]*]|<[a-zA-Z]*|[^a-zA-Z\s]', "", text).strip()
                words_per_segment = len(text.split()) / duration
                transcriptions[speaker][key]['words_per_segment'] = words_per_segment
                transcriptions[speaker][key]['text'] = text


def detect_segment_bounds(vad):
    """Calculate speech lengths of vad segments, join following segments that was not interrupted by second channel,
    and find hesitation in monolog"""
    vad_appended1 = np.append(np.append([1 - vad[0, 0]], vad[:, 0], axis=0), [1 - vad[-1, 0]], axis=0)
    speech_bounds1 = np.where(np.diff(vad_appended1))[0]
    speech_bounds1 = np.append(speech_bounds1[:-1][:, np.newaxis], speech_bounds1[1:][:, np.newaxis], axis=1)
    speech_bounds1 = np.compress(vad[speech_bounds1[:, 0], 0], speech_bounds1, axis=0)

    vad_appended2 = np.append(np.append([1 - vad[0, 1]], vad[:, 1], axis=0), [1 - vad[-1, 1]], axis=0)
    speech_bounds2 = np.where(np.diff(vad_appended2))[0]
    speech_bounds2 = np.append(speech_bounds2[:-1][:, np.newaxis], speech_bounds2[1:][:, np.newaxis], axis=1)
    speech_bounds2 = np.compress(vad[speech_bounds2[:, 0], 1], speech_bounds2, axis=0)

    # Join bounds of speech and invert second channel vad for finding non interrupted segments
    speech_bounds = (speech_bounds1, speech_bounds2)
    vad_inverted = 1 - vad
    vad1 = np.logical_or(vad[:, 0], vad_inverted[:, 1])
    vad2 = np.logical_or(vad[:, 1], vad_inverted[:, 0])
    vad_joined = (vad1, vad2)

    # Tuples to fill
    segments_hesitation = ([], [])
    segment_bounds_joined = ([], [])

    # Iterate over channels and search for segments according to second channel
    for k in range(len(speech_bounds)):
        index = 0
        # Iterate over segments of speech
        while index < speech_bounds[k].shape[0]:
            # Extract segment from channel 1
            segment1 = speech_bounds[k][index]
            start = segment1[0]
            to = segment1[1]
            for j in range(1, speech_bounds[k].shape[0] - index):
                # Extract segment from channel 2
                segment2 = speech_bounds[k][index + j]
                end = segment2[1]

                # Found not interrupted segment, Continue joining segments
                if np.all(vad_joined[k][start:end] == 1):
                    # Add hesitation bounds
                    segments_hesitation[k].append((to, segment2[0]))
                    to = end
                else:
                    # No more segments to join
                    segment_bounds_joined[k].append([start, to])
                    index += j
                    break
            else:
                # No more segments to join
                index += 1
                segment_bounds_joined[k].append([start, to])

    return speech_bounds, (np.array(segment_bounds_joined[0]), np.array(segment_bounds_joined[1])), (
        np.array(segments_hesitation[0]), np.array(segments_hesitation[1]))


def calculate_segments_len_distribution(bounds):
    """Calculate joined segments lengths distribution"""
    speech_bounds_len = (bounds[:, 1] - bounds[:, 0]).astype(np.float)
    speech_bounds_len *= params.window_stride

    bins = np.array([0, 2, 5, 10, 15, 20, 30, np.iinfo(np.int16).max])

    current_counts, _ = np.histogram(speech_bounds_len, bins=bins)
    current_counts = (current_counts / np.sum(current_counts)) * 100
    return current_counts


def calculate_reaction_times(bounds):
    """Find silence segments when one speaker waits for second speaker response"""
    responses = ([], [])

    for k in range(len(bounds)):
        i = 0
        j = 0
        while i < len(bounds[k]) - 1:
            # Get current segment and next segment and find interval in which we can wait for respond
            segment = bounds[k][i]
            segment_post = bounds[k][i + 1]
            wait_start = segment[1]
            wait_end = segment_post[0]

            while j < len(bounds[1 - k]):
                # Find response
                start = bounds[1 - k][j][0]
                end = bounds[1 - k][j][1]
                j += 1
                if wait_start < start < wait_end:
                    # Respond found
                    responses[k].append((wait_start, start))
                    i += 1
                    break
                elif end < wait_start:
                    # Segment in other channel is too small should find bigger bounds
                    continue
                else:
                    # Could not find segment in channel 2 that could lay between segments in channel 1
                    i += 1
                    break
            else:
                break
    return np.array(responses[0]), np.array(responses[1])


def get_texts(transcriptions):
    """Append texts to long string for word clouding"""
    texts = ['', '']
    for speaker in range(len(transcriptions)):
        for transcription in transcriptions[speaker]:
            text = transcriptions[speaker][transcription]['text']
            if text is not None:
                texts[speaker] += f' {text}'
    non_speech_activity = len(re.findall(r'[%_][a-zA-z]*', texts[0])), len(re.findall(r'[%_][a-zA-z]*', texts[1]))
    return texts, non_speech_activity


def calculate_loudness(energy, vad, transcriptions):
    """Add to transcriptions its "loudness" detected by finding the most energetically important segments"""
    energy_mean = [np.mean(energy[:, 0][vad[:, 0].astype(bool)]), np.mean(energy[:, 1][vad[:, 1].astype(bool)])]
    for speaker in range(len(transcriptions)):
        for transcription in transcriptions[speaker]:
            t_start, t_end = transcription
            transcriptions[speaker][transcription]['energy'] = np.mean(
                energy[t_start: t_end, speaker] / energy_mean[speaker])


def calculate_client_mood(transcriptions):
    """Calculate client mood from transcriptions"""
    emotions = np.zeros((len(transcriptions), 5))
    for index, transcription in enumerate(transcriptions):
        text = transcriptions[transcription]['text']
        if text:
            emotions[index] = list(te.get_emotion(text).values())

    return emotions


def interruption_len_hist(interruptions):
    """Return counts of interruptions"""
    interruptions_len = (interruptions[:, 1] - interruptions[:, 0]).astype(np.float)
    interruptions_len *= params.window_stride
    bins = np.array([0, 0.2, 0.5, 0.8, 1, np.iinfo(np.int16).max])
    counts, _ = np.histogram(interruptions_len, bins=bins)
    return counts


def get_stats(vad, transcription, energy):
    """Calculate all available statistics"""
    texts, fills = get_texts(transcription)
    interruptions, interruptions_texts = detect_cross_talks(vad, params.min_crosstalk, transcription)
    interruptions_len = interruption_len_hist(interruptions[0]), interruption_len_hist(interruptions[1])
    volume_changes = process_volume_changes(energy, vad)
    calculate_loudness(energy, vad, transcription)
    calculate_speech_speed(transcription)
    speech_bounds, segment_bounds_joined, hesitations = detect_segment_bounds(vad)
    speech_len = calculate_segments_len_distribution(segment_bounds_joined[0]), calculate_segments_len_distribution(
        segment_bounds_joined[1])
    reactions = calculate_reaction_times(segment_bounds_joined)
    client_mood = calculate_client_mood(transcription[0])

    speech_ratio = calculate_speech_ratio(vad)
    return {'interruptions': interruptions, 'interruptions_len': interruptions_len, 'transcription': transcription,
            'speech': speech_bounds, 'speech_joined': segment_bounds_joined, 'hesitations': hesitations,
            'reactions': reactions, 'volume_changes': volume_changes, 'client_mood': client_mood,
            'speech_len': speech_len, 'speech_ratio': speech_ratio, 'texts': texts, 'fills': fills,
            'signal': {'len': vad.shape[0] * params.window_stride}, 'interruptions_texts': interruptions_texts}
