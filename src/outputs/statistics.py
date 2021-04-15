import numpy as np
from audio_processing.feature_extraction import normalize_energy_to_1
import params


def detect_cross_talks(vad, min_len):
    """Detects cross talks in vad segments"""
    cross_talks = np.logical_and(vad[:, 0], vad[:, 1])
    cross_talks = np.append(np.append([0], cross_talks), [0])
    cross_talks_bounds = np.where(np.diff(cross_talks))[0].reshape(-1, 2)
    cross_talks_bounds = cross_talks_bounds[cross_talks_bounds[:, 1] - cross_talks_bounds[:, 0] > min_len]
    pre_cross_talk = cross_talks_bounds[:, 0] - 1
    cross_talk_origin = (vad[pre_cross_talk, 1] == 0).astype('i1')[:, np.newaxis]

    cross_talks = np.append(cross_talk_origin, cross_talks_bounds, axis=1)

    return cross_talks[cross_talks[:, 0] == 0][:, 1:], cross_talks[cross_talks[:, 0] == 1][:, 1:]


def detect_interruptions(vad):
    """Detects interruptions (speaker 1 is speaking from 0-10, speaker 2 from 3-6) in vad segments"""
    cross_talks = np.logical_and(vad[:, 0], vad[:, 1])
    cross_talks = np.append(np.append([0], cross_talks), [0])
    cross_talks_bounds = np.where(np.diff(cross_talks))[0].reshape(-1, 2)
    vad_appended = np.append(vad, [[0, 0]], axis=0)
    pre_cross_talk = vad_appended[cross_talks_bounds[:, 0] - 1]
    post_cross_talk = vad_appended[cross_talks_bounds[:, 1]]

    interruptions = np.all(post_cross_talk == pre_cross_talk, axis=1)
    interruptions_arg = np.argwhere(interruptions)
    interruption_origin = pre_cross_talk[interruptions_arg, 0]

    interruptions = np.append(interruption_origin, cross_talks_bounds[np.squeeze(interruptions_arg)], axis=1)

    return interruptions[interruptions[:, 0] == 0][:, 1:], interruptions[interruptions[:, 0] == 1][:, 1:]


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
    """Get normalized energy of voice active segments"""
    log_energy1 = normalize_energy_to_1(energy[:, 0])
    log_energy2 = normalize_energy_to_1(energy[:, 1])
    energy_active_segments1 = np.where(vad[:, 0], log_energy1, 0)
    energy_active_segments2 = np.where(vad[:, 1], log_energy2, 0)

    return energy_active_segments1, energy_active_segments2


def calculate_speech_ratio(vad):
    """Calculate percentage of time when each speaker is active"""
    speaker1 = np.sum(vad[:, 0]) / vad.shape[0]
    speaker2 = np.sum(vad[:, 1]) / vad.shape[0]

    speakers = speaker1 + speaker2
    return np.array([[speaker1 / speakers, speaker2 / speakers], [speaker1, speaker2]])


def calculate_speech_speed(transcriptions):
    """Calculate on transcribed segments speed of speech"""
    for speaker in transcriptions:
        for key in speaker:
            duration = key[1] - key[0]
            text = speaker[key]['text']

            if text is None:
                speaker[key]['words_per_segment'] = None
            else:
                words_per_segment = len(text.split(' ')) / duration
                speaker[key]['words_per_segment'] = words_per_segment
    return transcriptions


def calculate_speech_len(vad):
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
    for index, _ in enumerate(transcriptions):
        for transcription in transcriptions[f'{index}']:
            text = transcriptions[f'{index}'][transcription]['text']
            if text is not None:
                texts[index] += f' {text}'
    return tuple(texts)


def find_loud_transcriptions(loudness, energy, vad, transcriptions):
    """Add to transcriptions its "loudness" detected by finding the most energetically important segments"""
    segments = ({}, {})
    energy_mean = [np.mean(energy[:, 0][vad[:, 0]]), np.mean(energy[:, 1][vad[:, 1]])]
    for index, speaker in enumerate(loudness):
        # iterate over loud segments
        for segment in speaker:
            start = segment[0]
            end = segment[1]

            # find belonging transcription
            for transcription in transcriptions[f'{index}']:
                t_start, t_end = transcription

                # we have not found any transcription for loud segment
                if t_start > end:
                    break

                elif t_start <= start and t_end >= end:
                    # add segment len to transcription ratio
                    text = transcriptions[f'{index}'][transcription]['text']
                    if text is not None:
                        if (t_start, t_end) in segments[index]:
                            pre_ratio = segments[index][(t_start, t_end)]['ratio']
                        else:
                            pre_ratio = 0
                        segments[index][(t_start, t_end)] = {'text': text,
                                                             'ratio': pre_ratio + end - start,
                                                             'energy': np.sum(energy[t_start: t_end, index]) / (
                                                                     energy_mean[index] * (t_end - t_start))}
    return segments


def get_stats(vad, transcription, energy, min_segment_len):
    """Calculate all available statistics"""
    cross_talks = detect_cross_talks(vad, min_segment_len)
    interruptions = detect_interruptions(vad)
    loudness = detect_loudness(energy, vad, percentile=80, min_len=min_segment_len)
    volume_changes = process_volume_changes(energy, vad)
    loud_transcription_with_speed = calculate_speech_speed(
        find_loud_transcriptions(loudness, energy, vad, transcription))
    texts = get_texts(transcription)
    speech_bounds, segment_bounds_joined, hesitations = calculate_speech_len(vad)
    reactions = calculate_reaction_times(segment_bounds_joined)

    speech_ratio = calculate_speech_ratio(vad)
    return {'cross_talks': cross_talks, 'interruptions': interruptions, 'loudness': loud_transcription_with_speed,
            'speech': speech_bounds, 'speech_joined': segment_bounds_joined, 'hesitations': hesitations,
            'reactions': reactions, 'volume_changes': volume_changes,
            'speech_ratio': speech_ratio, 'texts': texts,
            'signal': {'len': vad.shape[0] * params.window_stride}}
