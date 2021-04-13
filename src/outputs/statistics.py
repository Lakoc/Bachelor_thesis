import numpy as np


def detect_cross_talks(vad):
    """Detects cross talks in vad segments"""
    cross_talks = np.logical_and(vad[:, 0], vad[:, 1])
    cross_talks = np.append(np.append([0], cross_talks), [0])
    cross_talks_bounds = np.where(np.diff(cross_talks))[0].reshape(-1, 2)
    pre_cross_talk = cross_talks_bounds[:, 0] - 1
    cross_talk_origin = (vad[pre_cross_talk, 1] == 0).astype('i1')[:, np.newaxis]

    cross_talks = np.append(cross_talk_origin, cross_talks_bounds, axis=1)

    return cross_talks


def detect_interruptions(vad):
    """Detects interruptions (speaker 1 is speaking from 0-10, speaker 2 from 3-6) in vad segments"""
    cross_talks = np.logical_and(vad[:, 0], vad[:, 1])
    cross_talks = np.append(np.append([0], cross_talks), [0])
    cross_talks_bounds = np.where(np.diff(cross_talks))[0].reshape(-1, 2)
    vad_appended = np.append(vad, [[0,0]], axis=0)
    pre_cross_talk = vad_appended[cross_talks_bounds[:, 0] - 1]
    post_cross_talk = vad_appended[cross_talks_bounds[:, 1]]

    interruptions = np.all(post_cross_talk == pre_cross_talk, axis=1)
    interruptions_arg = np.argwhere(interruptions)
    interruption_origin = pre_cross_talk[interruptions_arg, 0]

    interruptions = np.append(interruption_origin, cross_talks_bounds[np.squeeze(interruptions_arg)], axis=1)

    return interruptions


def detect_loudness(energy, vad, percentile=95, min_len=20):
    """Detect loud parts of speech"""
    energy_active_segments1 = np.where(vad[:, 0], energy[:, 0], 0)
    energy_active_segments2 = np.where(vad[:, 1], energy[:, 1], 0)

    loudness1 = np.squeeze(np.argwhere(energy_active_segments1 > np.percentile(energy_active_segments1, percentile)))
    loudness2 = np.squeeze(np.argwhere(energy_active_segments2 > np.percentile(energy_active_segments2, percentile)))

    sequences1 = np.split(loudness1, np.array(np.where(np.diff(loudness1) > 1)[0]) + 1)
    sequences2 = np.split(loudness2, np.array(np.where(np.diff(loudness2) > 1)[0]) + 1)
    loudness_bounds_filtered = ([], [])

    for s in sequences1:
        if len(s) >= min_len:
            loudness_bounds_filtered[0].append((np.min(s), np.max(s)))
    for s in sequences2:
        if len(s) >= min_len:
            loudness_bounds_filtered[1].append((np.min(s), np.max(s)))

    return loudness_bounds_filtered


def calculate_speech_ratio(vad):
    """Calculate percentage of time when each speaker is active"""
    speaker1 = np.sum(vad[:, 0]) / vad.shape[0]
    speaker2 = np.sum(vad[:, 1]) / vad.shape[0]

    speakers = speaker1 + speaker2
    return np.array([[speaker1 / speakers, speaker2 / speakers], [speaker1, speaker2]])


def calculate_speech_speed(transcriptions):
    """Calculate on transcribed segments speed of speech"""
    for speaker in transcriptions:
        for key in transcriptions[speaker]['segments']:
            duration = int(key.split('-')[1])
            text = transcriptions[speaker]['segments'][key]['text']

            if text is None:
                transcriptions[speaker]['segments'][key]['words_per_segment'] = None
            else:
                words_per_segment = len(text.split(' ')) / duration
                transcriptions[speaker]['segments'][key]['words_per_segment'] = words_per_segment
    return transcriptions['0']['segments'], transcriptions['1']['segments']


def calculate_speech_len(vad):
    vad_appended1 = np.append(np.append([1 - vad[0, 0]], vad[:, 0], axis=0), [1 - vad[-1, 0]], axis=0)
    speech_bounds1 = np.where(np.diff(vad_appended1))[0]
    speech_bounds1 = np.append(speech_bounds1[:-1][:, np.newaxis], speech_bounds1[1:][:, np.newaxis], axis=1)
    speech_bounds1 = np.compress(vad[speech_bounds1[:, 0], 0], speech_bounds1, axis=0)
    speech_lengths1 = (speech_bounds1[:, 1] - speech_bounds1[:, 0])[:, np.newaxis]

    vad_appended2 = np.append(np.append([1 - vad[0, 1]], vad[:, 1], axis=0), [1 - vad[-1, 1]], axis=0)
    speech_bounds2 = np.where(np.diff(vad_appended2))[0]
    speech_bounds2 = np.append(speech_bounds2[:-1][:, np.newaxis], speech_bounds2[1:][:, np.newaxis], axis=1)
    speech_bounds2 = np.compress(vad[speech_bounds2[:, 0], 1], speech_bounds2, axis=0)
    speech_lengths2 = (speech_bounds2[:, 1] - speech_bounds2[:, 0])[:, np.newaxis]

    speech_bounds = (speech_bounds1, speech_bounds2)
    vad_inverted = 1 - vad
    vad1 = np.logical_or(vad[:, 0], vad_inverted[:, 1])
    vad2 = np.logical_or(vad[:, 1], vad_inverted[:, 0])
    vad_joined = (vad1, vad2)
    segments = ({}, {})
    segments_len_joined = ([], [])
    segments_hesitation = ([], [])
    segment_bounds_joined = ([], [])
    for k in range(len(speech_bounds)):
        index = 0
        while index < speech_bounds[k].shape[0]:
            segment1 = speech_bounds[k][index]
            start = segment1[0]
            to = segment1[1]
            hesitation = []
            for j in range(1, speech_bounds[k].shape[0] - index):
                segment2 = speech_bounds[k][index + j]

                end = segment2[1]

                if np.all(vad_joined[k][start:end] == 1):
                    hesitation.append(segment2[0] - to)
                    segments_hesitation[k].append(segment2[0] - to)
                    to = end
                else:
                    segments[k][f'{start}-{to}'] = hesitation
                    segments_len_joined[k].append(to - start)
                    segment_bounds_joined[k].append([start, to])
                    index += j
                    break
            else:
                index += 1
                segments[k][f'{start}-{to}'] = hesitation
                segments_len_joined[k].append(to - start)
                segment_bounds_joined[k].append([start, to])

    return (speech_lengths1, speech_lengths2), segments_len_joined, segments_hesitation, segment_bounds_joined


def calculate_reaction_times(bounds):
    """Find silence segments when one speaker waits for second speaker response"""
    responses = ([], [])

    for k in range(len(bounds)):
        i = 0
        j = 0
        while i < len(bounds[k]) - 1:
            segment = bounds[k][i]
            segment_post = bounds[k][i + 1]
            wait_start = segment[1]
            wait_end = segment_post[0]

            while j < len(bounds[1 - k]):
                start = bounds[1 - k][j][0]
                end = bounds[1 - k][j][1]
                j += 1
                if wait_start < start < wait_end:
                    responses[k].append({'start': start, 'duration': start - wait_start})
                    i += 1
                    break
                elif end < wait_start:
                    continue
                else:
                    i += 1
                    break
            else:
                break
    return responses


def get_stats(vad, transcription, energy):
    """Calculate all available statistics"""
    cross_talks = detect_cross_talks(vad)
    interruptions = detect_interruptions(vad)
    loudness = detect_loudness(energy, vad)
    speed = calculate_speech_speed(transcription)
    speech_len, speech_len_joined, hesitations, segment_bounds_joined = calculate_speech_len(vad)
    reaction_time = calculate_reaction_times(segment_bounds_joined)

    speech_ratio = calculate_speech_ratio(vad)
    return {'cross_talks': cross_talks, 'interruptions': interruptions, 'loudness': loudness, 'speed': speed,
            'speech_len': speech_len, 'speech_len_joined': speech_len_joined, 'hesitations': hesitations,
            'reaction_time': reaction_time,
            'speech_ratio': speech_ratio}
