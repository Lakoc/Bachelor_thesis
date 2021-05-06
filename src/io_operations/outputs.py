import numpy as np
import params


def diarization_to_rttm_file(path, speaker, segment_time):
    """Save dictaphone diarization output to rttm file"""
    file_name = path.split(".")[0].split('/')[-1]
    path = f'{path.split(".")[0]}.rttm'
    with open(path, 'w') as file:
        for index, val in enumerate(speaker):
            if val == 0:
                continue
            else:
                file.write(
                    f'SPEAKER {file_name} {val} {segment_time[index]:.2f} {(float(segment_time[index + 1]) - float(segment_time[index])):.2f} <NA> <NA> {val} <NA> <NA>\n')


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


def online_session_to_rttm(path, vad):
    """Save online session diarization to rttm file"""
    file_name = path.split('/')[-1]
    path = f'{path}.rttm'
    speech_start_spk1 = np.empty(vad.shape[0], dtype=bool)
    speech_start_spk1[0] = True
    speech_start_spk1[1:] = np.not_equal(vad[:-1, 0], vad[1:, 0])
    segment_indexes1 = np.argwhere(speech_start_spk1).reshape(-1)

    speech_start_spk2 = np.empty(vad.shape[0], dtype=bool)
    speech_start_spk2[0] = True
    speech_start_spk2[1:] = np.not_equal(vad[:-1, 1], vad[1:, 1])
    segment_indexes2 = np.argwhere(speech_start_spk2).reshape(-1)

    # Append end of audio
    segment_indexes = np.append(segment_indexes1, segment_indexes2)
    start_time_index = np.argsort(segment_indexes)
    len_indexes1 = segment_indexes1.shape[0]

    segment_indexes1 = np.append(segment_indexes1, vad.shape[0])
    segment_indexes2 = np.append(segment_indexes2, vad.shape[0])
    with open(path, 'w') as file:
        for index in start_time_index:
            if index < len_indexes1:
                start_time = segment_indexes1[index]
                speech = vad[start_time, 0]
                if not speech:
                    continue
                speaker = 1
                end_time = segment_indexes1[index + 1]
            else:
                index = index - len_indexes1
                start_time = segment_indexes2[index]
                speech = vad[start_time, 1]
                if not speech:
                    continue
                speaker = 2
                end_time = segment_indexes2[index + 1]

            start_time *= params.window_stride
            end_time *= params.window_stride
            time = end_time - start_time

            file.write(
                f'SPEAKER {file_name} {speaker} {start_time:.2f} {time:.2f} <NA> <NA> {speaker} <NA> <NA>\n')
