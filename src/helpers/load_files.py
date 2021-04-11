import re
import numpy as np

rttm_regex = re.compile(r'SPEAKER \S* \d ([\d.]*) ([\d.]*) <NA> <NA> (\d).*')
transcription_regex = re.compile(r'Channel (\d) \(([\d.]*), ([\d.]*)\): (.*)')


def extract_rttm_line(line):
    match = rttm_regex.search(line)
    start = int(float(match[1]) * 1000)
    return start, start + int(float(match[2]) * 1000), int(match[3])

def extract_transcription_line(line):
    match = transcription_regex.search(line)
    start = int(float(match[2]) * 1000)
    return start, int(match[1]), None if match[4] == '<UNKNOWN>' else match[4]

def load_vad_from_rttm(path):
    with open(path, 'r') as file:
        lines = file.readlines()

        arr_extracted = []
        for line in lines:
            arr_extracted.append(extract_rttm_line(line))

        vad = np.zeros((arr_extracted[-1][1], 2))

        for speech_tuple in arr_extracted:
            vad[speech_tuple[0]: speech_tuple[1], speech_tuple[2] - 1] = 1

        return vad


def load_transcription(path):
    with open(path, 'r') as file:
        lines = file.readlines()

        arr_extracted = []
        for line in lines:
            arr_extracted.append(extract_transcription_line(line))

        transcriptions = {'0': {}, '1': {}}
        for speech_tuple in arr_extracted:
            transcriptions[f'{speech_tuple[1] - 1}'][f'{speech_tuple[0]}'] = speech_tuple[2]

        return transcriptions
