import re
import numpy as np
import json

from bs4 import BeautifulSoup

import params

rttm_regex = re.compile(r'SPEAKER \S* \d ([\d.]*) ([\d.]*) <NA> <NA> (\d).*')
transcription_regex = re.compile(r'Channel (\d) \(([\d.]*), ([\d.]*)\): (.*)')


def extract_rttm_line(line):
    match = rttm_regex.search(line)
    start = int(float(match[1]) / params.window_stride)
    return start, start + int(float(match[2]) / params.window_stride), int(match[3])


def extract_transcription_line(line):
    match = transcription_regex.search(line)
    start = int(float(match[2]) / params.window_stride)
    duration = int(float(match[3]) / params.window_stride)
    return start, start +duration, int(match[1]), None if match[4] == '<UNKNOWN>' else match[4]


def load_vad_from_rttm(path, size):
    with open(path, 'r') as file:
        lines = file.readlines()

        vad = np.zeros((size, 2))

        for line in lines:
            speech_tuple = extract_rttm_line(line)
            vad[speech_tuple[0]: speech_tuple[1], speech_tuple[2] - 1] = 1

        return vad.astype('i1')


def load_transcription(path):
    with open(path, 'r') as file:
        lines = file.readlines()

        arr_extracted = []
        for line in lines:
            arr_extracted.append(extract_transcription_line(line))

        transcriptions = {'0': {}, '1':{}}
        for speech_tuple in arr_extracted:
            transcriptions[f'{speech_tuple[2] - 1}'][(speech_tuple[0], speech_tuple[1])] = {
                'text': speech_tuple[3]}

        return transcriptions


def load_texts(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        return data


def load_energy(path):
    rmse = np.loadtxt(path)
    return rmse


def load_stats(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        return data


def load_template(template_path):
    with open(template_path, 'r') as file:
        soup = BeautifulSoup(file, "html.parser")
    return soup
