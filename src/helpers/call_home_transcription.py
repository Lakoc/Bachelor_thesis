import sys
from os import listdir, remove, makedirs
from os.path import isfile, exists, join as path_join
import re
from scipy.io.wavfile import write
from audio_processing.preprocessing import read_wav_file

path = sys.argv[1]

files = [f.split('.')[0] for f in listdir(path) if isfile(path_join(path, f)) and f.endswith(f'.wav')]

"""Clean from newlines and comments"""
for file in files:
    with open(f'{path_join(path, file)}.txt', 'r') as f:
        with open(f'{path_join(path, file)}.clean', 'w') as f1:
            for line in f:
                if line[0] != '#' and not line.isspace():
                    f1.write(line)

"""Create rttm files"""
line_re = re.compile(r'([\d.]*) ([\d.]*) ([AB])1?:.*')
for file in files:
    with open(f'{path_join(path, file)}.clean', 'r') as f:
        with open(f"{path_join(path, file)}.rttm", 'w') as f1:
            for line in f:
                match = re.search(line_re, line)
                start_time = float(match[1])
                end_time = float(match[2])
                speaker = 1 if match[3] == 'A' else 2
                if start_time < 300:
                    if end_time > 300:
                        end_time = 300
                    f1.write(
                        f'SPEAKER {file} {speaker} {start_time:.3f} {end_time - start_time:.3f} <NA> <NA> {speaker} <NA> <NA>\n')

"""Trim audio files and move rttm starts"""
rttm_reg = re.compile(r'(.*) (.*) (\d) ([\d.]*) ([\d.]*) (<NA>) (<NA>) (\d)')

separator = '/'
path_new = f'{separator.join(path.split("/")[0:-2])}/dev'

if not exists(path_new):
    makedirs(path_new)

for file in files:
    with open(f"{path_join(path, file)}.rttm", 'r') as f:
        with open(f"{path_join(path_new, file)}.rttm", 'w') as f1:
            lines = f.readlines()
            match = re.search(rttm_reg, lines[0])
            start_time = float(match[4])
            wav_file, sampling_rate = read_wav_file(f"{path_join(path, file)}.wav")
            start_index = int(start_time * sampling_rate)
            file_trimmed = wav_file[start_index:, :]
            write(f"{path_join(path_new, file)}.wav", sampling_rate, file_trimmed)
            for line in lines:
                match = re.search(rttm_reg, line)
                start_time_old = float(match[4])
                start_time_new = start_time_old - start_time
                f1.write(
                    f'{match[1]} {match[2]} {match[3]} {start_time_new:.3f} {match[5]} {match[6]} {match[7]} {match[8]}\n')

"""Move old files"""
for file in files:
    remove(f'{path_join(path, file)}_trimmed.wav')
"""Remove unnecessary files"""
for file in files:
    remove(f'{path_join(path, file)}_trimmed.wav')
