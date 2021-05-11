import sys
from os import listdir, remove
from os.path import isfile, join as path_join
import re
from scipy.io.wavfile import write
from audio_processing.preprocessing import read_wav_file
from helpers.dir_exist import create_if_not_exist

path = sys.argv[1]

files = [f.split('.')[0] for f in listdir(path) if isfile(path_join(path, f)) and f.endswith(f'.wav')]

"""Clean from newlines and comments"""
for file in files:
    with open(f'{path_join(path, file)}.txt', 'r') as f:
        with open(f'{path_join(path, file)}.clean', 'w') as f1:
            for line in f:
                if line[0] != '#' and not line.isspace():
                    f1.write(line)

"""Trim audio files and create rttm file"""
line_re = re.compile(r'([\d.]*) ([\d.]*) ([AB])[12]?: (.*)')
separator = '/'
path_new = f'{separator.join(path.split("/")[0:-2])}/callhome_trimmed'

create_if_not_exist(path_new)

for file in files:
    with open(f"{path_join(path, file)}.clean", 'r') as f:
        with open(f"{path_join(path_new, file)}.rttm", 'w') as f1:
            with open(f"{path_join(path_new, file)}.txt", 'w') as f2:
                lines = f.readlines()

                # Detect start and end of annotations
                match = re.search(line_re, lines[0])
                start_time = float(match[1])
                start_index = int(start_time * sampling_rate)

                match = re.search(line_re, lines[-1])
                end_time = float(match[2])
                end_index = int(end_time * sampling_rate)

                # Trim source wav file
                wav_file, sampling_rate = read_wav_file(f"{path_join(path, file)}.wav")
                file_trimmed = wav_file[start_index: end_index, :]
                write(f"{path_join(path_new, file)}.wav", sampling_rate, file_trimmed)

                # Save new annotations by subtracting start time
                for line in lines:
                    match = re.search(line_re, line)
                    start_time_old = float(match[1])
                    start_time_new = start_time_old - start_time
                    end_time = float(match[2])
                    end_time_new = end_time - start_time
                    speaker = 1 if match[3] == 'A' else 2
                    text = match[4]
                    f1.write(
                        f'SPEAKER {file} {speaker} {start_time_new:.3f} {end_time - start_time_old:.3f} <NA> <NA> {speaker} <NA> <NA>\n')
                    f2.write(f'Channel {speaker} {start_time_new:.2f}, {end_time_new:.2f}: {text}\n')

"""Remove unnecessary files"""
for file in files:
    remove(f'{path_join(path, file)}.clean')
