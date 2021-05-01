import sys
from os import listdir, remove
from os.path import isfile, join as path_join
import re
from scipy.io.wavfile import write
from audio_processing.preprocessing import read_wav_file
from helpers.dir_exist import create_if_not_exist

path = sys.argv[1]

files = [f.split('.')[0] for f in listdir(path) if isfile(path_join(path, f)) and f.endswith(f'.wav')]

"""Trim audio files and create rttm file"""
rttm_reg = re.compile(r'(.*) (.*) (\d) ([\d.]*) ([\d.]*) (<NA>) (<NA>) (\d)')
transcription_regex = re.compile(r'Channel (\d) ([\d.]*), ([\d.]*): (.*)')
separator = '/'
path_new = f'{separator.join(path.split("/")[0:-2])}/online_trimmed'

create_if_not_exist(path_new)

for file in files:
    with open(f"{path_join(path, file)}.txt", 'r') as f:
        with open(f"{path_join(path, file)}.rttm", 'r') as fr:
            with open(f"{path_join(path_new, file)}.txt", 'w') as f1:
                with open(f"{path_join(path_new, file)}.rttm", 'w') as fr1:
                    lines = f.readlines()

                    wav_file, sampling_rate = read_wav_file(f"{path_join(path, file)}.wav")
                    wav_size = wav_file.shape[0]
                    part = wav_size // 2

                    file_trimmed = wav_file[0: part, :]
                    write(f"{path_join(path_new, file)}.wav", sampling_rate, file_trimmed)

                    for line in lines:
                        match = re.search(transcription_regex, line)
                        start_time = float(match[2])
                        end_time = float(match[3])

                        if end_time < part / sampling_rate:
                            f1.write(f'Channel {match[1]} {start_time:.2f}, {end_time:.2f}: {match[4]}\n')
                        elif start_time < part / sampling_rate:
                            f1.write(f'Channel {match[1]} {start_time:.2f}, {part / sampling_rate:.2f}: {match[4]}\n')

                    lines = fr.readlines()
                    for line in lines:
                        match = rttm_reg.search(line)
                        start_time = float(match[4])
                        duration = float(match[5])
                        speaker = int(match[3])
                        file = match[2]
                        end_time = start_time + duration
                        if end_time < part / sampling_rate:
                            fr1.write(
                                f'SPEAKER {file} {speaker} {start_time:.2f} {duration:.2f} <NA> <NA> {speaker} <NA> <NA>\n')
                        elif start_time < part / sampling_rate:
                            fr1.write(
                                f'SPEAKER {file} {speaker} {start_time:.2f} {part / sampling_rate:.2f} <NA> <NA> {speaker} <NA> <NA>\n')
