import speech_recognition as sr
import re
from os.path import join
from scipy.io import wavfile
import tempfile
import shutil

regex = re.compile(r'SPEAKER \S* (\d) ([\d.]*) ([\d.]*) <NA> <NA> \d.*')


def extract(line):
    x = regex.search(line)
    return float(x[2]), float(x[3]), int(x[1])


def extract_channels(file_name, path, tmp_dir):
    fs, data = wavfile.read(f'{join(path, file_name)}.wav')

    wavfile.write(f'{tmp_dir}/left.wav', fs, data[:, 0])
    wavfile.write(f'{tmp_dir}/right.wav', fs, data[:, 1])


def clean(path):
    shutil.rmtree(path)


def process_file(src, dst, file, language):
    r = sr.Recognizer()
    file_path = join(src, file)

    tmp_dir = tempfile.mkdtemp()
    extract_channels(file, src, tmp_dir)

    with open(f'{file_path}.rttm', 'r') as f1:
        with open(f'{join(dst, file)}.txt', 'w') as f2:
            with sr.AudioFile(f'{tmp_dir}/left.wav') as left:
                with sr.AudioFile(f'{tmp_dir}/right.wav') as right:
                    offset_l = 0
                    offset_r = 0
                    annotations = f1.readlines()
                    for annotation in annotations:
                        start, duration, channel = extract(annotation)
                        if channel == 1:
                            offset = offset_l + start
                            offset_l -= offset + duration
                            source = left
                        else:
                            offset = offset_r + start
                            offset_r -= offset + duration
                            source = right
                        audio = r.record(source, offset=offset, duration=duration)
                        try:
                            out = r.recognize_google(audio, language=language)
                        except sr.UnknownValueError:
                            out = '<UNKNOWN>'

                        f2.write(
                            f'Channel {channel} ({start:.3f}, {duration:.3f}): {out}\n')
    clean(tmp_dir)
