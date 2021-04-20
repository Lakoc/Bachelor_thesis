import speech_recognition as sr
import re
from os.path import join
from scipy.io import wavfile
import tempfile
import shutil
import numpy as np
from progress.bar import Bar
import params
from helpers.load_files import load_vad_from_rttm

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


def join_segments(vad):
    """Join monolog segments"""
    vad_appended1 = np.append(np.append([1 - vad[0, 0]], vad[:, 0], axis=0), [1 - vad[-1, 0]], axis=0)
    speech_bounds1 = np.where(np.diff(vad_appended1))[0]
    speech_bounds1 = np.append(speech_bounds1[:-1][:, np.newaxis], speech_bounds1[1:][:, np.newaxis], axis=1)
    speech_bounds1 = np.compress(vad[speech_bounds1[:, 0], 0], speech_bounds1, axis=0)

    vad_appended2 = np.append(np.append([1 - vad[0, 1]], vad[:, 1], axis=0), [1 - vad[-1, 1]], axis=0)
    speech_bounds2 = np.where(np.diff(vad_appended2))[0]
    speech_bounds2 = np.append(speech_bounds2[:-1][:, np.newaxis], speech_bounds2[1:][:, np.newaxis], axis=1)
    speech_bounds2 = np.compress(vad[speech_bounds2[:, 0], 1], speech_bounds2, axis=0)

    speech_bounds = (speech_bounds1, speech_bounds2)
    vad_inverted = 1 - vad
    vad1 = np.logical_or(vad[:, 0], vad_inverted[:, 1])
    vad2 = np.logical_or(vad[:, 1], vad_inverted[:, 0])
    vad_joined = (vad1, vad2)

    segment_bounds_joined = []
    for k in range(len(speech_bounds)):
        index = 0
        while index < speech_bounds[k].shape[0]:
            segment1 = speech_bounds[k][index]
            start = segment1[0]
            to = segment1[1]
            for j in range(1, speech_bounds[k].shape[0] - index):
                segment2 = speech_bounds[k][index + j]

                end = segment2[1]

                if np.all(vad_joined[k][start:end] == 1):
                    to = end
                else:
                    segment_bounds_joined.append(
                        [k + 1, start * params.window_stride, (to - start) * params.window_stride])
                    index += j
                    break
            else:
                index += 1
                segment_bounds_joined.append([k + 1, start * params.window_stride, (to - start) * params.window_stride])
    return sorted(segment_bounds_joined, key=lambda l: l[1])


def process_file(src, dst, file, language):
    r = sr.Recognizer()
    file_path = join(src, file)

    tmp_dir = tempfile.mkdtemp()
    extract_channels(file, src, tmp_dir)

    with open(f'{join(dst, file)}.txt', 'w') as f2:
        with sr.AudioFile(f'{tmp_dir}/left.wav') as left:
            with sr.AudioFile(f'{tmp_dir}/right.wav') as right:
                offset_l = 0
                offset_r = 0
                vad = load_vad_from_rttm(f'{file_path}.rttm', int(left.DURATION / params.window_stride))
                segments = join_segments(vad)
                with Bar(f'Creating transcription for {file}', max=len(segments)) as bar:
                    for annotation in segments:
                        channel, start, duration = annotation
                        if channel == 1:
                            offset = offset_l + start
                            offset_l -= offset + duration
                            source = left
                        else:
                            offset = offset_r + start
                            offset_r -= offset + duration
                            source = right
                        audio = r.record(source, offset=offset, duration=duration)
                        if duration < 1.5:
                            out = '<UNKNOWN>'
                        else:
                            try:
                                out = r.recognize_sphinx(audio)
                                """Google API for testing purposes"""
                                # out = r.recognize_google(audio, language=language)
                            except sr.UnknownValueError:
                                out = '<UNKNOWN>'

                        f2.write(
                            f'Channel {channel} ({start:.2f}, {duration:.2f}): {out}\n')
                        bar.next()
    clean(tmp_dir)
