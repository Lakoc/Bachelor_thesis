import numpy as np
from scipy.io import wavfile


def read_wav_file(path):
    """Read wav file and returns numpy array and sampling rate"""
    sampling_rate, wav_file = wavfile.read(path)
    return wav_file, sampling_rate


def process_pre_emphasis(signal, coefficient):
    """Amplify high frequencies, and balance frequency spectrum"""
    return np.append([signal[0]], signal[1:] - coefficient * signal[:-1], axis=0)


def process_hamming(signal, sampling_rate, window_size, window_overlap):
    """Segment signal  it and apply window function"""

    if window_size < window_overlap:
        exit("Window size smaller than overlap!")

    window_size = int(round(sampling_rate * window_size))
    window_overlap = int(round(sampling_rate * window_overlap))
    window_stride = window_size - window_overlap

    signal_len = len(signal)
    # size_of_signal = ((number_of_frames - 1) * (window_size - window_overlap)) + window_size
    number_of_frames = int(np.ceil((signal_len - window_size) / window_stride + 1))
    signal_len_new = (number_of_frames - 1) * window_stride + window_size

    # np.arange ends on stop so add 1 more step
    indices = np.tile(np.arange(0, window_size), (number_of_frames, 1)) + np.tile(
        np.arange(0, signal_len_new - window_size + window_stride, window_stride),
        (window_size, 1)).T

    # pad zeros to end for evenly segmented windows
    zeros_to_append = signal_len_new - signal_len
    signal_padded = np.pad(signal, ((0, zeros_to_append), (0, 0)), 'constant', constant_values=0)

    # split to segments
    windows = signal_padded[indices.astype(np.int32, copy=False)]

    # apply hamming window function on segments
    windows = windows.transpose(0, 2, 1)
    windows *= np.hamming(window_size)
    return windows.transpose(0, 2, 1)
