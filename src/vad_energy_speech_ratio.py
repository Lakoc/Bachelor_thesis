from src.AudioProcessing import read_wav_file
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

SAMPLE_START = 0
SPEECH_START_BAND = 300
SPEECH_END_BAND = 3000
SAMPLE_WINDOW = 0.02
SAMPLE_OVERLAP = 0.01
THRESHOLD = 0.6

filename = '../data/hedepy_test/treatmentTherapistShort.wav'

wav_file, sampling_rate = read_wav_file(filename)

data = wav_file[:, 0]

plt.figure(figsize=(12, 8))
plt.plot(np.arange(len(data)), data)
plt.title("Raw audio signal")
plt.show()


def _calculate_frequencies(audio_data):
    data_freq = np.fft.fftfreq(len(audio_data), 1.0 / sampling_rate)
    data_freq = data_freq[1:]
    return data_freq


def _calculate_energy(audio_data):
    data_ampl = np.abs(np.fft.fft(audio_data))
    data_ampl = data_ampl[1:]
    return data_ampl ** 2


def _connect_energy_with_frequencies(data):
    data_freq = _calculate_frequencies(data)
    data_energy = _calculate_energy(data)

    energy_freq = {}
    for (i, freq) in enumerate(data_freq):
        if abs(freq) not in energy_freq:
            energy_freq[abs(freq)] = data_energy[i] * 2
    return energy_freq


def _sum_energy_in_band(energy_frequencies):
    sum_energy = 0
    for f in energy_frequencies.keys():
        if SPEECH_START_BAND < f < SPEECH_END_BAND:
            sum_energy += energy_frequencies[f]
    return sum_energy


SPEECH_WINDOW = 0.5


def _smooth_speech_detection(detected_voice):
    window = 0.02
    median_window = int(SPEECH_WINDOW / window)
    if median_window % 2 == 0:
        median_window = median_window - 1
    median_energy = signal.medfilt(detected_voice, median_window)
    return median_energy

speech_ratio_list = []
detected_voice = []
mean_data = []

while (SAMPLE_START < (len(data) - SAMPLE_WINDOW)):

    # Select only the region of the data in the window
    SAMPLE_END = int(SAMPLE_START + SAMPLE_WINDOW * sampling_rate)
    if SAMPLE_END >= len(data):
        SAMPLE_END = len(data) - 1

    data_window = data[SAMPLE_START:SAMPLE_END]
    mean_data.append(np.mean(data_window))

    # Full energy
    energy_freq = _connect_energy_with_frequencies(data_window)
    sum_full_energy = sum(energy_freq.values())

    # Voice energy
    sum_voice_energy = _sum_energy_in_band(energy_freq)

    # Speech ratio
    speech_ratio = sum_voice_energy / sum_full_energy
    speech_ratio_list.append(speech_ratio)
    detected_voice.append(speech_ratio > THRESHOLD)

    # Increment
    SAMPLE_START += int(SAMPLE_OVERLAP * sampling_rate)

detected_voice = _smooth_speech_detection(np.array(detected_voice))


plt.figure(figsize=(12,8))
plt.plot(speech_ratio_list)
plt.axhline(THRESHOLD, c='r')
plt.title("Speech ratio list vs. threshold")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(np.array(mean_data), alpha=0.4, label="Not detected")
plt.plot(np.array(detected_voice) * np.array(mean_data), label="Detected")
plt.legend()
plt.title("Detected vs. non-detected region")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(np.array(detected_voice), label="Detected")
plt.legend()
plt.title("Detected vs. non-detected region")
plt.show()
