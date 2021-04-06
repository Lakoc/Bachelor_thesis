import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def preemphasis(signal, pre_signal, Fs):
    s = signal[:, 0]
    s1 = pre_signal[:, 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.magnitude_spectrum(s, Fs=Fs)
    ax.set_ylabel('Relativní četnost')
    ax.set_xlabel('Frekvence [Hz]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/signal_spectrum.pdf')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.magnitude_spectrum(s1, Fs=Fs)
    ax.set_ylabel('Relativní četnost')
    ax.set_xlabel('Frekvence [Hz]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/signal_spectrum_pre.pdf')
    plt.show()


def segmentation(signal, sampling_rate, window_size, window_overlap):
    window_size = int(round(sampling_rate * window_size))
    window_overlap = int(round(sampling_rate * window_overlap))
    window_stride = window_size - window_overlap

    size = 3 * window_stride + window_size + 1
    frame1 = np.zeros(size)
    frame2 = np.zeros(size)
    frame3 = np.zeros(size)
    frame4 = np.zeros(size)

    frame1[1:window_size] = 250
    frame2[window_stride:window_stride + window_size] = 350
    frame3[2 * window_stride:2 * window_stride + window_size] = 450
    frame4[3 * window_stride:3 * window_stride + window_size] = 550

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(signal[:3 * window_stride + window_size, 0], label='Signál')
    ax.plot(frame1, color='black', label='Okno')
    ax.plot(frame2, color='black')
    ax.plot(frame3, color='black')
    ax.plot(frame4, color='black')
    ax.legend()
    fig.tight_layout()

    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/segmentation.pdf')

def hamming_rectangular():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(np.hamming(1000))
    fig.tight_layout()

    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/hamming.pdf')

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    window = np.hamming(100)
    A = np.fft.fft(window, 8000) / 50
    mag = np.abs(np.fft.fftshift(A))
    response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)
    plt.plot(response)
    ax.set_ylabel('Velikost [dB]')
    ax.set_xlabel('Frekvence [Hz]')
    fig.tight_layout()

    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/hamming_spektrum.pdf')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    x = np.zeros(1000)
    x[1:999] = 1
    ax.plot(x)

    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/rectangular.pdf')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    window = np.ones(100)
    A = np.fft.fft(window, 8000) / 50
    mag = np.abs(np.fft.fftshift(A))
    response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)
    ax.set_ylabel('Velikost [dB]')
    ax.set_xlabel('Frekvence [Hz]')
    plt.plot(response)

    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/rectangular_spectrum.pdf')

def energy(signal, energy, rmse):
    fig, axs = plt.subplots(nrows=2,figsize=(7,6), sharex=True)


    linear = np.linspace(0,len(signal), len(energy))

    axs[0].set_ylabel('Energie')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].plot(linear, energy, label='Energie')

    axs[1].set_ylabel('Střední kvadratická energie')
    axs[1].set_xlabel('Čas [n]')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].plot(linear, rmse, label='Střední kvadratická energie')
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/energy.pdf')

def mel_bank():
    pass