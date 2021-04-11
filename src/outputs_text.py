import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import fftpack, ndimage
from sklearn.mixture import GaussianMixture
from helpers.propagation import forward_backward, segments_filter
from scipy.special import logsumexp
import params


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
    frame1[:] = np.nan
    frame2[:] = np.nan
    frame3[:] = np.nan
    frame4[:] = np.nan

    frame1[0] = 0
    frame1[window_size] = 0
    frame1[1:window_size] = 250
    frame2[window_stride:window_stride + window_size] = 350
    frame2[window_stride] = 0
    frame2[window_stride + window_size] = 0
    frame3[2 * window_stride:2 * window_stride + window_size] = 450
    frame3[2 * window_stride] = 0
    frame3[2 * window_stride + window_size] = 0
    frame4[3 * window_stride:3 * window_stride + window_size] = 550
    frame4[3 * window_stride] = 0
    frame4[3 * window_stride + window_size] = 0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(signal[:3 * window_stride + window_size, 0], label='Signál')
    ax.plot(frame1, label='Okno 1', linewidth=3.0)
    ax.plot(frame2, label='Okno 2', linewidth=3.0)
    ax.plot(frame3, label='Okno 3', linewidth=3.0)
    ax.plot(frame4, label='Okno 4', linewidth=3.0)
    ax.plot(np.zeros(size), color='black')
    ax.legend()
    fig.tight_layout()

    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/segmentation.pdf')


def hamming_rectangular():
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)
    axs[0].set_title('Hammingovo okno')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_ylabel('Amplituda')
    axs[0].set_xlabel('Vzorek')

    axs[0].plot(np.hamming(1000))

    x = np.zeros(1000)
    x[1:999] = 1
    axs[1].set_title('Pravoúhlé okno')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_xlabel('Vzorek')
    axs[1].plot(x)
    fig.tight_layout()

    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/hamming_rectangular.pdf')

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)

    axs[0].set_title('Frekvenční odezva Hammingova okno')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    window = np.hamming(51)
    A = np.fft.fft(window, 2048) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(np.fft.fftshift(A / abs(A).max())))
    axs[0].plot(freq, response)
    axs[0].set_ylabel('Normalizovaná velikost [dB]')
    axs[0].set_xlabel('Normalizovaná frekvence [cyklů / vzorek]')

    axs[1].set_title('Frekvenční odezva pravoúhlého okna')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    window = np.ones(51)
    A = np.fft.fft(window, 2048) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(np.fft.fftshift(A / abs(A).max())))
    axs[1].plot(freq, response)
    axs[1].axis([-0.5, 0.5, -120, 0])
    axs[1].set_xlabel('Normalizovaná frekvence [cyklů / vzorek]')
    fig.tight_layout()
    plt.savefig(
        '../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/hamming_rectangular_spektrum.pdf')


def energy(signal, energy, rmse, Fs):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4), sharex=True)

    # energy = 20 * np.log10(energy)
    # rmse = 20 * np.log10(rmse)
    linear = np.linspace(0, len(signal) / Fs, len(energy))

    axs[0].set_ylabel('Energie')
    axs[0].set_xlabel('Čas [s]')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].plot(linear, energy, label='Energie')

    axs[1].set_ylabel('Střední kvadratická energie')
    axs[1].set_xlabel('Čas [s]')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].plot(linear, rmse, label='Střední kvadratická energie')
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/energy.pdf')
    # plt.show()


def mel_bank(segments):
    # Get mel bounds
    sampling_rate = 8000
    mel_filters = 10
    low_freq_mel = 0
    frequency_resolution = 512
    cepstral_coef_count = 10
    lifter = 22
    # Conversion from hz to mel
    high_freq_mel = (2595 * np.log10(1 + (sampling_rate / 2) / 700))

    # Equally space mel and then convert to Hz
    mel_points = np.linspace(low_freq_mel, high_freq_mel, mel_filters + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))

    # Round to nearest bins on sampled frequency via params.frequency_resolution
    freq_bins = np.floor((frequency_resolution + 1) * hz_points / sampling_rate)

    # Create zeros bank
    mel_bank = np.zeros((mel_filters, int(np.floor(frequency_resolution / 2 + 1))))

    """For each filter get left, center, right sampled frequency -
    The first filter bank will start at the first point, reach its peak at the second point, 
    then return to zero at the 3rd point.
    """
    for m in range(1, mel_filters + 1):
        f_left = int(freq_bins[m - 1])
        f_center = int(freq_bins[m])
        f_right = int(freq_bins[m + 1])

        """ Sample each filter by following equation
            0 if k < f_left                                                  -
            (k − f_left) / (f_center − f_left) if k is before center        - -
            (f_right - k) / (f_right − f_center) if k is after center      -   -
            0 if k > f_right.                                             -     -               
        """
        for k in range(f_left, f_center):
            mel_bank[m - 1, k] = (k - freq_bins[m - 1]) / (freq_bins[m] - freq_bins[m - 1])
        for k in range(f_center, f_right):
            mel_bank[m - 1, k] = (freq_bins[m + 1] - k) / (freq_bins[m + 1] - freq_bins[m])

    fig, ax = plt.subplots(figsize=(7, 4))
    linear = np.linspace(0, sampling_rate // 2, len(mel_bank.T))
    ax.plot(linear, mel_bank.T, label='banka')
    ax.set_ylabel('Velikost')
    ax.set_xlabel('Frekvence [Hz]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/mel_bank.pdf')

    magnitude = np.absolute(np.fft.rfft(segments, frequency_resolution, axis=1))
    power_spectrum = ((1.0 / frequency_resolution) * (magnitude ** 2))

    # Apply banks on power_spectrum

    filter_banks = np.transpose(np.dot(np.transpose(power_spectrum, [0, 2, 1]), mel_bank.T), [0, 2, 1])

    # fig, ax = plt.subplots()
    # ax.plot(mel_bank.transpose(1,0))
    # fig.show()
    # Replace 0 with smallest float for numerical stability and
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # Decorrelate filter banks and compress it
    mfcc = fftpack.dct(filter_banks, axis=1, norm='ortho')
    mfcc_compressed = mfcc[:, 1: (cepstral_coef_count + 1), :]

    # Sinusoidal liftering used to de-emphasize higher MFCCs
    (_, n_coef, _) = mfcc_compressed.shape
    n = np.arange(n_coef)
    lift = 1 + (lifter / 2) * np.sin(np.pi * n / lifter)
    mfcc_compressed = np.transpose(lift * np.transpose(mfcc_compressed, [0, 2, 1]), [0, 2, 1])

    # # Improve SNR by subtracting mean value
    filter_banks -= np.mean(filter_banks, axis=0)
    mfcc_compressed -= np.mean(mfcc_compressed, axis=0)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))

    linear_time = np.linspace(0, params.window_size * 1000, len(segments[0, :, 0]))
    axs[0, 0].set_title('Preemfázovaný vstupní signál')
    axs[0, 0].plot(linear_time, segments[0, :, 0], label='Signál')
    axs[0, 0].set_ylabel('Amplituda')
    axs[0, 0].set_xlabel('Čas [ms]')
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].spines['top'].set_visible(False)

    axs[0, 1].set_title('Fourierovo spektrum a použité banky')
    axs[0, 1].plot(linear, mel_bank.T * np.max(power_spectrum[0, :, 0]), label='Banka', color='black')
    axs[0, 1].plot(linear, power_spectrum[0, :, 0], label='Signál')
    axs[0, 1].set_ylabel('Velikost')
    axs[0, 1].set_xlabel('Frekvence [Hz]')
    axs[0, 1].spines['right'].set_visible(False)
    axs[0, 1].spines['top'].set_visible(False)

    axs[1, 0].set_title('Logaritmus energií na výstupech filtrů')
    axs[1, 0].plot(filter_banks[0, :, 0], label='Energie')
    axs[1, 0].set_ylabel('Velikost')
    axs[1, 0].set_xlabel('Koeficient')
    axs[1, 0].spines['right'].set_visible(False)
    axs[1, 0].spines['top'].set_visible(False)

    axs[1, 1].set_title('Mel-frekvenční cepstralní koefienty')
    axs[1, 1].plot(mfcc[0, :, 0], label='Energie')
    axs[1, 1].set_ylabel('Velikost')
    axs[1, 1].set_xlabel('Koeficient')
    axs[1, 1].spines['right'].set_visible(False)
    axs[1, 1].spines['top'].set_visible(False)
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/mfcc.pdf')
    # plt.show()


def VAD_sample(signal, Fs):
    fig, ax = plt.subplots(figsize=(7, 4))
    sig = signal[40000:100000, 0]
    window = np.zeros(len(sig))
    window[13000:37500] = 1
    window[45000:58000] = 1
    linear = np.linspace(0, 600000 / Fs, 60000)
    ax.plot(linear, sig / np.max(signal) * 10, label='Signál')
    ax.plot(linear, window, label='Řečová aktivita', color='black', linewidth=3.0)
    ax.set_ylabel('Velikost')
    ax.set_xlabel('Čas [s]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/signal_vad.pdf')
    # plt.show()


def VAD_sample_smooth(signal, Fs):
    fig, ax = plt.subplots(figsize=(7, 4))
    sig = signal[40000:500000, 0]
    window = np.zeros(len(sig))
    window[13000:125000] = 1
    window[160000:195000] = 1
    window[243000:390000] = 1
    window[440000:] = 1
    window[-1] = 0

    linear = np.linspace(0, (500000 - 40000) / Fs, (500000 - 40000))

    ax.plot(linear, sig / np.max(signal) * 10, label='Signál')
    ax.plot(linear, window, label='Vyhlazená detekce řečové aktivity', color='black', linewidth=3.0)
    ax.set_ylabel('Velikost')
    ax.set_xlabel('Čas [s]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/signal_vad_smooth.pdf')
    # plt.show()


def VAD_threshold(signal, Fs):
    fig, ax = plt.subplots(figsize=(7, 4))
    linear = np.linspace(0, (100000 - 40000) / Fs, (100000 - 40000))

    sig = signal[40000:100000, 0]
    window = np.zeros(len(sig))
    window[13200:36300] = 1
    window[45500:58000] = 1
    threshold = np.zeros(len(sig))
    threshold += 0.05
    ax.plot(linear, sig / np.max(signal) * 10, label='Signál')
    ax.plot(linear, window, label='Řečová aktivita', color='black', linewidth=3.0)
    ax.plot(linear, threshold, label='Práh', color='C1')
    ax.set_ylabel('Velikost')
    ax.set_xlabel('Čas [s]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/signal_vad_threshold.pdf')
    # plt.show()


def VAD_adaptive_threshold(energy_segments):
    energy_segments = energy_segments[0:1000, :]
    med_filter = 0.1  # 250 ms
    window_stride = 0.01  # 10ms

    # Initial energies
    e_min = np.array([np.finfo(float).max, np.finfo(float).max])
    e_max = np.array([np.finfo(float).min, np.finfo(float).min])
    e_max_ref = e_max

    # Scaling factors
    j = np.array([1, 1], dtype=float)
    k = np.array([1, 1], dtype=float)

    vad = np.zeros(energy_segments.shape, dtype="i1")
    threshold_arr = np.zeros(energy_segments.shape)
    e_min_arr = np.zeros(energy_segments.shape)
    e_max_arr = np.zeros(energy_segments.shape)

    # Lambda min in case of too fast scaling
    lam_min = np.array([0.95, 0.95])
    for index, segment in enumerate(energy_segments):
        # Actualize e_min and e_max
        e_min = e_min * j
        j = np.where(segment < e_min, 1, j)
        j *= 1.0001
        e_min = np.minimum(e_min, segment)

        e_max_ref = e_max_ref * k
        k = np.where(segment > e_max_ref, 1, k)
        k *= 0.999
        e_max = np.maximum(e_max, segment)
        e_max_ref = np.maximum(e_max_ref, segment)

        # Actualize lambda and calculate threshold
        lam = np.maximum(lam_min, (e_max_ref - e_min) / e_max_ref)
        threshold = (1 - lam) * e_max + lam * e_min

        threshold_arr[index] = threshold
        e_max_arr[index] = e_max_ref
        e_min_arr[index] = e_min

        # 1 - speech, 0 - non speech
        vad[index] = segment > threshold

    # apply median filter to remove unwanted
    vad = ndimage.median_filter(vad, int(round(med_filter / window_stride) * 4))

    time = np.linspace(0, 10, 1000)
    fig, axs = plt.subplots(nrows=2, figsize=(7, 4), sharex=True)
    axs[0].plot(time, e_min_arr[:, 0], label='Minimální energie')
    axs[0].plot(time, e_max_arr[:, 0], label='Maximální energie')

    axs[0].set_ylabel('Velikost')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].legend()

    axs[1].plot(time, energy_segments[:, 0], label='Energie')
    axs[1].plot(time, threshold_arr[:, 0], label='Práh')
    axs[1].plot(time, vad[:, 0] * np.max(energy_segments[:, 0]), label='Řečová aktivita', color='black', linewidth=3.0)
    axs[1].set_ylabel('Velikost')
    axs[1].set_xlabel('Čas [s]')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].legend()
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/vad_adaptive.pdf')
    # plt.show()


def VAD_gmm(energy, signal, Fs):
    """Estimate vad using gaussian mixture model"""
    # Initialization of model
    means = np.array([np.min(energy), (np.max(energy) + np.min(energy)) / 2, np.max(energy)])[:, np.newaxis]
    weights = np.array([1 / 3, 1 / 3, 1 / 3])

    gmm = GaussianMixture(n_components=3, max_iter=300, tol=1e-5, weights_init=weights, covariance_type='spherical',
                          means_init=means)

    # Fit and predict posterior probabilities for each channel
    gmm.fit(energy[:, 0][:, np.newaxis])
    likelihoods1 = gmm.predict_proba(energy[:, 0][:, np.newaxis])

    # Init propagation matrix for sil likelihood smoothing
    vad_min_speech_dur = params.vad_min_speech_dur
    n_tokens = likelihoods1.shape[1]
    sp = np.ones(n_tokens) / n_tokens
    ip = np.zeros(vad_min_speech_dur * n_tokens)
    ip[::vad_min_speech_dur] = sp
    tr = np.array(params.transition_matrix)

    # HMM backward, forward propagation, and threshold silence component
    likelihood_propagated1, _, _, _ = forward_backward(likelihoods1, tr, ip)
    vad1 = likelihood_propagated1[:, 0] < params.min_silence_likelihood

    # Smooth activity segments and create arr containing both speakers
    vad1 = segments_filter(segments_filter(vad1, params.filter_size_non, 1), params.filter_size_active, 0)

    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))
    height = np.max(signal[0:Fs * 20:, 0])
    time = np.linspace(0, 20, 2000)
    time1 = np.linspace(0, 20, Fs * 20)
    threshold = np.full(2000, params.min_silence_likelihood)

    axs[0].plot(time, likelihoods1[0:2000, 0], label='Ticho')
    axs[0].plot(time, likelihoods1[0:2000, 1], label='Šum / tichá řeč')
    axs[0].plot(time, likelihoods1[0:2000, 2], label='Řeč')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].legend()
    axs[0].set_ylabel('Pravděpodobnost')

    axs[1].plot(time, likelihood_propagated1[0:2000, 0], label='Pravděpodobnost ticha')
    axs[1].plot(time, threshold, label='Práh')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylabel('Velikost')
    axs[1].legend()

    axs[2].plot(time, vad1[0:2000], label='Řečová aktivita', color='black', linewidth=3)
    axs[2].plot(time1, signal[0:20 * Fs, 0] / height, label='Normalizovaný signál')
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].set_ylabel('Velikost')
    axs[2].legend()
    axs[2].set_xlabel('Čas [s]')
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/vad_gmm.pdf')

def diarization(signal, Fs):
    fig, ax = plt.subplots(figsize=(7, 4))
    sig = signal[40000:100000, 0]
    time = np.linspace(0, 60000 / Fs, 60000)
    window1 = np.zeros(len(sig))
    window2 = np.zeros(len(sig))
    window1[13000:37500] = 1
    window2[45000:58000] = 1
    ax.plot(time, sig / np.max(signal) * 10, label='Signál')
    ax.plot(time, window1, label='Řečová aktivita terapeuta', linewidth=3.0)
    ax.plot(time, window2, label='Řečová aktivita klienta', linewidth=3.0)
    ax.set_ylabel('Velikost')
    ax.set_xlabel('Čas [s]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/diarization.pdf')
    # plt.show()

def cross_talk(signal, Fs):

    fig, axs = plt.subplots(nrows=2, figsize=(7, 4), sharey=True, sharex=True)
    sig1 = signal[0:100000, 0]
    sig2 = signal[0:100000, 1]
    time = np.linspace(0, 100000 / Fs, 100000)

    axs[0].plot(time, sig1 , label='Kanál 1', color='C0')
    axs[0].set_ylabel('Velikost')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].legend()

    axs[1].plot(time, sig2, label='Kanál 2', color='C1')
    axs[1].set_ylabel('Velikost')
    axs[1].set_xlabel('Čas [s]')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].legend()
    fig.tight_layout()
    plt.savefig('../thesis_text/Anal-za-audio-hovoru-mezi-dv-ma-astn-ky/obrazky-figures/cross_talk.pdf')
    # plt.show()