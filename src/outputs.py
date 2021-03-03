import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from math import ceil

# 15 ms not overlapping frame -> corresponding to 1000 // 15
NORMALIZATION_COEFFICIENT = 67


def generate_graph(save_to, plot_name, fig):
    """Handler for graph generations"""
    fig.tight_layout()
    if save_to:
        plt.savefig(f'{save_to}/{plot_name}.png')
    else:
        fig.show()


def count_percentage_with_values(values):
    """Create function that add counts and percentages to pie charts"""

    def fun(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

    return fun


def plot_energy_with_wav(track, energies_per_segment):
    """Plots wav and energy of signal"""
    fig, axs = plt.subplots(2)
    axs[0].set_title('Vaw')
    track_shape = track.shape
    axs[0].plot(np.reshape(track, track_shape[0] * track_shape[1]))
    axs[1].set_title('Energy')
    axs[1].plot(energies_per_segment)
    fig.tight_layout()
    fig.show()


def plot_wav_with_detection(sampling_rate, wav, vad_segments, joined_vad, cross_talks, save_to):
    """Plot signals with vad and cross talks"""
    fig, axs = plt.subplots(7, sharex=True, figsize=(14, 12))
    wav_len = len(wav[:, 0])
    length_in_sec = wav_len / sampling_rate
    time_audio = np.linspace(0, length_in_sec, num=wav_len)
    time_vad = np.linspace(0, length_in_sec, num=len(vad_segments[0]))

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(bottom=False)

    axs[0].set_title('Therapist')
    axs[0].plot(time_audio, wav[:, 0], color='C1')
    axs[1].set_title('Therapist vad')
    axs[1].plot(time_vad, vad_segments[0], color='C1')
    axs[2].set_title('Therapist vad joined')
    axs[2].plot(time_vad, joined_vad[0], color='C1')
    axs[3].set_title('Client')
    axs[3].plot(time_audio, wav[:, 1], color='C1')
    axs[4].set_title('Client vad')
    axs[4].plot(time_vad, vad_segments[1], color='C1')
    axs[5].set_title('Client vad joined')
    axs[5].plot(time_vad, joined_vad[1], color='C1')
    axs[6].set_title('Cross talks')
    axs[6].set_xlabel("Time [s]")
    axs[6].tick_params(bottom=True)
    axs[6].plot(time_vad, cross_talks, color='C1')
    generate_graph(save_to, 'speech_vad', fig)

    # fig, ax = plt.subplots(1, figsize=(12, 3))
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.tick_params(bottom=False, left=False)
    # ax.plot(time_audio, wav[:, 0], color='k')
    # generate_graph(save_to, 'speech_Therapist', fig)
    #
    # fig, ax = plt.subplots(1, figsize=(12, 3))
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.tick_params(bottom=False, left=False)
    # ax.plot(time_audio, wav[:, 1], color='k')
    # generate_graph(save_to, 'speech_Client', fig)


def plot_speech_time_comparison(mean_energies, speech_time, save_to):
    """Plots pie charts to show speech comparison"""
    fig, ax = plt.subplots(1)
    # axs[0].set_title('Speech energy')
    # axs[0].pie(mean_energies, labels=['Therapist', 'Client'], autopct='%1.2f%%',
    #            startangle=90)
    ax.set_title('Speech time', fontsize=16)
    ax.pie(speech_time, labels=['Therapist', 'Client'], autopct='%1.2f%%',
           startangle=90)
    generate_graph(save_to, 'energy_and_speech_time', fig)


def create_time_histogram(data, titles, save_to, hist_name, bins):
    """Create histogram with normalized time values"""
    graphs = len(titles)
    fig, axes = plt.subplots(nrows=graphs, ncols=1, figsize=(8, 6), sharex=True, sharey=True)
    for index in range(graphs):
        data_normalized = data[index] / NORMALIZATION_COEFFICIENT
        counts, _ = np.histogram(data_normalized, bins=bins)
        labels = [f'{bins[i]:.1f} - {bins[i + 1]:.1f}' for i in range(len(bins) - 1)]
        count_sum = np.sum(counts)
        counts_probability = counts / count_sum
        axes[index].bar(labels, counts_probability, width=0.5, color='C1')
        axes[index].grid(axis='y', color='black', linewidth=.5, alpha=.5)
        axes[index].set_axisbelow(True)
        axes[index].spines["top"].set_visible(False)
        axes[index].spines["right"].set_visible(False)
        axes[index].tick_params(left=False, bottom=False)
        axes[index].set_ylabel("Probability")
        axes[index].set_title(titles[index], fontsize=20)
    axes[graphs - 1].set_xlabel("Length [s]")
    generate_graph(save_to, hist_name, fig)


def plot_responses_lengths(responses, sentences_lengths, save_to):
    """Plot sentences lengths and responses time"""
    create_time_histogram(responses, ['Therapist reaction time', 'Client reaction time'],
                          save_to, 'response_time', np.arange(0, 3.1, 0.3))
    max_val = max(sentences_lengths[0].max() / NORMALIZATION_COEFFICIENT,
                  sentences_lengths[1].max() / NORMALIZATION_COEFFICIENT)
    step = ceil(max_val / 10)
    max_val += step
    create_time_histogram(sentences_lengths,
                          ['Therapist sentences length', 'Client sentences length'], save_to,
                          'sentences_length', np.arange(0, max_val, step))


def plot_interruptions(interruptions, save_to):
    """Create pie chart with interruptions"""
    fig, ax = plt.subplots()
    ax.set_title('Interruptions', fontsize=16)
    ax.pie(interruptions, labels=['Therapist', 'Client'], autopct=count_percentage_with_values(interruptions),
           startangle=90)
    generate_graph(save_to, 'interruptions', fig)


def plot_monolog_hesitations_histogram(hesitations, save_to):
    """Create histogram of hesitations"""
    create_time_histogram(hesitations, ['Therapist monolog hesitations', 'Client monolog hesitations'],
                          save_to, 'monolog_hesitations', np.arange(0, 5.1, 0.5))


def plot_lpc_ftt(sig, sampling_frequency, lpc, coefficients):
    """Plots wav and lpc"""
    # Read the wav file (mono)
    # Plot the signal read from wav file
    sampling_half = sampling_frequency // 2
    weight = np.linspace(start=(2 * -1j * np.pi) / sampling_half, stop=(2 * coefficients * -1j * np.pi) / sampling_half,
                         num=coefficients)
    frequencies = np.arange(0, sampling_half)
    log_spectrum = np.empty((len(frequencies)))
    for index, freq in enumerate(frequencies):
        # print(weight * freq)
        # print(lpc @ np.exp(weight * freq))
        log_spectrum[index] = 1 / np.power(np.absolute(1 - lpc @ np.exp(weight * freq)), 2)
    # log_spectrum = np.empty(sampling_half)
    # for i in range(0, sampling_half):
    #     log_spectrum[i] = 10 * np.log10(
    #         1 / np.power(np.absolute(1 - lpc @ weight), 2))

    # The FFT of the signal
    sig_fft = fftpack.fft(sig)
    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft) ** 2
    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(sig.size, d=1 / sampling_frequency)

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(sample_freq, power)
    axes[1].plot(log_spectrum)
    # ax.set_xlabel('Frequency in Hertz [Hz]')
    # ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    axes[0].set_xlim(0, 5000)
    fig.show()
