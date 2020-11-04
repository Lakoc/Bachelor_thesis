import matplotlib.pyplot as plt
import numpy as np

# 15 ms not overlapping frame we get that by dividing by this number
NORMALIZATION_COEFFICIENT = 67


def generate_graph(save_to, plot_name, fig):
    """Handler for graph generations"""
    if save_to:
        plt.savefig(f'{save_to}/{plot_name}.png', dpi=1200)
    fig.show()


def count_percentage_with_values(values):
    """Create function that add counts and percentages to pie charts"""

    def fun(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

    return fun


def plot_wav_with_detection(sampling_rate, wav, vad_segments, cross_talks, save_to):
    """Plot signals with vad and cross talks"""
    fig, axs = plt.subplots(5, sharex=True)
    wav_len = len(wav[:, 0])
    length_in_sec = wav_len / sampling_rate
    time_audio = np.linspace(0, length_in_sec, num=wav_len)
    time_vad = np.linspace(0, length_in_sec, num=len(vad_segments[0]))
    axs[0].set_title('Left channel')
    axs[0].plot(time_audio, wav[:, 0])
    axs[1].set_title('Left channel vad')
    axs[1].plot(time_vad, vad_segments[0])
    axs[2].set_title('Right channel')
    axs[2].plot(time_audio, wav[:, 1])
    axs[3].set_title('Right channel vad')
    axs[3].plot(time_vad, vad_segments[1])
    axs[4].set_title('Cross talks')
    axs[4].set_xlabel("Time (s)")
    axs[4].plot(time_vad, cross_talks)
    generate_graph(save_to, 'speech_vad', fig)


def plot_speech_time_comparison(mean_energies, speech_time, save_to):
    """Plots pie charts to show speech comparison"""
    fig, axs = plt.subplots(2)
    axs[0].set_title('Speech energy')
    axs[0].pie(mean_energies, labels=['Therapist', 'Client'], autopct='%1.2f%%',
               startangle=90)
    axs[1].set_title('Speech time')
    axs[1].pie(speech_time, labels=['Therapist', 'Client'], autopct='%1.2f%%',
               startangle=90)
    generate_graph(save_to, 'energy_and_speech_time', fig)


def create_histogram(data, titles, save_to, hist_name):
    """Plots histogram"""
    graphs = len(titles)
    fig, axes = plt.subplots(nrows=graphs, ncols=1, figsize=(8, 6), sharex=True, sharey=True)
    for index in range(graphs):
        labels, counts = np.unique(data[index] / NORMALIZATION_COEFFICIENT, return_counts=True)
        counts_probability = counts / counts.shape[0]
        axes[index].bar(labels, counts_probability, width=0.1)
        axes[index].grid(axis='y', color='black', linewidth=.5, alpha=.5)
        axes[index].spines["top"].set_visible(False)
        axes[index].spines["right"].set_visible(False)
        axes[index].tick_params(left=False, bottom=False)
        axes[index].set_ylabel("Probability")
        axes[index].set_xlabel("Length (s)")
        axes[index].set_title(titles[index], fontsize=20)
    generate_graph(save_to, hist_name, fig)


def plot_space_lengths(spaces, sentences_lengths, save_to):
    create_histogram(spaces, ['Client response time', 'Therapist response time'], save_to, 'response_time')
    create_histogram(sentences_lengths, ['Therapist sentences length', 'Client sentences length'], save_to,
                     'sentences_length')


def plot_interruptions(interruptions, save_to):
    fig, ax = plt.subplots()
    ax.set_title('Interruptions')
    ax.pie(interruptions, labels=['Therapist', 'Client'], autopct=count_percentage_with_values(interruptions),
           startangle=90)
    generate_graph(save_to, 'interruptions', fig)
