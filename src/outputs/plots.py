import numpy as np
import params
from helpers.propagation import mean_filter
from matplotlib import pyplot as plt
from helpers.plot_helpers import gauge, generate_graph


def speech_time_comparison(speech_time, path):
    """Plots pie charts to show speech comparison"""
    fig, ax = plt.subplots(1)
    ax.pie(speech_time, labels=['Terapeut', 'Klient'], autopct='%1.2f%%',
           startangle=90, textprops={'fontsize': 16}, colors=['C0', 'C1'],
           wedgeprops={'alpha': 0.5})
    generate_graph(path, fig)


def silence_ratio(speech, path):
    """Plots pie charts to show silence/speech comparison"""
    fig, ax = plt.subplots(1)
    ax.pie([speech, 1 - speech], labels=['Řeč', 'Ticho'], autopct='%1.2f%%',
           startangle=90, textprops={'fontsize': 16}, colors=['C0', 'C1'],
           wedgeprops={'alpha': 0.5})
    generate_graph(path, fig)


def speech_time_comparison_others(speech_time, speech_time_mean, path):
    """Plots pie charts to show speech comparison"""

    def tickers_format(value):
        return f'{value * 100:.1f}%'

    def val_format(value):
        return f'{value * 100:.3f}%'

    fig, ax = gauge(labels=['Nízký', 'Normální', 'Vysoký'], colors=['C0', 'C2', 'C1'],
                    min_val=speech_time_mean[0] - 5 * speech_time_mean[1],
                    max_val=speech_time_mean[0] + 5 * speech_time_mean[1],
                    value=speech_time, tickers_format=tickers_format, val_format=val_format)
    generate_graph(path, fig)


def volume_changes(energy_over_segments, interruptions, size, path):
    """Plot volume changes over time for better finding of most energetic segments"""
    speech = energy_over_segments != 0
    energy_over_segments = mean_filter(energy_over_segments, 150)
    energy_over_segments /= np.max(energy_over_segments)
    speech = np.where(speech, energy_over_segments, 0)
    interruptions_arr = np.empty(shape=energy_over_segments.shape[0])
    interruptions_arr[:] = np.nan
    for interruption in interruptions:
        if interruption[1] - interruption[0] > 50:
            interruptions_arr[interruption[0] - 1] = 0
            interruptions_arr[interruption[0]: interruption[1]] = 1
            interruptions_arr[interruption[1] - 1] = 0

    time = np.linspace(0, size / 60, energy_over_segments.shape[0])
    zeros = np.zeros_like(speech)

    fig, ax = plt.subplots(figsize=(24, 4))
    ax.fill_between(time, zeros, energy_over_segments, color='C0', alpha=0.5, label='Hlasitost')
    ax.plot(time,  interruptions_arr, color='C1', label='Skoky do řeči druhému mluvčímu')
    ax.set_ylabel('Normalizovaná velikost', fontsize=16)
    ax.set_xlabel('Čas [min]', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(fontsize=16)
    generate_graph(path, fig)


def interruptions_histogram(interruptions_counts, interruptions_overall_counts, path):
    """Plot cross talk histogram as linear plot for
    better visualisation of dependence between current and overall stats"""

    bins = np.array([0, 0.2, 0.5, 0.8, 1, np.iinfo(np.int16).max])
    labels = [f'{bins[i]:} - {bins[i + 1]} s' for i in range(len(bins) - 2)]
    labels.append('> 1 s')
    ind = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots()
    ax.bar(ind, interruptions_counts, width, label='Aktuální sezení')
    ax.bar(ind + width + 0.05, interruptions_overall_counts, width, label='Průměr')
    plt.xticks(ind + width / 2 + 0.025, labels, rotation=45)
    plt.xticks(rotation=45)
    ax.grid(axis='y', color='black', linewidth=.5, alpha=.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel("Výskyt", fontsize=16)
    ax.legend(fontsize=16)
    ax.set_xlabel("Délka [s]", fontsize=16)
    generate_graph(path, fig)


def reaction_time_comparison(reactions, reaction_time_mean, path):
    """Plot reaction time comparison"""
    if reactions.shape[0] < 1:
        reactions = np.array([[0, 0]])
    reaction_time = np.mean(reactions[:, 1] - reactions[:, 0]) * params.window_stride

    fig, ax = gauge(['Nízká', 'Normální', 'Vysoká'], ['C0', 'C2', 'C1'], value=reaction_time,
                    min_val=reaction_time_mean[0] - 5 * reaction_time_mean[1],
                    max_val=reaction_time_mean[0] + 5 * reaction_time_mean[1], tickers_format=lambda l: f'{l:.2f}',
                    val_format=lambda l: f'{l:.3f} s')
    generate_graph(path, fig)


def speech_bounds_lengths(current_counts, overall_counts, path):
    """Plots speech lengths"""
    bins = np.array([0, 2, 5, 10, 15, 20, 30, np.iinfo(np.int16).max])
    labels = [f'{bins[i]:} - {bins[i + 1]:}' for i in range(len(bins) - 2)]
    labels.append('> 30')
    ind = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots()
    plt.xticks(ind + width / 2 + 0.025, labels, rotation=45)
    ax.bar(ind, current_counts, width, label='Aktuální sezení')
    ax.bar(ind + width + 0.05, overall_counts, width, label='Průměr')
    ax.grid(axis='y', color='black', linewidth=.5, alpha=.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel("Výskyt [%]", fontsize=16)
    ax.legend(fontsize=16)
    ax.set_xlabel("Délka [s]", fontsize=16)
    generate_graph(path, fig)


def speed_changes(speed, intervals, size, path):
    """Plots speech speed"""
    speed_per_parts = [[] for _ in range(intervals)]
    interval_size = (size / intervals) / params.window_stride
    for key in speed.keys():
        index = int(key[1] / interval_size)
        if index == intervals:
            if key[1] * params.window_stride - 10 < size:
                index -= 1
            else:
                raise ValueError('Transcriptions and signal not aligned')
        speed_per_parts[index].append(speed[key]['words_per_segment'])

    speed_per_parts = np.array([sum(part) / len(part) if len(part) > 0 else np.nan for part in speed_per_parts])
    time = np.linspace(0, size / 60, intervals)
    speed_per_parts = (speed_per_parts / params.window_stride) * 60
    speed_mean = np.empty(speed_per_parts.shape[0])
    speed_mean[:] = np.mean(speed_per_parts)

    fig, ax = plt.subplots()
    ax.plot(time, speed_per_parts, label='Rychlost řeči')
    ax.plot(time, speed_mean, alpha=.4, label='Průměrná rychlost řeči')
    ax.grid(axis='y', color='black', linewidth=.5, alpha=.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.legend(fontsize=16)
    ax.set_ylabel("Rychlost [slov/minuta]", fontsize=16)
    ax.set_xlabel("Čas [min]", fontsize=16)
    generate_graph(path, fig)


def hesitations_difference(hesitations, signal_len, hesitations_mean, path):
    """Plot hesitations count comparison"""
    hesitations = hesitations.shape[0] / signal_len * 60

    fig, ax = gauge(['Nízký', 'Normální', 'Vysoký'], ['C0', 'C2', 'C1'], value=hesitations,
                    min_val=hesitations_mean[0] - 5 * hesitations_mean[1],
                    max_val=hesitations_mean[0] + 5 * hesitations_mean[1],
                    tickers_format=lambda l: f'{l:.1f}', val_format=lambda l: f'{l:.2f} váhání za minutu')
    generate_graph(path, fig)


def fills_difference(fills, signal_len, fills_mean, path):
    """Plot hesitations count comparison"""
    fills = fills / (signal_len / 60)

    fig, ax = gauge(['Nízký', 'Normální', 'Vysoký'], ['C0', 'C2', 'C1'], value=fills,
                    min_val=fills_mean[0] - 5 * fills_mean[1],
                    max_val=fills_mean[0] + 5 * fills_mean[1],
                    tickers_format=lambda l: f'{l:.1f}', val_format=lambda l: f'{l:.2f}x za minutu')
    generate_graph(path, fig)


def client_mood(mood, path):
    labels = ['Hněv', 'Strach', 'Štěstí', 'Smutek', 'Překvapení']
    smoothing_kernel = np.ones(10) / 10
    positive = np.zeros(mood.shape[0])

    fig, axs = plt.subplots(nrows=2, figsize=(16, 8), sharex=True)
    for column, label in zip(mood.T, labels):
        column = np.convolve(column, smoothing_kernel, mode='same')
        if label in ['Štěstí', 'Překvapení']:
            positive += column
        else:
            positive -= column
        axs[0].plot(column, label=label, alpha=0.5)
    axs[0].legend(fontsize=16)
    axs[0].set_ylabel('Pravděpodobnost', fontsize=16)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].grid(axis='y', color='black', linewidth=.3, alpha=.5)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    positive_over = np.where(positive > 0, positive, 0)
    positive_under = np.where(positive < 0, positive, 0)
    zeros = np.zeros(positive_over.shape[0])
    time = np.arange(0, zeros.shape[0])

    axs[1].fill_between(time, zeros, positive_over, color='C2', alpha=0.5, label='Dobrá nálada')
    axs[1].fill_between(time, zeros, positive_under, color='C1', alpha=0.5, label='Špatná nálada')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[1].set_ylabel('Celková nálada', fontsize=16)
    axs[1].legend(fontsize=16)
    axs[1].grid(axis='y', color='black', linewidth=.3, alpha=.5)
    generate_graph(path, fig)
