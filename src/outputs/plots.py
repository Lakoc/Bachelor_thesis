import numpy as np

import params
from helpers.propagation import mean_filter
from matplotlib import pyplot as plt

from helpers.plot_helpers import gauge, generate_graph, value_to_angle


def plot_speech_time_comparison(speech_time, path):
    """Plots pie charts to show speech comparison"""
    fig, ax = plt.subplots(1)
    ax.pie(speech_time, labels=['Terapeut', 'Klient'], autopct='%1.2f%%',
           startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'}, colors=['C0', 'C1'],
           wedgeprops={'alpha': 0.5})

    generate_graph(path, fig)


def plot_speech_time_comparison_others(speech_time, speech_time_mean, path):
    """Plots pie charts to show speech comparison"""

    def tickers_format(value):
        return f'{value * 100:.1f}%'

    def val_format(value):
        return f'{value * 100:.3f}%'

    gauge(labels=['Nízký', 'Normální', 'Vysoký'], colors=['C0', 'C2', 'C1'],
          min_val=speech_time_mean[0] - 5 * speech_time_mean[1], max_val=speech_time_mean[0] + 5 * speech_time_mean[1],
          value=speech_time, tickers_format=tickers_format, val_format=val_format, path=path)


def plot_volume_changes(energy_over_segments, interruptions, size, path):
    """Plot volume changes over time for better finding of most energetic segments"""
    fig, ax = plt.subplots(figsize=(16, 4))
    silence = np.where(energy_over_segments == 0, 1, 0)
    energy_over_segments = mean_filter(energy_over_segments, 1000)
    energy_over_segments = np.where(silence, 0, energy_over_segments)
    energy_over_segments /= np.max(energy_over_segments)
    interruptions_arr = np.empty(shape=energy_over_segments.shape[0])
    interruptions_arr[:] = np.nan
    for interruption in interruptions:
        if interruption[1] - interruption[0] > 50:
            interruptions_arr[interruption[0] - 1] = 0
            interruptions_arr[interruption[0]: interruption[1]] = 1
            interruptions_arr[interruption[1]] = 0

    silence = np.where(silence, 0, np.nan)

    time = np.linspace(0, size / 60, energy_over_segments.shape[0])
    ax.plot(time, energy_over_segments, label='Hlasitost aktivních segmentů')
    ax.plot(time, interruptions_arr, label='Skoky do řeči druhému mluvčímu')
    ax.plot(time, silence, label='Neaktivní segmenty')
    ax.set_ylabel('Normalizovaná hlasitost')
    ax.set_xlabel('Čas [m]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    generate_graph(path, fig)


def plot_cross_talk_histogram(cross_talks, cross_talks_overall, bins_count, path):
    """Plot cross talk histogram as linear plot for
    better visualisation of dependence between current and overall stats"""
    fig, ax = plt.subplots()
    cross_talks_len_current = (cross_talks[:, 1] - cross_talks[:, 0]).astype(np.float)
    cross_talks_len_current *= params.window_stride

    cross_talks_len_overall = (cross_talks_overall[:, 1] - cross_talks_overall[:, 0]).astype(np.float)
    cross_talks_len_overall *= params.window_stride
    bins = np.linspace(0, max(np.max(cross_talks_len_current), np.max(cross_talks_len_overall)), bins_count)
    labels = [f'{bins[i]:.1f} - {bins[i + 1]:.1f}' for i in range(len(bins) - 1)]

    current_counts, _ = np.histogram(cross_talks_len_current, bins=bins)
    ax.plot(labels, current_counts, label='Aktuální sezení')

    overall_counts, _ = np.histogram(cross_talks_len_overall, bins=bins)
    ax.plot(labels, overall_counts, label='Průměr mezi sezeními')

    ax.grid(axis='y', color='black', linewidth=.5, alpha=.5)
    plt.xticks(rotation=45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel("Výskyt")
    ax.legend()
    ax.set_xlabel("Délka [s]")
    generate_graph(path, fig)


def plot_reaction_time_comparison(reactions, reaction_time_mean, path):
    """Plot reaction time comparison"""
    reaction_time = np.mean(reactions[:, 1] - reactions[:, 0]) * params.window_stride

    gauge(['Nízká', 'Normální', 'Vysoká'], ['C0', 'C2', 'C1'], value=reaction_time,
          min_val=reaction_time_mean[0] - 5 * reaction_time_mean[1],
          max_val=reaction_time_mean[0] + 5 * reaction_time_mean[1], tickers_format=lambda l: f'{l:.2f}',
          val_format=lambda l: f'{l:.3f}', path=path)


def plot_speech_bounds_len(speech_bounds, path):
    """Plots speech lengths"""
    fig, ax = plt.subplots()
    speech_bounds_len = (speech_bounds[:, 1] - speech_bounds[:, 0]).astype(np.float)
    speech_bounds_len *= params.window_stride

    bins = np.array([0, 2, 5, 10, 15, 20, 30, np.iinfo(np.int16).max])
    labels = [f'{bins[i]:} - {bins[i + 1]:}' for i in range(len(bins) - 2)]
    labels.append('> 30')

    ind = np.arange(len(labels))

    width = 0.3
    current_counts, _ = np.histogram(speech_bounds_len, bins=bins)
    current_counts = (current_counts / np.sum(current_counts)) * 100
    ax.bar(ind, current_counts, width, label='Aktuální sezení')
    ax.bar(ind + width + 0.05, current_counts, width, label='Průměr')
    plt.xticks(ind + width / 2 + 0.025, labels, rotation=45)

    ax.grid(axis='y', color='black', linewidth=.5, alpha=.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel("Výskyt [%]")
    ax.legend()
    ax.set_xlabel("Délka [s]")
    generate_graph(path, fig)


def plot_speed(speed, intervals, size, path):
    """Plots speech speed"""
    speed = np.array([speed[segment]['words_per_segment'] for segment in speed if speed[segment]['words_per_segment']])

    speed_percentiles = np.empty(intervals)
    interval = int(speed.shape[0] / intervals)
    for i in range(0, intervals):
        speed_percentiles[i] = np.mean(speed[interval * i:interval * (i + 1)])

    time = np.linspace(0, size / 60, intervals + 1)
    speed_percentiles = (speed_percentiles / params.window_stride) * 60
    speed_overall = np.empty(speed_percentiles.shape[0])
    speed_overall[:] = 124
    speed_mean = np.empty(speed_percentiles.shape[0])
    speed_mean[:] = np.mean(speed_percentiles)
    fig, ax = plt.subplots()

    ax.plot(speed_percentiles, label='Rychlost řeči')
    ax.plot(speed_overall, alpha=.4, label='Průměrná rychlost řeči mezi nahrávkami')
    ax.grid(axis='y', color='black', linewidth=.5, alpha=.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.legend()
    ax.set_ylabel("Délka [slov/minuta]")
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    generate_graph(path, fig)


def plot_hesitations(hesitations, signal_len, hesitations_mean, path):
    """Plot hesitations count comparison"""
    hesitations = hesitations.shape[0] / signal_len * 60

    gauge(['Nízký', 'Normální', 'Vysoký'], ['C0', 'C2', 'C1'], value=hesitations,
          min_val=hesitations_mean[0] - 5 * hesitations_mean[1], max_val=hesitations_mean[0] + 5 * hesitations_mean[1],
          path=path, tickers_format=lambda l: f'{l:.1f}', val_format=lambda l: f'{l:.2f}')
