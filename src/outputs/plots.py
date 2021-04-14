import numpy as np
from matplotlib import pyplot as plt

from helpers.plot_helpers import gauge, generate_graph


def plot_speech_time_comparison(speech_time, path):
    """Plots pie charts to show speech comparison"""
    fig, ax = plt.subplots(1)
    ax.pie(speech_time, labels=['Terapeut', 'Klient'], autopct='%1.2f%%',
           startangle=90)
    generate_graph(path, fig)


def plot_speech_time_comparison_others(speech_time, speech_time_mean, path):
    """Plots pie charts to show speech comparison"""
    center = speech_time_mean[0]
    var_center = speech_time_mean[1]
    angle = (center - speech_time) / var_center * 5 / 9

    angle = min(max(angle, -1), 1)
    angle += 1
    angle *= 90
    gauge(['Nízký', 'Normální', 'Vysoký'], ['C0', 'C2', 'C1'], angle, path)


def plot_volume_changes(energy_over_segments, size, path):
    fig, ax = plt.subplots(figsize=(16, 4))
    time = np.linspace(0, size, energy_over_segments.shape[0])
    ax.plot(time, energy_over_segments / np.max(energy_over_segments))
    ax.set_ylabel('Relativní hlasitost')
    ax.set_xlabel('Čas [s]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    generate_graph(path, fig)
