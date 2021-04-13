from matplotlib import pyplot as plt

from helpers.gauge import gauge

from outputs.outputs import generate_graph


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
    angle = min(max((var_center - center + speech_time) * 180, 0), 180)
    gauge(labels=['Nízký', 'Normalní', 'Vysoký'], colors=['C0', 'C2', 'C1'], angle=180 - angle, fname=path)
