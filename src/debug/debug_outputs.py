import matplotlib.pyplot as plt
import numpy as np


def plot_6_7k(likelihoods, active):
    fig, ax = plt.subplots(figsize=(40, 20))
    ax.set_title('Speaker diarization', fontsize=60, fontweight='bold')
    likelihoods *= active[:, np.newaxis]
    data = likelihoods[6000:7105, 1] - likelihoods[6000:7105, 0]
    ax.plot(data)
    ax.vlines([262, 462, 647, 730, 834, 938, 975], np.min(data), np.max(data), linewidth=2, color='r')
    y = (np.min(data) + np.max(data)) / 2
    ax.text(131 - 10, y, '1', fontsize=40)
    ax.text(362 - 10, y, '2', fontsize=40)
    ax.text(555 - 10, y, '1', fontsize=40)
    ax.text(689 - 10, y, '0', fontsize=40)
    ax.text(782 - 10, y, '1', fontsize=40)
    ax.text(886 - 10, y, '2', fontsize=40)
    ax.text(957 - 10, y, '1', fontsize=40)
    ax.text(1040 - 10, y, '2', fontsize=40)
    ax.plot(np.zeros(1105))
    fig.show()
