import matplotlib.pyplot as plt
import numpy as np


def plot_wav_with_detection(sampling_rate, wav, vad_segments, cross_talks):
    fig, axs = plt.subplots(5)
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
    axs[4].plot(time_vad, cross_talks)
    # plt.savefig('output/plots.png', dpi=1200)
    fig.show()


def generate_statistics(energy_over_segments, vad, spaces):
    channel1_energy = np.mean(energy_over_segments[0])
    channel2_energy = np.mean(energy_over_segments[1])
    is_channel1_louder = channel1_energy > channel2_energy
    print(f'Channel 1 energy: {channel1_energy}')
    print(f'Channel 2 energy: {channel2_energy}')
    print(
        f'Overall channel {1 if is_channel1_louder else 2} produced '
        f'{channel1_energy / channel2_energy if is_channel1_louder else channel2_energy / channel1_energy} '
        f'times more energy.')
    # 20 ms not overlapping frame
    print(f'Channel 1 speech time: {np.sum(vad[0]) / 50}s')
    print(f'Channel 2 speech time: {np.sum(vad[1]) / 50}s')
    print(f'Channel 1 spaces: min={spaces[0]["min"]}s mean={spaces[0]["mean"]}s max={spaces[0]["max"]}s')
    print(f'Channel 2 spaces: min={spaces[1]["min"]}s mean={spaces[1]["mean"]}s max={spaces[1]["max"]}s')
