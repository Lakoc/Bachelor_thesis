from os import listdir
from os.path import isfile, join
from helpers.dir_exist import create_if_not_exist
import numpy as np

from audio_processing import vad as vad_module
from outputs import outputs
import params
from audio_processing.preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from audio_processing.feature_extraction import calculate_rmse, normalize_energy_to_0_1, normalize_energy
from progress.bar import Bar
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Module for processing diarization over wav files in provided directory.')
    parser.add_argument('src', type=str, help='source path of wav files')
    parser.add_argument('dest', type=str, help='destination path for rttm files')

    args = parser.parse_args()

    create_if_not_exist(args.dest)

    print(f'Listing files in {args.src}')
    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.wav')]

    if len(files) < 1:
        raise FileNotFoundError(f'No wav files found in {args.src}')

    with Bar(f'Processing files in {args.src}', max=len(files)) as bar:
        for file in files:

            # Load wav file
            file_name = file.split(".wav")[0]
            wav_file, sampling_rate = read_wav_file(join(args.src, file))

            # Check shape
            if wav_file.shape[0] < sampling_rate:
                raise ValueError(f'Wav file {file_name} is too short')
            if wav_file.shape[1] != 2:
                raise ValueError(f'Wav file {file_name} does not have 2 channels.')

            # Preprocessing
            signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
            segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                               params.window_overlap)

            # Clean memory
            del signal

            # Calculate root mean square energy and clean segments
            root_mean_squared_energy = calculate_rmse(segmented_tracks)
            del segmented_tracks

            # Save energy for stats generation
            np.savetxt(f'{join(args.dest, file_name)}.energy', root_mean_squared_energy)

            """Thresholding"""
            # vad = vad_module.energy_vad_threshold(normalize_energy_to_0_1(
            #     root_mean_squared_energy), threshold=params.energy_threshold)
            """Adaptive threshold"""
            # vad = vad_module.energy_vad_threshold_with_adaptive_threshold(normalize_energy_to_0_1(
            #     root_mean_squared_energy))
            """Gmm based VAD"""
            vad = vad_module.energy_gmm_based(
                normalize_energy_to_0_1(root_mean_squared_energy), propagation=True)

            """Smoothing the vad output"""
            vad = vad_module.apply_median_filter(vad)
            vad = vad_module.apply_silence_speech_removal(vad)

            # Save outputs to rttm file
            outputs.online_session_to_rttm(join(args.dest, file_name), vad)
            bar.next()
