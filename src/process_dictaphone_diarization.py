from os import listdir
from os.path import isfile, join

import python_speech_features

from io_operations import outputs
import params
from audio_processing.preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from audio_processing.feature_extraction import calculate_rmse, calculate_mfcc, calculate_energy_over_segments, \
    normalize_energy_to_0_1
import audio_processing.diarization as diarization_module
from io_operations.outputs import diarization_with_timing
from progress.bar import Bar
from argparse import ArgumentParser
import numpy as np
from helpers.dir_exist import create_if_not_exist
from io_operations.load_files import load_vad_from_rttm

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
            file_name = file.split('.wav')[0]
            wav_file, sampling_rate = read_wav_file(join(args.src, file))

            # Check wav shape
            if wav_file.shape[0] < sampling_rate:
                raise ValueError(f'Wav file {file_name} is too short')
            if wav_file.shape[1] != 2:
                raise ValueError(f'Wav file {file_name} does not have 2 channels.')

            # Preprocessing
            signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
            segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                               params.window_overlap)

            # Delete signal, preventing ram memory exceeding
            mfcc_0 = python_speech_features.mfcc(signal[:, 0], samplerate=sampling_rate, winlen=params.window_size,
                                                 winstep=params.window_stride,
                                                 nfft=1024,
                                                 numcep=params.cepstral_coef_count)
            mfcc_1 = python_speech_features.mfcc(signal[:, 1], samplerate=sampling_rate, winlen=params.window_size,
                                                 winstep=params.window_stride,
                                                 nfft=1024,
                                                 numcep=params.cepstral_coef_count)
            mfcc = np.vstack([mfcc_0[np.newaxis, ...], mfcc_1[np.newaxis, ...]]).transpose((1, 2, 0))
            del signal
            root_mean_squared_energy = calculate_rmse(segmented_tracks)
            np.savetxt(f'{join(args.dest, file_name)}.energy', root_mean_squared_energy)
            # _, mfcc, _ = calculate_mfcc(segmented_tracks, sampling_rate, params.cepstral_coef_count)

            # Delete segments, preventing ram memory exceeding
            del segmented_tracks

            """Thresholding"""
            # vad = vad_module.energy_vad_threshold(normalize_energy_to_0_1(
            #     root_mean_squared_energy), threshold=params.energy_threshold)
            """Adaptive threshold"""
            # vad = vad_module.energy_vad_threshold_with_adaptive_threshold(normalize_energy_to_0_1(
            #     root_mean_squared_energy))
            """Gmm based VAD"""
            # vad = vad_module.energy_gmm_based(
            #     normalize_energy_to_0_1(root_mean_squared_energy), propagation=True)

            vad = load_vad_from_rttm(f'{join(args.src, file_name)}.rttm', root_mean_squared_energy.shape[0])
            diar = np.argmax(vad, axis=1)
            diar += 1
            vad[:, 0] = vad[:, 1] = np.logical_or(vad[:, 0], vad[:, 1])
            diar *= vad[:, 0]
            """Smoothing the vad output"""
            # vad = vad_module.apply_median_filter(vad)
            # vad = vad_module.apply_silence_speech_removal(vad)

            """Energy based diarization"""
            # diarization = diarization_module.energy_based_diarization(
            #     root_mean_squared_energy, vad)

            diarization = diarization_module.gmm_mfcc_diarization_1channel(mfcc, vad, root_mean_squared_energy, diar)

            #
            # diarization = diarization_module.gmm_mfcc_diarization_2channels(mfcc, vad,
            #                                                                 root_mean_squared_energy)

            # Save outputs to rttm file
            outputs.diarization_to_rttm_file(join(args.dest, file_name), *diarization_with_timing(diarization))
            bar.next()
