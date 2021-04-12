from os import listdir, makedirs
from os.path import isfile, join, exists
from audio_processing import vad as vad_module
from outputs import outputs
import params
from audio_processing.preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from audio_processing.feature_extraction import calculate_rmse, calculate_mfcc
from audio_processing.diarization import gmm_mfcc_diarization_no_interruptions_2channels_single_iteration
from outputs.outputs import diarization_with_timing
from progress.bar import Bar
from argparse import ArgumentParser
import numpy as np
from helpers.dir_exist import create_if_not_exist

if __name__ == '__main__':
    parser = ArgumentParser(description='Module for processing diarization over wav files in provided directory.')
    parser.add_argument('--src', type=str, required=True,
                        help='source path of wav files')
    parser.add_argument('--dest', type=str, required=True,
                        help='destination path for rttm files')

    args = parser.parse_args()

    create_if_not_exist(args.dest)

    print(f'Listing files in {args.source}')
    files = [f for f in listdir(args.source) if isfile(join(args.source, f)) and f.endswith(f'.wav')]

    with Bar(f'Processing files in {args.source}', max=len(files)) as bar:
        for file in files:
            file_name = file.split('.wav')[0]
            wav_file, sampling_rate = read_wav_file(join(args.src, file))
            signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
            segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                               params.window_overlap)
            root_mean_squared_energy = calculate_rmse(segmented_tracks)
            np.savetxt(f'{join(args.dest, file_name)}.energy', root_mean_squared_energy)
            _, mfcc, _ = calculate_mfcc(segmented_tracks, sampling_rate, params.cepstral_coef_count)
            vad = vad_module.energy_gmm_based_vad_propagation(root_mean_squared_energy)
            diarization = gmm_mfcc_diarization_no_interruptions_2channels_single_iteration(mfcc, vad,
                                                                                           root_mean_squared_energy)
            outputs.diarization_to_rttm_file(join(args.dst, file_name), *diarization_with_timing(diarization))
            bar.next()
