import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
import outputs
import params
from preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from feature_extraction import calculate_rmse, calculate_mfcc
import vad as vad_module
from diarization import gmm_mfcc_diarization_no_interruptions_2channels_single_iteration
from statistics import diarization_with_timing
from progress.bar import Bar
from tests.transcription_test_rttm import calculate_success_rate
import numpy as np

path = sys.argv[1]
dictaphone = sys.argv[2]
extension = sys.argv[3]
out_dir = sys.argv[4]
success = sys.argv[5]
verbose = False


def process_dictaphone(name):
    wav_file, sampling_rate = read_wav_file(join(path, name))
    signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
    segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                       params.window_overlap)
    root_mean_squared_energy = calculate_rmse(segmented_tracks)
    _, mfcc, _ = calculate_mfcc(segmented_tracks, sampling_rate, params.cepstral_coef_count)
    vad = vad_module.energy_gmm_based_vad_propagation(root_mean_squared_energy)
    diarization = gmm_mfcc_diarization_no_interruptions_2channels_single_iteration(mfcc, vad, root_mean_squared_energy)
    outputs.diarization_to_rttm_file(join(out_dir, name), *diarization_with_timing(diarization))


def process_no_crosstalk(name):
    wav_file, sampling_rate = read_wav_file(join(path, name))
    signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
    segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                       params.window_overlap)
    root_mean_squared_energy = calculate_rmse(segmented_tracks)
    vad = vad_module.energy_gmm_based_vad_propagation(root_mean_squared_energy)
    outputs.online_session_to_rttm(join(out_dir, name), vad)


if not exists(out_dir):
    print(f'Creating directory {out_dir}')
    makedirs(out_dir)

print(f'Listing files in {path}')
files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(f'.wav')]

with Bar(f'Processing files in {path}', max=len(files)) as bar:
    for file_name in files:
        if dictaphone == 'yes':
            process_dictaphone(file_name)
        else:
            process_no_crosstalk(file_name)
        bar.next()
        break

# miss_percentage, false_alarm_percentage, confusion_percentage, der
error_rate = np.zeros(4)
if success == 'yes':
    with Bar(f'Calculating success_rate', max=len(files)) as bar:
        for file in files:
            file_name = f'{file.split(".")[0].split("/")[-1]}.rttm'
            error_rate += calculate_success_rate(path, file_name, out_dir, verbose)
            bar.next()
            break
        error_rate /= len(files)
        print(f"""\nOverall error rates\n 
        Miss rate {error_rate[0] *100:.3f}%
        False alarm rate {error_rate[1] *100:.3f}%
        Confusion error rate {error_rate[2] *100:.3f}%
        Diarization error rate {error_rate[3] *100:.3f}%
        """)
