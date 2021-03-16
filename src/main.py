import outputs
import params
from preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from feature_extraction import calculate_energy_over_segments, normalize_energy, calculate_rmse, calculate_mfcc, \
    calculate_delta, append_delta
from vad import energy_vad_threshold_with_adaptive_threshold, energy_gmm_based_vad
from diarization import gmm_mfcc_diarization_no_interruptions_2channels_single_iteration
from statistics import diarization_with_timing
from helpers.diarization_helpers import smoother_diarization
from tests.calculate_success_rate import calculate_success_rate

"""Params check"""
outputs.check_params()

"""Load wav file"""
wav_file, sampling_rate = read_wav_file(params.file_path)
# wav_file = wav_file[:sampling_rate * 60, :]

"""Segmentation"""
signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                   params.window_overlap)

"""Feature extraction"""
energy = calculate_energy_over_segments(segmented_tracks)
normalized_energy = normalize_energy(energy)
# root_mean_squared_energy = calculate_rmse(segmented_tracks)

filter_banks, mfcc, power_spectrum = calculate_mfcc(segmented_tracks, sampling_rate)
delta_mfcc = calculate_delta(mfcc)
mfcc_dd = append_delta(mfcc, delta_mfcc, calculate_delta(delta_mfcc))

"""Voice activity detection"""
# vad = energy_vad_threshold_with_adaptive_threshold(root_mean_squared_energy)
vad = energy_gmm_based_vad(normalized_energy)

"""Diarization"""
# diarization = energy_based_diarization_no_interruptions(energy_over_segments, vad)
diarization = gmm_mfcc_diarization_no_interruptions_2channels_single_iteration(mfcc, vad, energy)

"""Outputs"""
outputs.diarization_to_files(*diarization_with_timing(smoother_diarization(diarization)))
# outputs.plot_energy_with_wav(segmented_tracks[:, :, 0], energy_over_segments[:, 0])
# outputs.plot_vad_energy_with_wav(signal, energy, vad)

"""Tests"""
calculate_success_rate()
