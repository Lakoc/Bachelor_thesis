from outputs import outputs, outputs_text
import params
from audio_processing.preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from audio_processing.feature_extraction import calculate_energy_over_segments, normalize_energy, calculate_rmse, calculate_mfcc, \
    calculate_delta, append_delta
from audio_processing.vad import energy_gmm_based_vad_propagation
from audio_processing.diarization import gmm_mfcc_diarization_no_interruptions_2channels_single_iteration
from tests.transcription_test import calculate_success_rate
import outputs.outputs_text


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
root_mean_squared_energy = calculate_rmse(segmented_tracks)

outputs_text.VAD_adaptive_threshold(root_mean_squared_energy)
exit(0)
filter_banks, mfcc, power_spectrum = calculate_mfcc(segmented_tracks, sampling_rate, params.cepstral_coef_count)
delta_mfcc = calculate_delta(mfcc)
mfcc_dd = append_delta(mfcc, delta_mfcc, calculate_delta(delta_mfcc))

"""Voice activity detection"""
# vad = energy_vad_threshold(energy)
# vad = energy_vad_threshold_with_adaptive_threshold(root_mean_squared_energy)
# vad = energy_gmm_based_vad(root_mean_squared_energy)
vad = energy_gmm_based_vad_propagation(root_mean_squared_energy)
# outputs_text.VAD_gmm(root_mean_squared_energy, signal, sampling_rate)
# exit(0)
"""Diarization"""
# diarization = energy_based_diarization_no_interruptions(root_mean_squared_energy, vad)
# diarization = gmm_mfcc_diarization_no_interruptions_1channel(mfcc, vad, root_mean_squared_energy)
diarization = gmm_mfcc_diarization_no_interruptions_2channels_single_iteration(mfcc_dd, vad, root_mean_squared_energy)
# diarization = gmm_mfcc_diarization_no_interruptions_1channel_k_means(mfcc[:, :, 0], vad, root_mean_squared_energy[:, 0])
# diarization = gmm_mfcc_diarization_no_interruptions_2channels_2iterations(mfcc_dd, vad, root_mean_squared_energy)

"""Outputs"""
outputs.diarization_to_file(*diarization_with_timing(diarization))
# outputs.plot_energy_with_wav(segmented_tracks[:, :, 0], energy_over_segments[:, 0])
outputs.plot_vad_wav(signal, vad)

"""Tests"""
# vad_test(vad)
calculate_success_rate()

"""Matlab difference"""
# vad = np.loadtxt('matlab_test/vad.txt')
# vad = vad[:, np.newaxis]
# vad = np.append(vad, vad, axis=1)
# vad_test(vad)
