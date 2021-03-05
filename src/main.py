import src.outputs as outputs
import src.params as params

from src.preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from feature_extraction import calculate_energy_over_segments, normalize_energy, calculate_rmse, calculate_mfcc, calculate_delta
from vad import energy_vad_threshold, energy_vad_threshold_with_adaptive_threshold

outputs.check_params()

# get wav file and segment it
wav_file, sampling_rate = read_wav_file(params.file_path)
signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                   params.window_overlap)
#
# # extract features
energy_over_segments = normalize_energy(calculate_energy_over_segments(segmented_tracks))

rmse = calculate_rmse(segmented_tracks)
vad = energy_vad_threshold_with_adaptive_threshold(rmse)

# outputs.plot_energy_with_wav(segmented_tracks[:, :, 0], energy_over_segments[:, 0])
outputs.plot_vad_energy_with_wav(signal, energy_over_segments, vad)
# filter_banks, mfcc, power_spectrum = calculate_mfcc(segmented_tracks, sampling_rate)
# delta_mfcc = calculate_delta(mfcc)
# delta_delta_mfcc = calculate_delta(delta_mfcc)
