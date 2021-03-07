import outputs as outputs
import params as params

from preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from feature_extraction import calculate_energy_over_segments, normalize_energy, calculate_rmse, calculate_mfcc, \
    calculate_delta, append_delta
from vad import energy_vad_threshold, energy_vad_threshold_with_adaptive_threshold, energy_gmm_based_vad
from diarization import energy_based_diarization_no_interruptions, gmm_mfcc_diarization_no_interruptions

outputs.check_params()

# get wav file and segment it
wav_file, sampling_rate = read_wav_file(params.file_path)

wav_file = wav_file[:sampling_rate * 60, :]
signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                   params.window_overlap)

# # extract features
energy_over_segments = normalize_energy(calculate_energy_over_segments(segmented_tracks))
root_mean_squared_energy = calculate_rmse(segmented_tracks)
filter_banks, mfcc, power_spectrum = calculate_mfcc(segmented_tracks, sampling_rate)

delta_mfcc = calculate_delta(mfcc)
mfcc_dd = append_delta(mfcc, delta_mfcc, calculate_delta(delta_mfcc))

# voice activity detection
vad = energy_vad_threshold_with_adaptive_threshold(root_mean_squared_energy)
# vad, llhs = energy_gmm_based_vad(wav_file[:, 0].astype(float))


# diarization
# diarization = energy_based_diarization_no_interruptions(energy_over_segments, vad)
diarization = gmm_mfcc_diarization_no_interruptions(mfcc_dd, vad)


# outputs
outputs.diarization_to_files(diarization)

# outputs.plot_energy_with_wav(segmented_tracks[:, :, 0], energy_over_segments[:, 0])
# outputs.plot_vad_energy_with_wav(signal, energy_over_segments, vad)
