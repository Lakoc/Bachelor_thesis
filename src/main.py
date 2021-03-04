import src.outputs as outputs
import src.params as params

from src.preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from feature_extraction import calculate_energy_over_segments, calculate_mfcc

outputs.check_params()

# get wav file
wav_file, sampling_rate = read_wav_file(params.file_path)
signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                   params.window_overlap)

# extract features
energy_over_segments = calculate_energy_over_segments(segmented_tracks)
# outputs.plot_energy_with_wav(segmented_tracks[:, :, 0], energy_over_segments[:, 0])

mfcc = calculate_mfcc(segmented_tracks, sampling_rate)
