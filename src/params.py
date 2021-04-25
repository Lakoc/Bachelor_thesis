# input file path
file_path = 'data_to_delete/transcribed/same_room/session1/stereo.wav'
transcription_path = 'data_to_delete/transcribed/same_room/session1/transcription.trs'
# preprocessing
pre_emphasis_coefficient = 0.97

window_size = 0.02  # 20ms
window_overlap = 0.01  # 10ms
window_stride = window_size - window_overlap  # 10ms

# mfcc
frequency_resolution = 512
cepstral_coef_count = 10
lifter = 22
delta_neighbours = 2

# energy vad
energy_threshold_interval = 0.1  # 100ms
med_filter = 0.2


# maximum system sensitivity
min_silence_likelihood = 0.95
transition_matrix = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
filter_non_active = 0.1
filter_active = 0.1

# vad hmm
vad_min_speech_dur = 1
vad_loop_probability = 0.9

# diarization
gmm_components = 32
energy_diff_filter = 50
gmm_max_iterations = 100
gmm_error_rate = 1e-3
mean_filter_diar = 100  # Each segment represents 10 ms -> 100 segments = 1s
likelihood_percentile = 20  # Take 20% of highest likelihood frames
model_means_shift = 0.05
diar_min_speech_dur = 1
diar_loop_prob = 0.99

# output folder
output_folder = 'output'

# tests
transcription_plot = True
collar_size = 25  # x 10ms

# OLD
hesitations_max_size_max_size = 2  # seconds
number_of_lpc_coefficients = 10
displayEnergy = False

# get pitches sizes in segments
# remove_pitches_size = int(remove_pitches_size / window_size)
hesitations_max_size_max_size = int(hesitations_max_size_max_size / window_size)
