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
cepstral_coef_count = 20
lifter = 22
delta_neighbours = 2

# energy vad
# --thresholding
energy_threshold_interval = 0.1  # 100ms
energy_threshold = 0.006

# --adaptive
growth_coefficient = 1.000001
lam = 0.99

# --filtration
vad_med_filter = 0.5  # 100ms
vad_filter_non_active = 0.4  # 100ms
vad_filter_active = 0.1  # 100ms

# --forward-backward settings
vad_min_speech_dur = 1
p_loop = 0.95

# --gmm
vad_component_to_threshold = 0  # silence 0, noise 1, speech 2
vad_min_likelihood = 0.95
vad_max_iter = 300
vad_error_rate = 1e-5

# diarization
# --gmm settings
gmm_components = 32
gmm_max_iterations = 300
gmm_error_rate = 1e-3
gmm_verbose = False

# --filtration
diarization_output_mean_filter = 1
diarization_init_array_filter = 0.5

# --forward-backward
diar_min_speech_dur = 1
diar_loop_prob = 0.9

# --means updates
likelihood_percentile = 5
model_means_shift = 0.05

# outputs
min_crosstalk = 0.1
volume_table_items = 15
interruptions_table_items = 10
