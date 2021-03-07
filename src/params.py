# input file path
file_path = 'data_to_delete/ukazka1/prezentace/merged.wav'

# preprocessing
pre_emphasis_coefficient = 0.97

window_size = 0.025  # 25ms
window_overlap = 0.015  # 15ms
window_stride = window_size - window_overlap  # 10ms

# mfcc
frequency_resolution = 512
cepstral_coef_count = 12

lifter = 22

delta_neighbours = 2

# energy vad
energy_threshold_interval = 0.1  # 100ms
med_filter = 0.35  # 350 ms

# diarization
med_filter_diar = 100  # Each segment represents 10 ms -> 100 segments = 1s
second = 60
minute = second * 60
means_shift = 0.05

# output folder
output_folder = 'output'

# OLD
hesitations_max_size_max_size = 2  # seconds
number_of_lpc_coefficients = 10
displayEnergy = False

# get pitches sizes in segments
# remove_pitches_size = int(remove_pitches_size / window_size)
hesitations_max_size_max_size = int(hesitations_max_size_max_size / window_size)
