from AudioProcessing import process_hamming, read_wav_file, calculate_sign_changes, calculate_energy_over_segments, \
    process_voice_activity_detection_via_energy, detect_cross_talks, process_voice_activity_detection_via_gmm
import statistics

# config -> we will get from params
pre_emphasis_coefficient = 0.97
file_path = 'data/treatmentMerged.wav'
energy_threshold = 8
remove_pitches_size = 40
with_gmm = False
displayEnergy = True

# pre_emphasis_coefficient = 0.97
# file_path = 'data/treatmentMerged.wav'
# remove_pitches_size = 200
# with_gmm = True

# get wav file
wav_file, sampling_rate = read_wav_file(file_path)

if with_gmm:
    vad = process_voice_activity_detection_via_gmm(wav_file, remove_pitches_size)
else:
    # get parameters needed for next processing
    segmented_tracks = process_hamming(wav_file, pre_emphasis_coefficient, sampling_rate)
    energy_over_segments = calculate_energy_over_segments(segmented_tracks, displayEnergy)
    zero_crossings = calculate_sign_changes(segmented_tracks)
    vad = process_voice_activity_detection_via_energy(energy_over_segments, zero_crossings, energy_threshold,
                                                      remove_pitches_size)
    statistics.generate_statistics(energy_over_segments, vad)

cross_talks = detect_cross_talks(vad)

# generate statistics
statistics.plot_wav_with_detection(sampling_rate, wav_file, vad, cross_talks)
