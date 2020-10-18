from AudioProcessing import process_hamming, read_wav_file, calculate_sign_changes, calculate_energy_over_segments, \
    detect_spaces, process_voice_activity_detection_via_energy, detect_cross_talks
import statistics

pre_emphasis_coefficient = 0.97
file_path = 'data/treatmentMerged.wav'
energy_threshold = 8
remove_pitches_size = 40

wav_file, sampling_rate = read_wav_file(file_path)
segmented_tracks = process_hamming(wav_file, pre_emphasis_coefficient, sampling_rate)
energy_over_segments = calculate_energy_over_segments(segmented_tracks)
zero_crossings = calculate_sign_changes(segmented_tracks)
vad = process_voice_activity_detection_via_energy(energy_over_segments, zero_crossings, energy_threshold,
                                                  remove_pitches_size)
cross_talks = detect_cross_talks(vad)
spaces = detect_spaces(vad)

statistics.plot_wav_with_detection(sampling_rate, wav_file, vad, cross_talks)
statistics.generate_statistics(wav_file, energy_over_segments, vad)
