from AudioProcessing import process_hamming, read_wav_file, calculate_sign_changes, calculate_energy_over_segments, \
    process_voice_activity_detection_via_energy, detect_cross_talks, process_voice_activity_detection_via_gmm
import statistics
import ouputs

# config -> we will get from params
energy_threshold = 8
displayEnergy = False
pre_emphasis_coefficient = 0.97
file_path = 'data/treatmentMerged.wav'
remove_pitches_size = 20
save_to = False

with_gmm = False
# with_gmm = True

# get wav file
wav_file, sampling_rate = read_wav_file(file_path)

if with_gmm:
    vad = process_voice_activity_detection_via_gmm(wav_file, remove_pitches_size, win_length=sampling_rate // 40)
else:
    # get parameters needed for next processing
    segmented_tracks = process_hamming(wav_file, pre_emphasis_coefficient, sampling_rate)
    energy_over_segments = calculate_energy_over_segments(segmented_tracks, displayEnergy)
    zero_crossings = calculate_sign_changes(segmented_tracks)
    vad = process_voice_activity_detection_via_energy(energy_over_segments, zero_crossings, energy_threshold,
                                                      remove_pitches_size)
    cross_talks, interruptions = detect_cross_talks(vad)

    # generate statistics
    mean_energies = statistics.count_mean_energy_when_speaking(energy_over_segments, vad)
    speech_time = statistics.get_speech_time(vad)
    spaces, sentences_lengths = statistics.generate_response_and_speech_time_statistics(vad, save_to)

    ouputs.plot_wav_with_detection(sampling_rate, wav_file, vad, cross_talks, save_to)
    ouputs.plot_speech_time_comparison(mean_energies, speech_time, save_to)
    ouputs.plot_interruptions(interruptions, save_to)
    ouputs.plot_space_lengths(spaces, sentences_lengths, save_to)

    statistics.generate_text_statistics(mean_energies, speech_time)
