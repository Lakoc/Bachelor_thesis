from AudioProcessing import process_hamming, read_wav_file, calculate_sign_changes, calculate_energy_over_segments, \
    process_voice_activity_detection_via_energy, process_voice_activity_detection_via_gmm
import statistics
import outputs

# config -> get from params
file_path = 'data/treatmentMerged.wav'
window_size = 1 / 40  # 25ms
overlap_size = 1 / 100  # 10ms
remove_pitches_size = 0.5  # seconds
hesitations_max_size_max_size = 5  # seconds
pre_emphasis_coefficient = 0.97
energy_threshold = 8
number_of_lpc_coefficients = 14
displayEnergy = True
save_to = False
with_gmm = False

# get wav file
wav_file, sampling_rate = read_wav_file(file_path)

# get pitches sizes in segments
remove_pitches_size = int(remove_pitches_size / window_size)
hesitations_max_size_max_size = int(hesitations_max_size_max_size / window_size)

if with_gmm:
    vad = process_voice_activity_detection_via_gmm(wav_file, remove_pitches_size, win_length=sampling_rate // 40)
else:
    # get parameters needed for next processing
    segmented_tracks = process_hamming(wav_file, pre_emphasis_coefficient, sampling_rate, window_size, overlap_size)
    energy_over_segments = calculate_energy_over_segments(segmented_tracks, displayEnergy)
    zero_crossings = calculate_sign_changes(segmented_tracks)
    vad = process_voice_activity_detection_via_energy(energy_over_segments, zero_crossings, energy_threshold,
                                                      remove_pitches_size)
    cross_talks, interruptions = statistics.detect_cross_talks(vad)
    vad_joined, hesitations = statistics.join_vad_and_count_hesitations(vad, hesitations_max_size_max_size)

    # generate statistics
    mean_energies = statistics.count_mean_energy_when_speaking(energy_over_segments, vad)
    speech_time = statistics.get_speech_time(vad)
    responses, sentences_lengths = statistics.generate_response_and_speech_time_statistics(vad)
    lpc = statistics.calculate_lpc_over_segments(segmented_tracks, number_of_lpc_coefficients)
    # TODO: Plot lpc

    # plot outputs
    outputs.plot_wav_with_detection(sampling_rate, wav_file, vad, vad_joined, cross_talks, save_to)
    outputs.plot_speech_time_comparison(mean_energies, speech_time, save_to)
    outputs.plot_interruptions(interruptions, save_to)
    outputs.plot_responses_lengths(responses, sentences_lengths, save_to)
    outputs.plot_monolog_hesitations_histogram(hesitations, save_to)

    statistics.generate_text_statistics(mean_energies, speech_time, hesitations, responses, sentences_lengths,
                                        interruptions)
