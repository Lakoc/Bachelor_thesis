from src.AudioProcessing import process_hamming, read_wav_file, calculate_sign_changes, \
    calculate_energy_over_segments, process_voice_activity_detection_via_energy, \
    process_voice_activity_detection_via_gmm
import src.statistics as statistics
import src.outputs as outputs
import os
import src.params as params

# from gauge import gauge
#
# gauge(labels=['Too few', 'Normal', 'Too many'], colors=['C0', 'C2', 'C1'], arrow=2,
#       title='Interruptions per minute', fname='presentation/mean_values.png')
# exit(0)


# config -> get from params


if params.save_to and not os.path.exists(params.save_to):
    os.makedirs(params.save_to)

# get wav file
wav_file, sampling_rate = read_wav_file(params.file_path)

# get pitches sizes in segments
remove_pitches_size = int(params.remove_pitches_size / params.window_size)
hesitations_max_size_max_size = int(params.hesitations_max_size_max_size / params.window_size)

# TODO: VAD based on energy

# if params.with_gmm:
#     vad = process_voice_activity_detection_via_gmm(wav_file, remove_pitches_size, win_length=sampling_rate // 40)
# else:
#     # get parameters needed for next processing
#     segmented_tracks = process_hamming(wav_file, params.pre_emphasis_coefficient, sampling_rate, params.window_size,
#                                        params.overlap_size)
#     energy_over_segments = calculate_energy_over_segments(segmented_tracks, params.displayEnergy)
#     zero_crossings = calculate_sign_changes(segmented_tracks)
#     vad = process_voice_activity_detection_via_energy(energy_over_segments, zero_crossings, params.energy_threshold,
#                                                       remove_pitches_size)
#     cross_talks, interruptions = statistics.detect_cross_talks(vad)
#     vad_joined, hesitations = statistics.join_vad_and_count_hesitations(vad, hesitations_max_size_max_size)
#
#     # generate statistics
#     mean_energies = statistics.count_mean_energy_when_speaking(energy_over_segments, vad)
#     speech_time = statistics.get_speech_time(vad)
#     responses, sentences_lengths = statistics.generate_response_and_speech_time_statistics(vad)
#     # lpc = statistics.calculate_lpc_over_segments(segmented_tracks, number_of_lpc_coefficients)
#     # # Test lpc
#     # for i in range(100, 110):
#     #     outputs.plot_lpc_ftt(segmented_tracks[0][i], sampling_rate, lpc[0][i], number_of_lpc_coefficients)
#
#     # # plot outputs
#     outputs.plot_wav_with_detection(sampling_rate, wav_file, vad, vad_joined, cross_talks, params.save_to)
    # outputs.plot_speech_time_comparison(mean_energies, speech_time, save_to)
    # outputs.plot_interruptions(interruptions, save_to)
    # outputs.plot_responses_lengths(responses, sentences_lengths, save_to)
    # outputs.plot_monolog_hesitations_histogram(hesitations, save_to)
    #
    # statistics.generate_text_statistics(mean_energies, speech_time, hesitations, responses, sentences_lengths,
    #                                     interruptions)
