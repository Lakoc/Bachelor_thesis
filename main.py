from AudioProcessing import AudioProcessing, detect_spaces, process_voice_activity_detection, detect_cross_talks
import statistics

audio_processing = AudioProcessing('data/treatmentMerged.wav')
energy_over_segments = audio_processing.calculate_energy_over_segments()
zero_crossings = audio_processing.calculate_sign_changes()
vad = process_voice_activity_detection(energy_over_segments, zero_crossings)
cross_talks = detect_cross_talks(vad)
spaces = detect_spaces(vad)

statistics.plot_wav_with_detection(audio_processing.get_sampling_rate(), audio_processing.get_audio(), vad, cross_talks)
statistics.generate_statistics(energy_over_segments, vad, spaces)
