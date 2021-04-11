from os import listdir, makedirs
from os.path import isfile, join, exists
import outputs
import params
from preprocessing import process_hamming, read_wav_file, process_pre_emphasis
from feature_extraction import calculate_rmse
import vad as vad_module
from progress.bar import Bar
from argparse import ArgumentParser

# with open(join(path, f'{name.split(".")[0]}.rttm'), 'r') as reference_file:
#     reference = reference_file.readlines()
#     reference_extracted = []
#
#     regex = re.compile(r'SPEAKER \S* \d ([\d.]*) ([\d.]*) <NA> <NA> (\d).*')
#
#     def extract(line):
#         x = regex.search(line)
#         start = int(float(x[1]) * 1000)
#         return start, start + int(float(x[2]) * 1000), int(x[3])
#
#     for line in reference:
#         reference_extracted.append(extract(line))
#
#     max_val = reference_extracted[-1][1]
#
#     reference_arr = np.zeros(max_val)
#
#     for speech_tuple in reference_extracted:
#         from_t, to_t, value = speech_tuple
#         if value == 2:
#             reference_arr[from_t: to_t] = value
#
#     fig, axs = plt.subplots(nrows=2)
#     height = np.max(wav_file[:, 1])
#
#     time1 = np.linspace(0, wav_file.shape[0], vad.shape[0])
#     axs[0].plot(time1, vad[:, 1])
#     axs[0].plot(wav_file[:, 1] / height)
#
#     time2 = np.linspace(0, wav_file.shape[0], reference_arr.shape[0])
#     axs[1].plot(time2, reference_arr)
#     axs[1].plot(wav_file[:, 1] / height)
#     plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Module for processing diarization over wav files in provided directory.')
    parser.add_argument('--src', type=str,
                        help='source path of wav files')
    parser.add_argument('--dest', type=str,
                        help='destination path for rttm files')

    args = parser.parse_args()

    if not exists(args.dest):
        print(f'Creating directory {args.dest}')
        makedirs(args.dest)

    print(f'Listing files in {args.src}')
    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.wav')]

    with Bar(f'Processing files in {args.src}', max=len(files)) as bar:
        for file_name in files:
            wav_file, sampling_rate = read_wav_file(join(args.src, file_name))
            signal = process_pre_emphasis(wav_file, params.pre_emphasis_coefficient)
            segmented_tracks = process_hamming(signal, sampling_rate, params.window_size,
                                               params.window_overlap)
            root_mean_squared_energy = calculate_rmse(segmented_tracks)
            vad = vad_module.energy_gmm_based_vad_propagation(root_mean_squared_energy)
            outputs.online_session_to_rttm(join(args.dest, file_name), vad)
            bar.next()
