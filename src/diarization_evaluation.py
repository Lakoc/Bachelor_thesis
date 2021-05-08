import numpy as np
from progress.bar import Bar
from system_tests.diarization_vad_test import diar_dictaphone, diar_online, confusion_channel_error
from os import listdir
from os.path import isfile, join
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(description='Module for checking Diarization error rate.')
    parser.add_argument('hyp', type=str,
                        help='path of hypothesis rttm files')
    parser.add_argument('ref', type=str,
                        help='path of reference rttm files')
    parser.add_argument('mode', type=int, choices=range(1, 3),
                        help='type of wav files 1 - dictaphone, 2 - online Zoom')

    parser.add_argument('collar', type=int,
                        help='error collar size in ms')

    parser.add_argument('--verbose', action="store_true", help='print error for each file')

    args = parser.parse_args()

    error_rate = np.zeros(4)

    print(f'Listing files in {args.hyp}')
    files = [f for f in listdir(args.hyp) if isfile(join(args.hyp, f)) and f.endswith(f'.rttm')]

    if len(files) < 1:
        raise FileNotFoundError(f'No rttm files found in {args.hyp}')

    with Bar(f'Calculating diarization error rate', max=len(files)) as bar:
        for file in files:

            file_name = f'{file.split(".")[0].split("/")[-1]}.rttm'

            if args.verbose:
                print(f'File {file}\n')

            # Calculate err rate according to selected mode
            if args.mode == 2:
                error_rate += confusion_channel_error(args.ref, file_name, args.hyp, args.collar // 10,
                                              args.verbose)
            else:
                error_rate += diar_online(args.ref, file_name, args.hyp, args.collar // 10, args.verbose)

            if args.verbose:
                print('\n')
            bar.next()

        # Normalize error rate and print overall stats
        error_rate /= len(files)
        print(f"""\nOverall error rates\n
        Miss rate: {error_rate[0] * 100:.3f}%
        False alarm rate: {error_rate[1] * 100:.3f}%
        {f'Confusion error rate: {error_rate[2] * 100:.3f}%' if args.mode == 2 else ''}
        System accuracy: {error_rate[3] * 100:.3f}%
        """)
