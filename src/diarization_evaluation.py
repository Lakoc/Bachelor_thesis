import numpy as np
from progress.bar import Bar
from tests.transcription_test_overlap import calculate_success_rate_no_overlap, calculate_success_rate
from os import listdir
from os.path import isfile, join
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(description='Module for processing diarization over wav files in provided directory.')
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

    print(f'Listing files in {args.ref}')
    files = [f for f in listdir(args.ref) if isfile(join(args.ref, f)) and f.endswith(f'.rttm')]

    if len(files) < 1:
        raise FileNotFoundError(f'No rttm files found in {args.hyp}')

    with Bar(f'Calculating success_rate', max=len(files)) as bar:
        for file in files:
            file_name = f'{file.split(".")[0].split("/")[-1]}.rttm'
            print(f'File {file}\n')
            if args.mode == 2:
                error_rate += calculate_success_rate_no_overlap(args.hyp, file_name, args.ref, args.collar,
                                                                args.verbose)
            else:
                error_rate += calculate_success_rate(args.hyp, file_name, args.ref, args.collar, args.verbose)
            print('\n')
            bar.next()

        error_rate /= len(files)
        print(f"""\nOverall error rates\n
        Miss rate: {error_rate[0] * 100:.3f}%
        False alarm rate: {error_rate[1] * 100:.3f}%
        {f'Diarization error rate: {error_rate[2] * 100:.3f}%' if args.mode == 1 else ''}
        System accuracy: {error_rate[3] * 100:.3f}%
        """)
