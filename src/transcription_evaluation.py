import numpy as np
from progress.bar import Bar
from os import listdir
from os.path import isfile, join
from argparse import ArgumentParser
from system_tests.transcription_test import transcription_test

if __name__ == '__main__':

    parser = ArgumentParser(description='Module for checking word error rate.')
    parser.add_argument('hyp', type=str,
                        help='path of hypothesis rttm files')
    parser.add_argument('ref', type=str,
                        help='path of reference rttm files')

    parser.add_argument('--verbose', action="store_true", help='print error for each file')

    args = parser.parse_args()

    error_rate = np.zeros(2)

    print(f'Listing files in {args.hyp}')
    files = [f for f in listdir(args.hyp) if isfile(join(args.hyp, f)) and f.endswith(f'.txt')]

    if len(files) < 1:
        raise FileNotFoundError(f'No txt files found in {args.hyp}')

    with Bar(f'Calculating word error rate', max=len(files)) as bar:
        for file in files:
            file_name = f'{file.split(".")[0].split("/")[-1]}.txt'
            if args.verbose:
                print(f'File {file}\n')
            error_rate += transcription_test(args.ref, file_name, args.hyp,
                                             args.verbose)
            if args.verbose:
                print('\n')
            bar.next()

        error_rate /= len(files)
        print(f"""\nOverall error rate\n
        Word error rate channel 1: {error_rate[0] * 100:.3f}%
        Word error rate channel 2: {error_rate[1] * 100:.3f}%""")
