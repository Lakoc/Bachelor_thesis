from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
from progress.bar import Bar
from helpers.speech2word import process_file
from helpers.dir_exist import create_if_not_exist

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Module for processing transcription over wav and rttm files in provided directory.')
    parser.add_argument('--src', type=str, required=True,
                        help='source path of wav and rttm files')
    parser.add_argument('--dest', type=str, required=True,
                        help='destination path for txt files')

    parser.add_argument('--language', type=str, required=True,
                        help='language tag for recognition')

    args = parser.parse_args()

    create_if_not_exist(args.dest)

    print(f'Listing files in {args.src}')
    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.wav')]

    for file in files:
        file_name = file.split('.wav')[0]
        process_file(args.src, args.dest, file_name, args.language)