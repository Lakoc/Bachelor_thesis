from argparse import ArgumentParser
from os import makedirs, listdir
from os.path import isfile, exists, join
from progress.bar import Bar
from helpers.speech2word import process_file

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

    if not exists(args.dest):
        print(f'Creating directory {args.dest}')
        makedirs(args.dest)

    print(f'Listing files in {args.src}')
    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.wav')]

    with Bar(f'Processing files in {args.src}', max=len(files)) as bar:
        for file in files:
            file_name = file.split('.wav')[0]
            process_file(args.src, args.dest, file_name, args.language)
            bar.next()
