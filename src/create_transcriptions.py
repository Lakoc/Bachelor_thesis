from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
from helpers.speech2word import process_file
from helpers.dir_exist import create_if_not_exist

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Module for processing transcription over wav and rttm files in provided directory.')
    parser.add_argument('src', type=str, help='source path of wav and rttm files')
    parser.add_argument('dest', type=str, help='destination path for txt files')

    parser.add_argument('--language', type=str, default='en-US',
                        help='language tag for recognition')

    args = parser.parse_args()

    if args.language != 'en-US':
        raise SystemError(""" No offline ASR system available for current language.
    In case of using other offline ASR system remove this error.
    In case of testing - google recognize API could be used for other languages, 
    but data would not be secured afterwards. (Switch models on line 104 and 106 in speech2word)""")

    create_if_not_exist(args.dest)

    print(f'Listing files in {args.src}')
    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.wav')]

    if len(files) < 1:
        print('No files with extension .wav found')

    for file in files:
        file_name = file.split('.wav')[0]
        process_file(args.src, args.dest, file_name, args.language)
