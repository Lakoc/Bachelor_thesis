from argparse import ArgumentParser
from os import makedirs, listdir
from os.path import isfile, exists, join
from progress.bar import Bar
from helpers.load_files import load_transcription, load_vad_from_rttm
from helpers.create_html_output import create_output
from outputs.statistics import get_stats

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Module for creating html statistic')
    parser.add_argument('--src', type=str, required=True,
                        help='source path of wav, rttm and txt files')
    parser.add_argument('--dest', type=str, required=True,
                        help='destination path for html files')
    parser.add_argument('--template', type=str, required=True,
                        help='template for html file')
    parser.add_argument('--stats', type=str, required=True,
                        help='path of file containing overall stats')

    args = parser.parse_args()

    if not exists(args.dest):
        print(f'Creating directory {args.dest}')
        makedirs(args.dest)

    print(f'Listing files in {args.src}')
    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.wav')]

    with Bar(f'Processing files in {args.src}', max=len(files)) as bar:
        for file in files:
            file_name = file.split('.wav')[0]
            vad = load_vad_from_rttm(f'{join(args.src, file_name)}.rttm')
            transcription = load_transcription(f'{join(args.src, file_name)}.txt')
            stats = get_stats(vad, transcription)
            create_output(stats,args.template, f'{join(args.dest, file_name)}.html')
            bar.next()
