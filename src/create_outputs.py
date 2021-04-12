from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join

from progress.bar import Bar
from helpers.load_files import load_transcription, load_vad_from_rttm, load_energy
from helpers.create_html_output import create_output
from outputs.statistics import get_stats
from helpers.dir_exist import create_if_not_exist

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

    create_if_not_exist(args.dest)

    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.rttm')]

    with Bar(f'Processing files in {args.src}', max=len(files)) as bar:
        for file in files:
            file_name = file.split('.rttm')[0]
            energy = load_energy(f'{join(args.src, file_name)}.energy')
            vad = load_vad_from_rttm(f'{join(args.src, file_name)}.rttm', energy.shape[0])
            transcription = load_transcription(f'{join(args.src, file_name)}.txt')
            stats = get_stats(vad, transcription, energy)
            create_output(stats, args.template, args.dest, file_name)
            bar.next()
