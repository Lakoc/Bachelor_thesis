from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from progress.bar import Bar
from io_operations.load_files import load_stats_overall, load_texts, load_stats
from outputs.create_html_output import create_output
from helpers.dir_exist import create_if_not_exist, create_new_dir

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Module for creating html statistic')
    parser.add_argument('src', type=str,
                        help='source path of pickle files')
    parser.add_argument('dest', type=str,
                        help='destination path for pickle files')
    parser.add_argument('template', type=str,
                        help='template for html file')
    parser.add_argument('overall_stats', type=str,
                        help='path of file containing overall stats')
    parser.add_argument('text', type=str,
                        help='path of file containing texts')

    args = parser.parse_args()

    create_if_not_exist(args.dest)

    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.pkl')]

    if len(files) < 1:
        raise FileNotFoundError(f'No pkl files found in {args.src}')

    # Create new subdir for plots
    create_new_dir(join(args.dest, 'plots'), 'output plots')

    # Copy stylesheet file to destination
    copyfile(f'{args.template.split(".html")[0]}.css', join(args.dest, 'style.css'))
    print(f'File {join(args.dest, "style.css")} was created')

    with Bar(f'Processing files in {args.src}', max=len(files)) as bar:
        for file in files:

            # Load source files
            file_name = file.split('.pkl')[0]
            texts = load_texts(args.text)
            stats = load_stats(join(args.src, file))
            stats_overall = load_stats_overall(args.overall_stats)

            # Create output
            create_output(stats, stats_overall, texts, args.template, args.dest, file_name)
            bar.next()
            exit(0)