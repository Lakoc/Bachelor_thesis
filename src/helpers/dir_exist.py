from os import makedirs
from os.path import exists


def create_if_not_exist(path, verbose=True):
    """Create directory if it not exists"""
    if not exists(path):
        makedirs(path)
        if verbose:
            print(f'Directory {path} was created')


def create_new_dir(path, file_type):
    """Create new dir, ask if should be files added there if exists"""
    if exists(path):
        print(f'Directory {path} already exists. Should {file_type} be added there?\nPlease type [y/n]')
        answer = input()
        if answer == 'y':
            pass
        elif answer == 'n':
            print('Exiting with no output files ...')
            exit(0)
        else:
            raise ValueError('Unknown answer')
    else:
        makedirs(path)
        print(f'Directory {path} was created')
