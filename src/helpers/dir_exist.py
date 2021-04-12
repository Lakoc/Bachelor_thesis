from os import makedirs
from os.path import exists


def create_if_not_exist(path):
    if not exists(path):
        print(f'Creating directory {path}')
        makedirs(path)
