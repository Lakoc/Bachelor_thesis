import pickle
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
import numpy as np
import json

import params


def get_dictionary_paths_2_levels(dictionary):
    """Get all keys of 2 level dictionary"""
    dict_keys = dictionary.keys()
    keys_nested = [dictionary[dict_key].keys() for dict_key in dict_keys]
    return zip(dict_keys, keys_nested)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Module for calculating overall statistics over dataset')
    parser.add_argument('src', type=str,
                        help='source path of pickle files')
    parser.add_argument('dest', type=str,
                        help='destination path for json file')
    args = parser.parse_args()

    print(f'Listing files in {args.src}')
    files = [f for f in listdir(args.src) if isfile(join(args.src, f)) and f.endswith(f'.pkl')]

    stats_overall = {
        "speech_ratio": {
            "therapist": [
            ]
        },
        "reactions": {
            "therapist": [
            ],
            "client": [
            ]
        },
        'interruptions_len': {
            "therapist": [
            ],
            "client": [
            ]
        },
        "hesitations": {
            "therapist": [
            ],
            "client": [
            ]
        },
        "fills": {
            "therapist": [
            ],
            "client": [
            ]
        },
        "speech_len": {
            "therapist": [],
            "client": []
        }
    }

    # Extract needed information
    for file_name in files:
        with open(join(args.src, file_name), 'rb') as file:
            stats = pickle.load(file)
            try:
                stats_overall['speech_ratio']['therapist'].append(stats['speech_ratio'][0][0])
                stats_overall['reactions']['therapist'].append(np.mean(
                    stats['reactions'][0][:, 1] - stats['reactions'][0][:, 0]) * params.window_stride)
                stats_overall['reactions']['client'].append(np.mean(
                    stats['reactions'][1][:, 1] - stats['reactions'][1][:, 0]) * params.window_stride)
                stats_overall['hesitations']['therapist'].append(np.mean(
                    stats['hesitations'][0][:, 1] - stats['hesitations'][0][:, 0]) * params.window_stride)
                stats_overall['hesitations']['client'].append(np.mean(
                    stats['hesitations'][1][:, 1] - stats['hesitations'][1][:, 0]) * params.window_stride)
                stats_overall['fills']['therapist'].append(stats['fills'][0] / (stats['signal']['len'] / 60))
                stats_overall['fills']['client'].append(stats['fills'][1] / (stats['signal']['len'] / 60))
                stats_overall['speech_len']['therapist'].append(stats['speech_len'][0])
                stats_overall['speech_len']['client'].append(stats['speech_len'][1])
                stats_overall['interruptions_len']['therapist'].append(stats['interruptions_len'][0])
                stats_overall['interruptions_len']['client'].append(stats['interruptions_len'][1])
            except IndexError:
                pass
    keys = get_dictionary_paths_2_levels(stats_overall)

    # Calculate overall stats
    for key in keys:
        for sub_key in key[1]:
            values = stats_overall[key[0]][sub_key]
            if key[0] == 'speech_len' or key[0] == 'interruptions_len':
                stats_overall[key[0]][sub_key] = list(np.mean(values, axis=0))
            else:
                stats_overall[key[0]][sub_key] = [np.mean(values), np.var(values)]

    # Save file
    with open(args.dest, 'w') as file:
        json.dump(stats_overall, file)
