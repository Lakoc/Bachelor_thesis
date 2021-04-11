import re
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

regex = re.compile(r'SPEAKER \S* \d ([\d.]*) ([\d.]*) <NA> <NA> (\d).*')


def extract(line):
    x = regex.search(line)
    start = int(float(x[1]) * 1000)
    return start, start + int(float(x[2]) * 1000), int(x[3])


def calculate_success_rate(ref_path, filename, output_path, collar_size, verbose):
    with open(join(output_path, filename), 'r') as hypothesis_file:
        with open(join(ref_path, filename), 'r') as reference_file:

            hypothesis = hypothesis_file.readlines()
            reference = reference_file.readlines()

            hypothesis_extracted = []
            reference_extracted = []
            for line in hypothesis:
                hypothesis_extracted.append(extract(line))

            for line in reference:
                reference_extracted.append(extract(line))

            max_val = max(hypothesis_extracted[-1][1], reference_extracted[-1][1])

            reference_arr = np.zeros(max_val)
            collar_one_side = collar_size // 2

            for speech_tuple in reference_extracted:
                from_t, to_t, value = speech_tuple
                reference_arr[np.max([from_t - collar_one_side, 0]): from_t + collar_one_side] = -1
                reference_arr[from_t + collar_one_side: to_t + 1 - collar_one_side] = np.where(
                    reference_arr[from_t + collar_one_side: to_t + 1 - collar_one_side] == -1, -1, value)
                reference_arr[to_t + 1 - collar_one_side: np.min([to_t + 1 + collar_one_side, max_val])] = -1

            hypothesis_arr = np.zeros(max_val)

            for speech_tuple in hypothesis_extracted:
                hypothesis_arr[speech_tuple[0]: speech_tuple[1]] += speech_tuple[2]

            miss = np.logical_and(reference_arr > 0, hypothesis_arr == 0)
            false_alarm = np.logical_and(hypothesis_arr > 0, reference_arr == 0)
            confusion = np.logical_and(reference_arr != hypothesis_arr,
                                       np.logical_and(hypothesis_arr > 0, reference_arr > 0))

            miss = np.sum(miss)
            miss_percentage = miss / reference_arr.shape[0]
            false_alarm = np.sum(false_alarm)
            false_alarm_percentage = false_alarm / reference_arr.shape[0]
            confusion = np.sum(confusion)
            confusion_percentage = confusion / reference_arr.shape[0]

            der = (miss + false_alarm + confusion) / reference_arr.shape[0]

            if verbose:
                print(
                    f'Miss rate: {miss_percentage:.3f} (Marked as non-active, although speech was present)')
                print(
                    f'False alarm: {false_alarm_percentage:.3f} (Marked as active, although speech was '
                    f'not present)')
                print(
                    f'Confusion: {confusion_percentage:.3f} (System provided speaker tag and the '
                    f'reference speech was present, but does not match)')
                print(f'Diarization error rate:{der:.3f}')
                print(f'Diarization accuracy:{1 - der:.3f}')

            return miss_percentage, false_alarm_percentage, confusion_percentage, der


def calculate_success_rate_no_overlap(ref_path, filename, output_path, collar_size, verbose):
    with open(join(output_path, filename), 'r') as hypothesis_file:
        with open(join(ref_path, filename), 'r') as reference_file:

            hypothesis = hypothesis_file.readlines()
            reference = reference_file.readlines()

            hypothesis_extracted = []
            reference_extracted = []
            for line in hypothesis:
                hypothesis_extracted.append(extract(line))

            for line in reference:
                reference_extracted.append(extract(line))

            max_val = max(hypothesis_extracted[-1][1], reference_extracted[-1][1])

            reference_arr = np.zeros((max_val, 2))
            collar_one_side = collar_size // 2

            for speech_tuple in reference_extracted:
                from_t, to_t, value = speech_tuple
                value -= 1
                reference_arr[np.max([from_t - collar_one_side, 0]): from_t + collar_one_side, value] = -1
                reference_arr[from_t + collar_one_side: to_t + 1 - collar_one_side, value] = np.where(
                    reference_arr[from_t + collar_one_side: to_t + 1 - collar_one_side, value] == -1, -1, 1)
                reference_arr[to_t + 1 - collar_one_side: np.min([to_t + 1 + collar_one_side, max_val]), value] = -1

            hypothesis_arr = np.zeros((max_val, 2))

            for speech_tuple in hypothesis_extracted:
                hypothesis_arr[speech_tuple[0]: speech_tuple[1], speech_tuple[2] - 1] = 1

            miss1 = np.logical_and(reference_arr[:, 0] > 0, hypothesis_arr[:, 0] == 0)
            miss2 = np.logical_and(reference_arr[:, 1] > 0, hypothesis_arr[:, 1] == 0)
            false_alarm1 = np.logical_and(hypothesis_arr[:, 0] > 0, reference_arr[:, 0] == 0)
            false_alarm2 = np.logical_and(hypothesis_arr[:, 1] > 0, reference_arr[:, 1] == 0)

            miss = np.sum(miss1) + np.sum(miss2)
            miss_percentage = miss / (2 * reference_arr.shape[0])

            print(np.sum(false_alarm1), np.sum(false_alarm2))
            false_alarm = np.sum(false_alarm1) + np.sum(false_alarm2)
            false_alarm_percentage = false_alarm / (2 * reference_arr.shape[0])

            der = miss_percentage + false_alarm_percentage

            if verbose:
                print(
                    f'Miss rate over channels: {miss_percentage * 100:.3f}% (Marked as non-active, although speech was present)')
                print(
                    f'False alarm over channels: {false_alarm_percentage * 100:.3f}% (Marked as active, although speech was '
                    f'not present)')

                print(f'System accuracy: {(1 - der) * 100:.3f}%')

            return miss_percentage, false_alarm_percentage, 0, (1 - der)
