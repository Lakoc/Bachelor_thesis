import params
import re
import numpy as np
from bs4 import BeautifulSoup

regex = re.compile(r'(\d*\.\d*)\s*(\d*\.\d*)\s*spk(\d)')


def extract(line):
    x = regex.search(line)
    return int(float(x[1]) * 1000), int(float(x[2]) * 1000), int(x[3])


def extract_str(line):
    x = regex.search(line)
    return x[3], float(x[1]), float(x[2])


def extract_transcribed(line):
    return int(float(line[1]) * 1000), int(float(line[2]) * 1000), int(line[0][-1]) if line[0] is not None else 0


def extract_transcribed_str(line):
    return line[0][-1] if line[0] is not None else '0', float(line[1]), float(line[2])


def calculate_success_rate():
    # with open(f'data_to_delete/ukazka1/prezentace/spklab.2.diar', 'r') as f1:
    with open(f'{params.output_folder}/diarization', 'r') as f1:
        with open(f'data_to_delete/transcription/M_16k.2.trs', 'r') as reference_file:
            content = reference_file.read()
            soup = BeautifulSoup(content, "lxml")
            items = soup.findAll('turn'
                                 )
            reference = [(item.get("speaker"), item.get("starttime"), item.get("endtime")) for item in
                         items]

            lines1 = f1.readlines()

            lines11 = []
            for line in lines1:
                lines11.append(extract(line))

            max_val = max(lines11[len(lines11) - 1][1], int(float(reference[len(reference) - 1][2]) * 1000))
            hyp = np.zeros(max_val)
            for line in lines1:
                from_t, to_t, value = extract(line)
                hyp[from_t: to_t + 1] = value

            reference_val = np.zeros(max_val)
            collar = 250  # ms
            collar_one_side = collar // 2
            for line in reference:
                from_t, to_t, value = extract_transcribed(line)
                reference_val[from_t - collar_one_side: from_t + 1] = -1
                reference_val[from_t: to_t + 1] = np.where(reference_val[from_t: to_t + 1] == -1, -1, value)
                reference_val[to_t + 1: to_t + 1 + collar_one_side] = -1

            miss = np.sum(np.logical_and(reference_val > 0, hyp == 0))
            false_alarm = np.sum(np.logical_and(hyp > 0, reference_val == 0))
            confusion = np.sum(np.logical_and(reference_val != hyp, np.logical_and(hyp > 0, reference_val > 0)))
            der = (miss + false_alarm + confusion) / reference_val.shape[0]

            # # active_segments_count = np.sum(l2 > 0)
            # # vad_x = np.where(l1 > 0, 1, -1)
            # # vad_multiplied = vad_x * (l2 > 0).astype(int)
            # # vad = np.sum(vad_multiplied) / active_segments_count
            # false_positive_vad = np.sum(np.logical_and(l1 > 0, l2 == 0)) / max_val
            # true_negative_vad = np.sum(np.logical_and(l1 == 0, l2 > 0)) / max_val
            # vad = 1 - false_positive_vad - true_negative_vad
            #
            # overall_likelihood = np.mean(l1 == l2)
            # print(f'Vad hit rate:{vad:.3f}')
            print(
                f'Miss rate: {(miss / reference_val.shape[0]):.3f} (Marked as non-active, although speech was present)')
            print(
                f'False alarm: {(false_alarm / reference_val.shape[0]):.3f} (Marked as active, although speech was not present)')
            print(
                f'Confusion: {(confusion / reference_val.shape[0]):.3f} (System provided speaker tag and the reference speech was present, but does not match)')
            print(f'Diarization error rate:{der:.3f}')
            print(f'Diarization accuracy:{1 - der:.3f}')
            # print(f'System hit rate overall:{overall_likelihood:.3f}')


if __name__ == '__main__':
    calculate_success_rate()
