import params
import re
import numpy as np
from bs4 import BeautifulSoup

regex = re.compile(r'(\d*\.\d*)\s*(\d*\.\d*)\s*spk(\d)')


def extract(line):
    x = regex.search(line)
    return int(float(x[1]) * 1000), int(float(x[2]) * 1000), int(x[3])


def extract_transcribed(line):
    return int(float(line[1]) * 1000), int(float(line[2]) * 1000), int(line[0][-1]) if line[0] is not None else 0


def calculate_success_rate():
    with open(f'{params.output_folder}/diarization', 'r') as f1:
        with open(f'data_to_delete/transcription/M_16k.2.trs', 'r') as f2:
            content = f2.read()
            soup = BeautifulSoup(content, "lxml")
            items = soup.findAll('turn'
                                 )
            lines22 = [(item.get("speaker"), item.get("starttime"), item.get("endtime")) for item in
                       items]

            lines1 = f1.readlines()

            lines11 = []
            for line in lines1:
                lines11.append(extract(line))

            max_val = max(lines11[len(lines11) - 1][1], int(float(lines22[len(lines22) - 1][2]) * 1000))
            l1 = np.zeros(max_val)
            for line in lines1:
                from_t, to_t, value = extract(line)
                l1[from_t: to_t + 1] = value

            l2 = np.zeros(max_val)
            for line in lines22:
                from_t, to_t, value = extract_transcribed(line)
                l2[from_t: to_t + 1] = value

            # active_segments_count = np.sum(l2 > 0)
            # vad_x = np.where(l1 > 0, 1, -1)
            # vad_multiplied = vad_x * (l2 > 0).astype(int)
            # vad = np.sum(vad_multiplied) / active_segments_count
            false_positive_vad = np.sum(np.logical_and(l1 > 0, l2 == 0)) / max_val
            true_negative_vad = np.sum(np.logical_and(l1 == 0, l2 > 0)) / max_val
            vad = 1 - false_positive_vad - true_negative_vad

            overall_likelihood = np.mean(l1 == l2)
            print(f'Vad hit rate:{vad:.3f}')
            print(f'False positive vad:{false_positive_vad:.3f} (Marked as non-active, although speech was present)')
            print(f'True negative vad:{true_negative_vad:.3f} (Marked as active, although speech was not present)')
            print(f'Only diarization hit rate:{(overall_likelihood / vad):.3f}')
            print(f'System hit rate overall:{overall_likelihood:.3f}')


if __name__ == '__main__':
    calculate_success_rate()
