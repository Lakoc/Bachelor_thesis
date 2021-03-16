import params
import re
import numpy as np

regex = re.compile(r'(\d*\.\d*)\s*(\d*\.\d*)\s*spk(\d)')


def extract(l):
    x = regex.search(l)
    return int(float(x[1]) * 100), int(float(x[2]) * 100), 3 - int(x[3])


def calculate_success_rate():
    with open(f'{params.output_folder}/diarization', 'r') as f1:
        with open(f'data_to_delete/ukazka1/prezentace/spklab.2.diar', 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

            lines11 = []
            for line in lines1:
                lines11.append(extract(line))
            lines22 = []
            for line in lines2:
                lines22.append(extract(line))

            max_val = max(lines11[len(lines11) - 1][1], lines22[len(lines22) - 1][1])
            l1 = np.zeros(max_val)
            for line in lines1:
                from_t, to_t, value = extract(line)
                l1[from_t: to_t + 1] = value

            l2 = np.zeros(max_val)
            for line in lines2:
                from_t, to_t, value = extract(line)
                l2[from_t: to_t + 1] = value


            vad = np.sum(np.logical_and(l1, l2)) / np.sum(np.logical_or(l1, l2))
            different = np.logical_and(np.logical_and((l1 != l2), l1 != 0), l2 != 0)
            likelihood = 1 - np.mean(different)
            print(f'Vad hit rate:{vad}')
            print(f'Diarization hit rate:{likelihood}')

if __name__ == '__main__':
    calculate_success_rate()
