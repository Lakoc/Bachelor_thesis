import numpy as np
from bs4 import BeautifulSoup


def extract_transcribed(line):
    return int(float(line[1]) * 100), int(float(line[2]) * 100), int(line[0][-1]) if line[0] is not None else 0


def vad_test(vad):
    with open(f'data_to_delete/transcription/M_16k.2.trs', 'r') as f2:
        content = f2.read()
        soup = BeautifulSoup(content, "lxml")
        items = soup.findAll('turn'
                             )
        lines22 = [(item.get("speaker"), item.get("starttime"), item.get("endtime")) for item in
                   items]

        max_val = max(len(vad), int(float(lines22[len(lines22) - 1][2]) * 100))

        l2 = np.zeros(max_val)
        for line in lines22:
            from_t, to_t, value = extract_transcribed(line)
            l2[from_t: to_t + 1] = value

        l1 = np.logical_or(vad[:, 0], vad[:, 1])
        l1 = np.append(l1, np.zeros(max_val - len(l1)))
        # active_segments_count = np.sum(l2 > 0)
        # vad_x = np.where(l1 > 0, 1, -1)
        # vad_multiplied = vad_x * (l2 > 0).astype(int)
        # vad = np.sum(vad_multiplied) / active_segments_count
        false_positive_vad = np.sum(np.logical_and(l1 > 0, l2 == 0)) / max_val
        true_negative_vad = np.sum(np.logical_and(l1 == 0, l2 > 0)) / max_val
        vad = 1 - false_positive_vad - true_negative_vad

        print(f'Vad hit rate:{vad:.3f}')
        print(f'False positive vad:{false_positive_vad:.3f} (Marked as non-active, although speech was present)')
        print(f'True negative vad:{true_negative_vad:.3f} (Marked as active, although speech was not present)')
