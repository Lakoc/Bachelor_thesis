import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import params


def extract_transcribed(line):
    return int(float(line[1]) * 100), int(float(line[2]) * 100), int(line[0][-1]) if line[0] is not None else 0


def vad_test(vad):
    with open(params.transcription_path, 'r') as f2:
        content = f2.read()
        soup = BeautifulSoup(content, "lxml")
        items = soup.findAll('turn'
                             )
        lines22 = [(item.get("speaker"), item.get("starttime"), item.get("endtime")) for item in
                   items]

        max_val = max(len(vad), int(float(lines22[len(lines22) - 1][2]) * 100))

        reference = np.zeros(max_val)
        collar_one_side = params.collar_size // 2
        for line in lines22:
            from_t, to_t, value = extract_transcribed(line)
            reference[np.max([from_t - collar_one_side, 0]): from_t + collar_one_side] = -1
            reference[from_t + collar_one_side: to_t + 1 - collar_one_side] = value
            reference[to_t + 1 - collar_one_side: np.min([to_t + 1 + collar_one_side, max_val])] = -1

        hyp = np.logical_or(vad[:, 0], vad[:, 1])
        hyp = np.append(hyp, np.zeros(max_val - len(hyp)))

        miss = np.logical_and(reference > 0, hyp == 0)
        false_alarm = np.logical_and(hyp > 0, reference == 0)

        if params.transcription_plot:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(miss, label='miss', color='red')
            ax.plot(false_alarm, label='false_alarm', color='blue')
            fig.legend()
            fig.show()

        false_alarm = np.sum(false_alarm) / max_val
        miss = np.sum(miss) / max_val
        vad = 1 - miss - false_alarm
        print(f'Vad hit rate:{vad:.3f}')
        print(
            f'Miss rate: {miss:.3f} (Marked as non-active, although speech was present)')
        print(
            f'False alarm: {false_alarm:.3f} (Marked as active, although speech was not present)')
