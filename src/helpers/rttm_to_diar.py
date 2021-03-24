import re

with open('session2VBX.rttm', 'r') as file:
    lines = file.readlines()
    reg = re.compile(r'.* .* \d (\d+\.\d+) (\d+\.\d+) <NA> <NA> (\d)')
    with open('output/diarization', 'w') as out:
        for line in lines:
            match = reg.search(line)
            start, dur, spk = float(match[1]), float(match[2]), match[3]
            out.write(f'{start:.2f}\t{start + dur:.2f}\tspk{spk}\n')
