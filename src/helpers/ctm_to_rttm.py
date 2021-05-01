import sys
import re

name = sys.argv[1]
line_re = re.compile(r'(terapeut|client) \S ([\d.]*) ([\d.]*) (\S*) [\d.]*')

with open(name, 'r') as source:
    with open(f'{name.split(".ctm")[0]}.rttm', 'w') as out:
        lines = source.readlines()
        lines_processed = []
        for line in lines:
            match = line_re.search(line)
            lines_processed.append([2 if match[1] == "client" else 1, float(match[2]), float(match[3]), match[4]])

        lines_processed = sorted(lines_processed, key=lambda l: l[1])
        for line in lines_processed:
            out.write(
                f'SPEAKER {name.split(".ctm")[0].split("/")[-1]} {line[0]} {line[1]:.2f} {line[2]:.2f} <NA> <NA> {line[0]} <NA> <NA>\n')
