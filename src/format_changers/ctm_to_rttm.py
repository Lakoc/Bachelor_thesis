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

        lines_merged = []
        i = 0
        while i < len(lines_processed):
            speaker = lines_processed[i][0]
            start = lines_processed[i][1]
            end = start + lines_processed[i][2]
            text = lines_processed[i][3]
            j = i + 1
            while j < len(lines_processed):
                speaker_next = lines_processed[j][0]
                if speaker == speaker_next:
                    start_next = lines_processed[j][1]
                    if start_next - end < 0.5:
                        end_next = start_next + lines_processed[j][2]
                        end = end_next
                        text = f'{text} {lines_processed[j][3]}'
                        j += 1
                    else:
                        lines_merged.append([speaker, start, end -start, text])
                        break
                else:
                    lines_merged.append([speaker, start, end- start, text])
                    break
            i = j

        lines_merged = sorted(lines_merged, key=lambda l: l[1])
        for line in lines_merged:
            out.write(
                f'SPEAKER {name.split(".ctm")[0].split("/")[-1]} {line[0]} {line[1]:.2f} {line[2]:.2f} <NA> <NA> {line[0]} <NA> <NA>\n')
