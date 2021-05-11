import sys
from bs4 import BeautifulSoup

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as source:
    with open(output_file, 'w') as destination:
        content = source.read()
        soup = BeautifulSoup(content, "lxml")
        file_name = input_file.split('.wav')[0].split('/')[-1]
        items = soup.findAll('turn'
                             )
        reference = [(item.get("speaker"), item.get("starttime"), item.get("endtime")) for item in
                     items]
        for line in reference:
            if line[0] != 'spk2':
                speaker = int(line[0][-1])
                if speaker == 3:
                    speaker = 1
                else:
                    speaker = 2
                destination.write(f'SPEAKER {file_name} {speaker} {float(line[1]):.3f} {float(line[2]) - float(line[1]):.3f} <NA> <NA> {speaker} <NA> <NA>\n')
