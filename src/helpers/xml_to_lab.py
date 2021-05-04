import sys
from bs4 import BeautifulSoup

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as source:
    with open(output_file, 'w') as destination:
        content = source.read()
        soup = BeautifulSoup(content, "lxml")
        items = soup.findAll('turn'
                             )
        reference = [(item.get("speaker"), item.get("starttime"), item.get("endtime")) for item in
                     items]
        for line in reference:
            if line[0] != 'spk2':
                destination.write(f'{float(line[1]):.3f} {float(line[2]):.3f} sp\n')
