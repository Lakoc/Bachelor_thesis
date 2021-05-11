# Psychotherapy session analysis
Author: Alexander Polok

Date: 11.5.2021

This project performs analysis of psychotherapeutic sessions. 
Classifiers describing the therapy are extracted from the audio recordings. 
These are then aggregated, compared with other sessions, and graphically presented in a report summarizing the conversation. 
In this way, therapists are provided with feedback that can serve for professional growth and better psychotherapy in the future.

See demonstration outputs, which are located in the `sample_output/` directory.
For more details about analysis, see `thesis.pdf` file.
## Prerequisites

Python 3.8.5

## Installation

Install the required packages with:  
```pip install -r requirements.txt```

In case of using sphinx ASR install required library by:  
```pip install -U sphinx```

## Usage example

1. Diarization of online psychotherapy  
```
python3 src/process_online_diarization.py sample_data/en output
```

```
usage: process_online_diarization.py [-h] src dest

Module for processing diarization over wav files in provided directory.

positional arguments:
  src         source path of wav files
  dest        destination path for rttm files

optional arguments:
  -h, --help  show this help message and exit
```

2. Evaluation of diarization, 1 - online mode, 0 - 0ms collar 

```
python3 src/diarization_evaluation.py output sample_data/en 1 0 
```

```
usage: diarization_evaluation.py [-h] [--verbose] hyp ref {1,2,3} collar

Module for DER (diarization error rate) calculation.

positional arguments:
  hyp         path of hypothesis rttm files
  ref         path of reference rttm files
  {1,2,3}     type of evaluation (1 - online, 2 - dictaphone, 3 - dictaphone with reference)
  collar      error collar size in ms

optional arguments:
  -h, --help  show this help message and exit
  --verbose   print error rate for each file
```

3. Creation of transcriptions by using Google Cloud Api
```
cp sample_data/en/*.wav output
python3 src/create_transcriptions.py output output --language=en-US
```

```
usage: create_transcriptions.py [-h] [--language LANGUAGE] src dest

Module for processing transcription over wav according to rttm files in provided directory.

positional arguments:
  src                  source path of wav and rttm files
  dest                 destination path for txt files

optional arguments:
  -h, --help           show this help message and exit
  --language LANGUAGE  language tag of selected language
```

Or reference transcriptions could be used by copying them

```
cp sample_data/en/*.txt output
```

4. Calculate **Word error rate**
```
python3 src/transcription_evaluation.py output/ sample_data/en/
```

```
usage: transcription_evaluation.py [-h] [--verbose] hyp ref

Module for checking word error rate.

positional arguments:
  hyp         path of hypothesis rttm files
  ref         path of reference rttm files

optional arguments:
  -h, --help  show this help message and exit
  --verbose   print error for each file
```

5. Creation of stats for each session
```
python3 src/create_stats.py output output
```

```
usage: create_stats.py [-h] src dest

Module for calculation of statistic for each file in source destination.

positional arguments:
  src         source path of wav, rttm and txt files
  dest        destination path for pickle files

optional arguments:
  -h, --help  show this help message and exit
```

6. Calculation of overall stats
```
python3 src/calculate_overall_stats.py output overall_stats.json
```

```
usage: calculate_overall_stats.py [-h] src dest

Module for calculating overall statistics over dataset.

positional arguments:
  src         source path of pickle files
  dest        destination path for json file

optional arguments:
  -h, --help  show this help message and exit
```

7. Creation of HTML document containing information about session
```
python3 src/create_outputs.py output output templates/online_session.html sample_stats/callhome_stats.json templates/texts.json 
```

```
usage: create_outputs.py [-h] [--hide_emotions] [--hide_interruptions] src dest template overall_stats text

Module for creating html documents providing stats about each audio file. Generated plots could be found in subdir \plots.

positional arguments:
  src                   source path of pickle files
  dest                  destination path for pickle files
  template              html document template
  overall_stats         path of file containing overall stats
  text                  path of file containing template texts

optional arguments:
  -h, --help            show this help message and exit
  --hide_emotions       remove emotions from html document
  --hide_interruptions  remove interruptions from html document
```

## Project structure

    .
    ├── LICENSE                 # MIT license
    ├── README.md               # Contents of this file
    ├── requirements.txt        # Python libraries requirements
    ├── sample_data             # Directory containing sample data with its anotations
    ├── sample_stats            # Samples of overall stats calculated on datasets 
    ├── src                     # Source files
    └── templates               # Files used for creation of output HTML file

## Runnable scripts used for analysis
    src/
    ├── calculate_overall_stats.py
    ├── create_outputs.py
    ├── create_stats.py
    ├── create_transcriptions.py
    ├── diarization_evaluation.py
    ├── process_dictaphone_diarization.py
    ├── process_online_diarization.py
    └── transcription_evaluation.py


## License

This project is licensed under the terms of the MIT license.
