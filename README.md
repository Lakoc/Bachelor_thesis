# Psychotherapy session analysis
Author: Alexander Polok

Date: 7.5.2021


This project performs analysis of psychotherapeutic sessions. 
Classifiers describing the therapy are extracted from the audio recordings. 
These are then aggregated, compared with other sessions, and graphically presented in a report summarizing the conversation. 
In this way, therapists are provided with feedback that can serve for professional growth and better psychotherapy in the future.

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
python3 src/process_online_diarization.py sample_data/en sample_output
```
2. Evaluation of diarization, 1 - means online mode, 0 - means 0ms collar 

```
python3 src/diarization_evaluation.py sample_output sample_data/en 1 0 
```

3. Creation of transcriptions by using Google Cloud Api
```
cp sample_data/*.wav sample_output
python3 src/create_transcriptions.py sample_output sample_output --language=en-US
```
Or reference transcriptions could be used
```
cp sample_data/en/*.txt sample_output
```

4. Creation of stats for each session
```
python3 src/create_stats.py sample_output sample_output
```

5. Calculation of overall stats
```
python3 src/calculate_overall_stats.py sample_output overall_stats.json
```

6. Creation of HTML document containing information about session
```
python3 src/create_outputs.py sample_output sample_output templates/online_session.html callhome_stats.json texts.json 
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
