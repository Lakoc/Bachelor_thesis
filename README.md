## Ukázky spuštění skriptů

1. Diarizace
```bash
python3 src/process_online_diarization.py data_to_delete/callhome/callhome_trimmed/ CallHomeProcessed
```
2. Evaluace diarizace

```bash
python3 src/diarization_evaluation.py test/ data_to_delete/dictaphone/ 1 0 
```

3. Tvorba transkripce nebo případná kopie referenčních anotací
```bash
cp data_to_delete/callhome/callhome_trimmed/*.wav CallHomeProcessed
python3 src/create_transcriptions.py CallHomeProcessed CallHomeProcessed --language=en-US
```
```bash
cp data_to_delete/callhome/callhome_trimmed/*.txt CallHomeProcessed
```

4. Tvorba statistik pro příslušné nahrávky
```bash
python3 src/create_stats.py CallHomeProcessed/ CallHomeProcessed/
```

5. Kalkulace průměrných statistik
```bash
python3 src/calculate_overall_stats.py CallHomeProcessed/ callhome_stats.json
```

6. Tvorba souhrnných zpráv
```bash
python3 src/create_outputs.py CallHomeProcessed/ CallHomeProcessed/ templates/online_session.html callhome_stats.json texts.json 
```