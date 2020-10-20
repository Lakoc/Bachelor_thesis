#Zeptat se

Fonemovy rozpoznavac 32bit lib STK nejde zkompilovat vyzaduje g++ 4.1
Pokusit se instalovat rozsireni pro 32bit system?
Vydat se smerem trid nebo funkci?
Pridat na vedouciho na git?
Vytvorit jiru/trello pro tasky?

## Preprocessing
1. Preemfaze - provest jeste horni propust?
2. Hamming - upravit velikost okna?

## Analyza
1. Pruchody nulou - kmitocet, lze detekovat zacatek slov?
2. VAD pridat vic parametru, provest postprocessing pro odstraneni peaku? -> natrenovat komplikovanejsi model? 
https://wiki.aalto.fi/pages/viewpage.action?pageId=151500905
3. Formantove priznaky - Zakladni ton reci - pouzit LPC pro jejich vypocet?
Dokazeme pomoci nich rozlisit fonemy
Z zakladni frekvence dokazeme zjistit kdo mluvi v jakem je rozpolozeni .
4. Crosstalk - Vychazet zde z LPC a pomoci toho se snazit urcit kdo pravdepodobne mluvi vzhledem k cele mluve? Nebo se pokusit natrenovat nejaky model?

LPC - predpoved n-teho vzorku pomoci kombinace aktualnich dat
provest MFCC? 

# Statistiky
Pro rozpoznani mezer a delky vet natrenovat model? Threshholding v tomto pripade smysl moc nedava?

# TODO
pomer reci, procento kdy kdo mluvil,  cross talky- pocty , kruhovy graf, vad rozdelen a spojene , vahani v monologu - pocty prumery delky sumu delek,  procentualne zobrazovat, najit studie jak zobrazovat data, citelnost a preyentovatelnost dat,  cekani odstranit min a max, lpc 