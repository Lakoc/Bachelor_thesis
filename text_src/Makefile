# makefile pro preklad LaTeX verze Bc. prace
# makefile for compilation of the thesis
# (c) 2008 Michal Bidlo
# E-mail: bidlom AT fit vutbr cz
# Edited by: dytrych AT fit vutbr cz
#===========================================
# asi budete chtit prejmenovat / you will probably rename:
CO=projekt

all: $(CO).pdf

pdf: $(CO).pdf

$(CO).ps: $(CO).dvi
	dvips $(CO)

$(CO).pdf: clean
	pdflatex $(CO)
	-bibtex $(CO)
	pdflatex $(CO)
	pdflatex $(CO)

$(CO).dvi: $(CO).tex $(CO).bib
	latex $(CO)
	-bibtex $(CO)
	latex $(CO)
	latex $(CO)

clean:
	rm -f *.dvi *.log $(CO).blg $(CO).bbl $(CO).toc *.aux $(CO).out $(CO).lof $(CO).ptc
	rm -f $(CO).pdf
	rm -f *~

pack:
	tar czvf $(CO).tar.gz *.tex *.bib *.bst ./template-fig/* ./bib-styles/* ./cls/* zadani.pdf $(CO).pdf Makefile Changelog

rename:
	mv $(CO).tex $(NAME).tex
	mv $(CO)-kapitoly-chapters.tex $(NAME)-kapitoly-chapters.tex
	mv $(CO)-kapitoly-chapters-en.tex $(NAME)-kapitoly-chapters-en.tex
	mv $(CO)-literatura-bibliography.bib $(NAME)-literatura-bibliography.bib
	mv $(CO)-prilohy-appendices.tex $(NAME)-prilohy-appendices.tex
	mv $(CO)-prilohy-appendices-en.tex $(NAME)-prilohy-appendices-en.tex
	sed -i "s/$(CO)-kapitoly-chapters/$(NAME)-kapitoly-chapters/g" $(NAME).tex
	sed -i "s/$(CO)-kapitoly-chapters-en/$(NAME)-kapitoly-chapters-en/g" $(NAME).tex
	sed -i "s/$(CO)-literatura-bibliography/$(NAME)-literatura-bibliography/g" $(NAME).tex
	sed -i "s/$(CO)-prilohy-appendices/$(NAME)-prilohy-appendices/g" $(NAME).tex
	sed -i "s/$(CO)-prilohy-appendices-en/$(NAME)-prilohy-appendices-en/g" $(NAME).tex
	sed -i "s/$(CO)/$(NAME)/g" Makefile
	
# Pozor, vlna neresi vse (viz popis.txt) / Warning - vlna is not solving all problems (see description.txt)
vlna:
	vlna -l $(CO)-*.tex

# Spocita normostrany / Count of standard pages
normostrany:
	echo "scale=2; `detex -n $(CO)*.tex | wc -c`/1800;" | bc

