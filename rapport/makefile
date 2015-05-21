

all:
	pdflatex -shell-escape rapport.tex && pdflatex -shell-escape rapport.tex
	bibtex rapport
	pdflatex -shell-escape rapport.tex && pdflatex -shell-escape rapport.tex
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.toc
	rm -rf *.maf
	rm -rf *.mtc*
	rm -rf *.out
	rm -rf *.pyg
	rm -rf *.bbl
	rm -rf *.blg