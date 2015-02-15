all : compilRapport clean
compilRapport : rapport.pdf

rapport.pdf : rapport.tex
	pdflatex -shell-escape rapport.tex && pdflatex -shell-escape rapport.tex 
clean : 
	rm rapport.aux rapport.log rapport.maf rapport.mtc* rapport.out rapport.toc rapport.tex.* rapport.pyg