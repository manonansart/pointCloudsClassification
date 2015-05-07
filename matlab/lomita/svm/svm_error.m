function erreur = svm_error(pred, Ytest)
	erreur = (sum(pred .* Ytest < 0) / length(Ytest)) * 100
end