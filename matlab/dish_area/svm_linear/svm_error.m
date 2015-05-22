function erreur = svm_error(ypred, Ytest)
	erreur = (sum(ypred ~= Ytest) / length(Ytest)) * 100
end