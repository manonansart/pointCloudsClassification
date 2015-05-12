function pred = svm_predict_linear(X, w, b)

	pred = X * w + b;
	pred(find(pred > 0)) = 1;
	pred(find(pred < 0)) = -1;
end