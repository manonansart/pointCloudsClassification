function ypred = svm_predict_linear(Xtest, kernel, kerneloption, alpha, b, pos, Xapp, Yapp)

	%% Predictions on test set to calculate the error rate
	Kgrid = svmkernel(Xtest, kernel, kerneloption, Xapp(pos, :));
	ypred = Kgrid*(Yapp(pos).*alpha) + b;

	% Calculate new labels
	ypred = sign(ypred);
end