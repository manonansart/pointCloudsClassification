function [w, b] = svm_train_linear(Xapp, Yapp, C)

	[nApp, p] = size(Xapp);

	DY = diag(Yapp);
	G = Xapp * Xapp';
	H = DY * G * DY + 1e-8*eye(size(G));

	e = ones(nApp,1);
	lambda = eps^.5;

	[alpha, b, pos] = monqp(H, e, Yapp, 0, C, lambda, 0);    
	w = Xapp(pos, :)' *(Yapp(pos) .* alpha);

end