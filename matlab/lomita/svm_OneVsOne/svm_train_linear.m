function [w, b] = svm_train_linear(Xapp, Yapp, Xval, Yval)

	[nApp, p] = size(Xapp);
	[nVal, p] = size(Xval);

	DY = diag(Yapp);
	G = Xapp * Xapp';
	H = DY * G * DY + 1e-8*eye(size(G));

	e = ones(nApp,1);
	lambda = eps^.5;


	% Validation for C : rough C first
	C_listBig = logspace(log10(.1), log10(10000), 50);
	nbErrMin = inf;
	bestC = 0;

	for i = 1:length(C_listBig)
		C = C_listBig(i);
		[alpha, b, pos] = monqp(H, e, Yapp, 0, C, lambda, 0);    
		w = Xapp(pos, :)' *(Yapp(pos) .* alpha);
		nbErr = sum((Xval * w + b) .* Yval < 0);
		if (nbErr < nbErrMin)
			nbErrMin = nbErr;
			bestC = C_listBig(i);
		end
	end


	% Validation for C : precise C next
	C_listSmall = logspace(log10(0.9 * bestC), log10(1.1 * bestC), 50);
	nbErrMin = inf;
	bestC = 0;

	for i = 1:length(C_listSmall)
		C = C_listSmall(i);
		[alpha, b, pos] = monqp(H, e, Yapp, 0, C, lambda, 0);    
		w = Xapp(pos, :)' *(Yapp(pos) .* alpha);
		nbErr = sum((Xval * w + b) .* Yval < 0);
		if (nbErr < nbErrMin)
			nbErrMin = nbErr;
			bestC = C_listSmall(i);
		end
	end

	% Recalculate the parameters with the best C
	[alpha, b, pos] = monqp(H, e, Yapp, 0, C, lambda, 0);    
	w = Xapp(pos, :)' *(Yapp(pos) .* alpha);

end