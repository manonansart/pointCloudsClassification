function [w, b, moyenne, variance] = svm_train_linear(X, Y)

	[Xapp, Yapp, Xval, Yval] = splitdata(X, Y, 0.7);

	[nApp, p] = size(Xapp);
	[nVal, p] = size(Xval);

	moyenne = mean(Xapp);
	variance = std(Xapp);

	% Center and reduce
	Xapp = (Xapp - ones(nApp, 1) * moyenne) ./ (ones(nApp, 1) * variance); 
	Xval = (Xval - ones(nVal, 1) * moyenne) ./ (ones(nVal, 1) * variance);
		

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
		C = construireC(Yapp, C_listBig(i));
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
		C = construireC(Yapp, C_listSmall(i));
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