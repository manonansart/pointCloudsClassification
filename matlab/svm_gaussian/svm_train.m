function [alpha, b, pos, moyenne, variance] = svm_train(X, Y)

	[Xapp, Yapp, Xval, Yval] = splitdata(X, Y, 0.7);

	[nApp, p] = size(Xapp);
	[nVal, p] = size(Xval);

	moyenne = mean(Xapp);
	variance = std(Xapp);

	% Center and reduce
	Xapp = (Xapp - ones(nApp, 1) * moyenne) ./ (ones(nApp, 1) * variance); 
	Xval = (Xval - ones(nVal, 1) * moyenne) ./ (ones(nVal, 1) * variance);
		

	%% Calculate K kernel and G matrix
	kernel = 'gaussian';
	kerneloption = 2;

	K = svmkernel(Xapp, kernel, kerneloption, Xapp);
	G = (Yapp*Yapp').*K;

	% Initiation the parameters
	e = ones(nApp,1);
	epsilon = 10^-5;
	lambda = eps^.5;

	nbErrMin = inf;
	bestC1 = 0;
	bestCMoins1 = 0;

	% Values of C and CMoins1 to try for validation
	C_list = logspace(log10(.1), log10(1000), 50);
	CMoins_list = logspace(log10(.1), log10(1000), 50);

	for i = 1:length(C_list)
		C1 = C_list(i);
		vecteurC = zeros(nApp, 1);
		vecteurC(find(Yapp == 1)) = C1;
		for j= 1:length(CMoins_list)
			CMoins1 = CMoins_list(i);
			vecteurC(find(Yapp == -1)) = CMoins1;
			matriceC = diag(1 ./ vecteurC);
			H = G + matriceC;

			%% Solve with monqp
			[alpha, b, pos] = monqp(H, e, Yapp, 0, inf, lambda, 0);
		    
		    ypred = svm_predict(Xval, kernel, kerneloption, alpha, b, pos, Xapp, Yapp);

			% Number of errors
			nbErr = sum(ypred ~= Yval);

			% Updates the parameters if better result
			if (nbErr < nbErrMin)
				nbErrMin = nbErr;
				bestC1 = C1;
				bestCMoins1 = CMoins1;
			end
		end
		if (i == 25)
			disp('Hang in there, you are half-way throw the calculation')
		end
	end

	% Recalculate the parameters with the best C
	vecteurC(find(Yapp == 1)) = bestC1;
	vecteurC(find(Yapp == -1)) = bestCMoins1;
	matriceC = diag(1 ./ vecteurC);
	H = G + matriceC;

	%% Solve with monqp
	[alpha, b, pos] = monqp(H, e, Yapp, 0, inf, lambda, 0);

end