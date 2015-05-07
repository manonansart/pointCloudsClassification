close all
clear all
clc

% Note: don't forget to add splitdata, monqp and svmkernel to the path.
% Results: first run : error rate 0.16% and runtime 360s
%          second run : error rate 0.15% and runtime 384s

%%  Load the data and replace text labels

data = load('../dataset/dish_area_dataset/attributesSmall.csv');

% Gets the columns from the text file
Y = load('../dataset/dish_area_dataset/labelsSmall.csv');

% Replaces the label background (1) with -1 and the label car (2) with 1
Y = (Y - 1) * 2 - 1;

%% Split the data into app, val and test
[Xapp, Yapp, Xtest, Ytest] = splitdata(data, Y, 0.5);
[Xval, Yval, Xtest, Ytest] = splitdata(Xtest, Ytest, 0.5);

[nApp, p] = size(Xapp);
[nTest, p] = size(Xtest);
[nVal, p] = size(Xval);

moyenne = mean(Xapp);
variance = std(Xapp);

% Center and reduce
Xapp = (Xapp - ones(nApp, 1) * moyenne) ./ (ones(nApp, 1) * variance); 
Xval = (Xval - ones(nVal, 1) * moyenne) ./ (ones(nVal, 1) * variance); 
Xtest = (Xtest - ones(nTest, 1) * moyenne) ./ (ones(nTest, 1) * variance);


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
	    
	    % pos correspond aux positions des alphas differents de 0. alpha = a(pos)
		a4 = zeros(nApp, 1);
		a4(pos) = alpha;

		
		%% Predictions on validation set
		Kgrid = svmkernel(Xval, kernel, kerneloption, Xapp(pos, :));
		ypred = Kgrid*(Yapp(pos).*alpha) + b;

		% Calculate new labels
		ypred(find(ypred > 0)) = 1;
		ypred(find(ypred < 0)) = -1;

		% Number of errors
		nbErr = length(find(ypred - Yval > 0.0001));

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

vecteurC = zeros(nApp, 1);
vecteurC(find(Yapp == 1)) = bestC1;
vecteurC(find(Yapp == -1)) = bestCMoins1;
matriceC = diag(1 ./ vecteurC);
H = G + matriceC;

%% Solve with monqp
[alpha, b, pos] = monqp(H, e, Yapp, 0, inf, lambda, 0);

% pos correspond aux positions des alphas differents de 0. alpha = a(pos)
a4 = zeros(nApp, 1);
a4(pos) = alpha;

%% Predictions on test set to calculate the error rate
Kgrid = svmkernel(Xtest, kernel, kerneloption, Xapp(pos, :));
ypred = Kgrid*(Yapp(pos).*alpha) + b;

% Calculate new labels
ypred(find(ypred > 0)) = 1;
ypred(find(ypred < 0)) = -1;

% Calculate the final error rate (in percent)
erreur = (length(find(ypred - Ytest > 0.0001)) / nTest) * 100
bestCMoins1  
bestC1