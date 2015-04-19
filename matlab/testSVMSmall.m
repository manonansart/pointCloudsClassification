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

%% Split the data into app and test
[Xapp, Yapp, Xtest, Ytest] = splitdata(data, Y, 0.67);

[nApp, p] = size(Xapp);
[nTest, p] = size(Xtest);

moyenne = mean(Xapp);
variance = std(Xapp);

% Center and reduce
Xapp = (Xapp - ones(nApp, 1) * moyenne) ./ (ones(nApp, 1) * variance); 
Xtest = (Xtest - ones(nTest, 1) * moyenne) ./ (ones(nTest, 1) * variance); 


%% Calculate K kernel and G matrix
kernel = 'gaussian';
kerneloption = 2;

K = svmkernel(Xapp, kernel, kerneloption, Xapp);

C1 = 1000;
CMoins1 = 1000;

G = (Yapp*Yapp').*K;
vecteurC = zeros(nApp, 1);
vecteurC(find(Yapp == 1)) = C1;
vecteurC(find(Yapp == -1)) = CMoins1;
matriceC = diag(1 ./ vecteurC);

H = G + matriceC;

e = ones(nApp,1);
epsilon = 10^-5;

%% Solve with monqp
tic
	lambda = eps^.5;
	[alpha, b, pos] = monqp(H, e, Yapp, 0, inf, lambda, 0);
    
    % pos correspond aux positions des alphas differents de 0. alpha = a(pos)
	a4 = zeros(nApp, 1);
	a4(pos) = alpha;
toc

Kgrid = svmkernel(Xtest, kernel, kerneloption, Xapp(pos, :));
ypred = Kgrid*(Yapp(pos).*alpha) + b;

% Calculate new labels
ypred(find(ypred > 0)) = 1;
ypred(find(ypred < 0)) = -1;

% Calculate the error rate (in percent)
erreur = (length(find(ypred - Ytest > 0.0001)) / nTest) * 100  
