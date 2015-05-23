% Note : About 3% error (on 10 runs) with the small dataset w/o unlabeled but with the
% class background over-represented.
% We now need to test it on the entire dataset w/o unlabeled and with a
% class smaller background class 
clear all
close all
clc

%% Load and split the data

data = load('../../../dataset/lomita/attributes_without_unlabeled_and_reduced_background');

Y = load('../../../dataset/lomita/labels_without_unlabeled_and_reduced_background.csv');

% Split the data into app and test
[Xapp, Yapp, Xtest, Ytest] = splitdata(data, Y, 0.60);

[n, p] = size(Xapp);
[nt, pt] = size(Xtest);

moyenne = mean(Xapp);
variance = std(Xapp);

% Center and reduce
Xapp = (Xapp - ones(n, 1) * moyenne) ./ (ones(n, 1) * variance);
Xtest = (Xtest - ones(nt, 1) * moyenne) ./ (ones(nt, 1) * variance);

%% 1 - Clean data (to be refactored)

X1 = Xapp(Yapp==1);
Y1 = Yapp(Yapp==1);
X2 = Xapp(Yapp==2);
X3 = Xapp(Yapp==3);
X4 = Xapp(Yapp==4);

[n1, p1] = size(X1);
[n2, p2] = size(X2);
[n3, p3] = size(X3);
[n4, p4] = size(X4);

y1 = [ones(n1,1) -ones(n1,1) -ones(n1,1) -ones(n1,1)]; 
y2 = [-ones(n2,1) ones(n2,1) -ones(n2,1) -ones(n2,1)];
y3 = [-ones(n3,1) -ones(n3,1) ones(n3,1) -ones(n3,1)];
y4 = [-ones(n4,1) -ones(n4,1) -ones(n4,1) ones(n4,1)];
yi = [y1 ; y2 ; y3 ; y4];
    


%% 2 - One versus all  support vector machine (1vsAll SVM)

% Build 4 SVM (one class vs the 3 others);

kernel='gaussian' ; d=1;
C=10e9;
lambda=1e-6;
[xsup1, w1, w01, ind_sup1, a1] = svmclass(Xapp, yi(:,1), C, lambda, kernel, d, 0);
[xsup2, w2, w02, ind_sup2, a2] = svmclass(Xapp, yi(:,2), C, lambda, kernel, d, 0);
[xsup3, w3, w03, ind_sup3, a3] = svmclass(Xapp, yi(:,3), C, lambda, kernel, d, 0);
[xsup4, w4, w04, ind_sup4, a4] = svmclass(Xapp, yi(:,4), C, lambda, kernel, d, 0);

% Retrieve all the support vectors

vsup = [ind_sup1; ind_sup2; ind_sup3; ind_sup4];

% Calculate the prediction of the 1vsAll SVM on the test set

ypred1 = svmval(Xtest, xsup1, w1, w01, kernel, d);
ypred2 = svmval(Xtest, xsup2, w2, w02, kernel, d);
ypred3 = svmval(Xtest, xsup3, w3, w03, kernel, d);
ypred4 = svmval(Xtest, xsup4, w4, w04, kernel, d);

[v yc] = max([ypred1, ypred2, ypred3, ypred4]');

% Calculate the error rate on the test set

nbbienclasse=length(find(Ytest == yc'));
freq_err = 1 - nbbienclasse /(nt);
perc_err = freq_err * 100

fprintf('Erreur de classification avec un SVM one v. all (données Lomita réduites) : %f %% \n',  perc_err);
fprintf('Kernel utilisé : %s \n',kernel);


%% Matrice de confusion
nbClasses = length(unique(Y));
matConf = zeros(nbClasses, nbClasses);
for i = 1:nbClasses
	for j = 1:nbClasses
		matConf(i, j) = sum((yc' == j) .* (Ytest == i));
	end
end

matConf