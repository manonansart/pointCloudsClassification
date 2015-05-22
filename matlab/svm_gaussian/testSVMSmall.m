close all
clear all
clc

% Note: don't forget to add splitdata, monqp and svmkernel to the path.
% Results: first run : error rate 0.16% and runtime 360s
%          second run : error rate 0.15% and runtime 384s

%%  Load the data and replace text labels

data = load('../../dataset/dish_area_dataset/attributesSmall.csv');

% Gets the columns from the text file
Y = load('../../dataset/dish_area_dataset/labelsSmall.csv');

% Replaces the label background (1) with -1 and the label car (2) with 1
Y = (Y - 1) * 2 - 1;

%% Split the data into app, val and test
[Xapp, Yapp, Xtest, Ytest] = splitdata(data, Y, 0.75);

%% Train svm with validation
tic
	[alpha, b, pos, moyenne, variance] = svm_train(Xapp, Yapp);
toc

%% Error calculations
% Center and reduce test set
[nTest, p] = size(Xtest);
Xtest = (Xtest - ones(nTest, 1) * moyenne) ./ (ones(nTest, 1) * variance);

% Calculate new labels
kernel = 'gaussian';
kerneloption = 2;

ypred = svm_predict(Xtest, kernel, kerneloption, alpha, b, pos, Xapp, Yapp);

% Calculate the final error rate (in percent)
erreur = svm_error(ypred, Ytest)

disp(strcat('Prediction: background | Truth : background : ', num2str(sum((ypred == -1) .* (Ytest == -1)))))
disp(strcat('Prediction: background | Truth : car : ', num2str(sum((ypred == -1) .* (Ytest == 1)))))
disp(strcat('Prediction: car | Truth : car : ', num2str(sum((ypred == 1) .* (Ytest == 1)))))
disp(strcat('Prediction: car | Truth : background : ', num2str(sum((ypred == 1) .* (Ytest == -1)))))

nbClasses = length(unique(Y));
matConf = zeros(nbClasses, nbClasses);
for i = 1:nbClasses
	for j = 1:nbClasses
		matConf(i, j) = sum((ypred == j) .* (Ytest == i));
	end
end

matConf