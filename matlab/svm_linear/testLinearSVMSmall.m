close all
clear all
clc

% Note: don't forget to add splitdata, monqp and svmkernel to the path.
% Results: error rate 1%, bestC 4.8 and runtime 30s -> without rotation
% Results: error rate 4%, bestC 9 and runtime 30s -> with rotation
% Resultats can change a lot depanding on the split because the svm is performed on a small dataset
% C varies between 4 and 50, the error rates stays around 1-1.5 and runtime is constant


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
	[w, b, moyenne, variance] = svm_train_linear(Xapp, Yapp);
toc

%% Error calculations
% Center and reduce test set
[nTest, p] = size(Xtest);
Xtest = (Xtest - ones(nTest, 1) * moyenne) ./ (ones(nTest, 1) * variance);

% Calculate new labels
pred = svm_predict_linear(Xtest, w, b);

% Calculate the errors
erreur = svm_error(pred, Ytest);

% Matrice de confusion
disp(strcat('Prediction: background | Truth : background : ', num2str(sum((pred == -1) .* (Ytest == -1)))))
disp(strcat('Prediction: background | Truth : car : ', num2str(sum((pred == -1) .* (Ytest == 1)))))
disp(strcat('Prediction: car | Truth : car : ', num2str(sum((pred == 1) .* (Ytest == 1)))))
disp(strcat('Prediction: car | Truth : background : ', num2str(sum((pred == 1) .* (Ytest == -1)))))

nbClasses = length(unique(Y));
matConf = zeros(nbClasses, nbClasses);
for i = 1:nbClasses
	for j = 1:nbClasses
		matConf(i, j) = sum((pred == j) .* (Ytest == i));
	end
end

matConf