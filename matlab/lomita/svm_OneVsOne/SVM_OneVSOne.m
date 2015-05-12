close all
clear all
clc

% Note: don't forget to add splitdata, monqp and svmkernel to the path.
% Results: error rate 1%, bestC 4.8 and runtime 30s -> without rotation
% Results: error rate 4%, bestC 9 and runtime 30s -> with rotation
% Resultats can change a lot depanding on the split because the svm is performed on a small dataset
% C varies between 4 and 50, the error rates stays around 1-1.5 and runtime is constant


%%  Load the data and replace text labels

data = load('../../../dataset/lomita/attributesSmall.csv');

% Gets the columns from the text file
Y = load('../../../dataset/lomita/labelsSmall.csv');


%% Split the data into app, val and test
[Xapp, Yapp, Xtest, Ytest] = splitdata(data, Y, 0.75);

W = zeros(size(Xapp, 2), 25);
b = zeros(1, 25);

% Train all svm
for i = 1 : 5
	if (i == 3)
		disp('Hang in there, you are half-way throw the calculation')
	end
	for j = 1 : 5
		if (j ~= i)
			ind_i = find(Yapp == i);
			ind_j = find(Yapp == j);

			Y_tmp = Yapp;

			Y_tmp(ind_i, :) = 1;
			Y_tmp(ind_j, :) = -1;

			X_ij = [Xapp(ind_i, :); Xapp(ind_j, :)];
			Y_ij = [Y_tmp(ind_i, :); Y_tmp(ind_j, :)];

			[w, b, moyenne, variance] = svm_train_linear(X_ij, Y_ij);
			W(:, (i-1)*5 + j) = w;
			B((i-1)*5 + j) = b;
		end	
	end
end

%% Error calculations
% Center and reduce test set
[nTest, p] = size(Xtest);
Xtest = (Xtest - ones(nTest, 1) * mean(Xtest)) ./ (ones(nTest, 1) * std(Xtest));

preds = [];

for i = 1 : 5
	for j = 1 : 5
		if (i ~= j)
			pred = svm_predict_linear(Xtest, W(:, (i-1)*5 + j), B((i-1)*5 + j));
			% Replaces 1 with i and -1 with j
			pred = (pred + 1) / 2 * i - (pred - 1) / 2 * j;
			preds = [preds pred];
		end
	end
end


pred_by_class = [];
for i = 1 : 5
	pred_by_class = [pred_by_class sum((preds == i), 2)];
end

[val, pred_final] = max(pred_by_class');
pred_final = pred_final';

erreur = sum(pred_final ~= Ytest) / nTest * 100


disp(strcat('Prediction: background | True : ', num2str(sum((pred_final == 1) .* (Ytest == 1)))))
disp(strcat('Prediction: background | False : ', num2str(sum((Ytest == 1)) - sum((pred_final == 1) .* (Ytest == 1)))))
disp(strcat('Prediction: bicyclist | True : ', num2str(sum((pred_final == 2) .* (Ytest == 2)))))
disp(strcat('Prediction: bicyclist | False : ', num2str(sum((Ytest == 2)) - sum((pred_final == 2) .* (Ytest == 2)))))
disp(strcat('Prediction: car | True : ', num2str(sum((pred_final == 3) .* (Ytest == 3)))))
disp(strcat('Prediction: car | False : ', num2str(sum((Ytest == 3)) - sum((pred_final == 3) .* (Ytest == 3)))))
disp(strcat('Prediction: pedestrian | True : ', num2str(sum((pred_final == 4) .* (Ytest == 4)))))
disp(strcat('Prediction: pedestrian | False : ', num2str(sum((Ytest == 4)) - sum((pred_final == 4) .* (Ytest == 4)))))
disp(strcat('Prediction: unlabeled | True : ', num2str(sum((pred_final == 5) .* (Ytest == 5)))))
disp(strcat('Prediction: unlabeled | False : ', num2str(sum((Ytest == 5)) - sum((pred_final == 5) .* (Ytest == 5)))))
