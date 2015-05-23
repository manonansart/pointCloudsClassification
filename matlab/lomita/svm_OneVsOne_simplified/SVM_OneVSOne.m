close all
clear all
clc

% Note: don't forget to add SVM-KM and splitdata to the path.
% With this svm 1 vs 1, the validation for C is made on the whole multiclass svm, not on each svm individually.
% The result is that we have one C for all svm and that it is less time consuming.

% About 4.5% error (on 1 run) with the small dataset w/o unlabeled but with the
% class background over-represented.
% About 0% error (on 1 run) with the big dataset w/o unlabeled and with the
% class background balanced.


%% Load and split the data
data = load('../../../dataset/lomita/attributes_without_unlabeled_and_reduced_background');
Y = load('../../../dataset/lomita/labels_without_unlabeled_and_reduced_background.csv');


%% Split the data into app, val and test
[Xapp, Yapp, Xtest, Ytest] = splitdata(data, Y, 0.75);
[Xapp, Yapp, Xval, Yval] = splitdata(Xapp, Yapp, 0.7);

[nApp, p] = size(Xapp);
[nVal, p] = size(Xval);
[nTest, p] = size(Xtest);

moyenne = mean(Xapp);
variance = std(Xapp);

% Center and reduce
Xapp = (Xapp - ones(nApp, 1) * moyenne) ./ (ones(nApp, 1) * variance);
Xval = (Xval - ones(nVal, 1) * moyenne) ./ (ones(nVal, 1) * variance);
Xtest = (Xtest - ones(nTest, 1) * moyenne) ./ (ones(nTest, 1) * variance);


tic
% Validation for C : rough C first
C_list = logspace(log10(.01), log10(100), 50);
nbErrMin = inf;
bestC = 0;

for i = 1:length(C_list)
	if (i == length(C_list)/2)
		disp('Hang in there, you are half-way throw the calculation')
	end
	C = C_list(i);
	W = zeros(size(Xapp, 2), 25);
	b = zeros(1, 25);

	% Train all svm
	for i = 1 : 4
		for j = 1 : 5
			if (j > i)
				% Find the data with label i and j and put label -1 and 1
				ind_i = find(Yapp == i);
				ind_j = find(Yapp == j);

				Yapp_tmp = Yapp;

				Yapp_tmp(ind_i, :) = 1;
				Yapp_tmp(ind_j, :) = -1;

				Xapp_ij = [Xapp(ind_i, :); Xapp(ind_j, :)];
				Yapp_ij = [Yapp_tmp(ind_i, :); Yapp_tmp(ind_j, :)];

				[w, b] = svm_train_linear(Xapp_ij, Yapp_ij, C);
				W(:, (i-1)*5 + j) = w;
				B((i-1)*5 + j) = b;
			end
		end
	end

	%% Validation error calculation
	preds = [];

	for i = 1 : 4
		for j = 1 : 5
			if (j > i)
				pred = svm_predict_linear(Xval, W(:, (i-1)*5 + j), B((i-1)*5 + j));
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

	nbErr = sum(pred_final ~= Yval);
	if (nbErr < nbErrMin)
		nbErrMin = nbErr;
		bestC = C;
	end
end

%% Test error calculation
preds = [];

for i = 1 : 4
	for j = 1 : 5
		if (j > i)
			pred = svm_predict_linear(Xtest, W(:, (i-1)*5 + j), B((i-1)*5 + j));
			% Replaces 1 with i and -1 with j
			pred = (pred + 1) / 2 * i - (pred - 1) / 2 * j;
			preds = [preds pred];
		end
	end
end

toc
pred_by_class = [];
for i = 1 : 5
	pred_by_class = [pred_by_class sum((preds == i), 2)];
end

[val, pred_final] = max(pred_by_class');
pred_final = pred_final';

erreur = sum(pred_final ~= Ytest) / nTest * 100
toc

disp(strcat('Truth: background | Good prediction : ', num2str(sum((pred_final == 1) .* (Ytest == 1)))))
disp(strcat('Truth: background | Wrong prediction : ', num2str(sum((Ytest == 1)) - sum((pred_final == 1) .* (Ytest == 1)))))
disp(strcat('Truth: bicyclist | Good prediction : ', num2str(sum((pred_final == 2) .* (Ytest == 2)))))
disp(strcat('Truth: bicyclist | Wrong prediction : ', num2str(sum((Ytest == 2)) - sum((pred_final == 2) .* (Ytest == 2)))))
disp(strcat('Truth: car | Good prediction : ', num2str(sum((pred_final == 3) .* (Ytest == 3)))))
disp(strcat('Truth: car | Wrong prediction : ', num2str(sum((Ytest == 3)) - sum((pred_final == 3) .* (Ytest == 3)))))
disp(strcat('Truth: pedestrian | Good prediction : ', num2str(sum((pred_final == 4) .* (Ytest == 4)))))
disp(strcat('Truth: pedestrian | Wrong prediction : ', num2str(sum((Ytest == 4)) - sum((pred_final == 4) .* (Ytest == 4)))))

nbClasses = length(unique(Y));
matConf = zeros(nbClasses, nbClasses);
for i = 1:nbClasses
	for j = 1:nbClasses
		matConf(i, j) = sum((pred_final == j) .* (Ytest == i));
	end
end

matConf