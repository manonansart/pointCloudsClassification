close all
clear all

data = load('../../../dataset/dish_area_dataset/attributes.csv');
Y = load('../../../dataset/dish_area_dataset/labels.csv');

clusters = kmeans(data, 2);


nbClasses = 2;
nvLabels = [];
for i = 1:nbClasses
	compteur = [];
	for j = 1:nbClasses
		compteur = [compteur sum((Y == i) .* (clusters == j))];
	end
	[tmp nvLabel] = max(compteur);
	nvLabels = [nvLabels nvLabel];
end

nvClusters = clusters;
for i = 1:nbClasses
	nvClusters(find(clusters == nvLabels(i))) = i;
end

% Calculate the error rate
erreur = sum(Y ~= clusters) / length(clusters) * 100

% Matrice de confusion
disp(strcat('Prediction: background | Truth : background : ', num2str(sum((clusters == 1) .* (Y == 1)))))
disp(strcat('Prediction: background | Truth : car : ', num2str(sum((clusters == 1) .* (Y == 2)))))
disp(strcat('Prediction: car | Truth : car : ', num2str(sum((clusters == 2) .* (Y == 2)))))
disp(strcat('Prediction: car | Truth : background : ', num2str(sum((clusters == 2) .* (Y == 1)))))

nbClasses = length(unique(Y));
matConf = zeros(nbClasses, nbClasses);
for i = 1:nbClasses
	for j = 1:nbClasses
		matConf(i, j) = sum((clusters == j) .* (Y == i));
	end
end

matConf