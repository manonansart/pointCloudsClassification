close all
clear all

data = load('../../dataset/lomita/attributes_without_unlabeled.csv');
Y = load('../../dataset/lomita/labels_without_unlabeled.csv');

tic
clusters = kmeans(data, 2);


nbClasses = 4;
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
erreur = sum(Y ~= nvClusters) / length(clusters) * 100

% Matrice de confusion
disp(strcat('Truth: background | Good prediction : ', num2str(sum((clusters == 1) .* (Y == 1)))))
disp(strcat('Truth: background | Wrong prediction : ', num2str(sum((Y == 1)) - sum((clusters == 1) .* (Y == 1)))))
disp(strcat('Truth: bicyclist | Good prediction : ', num2str(sum((clusters == 2) .* (Y == 2)))))
disp(strcat('Truth: bicyclist | Wrong prediction : ', num2str(sum((Y == 2)) - sum((clusters == 2) .* (Y == 2)))))
disp(strcat('Truth: car | Good prediction : ', num2str(sum((clusters == 3) .* (Y == 3)))))
disp(strcat('Truth: car | Wrong prediction : ', num2str(sum((Y == 3)) - sum((clusters == 3) .* (Y == 3)))))
disp(strcat('Truth: pedestrian | Good prediction : ', num2str(sum((clusters == 4) .* (Y == 4)))))
disp(strcat('Truth: pedestrian | Wrong prediction : ', num2str(sum((Y == 4)) - sum((clusters == 4) .* (Y == 4)))))
toc