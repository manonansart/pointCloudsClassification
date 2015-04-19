clear all
close

attributes = load('../dataset/dish_area_dataset/attributes.csv');
[nb1, nb2, labels] = textread('../dataset/dish_area_labels.txt', '%f %f %s');
labelsNum = strcmp(labels, 'background') + 2 * strcmp(labels, 'car');
indicesCar = find(labelsNum == 2);
indicesBack = find(labelsNum == 1);

disp('Mean of the minimum of intensity for cars VS background:');
disp([mean(attributes(indicesCar, 1)) mean(attributes(indicesBack, 1))]);
disp('Mean of the maximum of intensity for cars VS background:');
disp([mean(attributes(indicesCar, 2)) mean(attributes(indicesBack, 2))]);
disp('Mean of the mean of intensity for cars VS background:');
disp([mean(attributes(indicesCar, 3)) mean(attributes(indicesBack, 3))]);
disp('Mean of the variance of intensity for cars VS background:');
disp([mean(attributes(indicesCar, 4)) mean(attributes(indicesBack, 4))]);