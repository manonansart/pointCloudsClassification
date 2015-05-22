data = load('../../dataset/dish_area_dataset/attributes.csv');

% Gets the columns from the text file
[nb1, nb2, labels] = textread('../../dataset/dish_area_labels.txt', '%f %f %s');

% Replaces the label background with 1 and the label car with 2
labelsNum = strcmp(labels, 'background') + 2 * strcmp(labels, 'car');

[X, Y, tmp, tmp] = splitdata(data, labelsNum, 0.08);

csvwrite('../../dataset/dish_area_dataset/attributesSmall.csv', X);
csvwrite('../../dataset/dish_area_dataset/labelsSmall.csv', Y);