% Possible labels for dataset lomita : background bicyclist car pedestrian unlabeled

% Gets the columns from the text file
[nb1, labels, tmp] = textread('../../dataset/lomita/labels.txt', '%f %s %s');

% Replaces the string labels numbers
labelsNum = strcmp(labels, 'background') + 2 * strcmp(labels, 'bicyclist') + 3 * strcmp(labels, 'car') + 4 * strcmp(labels, 'pedestrian') + 5 * strcmp(labels, 'unlabeled');

% Saves the labels with csv format
csvwrite('../../dataset/lomita/labels.csv', labelsNum);

data = load('../dataset/dish_area_dataset/attributes_2.csv');
