close all
clear all

data = load('../dataset/dish_area_dataset/attributes.csv');

clusters = kmeans(data, 2);

% Gets the columns from the text file
[nb1, nb2, labels] = textread('dish_area_labels.txt', '%f %f %s')

% Replaces the label background with 1 and the label car with 2
labelsNum = strcmp(labels, 'background') + 2 * strcmp(labels, 'car');

% Calculate the error rate
err = sum(labelsNum ~= clusters) / length(clusters) * 100;