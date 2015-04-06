close all
clear all

data = load('../dataset/dish_area_dataset/attributes.csv');

clusters = kmeans(data, 2);

[nb1, nb2, labels] = textread('dish_area_labels.txt', '%f %f %s')

labelsNum = strcmp(labels, 'background') + 2 * strcmp(labels, 'car');

err = sum(labelsNum ~= clusters) / length(clusters) * 100;