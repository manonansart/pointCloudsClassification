clear all
close all
clc

%% load data
X = load('../../../dataset/lomita/attributesSmall_without_unlabeled.csv');

% Gets the columns from the teXtest file
Y = load('../../../dataset/lomita/labelsSmall_without_unlabeled.csv');

data = [X Y];

X1 = data(Y==1, :);
X2 = data(Y==2, :);
X3 = data(Y==3, :);
X4 = data(Y==4, :);

[n, p] = size(X1);
%% Reduce class background
% random index
i=randperm(n);

% write the size wanted for the class background
size_background_wanted = 200;

% New background (reduced) 
result=X1(i(1:size_background_wanted), :)

%% Write in a csv
newData = [result; X2; X3; X4];
csvwrite('../../../dataset/lomita/attributes_and_labels_without_unlabeled_and_reduced_background_small.csv', newData)

