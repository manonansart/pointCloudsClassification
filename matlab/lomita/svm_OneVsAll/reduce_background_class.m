clear all
close all
clc

%% load data
data = load('../../../dataset/lomita/attributes_and_labels_without_unlabeled.csv');
X = data(:, 1:10);
Y = data(:, 11);
X1 = data(Y==1, :);
X2 = data(Y==2, :);
X3 = data(Y==3, :);
X4 = data(Y==4, :);

[n, p] = size(X1);
%% Reduce class background
% random index
i=randperm(n);

% write the size wanted for the class background
size_background_wanted = 7000;

% New background (reduced) 
result=X1(i(1:size_background_wanted), :)

%% Write in a csv
newData = [result; X2; X3; X4];
csvwrite('../../../dataset/lomita/attributes_and_labels_without_unlabeled_and_reduced_background.csv', newData)

