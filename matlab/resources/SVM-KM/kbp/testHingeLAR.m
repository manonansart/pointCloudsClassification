
clear all;
close all;

randn('seed',2);
rand('seed',2);

tol_cor        = 1e-3;
tol_diff_corA  = 1e-3;

sigma = 2;
nbapp = 200;
nbtest = 3;


ngrid = 50;
[xapp,yapp,xtest,ytest]=dataset('checkers',nbapp,nbtest,sigma);
[xgrid_1,xgrid_2] = meshgrid(linspace(-2,2,ngrid));
xgrid = [reshape(xgrid_1,ngrid*ngrid,1) reshape(xgrid_2,ngrid*ngrid,1)];



kernel = 'gaussian';
kerneloption = [0.1 0.3 0.9];
regterm = 1e-6;

n = nbapp;
y = yapp;

xforplot = xapp;


x = [];
xt = [];
for i=1:length(kerneloption)
    x   = [x svmkernel(xapp,kernel,kerneloption(i))];
    xt  = [xt svmkernel(xgrid,kernel,kerneloption(i),xapp)];
end
var = std(x,[],1);
% K = K - ones(size(K,1),1)*moy;
x = x ./ (ones(size(x,1),1)*var);
xt = xt ./ (ones(size(xt,1),1)*var);

AllBorne{1}.type = 'nbSV';
AllBorne{1}.borne = [2:2:30];
lambda = regterm;
verbose=1;
[solution, solution_OLS] = HingeLAR(x,y, AllBorne, [], lambda,verbose);

for i=1:length(solution),
    ft = LARval(xt, solution{i});
%     [solution{i}.indxsup'; solution{i}.Beta]
    plot2Ddec(xforplot, y, solution{i}.indxsup, xgrid_1, xgrid_2, ft,1,1);
     pause(0.3)
end
