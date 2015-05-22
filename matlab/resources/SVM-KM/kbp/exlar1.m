%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

% Multiple Kernel Estimation using the KBP 
% the plotting describes the regularization path
%
%
% Paper :
% V. Guigue, A. Rakotomamonjy, S. Canu, Kernel Basis Pursuit. European Conference on Machine Learning, Porto, 2005.
% 

% 18/12/2005 AR

close all;
clear all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% signals to be estimated

nbapp=200;
nbtest=500;
sigma=0.1;
xapp=sort(rand(nbapp,1)*4);
xtest=linspace(0,4,nbtest)';
yapp=cos(exp(xapp)) + sigma*randn(nbapp,1);
ytest=cos(exp(xtest));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% multiple kernels

kernel='gaussian';
kerneloption=[0.05 0.1 0.5 1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAR  PARAMETRE

%

LARMKpar{1}.type          = 'nbSV';
LARMKpar{1}.borne         = [1: 20];

%% uncomment this for having the regularization path involving this two 
%% stopping criterion

% LARMKpar{2}.type          = 'trapscale';
% LARMKpar{2}.borne         = [1:3];
% LARMKpar{2}.indTS         = length(kerneloption)+1;
% kerneloption=[kerneloption 0.001];

Limites=[];
verbose   = 0;
lambda    = 1e-9;

% Learning
Kapp  = multiplekernel(xapp,kernel,kerneloption);
[solution, solution_OLS] = LAR(Kapp,yapp, LARMKpar, Limites, lambda, verbose);




% Test
for i=1:length(solution)
    indxsup=solution_OLS{i}.indxsup;
    Ktest  = multiplekernel(xtest,kernel,kerneloption,xapp,solution_OLS{i});
    ypred=LARval(Ktest,solution_OLS{i});
    plot(xapp,yapp,'b',xtest,ytest,'g',xtest,ypred,'k'); 
    legend('Training points','Test curve', 'Test prediction'); 
    title(['step : ' int2str(i)]);  
    
    pause
end;