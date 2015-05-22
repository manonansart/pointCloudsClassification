%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simple example of Multiple Kernel Estimation using the KBP 
% 
%
%
% Paper :
% V. Guigue, A. Rakotomamonjy, S. Canu, Kernel Basis Pursuit. European Conference on Machine Learning, Porto, 2005.
% 
% 18/12/2005
close all;
clear all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fabrication du signal à analyser

nbapp=200;
nbtest=500;
sigma=0.1;
xapp=sort(rand(nbapp,1)*4);
xtest=linspace(0,4,nbtest)';
yapp=cos(exp(xapp)) + sigma*randn(nbapp,1);
ytest=cos(exp(xtest));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fabrication du noyau

kernel='gaussian';
kerneloption=[0.05 0.1 0.5 1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAR  PARAMETRE

% select the type of stopping criterion by uncommenting

LARMKpar{1}.type          = 'nbSV';
LARMKpar{1}.borne         = [20];


%  LARMKpar{1}.type          = 'trapscale';
%  LARMKpar{1}.borne         = [2];
%  LARMKpar{1}.indTS         = length(kerneloption)+1;
%  kerneloption=[kerneloption 0.001];
 


Limites=[];
verbose   = 0;
lambda    = 1e-9;

% Learning
Kapp  = multiplekernel(xapp,kernel,kerneloption);
[Kapp,meanK,stdK]=normalizekernelLAR(Kapp);


[solution, solution_OLS] = LAR(Kapp,yapp, LARMKpar, Limites, lambda, verbose);
ypredapp=LARval(Kapp,solution{1});


% Test 
Ktest  = multiplekernel(xtest,kernel,kerneloption,xapp,solution_OLS{1});
[Ktest]=normalizekernelLAR(Ktest,meanK,stdK,solution_OLS{1});
ypred=LARval(Ktest,solution_OLS{1});


% Plotting stuff
plot(xapp,yapp,'b',xtest,ytest,'g',xtest,ypred,'k'); %,xapp,ypredapp,'r'
legend('Training points','Test curve', 'Test prediction'); %,'Learning prediction'