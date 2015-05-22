%  Example function of ranking
% on toy problem
%


close all
clear all


tic
n=10000;
GDitermax=30;
nbapp=20;
nbGoodVariable=5;
nbdimnoise=50;
N=5;

load('./data/toydata.mat');
load(['./data/CV-nbapp' int2str(nbapp) '-nbdata' int2str(n) '.mat']);
x=x(:,1:(nbGoodVariable+nbdimnoise));
GoodVariable=[1:5];

featseloption.AlphaApprox=0;   % if zero, alpha is calculated at each step 
featseloption.RemoveChunks='half';
featseloption.StopChunks=10;
featseloption.GDitermax=GDitermax;
featseloption.GDeta=0.01;
featseloption.GDthresh=0.01;
%----------------------Learning Parameters -------------------
C =10;  lambda = 1e-8; 
epsilon = 0.01; % 0.5
kerneloption =10; %5;
kernel='gaussian';
verbose=0;
tic
%------------------------------------------------------------
for i=1:N;
    
    fprintf('%d',i);
    indtest=1:n; indtest(indapp(i,:))=[];
    xapp=x(indapp(i,:),:);
    yapp=y(indapp(i,:));
    xtest=x(indtest,:);
    ytest=y(indtest);
    [xapp,xtest]=normalizemeanstd(xapp,xtest);
   [RankedVariables(i,:)]=FeatSelregalpha(xapp,yapp,C,epsilon,kernel,kerneloption,verbose,featseloption);
  % [RankedVariables(i,:)]=FeatSelregr2w2GD(xapp,yapp,C,epsilon,kernel,kerneloption,verbose,featseloption);

   %[RankedVariables(i,:)]=FeatSelregspanboundGDrandom(xapp,yapp,C,epsilon,kernel,kerneloption,verbose,featseloption);
     NbGoodRank(i)=length(intersect(GoodVariable,RankedVariables(i,1:nbGoodVariable)));
   
end
fprintf('\n----------------10 Top Ranked Variables---------------- \n');
RankedVariables(:,1:10)
mean(NbGoodRank)
toc