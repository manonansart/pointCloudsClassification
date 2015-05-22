function [AUC,tpr,fpr,b]=rankroccurve(ypred,ytest)

% USAGE
%
% [AUC,tpr,fpr,b]=rankroccurve(ypred,ytest)
%
% the outputs
%
% AUC       Area under curve value
% tpr,fpr   true positive and false positive vector for ROC curve plotting purpose
% b         a new bias for the decision function. b can replace w0 and it has been
%           processed so that the decision function corresponds to the one where the roc
%            curve and the (1,0)-(0,1) diagonal meets
%

% 30/07/2004 A. Rakotomamonjy


eps=1e-7;
classcode=[1 -1];
w0=0;

npos=length(find(ytest==classcode(1)));
nneg=length(find(ytest==classcode(2)));
N=npos+nneg;
  


ytest=ytest>0;
[ypred,ind] = sort(ypred);
ytest       = ytest(ind);


fpr = cumsum(ytest)/sum(ytest);
tpr = cumsum(1-ytest)/sum(1-ytest);
tpr = [0 ; tpr ; 1];
fpr = [0 ; fpr ; 1];
n = size(tpr, 1);
AUC = sum((fpr(2:n) - fpr(1:n-1)).*(tpr(2:n)+tpr(1:n-1)))/2;
b=[min(ypred)-1;ypred];
[aux,indice]=min(abs(1-fpr-tpr)); % intersection entre la courbe roc et la diagonale.
b=w0-b(indice) + eps;
